//! Resolve requirements to wheel URLs via `uv pip compile --format pylock.toml`.
//!
//! Single call to uv gives us URLs, sizes, and hashes — no PyPI lookups needed.
//! This runs entirely without Python — just needs `uv` on PATH.

use anyhow::{Context, Result};
use std::io::Write;
use std::process::Command;

use crate::manifest::WheelSpec;

/// Size threshold: wheels larger than this go through the streaming daemon.
/// Smaller wheels are installed via `uv pip install` (instant from cache).
const DAEMON_THRESHOLD: u64 = 5 * 1024 * 1024; // 5MB

/// Resolved artifacts split into small (uv) and large (daemon) buckets.
pub struct ResolvedPlan {
    /// All resolved wheel specs
    pub all: Vec<WheelSpec>,
    /// Small wheels — install via `uv pip install`
    pub uv_specs: Vec<String>,
    /// Large wheels — stream via daemon
    pub daemon_wheels: Vec<WheelSpec>,
}

/// Resolve requirements text to a plan with URLs, sizes, and hashes.
///
/// Uses `uv pip compile --format pylock.toml` which gives everything in one call.
pub fn resolve_requirements(
    requirements: &[String],
    python_version: &str,
    platform: &str,
) -> Result<ResolvedPlan> {
    if requirements.is_empty() {
        return Ok(ResolvedPlan {
            all: Vec::new(),
            uv_specs: Vec::new(),
            daemon_wheels: Vec::new(),
        });
    }

    let specs = uv_resolve_pylock(requirements, python_version, platform)?;

    let mut uv_specs = Vec::new();
    let mut daemon_wheels = Vec::new();

    for spec in &specs {
        if spec.size > DAEMON_THRESHOLD {
            daemon_wheels.push(spec.clone());
        } else {
            uv_specs.push(format!("{}=={}", spec.distribution, spec.version));
        }
    }

    Ok(ResolvedPlan {
        all: specs,
        uv_specs,
        daemon_wheels,
    })
}

/// Parse requirements from a pyproject.toml string.
///
/// Extracts `[project.dependencies]` list.
pub fn parse_pyproject_dependencies(content: &str) -> Result<Vec<String>> {
    let table: toml::Value =
        toml::from_str(content).context("failed to parse pyproject.toml")?;

    let deps = table
        .get("project")
        .and_then(|p| p.get("dependencies"))
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    Ok(deps)
}

/// Run `uv pip compile --format pylock.toml` and parse the result.
///
/// Returns WheelSpecs with URLs, sizes, and hashes — no PyPI lookups needed.
fn uv_resolve_pylock(
    requirements: &[String],
    python_version: &str,
    platform: &str,
) -> Result<Vec<WheelSpec>> {
    // Write requirements to temp file
    let mut tmp = tempfile::NamedTempFile::new().context("failed to create temp file")?;
    for req in requirements {
        writeln!(tmp, "{req}")?;
    }
    tmp.flush()?;

    let platform_tag = platform_to_uv_tag(platform);

    let output = Command::new("uv")
        .args([
            "pip",
            "compile",
            tmp.path().to_str().unwrap_or("-"),
            "--format",
            "pylock.toml",
            "--python-version",
            python_version,
            "--python-platform",
            &platform_tag,
            "--no-header",
        ])
        .output()
        .context("failed to run uv pip compile — is uv installed?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("uv pip compile failed: {}", &stderr[..stderr.len().min(500)]);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_pylock(&stdout)
}

/// Parse a pylock.toml string into WheelSpecs.
fn parse_pylock(content: &str) -> Result<Vec<WheelSpec>> {
    let table: toml::Value = toml::from_str(content).context("failed to parse pylock.toml")?;

    let packages = table
        .get("packages")
        .and_then(|p| p.as_array())
        .context("no [[packages]] in pylock.toml")?;

    let mut specs = Vec::new();

    for pkg in packages {
        let name = pkg
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let version = pkg
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        // Get wheels array — pick the best one
        let wheels = pkg.get("wheels").and_then(|w| w.as_array());

        if let Some(wheels) = wheels {
            if let Some((url, size, hash)) = pick_best_wheel(wheels) {
                let import_roots = guess_import_roots(&name);
                specs.push(WheelSpec {
                    url,
                    distribution: name,
                    version,
                    import_roots,
                    size,
                    hash,
                });
                continue;
            }
        }

        // sdist-only or no matching wheel — include with empty URL so uv handles it
        let import_roots = guess_import_roots(&name);
        specs.push(WheelSpec {
            url: String::new(),
            distribution: name,
            version,
            import_roots,
            size: 0, // forces into uv_specs bucket (< DAEMON_THRESHOLD)
            hash: None,
        });
    }

    Ok(specs)
}

/// Pick the best wheel from a pylock.toml wheels array.
///
/// uv already filtered for the target platform, so usually there's just one.
/// If multiple, prefer the first (uv orders by preference).
fn pick_best_wheel(wheels: &[toml::Value]) -> Option<(String, u64, Option<String>)> {
    for wheel in wheels {
        let url = wheel.get("url").and_then(|v| v.as_str())?;
        let size = wheel
            .get("size")
            .and_then(|v| v.as_integer())
            .unwrap_or(0) as u64;

        // Extract hash from hashes table (prefer sha256)
        let hash = wheel
            .get("hashes")
            .and_then(|h| h.as_table())
            .and_then(|h| h.get("sha256"))
            .and_then(|v| v.as_str())
            .map(|s| format!("sha256:{s}"));

        return Some((url.to_string(), size, hash));
    }
    None
}

/// Detect the current machine's CPU architecture.
fn detect_arch() -> &'static str {
    std::env::consts::ARCH // "x86_64", "aarch64", etc.
}

/// Convert a platform string to uv's `--python-platform` tag.
fn platform_to_uv_tag(platform: &str) -> String {
    if platform.contains("-unknown-") || platform.contains("-apple-") {
        return platform.to_string();
    }
    let arch = detect_arch();
    match platform {
        "linux" => format!("{arch}-unknown-linux-gnu"),
        "macos" | "macosx" => format!("{arch}-apple-darwin"),
        _ => format!("{arch}-unknown-{platform}-gnu"),
    }
}

/// Guess import root names from distribution name.
fn guess_import_roots(distribution: &str) -> Vec<String> {
    match distribution.to_lowercase().as_str() {
        "pyyaml" => vec!["yaml".to_string()],
        "pillow" => vec!["PIL".to_string()],
        "scikit-learn" => vec!["sklearn".to_string()],
        "python-dateutil" => vec!["dateutil".to_string()],
        "beautifulsoup4" => vec!["bs4".to_string()],
        "attrs" => vec!["attr".to_string(), "attrs".to_string()],
        _ => vec![distribution.replace('-', "_").to_lowercase()],
    }
}

/// Detect the running platform as a short string.
pub fn detect_platform() -> &'static str {
    if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux" // default for GPU containers
    }
}

/// Detect Python version from a python binary.
pub fn detect_python_version(python: &std::path::Path) -> Result<String> {
    let output = Command::new(python)
        .args(["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"])
        .output()
        .context("failed to detect python version")?;

    if !output.status.success() {
        anyhow::bail!("python version detection failed");
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pyproject_dependencies() {
        let content = r#"
[project]
name = "myapp"
version = "0.1.0"
dependencies = [
    "torch>=2.0",
    "transformers",
    "safetensors",
]
"#;
        let deps = parse_pyproject_dependencies(content).unwrap();
        assert_eq!(deps, vec!["torch>=2.0", "transformers", "safetensors"]);
    }

    #[test]
    fn test_parse_pylock() {
        let content = r#"
lock-version = "1.0"
created-by = "uv"

[[packages]]
name = "click"
version = "8.3.1"
wheels = [{ url = "https://example.com/click-8.3.1-py3-none-any.whl", size = 108274, hashes = { sha256 = "abc123" } }]

[[packages]]
name = "torch"
version = "2.0.0"
wheels = [{ url = "https://example.com/torch-2.0.0-cp311-linux_x86_64.whl", size = 800000000, hashes = { sha256 = "def456" } }]
"#;
        let specs = parse_pylock(content).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].distribution, "click");
        assert_eq!(specs[0].size, 108274);
        assert_eq!(specs[0].hash, Some("sha256:abc123".to_string()));
        assert_eq!(specs[1].distribution, "torch");
        assert_eq!(specs[1].size, 800_000_000);
    }

    #[test]
    fn test_guess_import_roots() {
        assert_eq!(guess_import_roots("pyyaml"), vec!["yaml"]);
        assert_eq!(guess_import_roots("Pillow"), vec!["PIL"]);
        assert_eq!(guess_import_roots("my-package"), vec!["my_package"]);
    }
}
