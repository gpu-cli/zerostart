//! Resolve requirements to wheel URLs via `uv pip compile --format pylock.toml`.
//!
//! Single call to uv gives us URLs, sizes, and hashes — no PyPI lookups needed.
//! This runs entirely without Python — just needs `uv` on PATH.

use anyhow::{Context, Result};
use std::io::Write;
use std::process::Command;

use crate::manifest::WheelSpec;

/// Fallback: look up a wheel URL from PyPI JSON API for sdist-only packages.
///
/// Some packages (e.g. vllm with cp38-abi3 wheels) have wheels that
/// `uv pip compile --format pylock.toml` misses. This is a targeted
/// fallback — only called for packages without wheels in pylock output.
fn lookup_pypi_wheel(
    name: &str,
    version: &str,
    python_version: &str,
    platform: &str,
) -> Option<(String, u64)> {
    let url = format!("https://pypi.org/pypi/{name}/{version}/json");
    let resp = reqwest::blocking::get(&url).ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let data: serde_json::Value = resp.json().ok()?;

    let arch = detect_arch();
    let py_major_minor = format!("cp{}", python_version.replace('.', ""));
    let plat_tag = if platform.contains("linux") {
        "manylinux"
    } else if platform.contains("macos") || platform.contains("darwin") {
        "macosx"
    } else {
        platform
    };

    let mut best_abi3: Option<(String, u64)> = None;
    let mut best_specific: Option<(String, u64)> = None;
    let mut best_universal: Option<(String, u64)> = None;

    for file_info in data.get("urls")?.as_array()? {
        let filename = file_info.get("filename")?.as_str()?;
        if !filename.ends_with(".whl") {
            continue;
        }

        let file_url = file_info.get("url")?.as_str()?.to_string();
        let file_size = file_info
            .get("size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        if filename.contains("none-any") {
            if best_universal.is_none() {
                best_universal = Some((file_url, file_size));
            }
            continue;
        }

        // Must match platform + arch
        let arch_match = filename.contains(arch)
            || (platform.contains("macos") && filename.contains("universal2"));
        if !filename.contains(plat_tag) || !arch_match {
            continue;
        }

        // abi3 wheels are compatible with any Python >= the tag
        if filename.contains("-abi3-") {
            if best_abi3.is_none() {
                best_abi3 = Some((file_url, file_size));
            }
            continue;
        }

        // Python-version-specific
        if filename.contains(&format!("-{py_major_minor}-"))
            || filename.contains("-py3-")
        {
            if best_specific.is_none() {
                best_specific = Some((file_url, file_size));
            }
        }
    }

    best_specific.or(best_abi3).or(best_universal)
}

/// Resolved artifacts split by whether we have a wheel URL.
///
/// Packages with wheel URLs go through our streaming daemon (parallel download+extract).
/// Packages without URLs (sdist-only) go through uv pip install (can build from source).
pub struct ResolvedPlan {
    /// All resolved wheel specs
    pub all: Vec<WheelSpec>,
    /// sdist-only packages — install via `uv pip install` (can build from source)
    pub uv_specs: Vec<String>,
    /// Packages with wheel URLs — stream via daemon (parallel download+extract)
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

    let mut specs = uv_resolve_pylock(requirements, python_version, platform)?;

    // For sdist-only packages (url is empty), try PyPI JSON fallback.
    // This catches packages like vllm that have abi3 wheels pylock misses.
    for spec in &mut specs {
        if spec.url.is_empty() && !spec.version.is_empty() {
            if let Some((url, size)) =
                lookup_pypi_wheel(&spec.distribution, &spec.version, python_version, platform)
            {
                tracing::info!(
                    "PyPI fallback found wheel for {}=={} ({:.1} MB)",
                    spec.distribution,
                    spec.version,
                    size as f64 / 1024.0 / 1024.0
                );
                spec.url = url;
                spec.size = size;
            }
        }
    }

    let mut uv_specs = Vec::new();
    let mut daemon_wheels = Vec::new();

    for spec in &specs {
        if spec.url.is_empty() {
            // sdist-only — uv builds from source
            uv_specs.push(format!("{}=={}", spec.distribution, spec.version));
        } else {
            // Has wheel URL — stream via daemon (parallel download+extract)
            daemon_wheels.push(spec.clone());
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
            size: 0,
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
