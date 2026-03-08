//! Resolve requirements to wheel URLs via `uv pip compile` + PyPI JSON API.
//!
//! This runs entirely without Python — just needs `uv` on PATH.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::Write;
use std::process::Command;

use crate::manifest::WheelSpec;

/// Resolve requirements text to wheel specs.
///
/// Accepts newline-separated requirements (like requirements.txt content).
/// Shells out to `uv pip compile` for resolution, then looks up each
/// package on PyPI JSON API for download URLs and sizes.
pub async fn resolve_requirements(
    requirements_text: &str,
    python_version: &str,
    platform: &str,
) -> Result<Vec<WheelSpec>> {
    let requirements: Vec<&str> = requirements_text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with('-'))
        .collect();

    if requirements.is_empty() {
        return Ok(Vec::new());
    }

    // Step 1: Resolve with uv pip compile
    let resolved = uv_resolve(&requirements, python_version, platform)?;
    if resolved.is_empty() {
        anyhow::bail!("uv pip compile returned no results");
    }

    // Step 2: Look up wheel URLs from PyPI JSON API (concurrent)
    let client = reqwest::Client::new();
    let mut handles = Vec::new();

    for (dist, version) in resolved {
        let client = client.clone();
        let py_ver = python_version.to_string();
        let plat = platform.to_string();

        handles.push(tokio::spawn(async move {
            match lookup_pypi_wheel(&client, &dist, &version, &py_ver, &plat).await {
                Ok(spec) => Some(spec),
                Err(e) => {
                    tracing::warn!("could not find wheel for {dist}=={version}: {e}");
                    None
                }
            }
        }));
    }

    let mut specs = Vec::new();
    for handle in handles {
        if let Some(spec) = handle.await? {
            specs.push(spec);
        }
    }

    Ok(specs)
}

/// Parse requirements from a pyproject.toml string.
///
/// Extracts `[project.dependencies]` list.
pub fn parse_pyproject_dependencies(content: &str) -> Result<Vec<String>> {
    // Simple TOML parsing — just extract the dependencies array
    // We use a minimal approach since we only need [project.dependencies]
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

/// Run `uv pip compile` to resolve requirements to pinned versions.
fn uv_resolve(
    requirements: &[&str],
    python_version: &str,
    platform: &str,
) -> Result<Vec<(String, String)>> {
    // Write requirements to temp file
    let mut tmp = tempfile::NamedTempFile::new().context("failed to create temp file")?;
    for req in requirements {
        writeln!(tmp, "{req}")?;
    }
    tmp.flush()?;

    let platform_tag = format!("x86_64-unknown-{platform}-gnu");

    let output = Command::new("uv")
        .args([
            "pip",
            "compile",
            tmp.path().to_str().unwrap_or("-"),
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
    let mut resolved = Vec::new();

    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Remove inline comments
        let line = line.split('#').next().unwrap_or("").trim();
        // Parse "package==version"
        if let Some((name, version)) = line.split_once("==") {
            resolved.push((name.to_string(), version.to_string()));
        }
    }

    Ok(resolved)
}

/// PyPI JSON API response (partial).
#[derive(Deserialize)]
struct PyPIResponse {
    urls: Vec<PyPIFile>,
}

#[derive(Deserialize)]
struct PyPIFile {
    filename: String,
    url: String,
    #[serde(default)]
    size: u64,
}

/// Look up the best wheel URL from PyPI JSON API.
async fn lookup_pypi_wheel(
    client: &reqwest::Client,
    distribution: &str,
    version: &str,
    python_version: &str,
    platform: &str,
) -> Result<WheelSpec> {
    let url = format!("https://pypi.org/pypi/{distribution}/{version}/json");
    let resp = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await?
        .error_for_status()?;
    let data: PyPIResponse = resp.json().await?;

    let py_tag = format!("cp{}", python_version.replace('.', ""));

    // Priority: platform-specific + python version > none-any > first .whl
    let mut best: Option<&PyPIFile> = None;

    for f in &data.urls {
        if !f.filename.ends_with(".whl") {
            continue;
        }
        if platform == "linux" && f.filename.contains("linux") && f.filename.contains(&py_tag) {
            best = Some(f);
            break;
        }
        if f.filename.contains("none-any") && best.is_none() {
            best = Some(f);
        }
    }

    // Fallback to first .whl
    if best.is_none() {
        best = data.urls.iter().find(|f| f.filename.ends_with(".whl"));
    }

    let file = best.context(format!("no wheel found for {distribution}=={version}"))?;

    let import_roots = guess_import_roots(distribution);

    Ok(WheelSpec {
        url: file.url.clone(),
        distribution: distribution.to_string(),
        import_roots,
        size: file.size,
        hash: None,
    })
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
    fn test_parse_pyproject_no_deps() {
        let content = r#"
[project]
name = "myapp"
"#;
        let deps = parse_pyproject_dependencies(content).unwrap();
        assert!(deps.is_empty());
    }

    #[test]
    fn test_guess_import_roots() {
        assert_eq!(guess_import_roots("pyyaml"), vec!["yaml"]);
        assert_eq!(guess_import_roots("Pillow"), vec!["PIL"]);
        assert_eq!(guess_import_roots("scikit-learn"), vec!["sklearn"]);
        assert_eq!(guess_import_roots("my-package"), vec!["my_package"]);
        assert_eq!(guess_import_roots("torch"), vec!["torch"]);
    }
}
