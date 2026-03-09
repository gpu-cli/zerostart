use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// A single wheel to install, as specified in the manifest.
#[derive(Debug, Clone, Deserialize)]
pub struct WheelSpec {
    /// Download URL for the wheel
    pub url: String,
    /// Distribution name (e.g., "torch", "requests")
    pub distribution: String,
    /// Package version (e.g., "2.4.1")
    #[serde(default)]
    pub version: String,
    /// Top-level import names (e.g., ["torch"], ["PIL"])
    #[serde(default)]
    pub import_roots: Vec<String>,
    /// Wheel size in bytes (used for scheduling priority)
    #[serde(default)]
    pub size: u64,
    /// Hash for integrity verification (e.g., "sha256:abcdef...")
    #[serde(default)]
    pub hash: Option<String>,
}

/// The manifest file written by the Python orchestrator for the Rust daemon.
#[derive(Debug, Deserialize)]
pub struct Manifest {
    /// Target site-packages directory
    pub site_packages: PathBuf,
    /// Wheels to install
    pub wheels: Vec<WheelSpec>,
}

impl Manifest {
    /// Parse a manifest from a JSON file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read manifest: {}", path.display()))?;
        let manifest: Manifest = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse manifest: {}", path.display()))?;

        if manifest.wheels.is_empty() {
            anyhow::bail!("manifest contains no wheels");
        }

        Ok(manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_parse_full_manifest() {
        let tmp = TempDir::new().unwrap();
        let manifest_path = tmp.path().join("manifest.json");
        std::fs::write(
            &manifest_path,
            r#"{
                "site_packages": "/tmp/sp",

                "wheels": [
                    {
                        "url": "https://files.pythonhosted.org/packages/.../torch-2.4.1.whl",
                        "distribution": "torch",
                        "import_roots": ["torch"],
                        "size": 815000000,
                        "hash": "sha256:abcdef"
                    },
                    {
                        "url": "https://files.pythonhosted.org/packages/.../requests-2.32.5.whl",
                        "distribution": "requests",
                        "import_roots": ["requests"],
                        "size": 63000
                    }
                ]
            }"#,
        )
        .unwrap();

        let manifest = Manifest::from_file(&manifest_path).unwrap();
        assert_eq!(manifest.wheels.len(), 2);
        assert_eq!(manifest.wheels[0].distribution, "torch");
        assert_eq!(manifest.wheels[0].size, 815000000);
        assert_eq!(
            manifest.wheels[0].hash.as_deref(),
            Some("sha256:abcdef")
        );
        assert_eq!(manifest.wheels[1].distribution, "requests");
        assert!(manifest.wheels[1].hash.is_none());
        assert_eq!(manifest.site_packages, PathBuf::from("/tmp/sp"));
    }

    #[test]
    fn test_parse_minimal_manifest() {
        let tmp = TempDir::new().unwrap();
        let manifest_path = tmp.path().join("manifest.json");
        std::fs::write(
            &manifest_path,
            r#"{
                "site_packages": "/tmp/sp",

                "wheels": [
                    {
                        "url": "https://example.com/pkg-1.0.whl",
                        "distribution": "pkg"
                    }
                ]
            }"#,
        )
        .unwrap();

        let manifest = Manifest::from_file(&manifest_path).unwrap();
        assert_eq!(manifest.wheels.len(), 1);
        assert_eq!(manifest.wheels[0].distribution, "pkg");
        assert_eq!(manifest.wheels[0].size, 0); // default
        assert!(manifest.wheels[0].import_roots.is_empty()); // default
        assert!(manifest.wheels[0].hash.is_none()); // default
    }

    #[test]
    fn test_reject_empty_wheels() {
        let tmp = TempDir::new().unwrap();
        let manifest_path = tmp.path().join("manifest.json");
        std::fs::write(
            &manifest_path,
            r#"{
                "site_packages": "/tmp/sp",

                "wheels": []
            }"#,
        )
        .unwrap();

        let result = Manifest::from_file(&manifest_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no wheels"));
    }

    #[test]
    fn test_reject_invalid_json() {
        let tmp = TempDir::new().unwrap();
        let manifest_path = tmp.path().join("manifest.json");
        std::fs::write(&manifest_path, "not json").unwrap();

        let result = Manifest::from_file(&manifest_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_missing_file() {
        let result = Manifest::from_file(Path::new("/tmp/nonexistent-zs-manifest-12345.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_missing_required_fields() {
        let tmp = TempDir::new().unwrap();
        let manifest_path = tmp.path().join("manifest.json");
        // Missing "distribution" field
        std::fs::write(
            &manifest_path,
            r#"{
                "site_packages": "/tmp/sp",

                "wheels": [{"url": "https://example.com/pkg.whl"}]
            }"#,
        )
        .unwrap();

        let result = Manifest::from_file(&manifest_path);
        assert!(result.is_err());
    }
}
