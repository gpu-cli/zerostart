use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Manages the status directory for per-package completion signaling.
///
/// Layout:
/// ```text
/// status_dir/
///   installing           # sentinel: exists while daemon is running
///   demand               # append-only file: import hook writes demanded distributions
///   done/                # one empty file per completed distribution
///   failed/              # one file per failed distribution (contains error message)
/// ```
pub struct StatusDir {
    root: PathBuf,
}

impl StatusDir {
    /// Create a new status directory, including done/ and failed/ subdirs.
    pub fn create(root: &Path) -> Result<Self> {
        fs::create_dir_all(root.join("done"))
            .with_context(|| format!("failed to create {}/done", root.display()))?;
        fs::create_dir_all(root.join("failed"))
            .with_context(|| format!("failed to create {}/failed", root.display()))?;
        Ok(Self {
            root: root.to_path_buf(),
        })
    }

    /// Open an existing status directory (no creation).
    pub fn open(root: &Path) -> Result<Self> {
        if !root.exists() {
            anyhow::bail!("status directory does not exist: {}", root.display());
        }
        Ok(Self {
            root: root.to_path_buf(),
        })
    }

    /// Create the `installing` sentinel file.
    pub fn mark_installing(&self) -> Result<()> {
        fs::write(self.root.join("installing"), b"")
            .with_context(|| "failed to create installing sentinel")
    }

    /// Remove the `installing` sentinel (signals all work complete).
    pub fn clear_installing(&self) -> Result<()> {
        let path = self.root.join("installing");
        if path.exists() {
            fs::remove_file(&path).with_context(|| "failed to remove installing sentinel")?;
        }
        Ok(())
    }

    /// Mark a distribution as successfully installed.
    pub fn mark_done(&self, distribution: &str) -> Result<()> {
        let path = self.root.join("done").join(distribution);
        fs::write(&path, b"")
            .with_context(|| format!("failed to write done marker for {distribution}"))
    }

    /// Mark a distribution as failed with an error message.
    pub fn mark_failed(&self, distribution: &str, error: &str) -> Result<()> {
        let path = self.root.join("failed").join(distribution);
        fs::write(&path, error.as_bytes())
            .with_context(|| format!("failed to write failed marker for {distribution}"))
    }

    /// Check if a distribution is done.
    pub fn is_done(&self, distribution: &str) -> bool {
        self.root.join("done").join(distribution).exists()
    }

    /// Check if a distribution has failed.
    pub fn is_failed(&self, distribution: &str) -> bool {
        self.root.join("failed").join(distribution).exists()
    }

    /// Check if the installer is still running.
    pub fn is_installing(&self) -> bool {
        self.root.join("installing").exists()
    }

    /// Get the path to the demand file.
    pub fn demand_path(&self) -> PathBuf {
        self.root.join("demand")
    }

    /// Get the root path.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_status_dir() {
        let tmp = TempDir::new().unwrap();
        let status = StatusDir::create(tmp.path()).unwrap();

        assert!(tmp.path().join("done").is_dir());
        assert!(tmp.path().join("failed").is_dir());
        assert!(!status.is_installing());
    }

    #[test]
    fn test_installing_lifecycle() {
        let tmp = TempDir::new().unwrap();
        let status = StatusDir::create(tmp.path()).unwrap();

        assert!(!status.is_installing());
        status.mark_installing().unwrap();
        assert!(status.is_installing());
        status.clear_installing().unwrap();
        assert!(!status.is_installing());
    }

    #[test]
    fn test_done_marker() {
        let tmp = TempDir::new().unwrap();
        let status = StatusDir::create(tmp.path()).unwrap();

        assert!(!status.is_done("torch"));
        status.mark_done("torch").unwrap();
        assert!(status.is_done("torch"));
        assert!(!status.is_failed("torch"));
    }

    #[test]
    fn test_failed_marker() {
        let tmp = TempDir::new().unwrap();
        let status = StatusDir::create(tmp.path()).unwrap();

        assert!(!status.is_failed("badpkg"));
        status
            .mark_failed("badpkg", "download failed: 404")
            .unwrap();
        assert!(status.is_failed("badpkg"));
        assert!(!status.is_done("badpkg"));

        // Check error message is stored
        let msg = fs::read_to_string(tmp.path().join("failed").join("badpkg")).unwrap();
        assert_eq!(msg, "download failed: 404");
    }

    #[test]
    fn test_clear_installing_idempotent() {
        let tmp = TempDir::new().unwrap();
        let status = StatusDir::create(tmp.path()).unwrap();

        // Should not error even if installing doesn't exist
        status.clear_installing().unwrap();
        status.clear_installing().unwrap();
    }

    #[test]
    fn test_open_nonexistent() {
        let result = StatusDir::open(Path::new("/tmp/nonexistent-zs-test-dir-12345"));
        assert!(result.is_err());
    }

    #[test]
    fn test_demand_path() {
        let tmp = TempDir::new().unwrap();
        let status = StatusDir::create(tmp.path()).unwrap();
        assert_eq!(status.demand_path(), tmp.path().join("demand"));
    }
}
