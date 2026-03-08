use std::path::PathBuf;

use crate::error::SnapshotError;
use crate::types::{SnapshotConfig, SnapshotId, SnapshotMetadata};

/// Returns the root directory for a given snapshot.
pub fn snapshot_dir(config: &SnapshotConfig, id: &SnapshotId) -> PathBuf {
    config.cache_dir.join("snapshots").join(&id.intent_hash)
}

/// Returns the CRIU images directory for a given snapshot.
pub fn images_dir(config: &SnapshotConfig, id: &SnapshotId) -> PathBuf {
    snapshot_dir(config, id).join("images")
}

/// Loads snapshot metadata from disk. Returns `Ok(None)` if the snapshot
/// directory or metadata file does not exist.
pub fn load_metadata(
    config: &SnapshotConfig,
    id: &SnapshotId,
) -> Result<Option<SnapshotMetadata>, SnapshotError> {
    let meta_path = snapshot_dir(config, id).join("metadata.json");
    if !meta_path.exists() {
        return Ok(None);
    }
    let contents = std::fs::read_to_string(&meta_path)?;
    let metadata: SnapshotMetadata = serde_json::from_str(&contents)?;
    Ok(Some(metadata))
}

/// Writes snapshot metadata to disk, creating directories as needed.
pub fn write_metadata(
    config: &SnapshotConfig,
    id: &SnapshotId,
    metadata: &SnapshotMetadata,
) -> Result<(), SnapshotError> {
    let dir = snapshot_dir(config, id);
    std::fs::create_dir_all(&dir)?;
    let meta_path = dir.join("metadata.json");
    let json = serde_json::to_string_pretty(metadata)?;
    std::fs::write(&meta_path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::time::Duration;

    #[test]
    fn metadata_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let config = SnapshotConfig {
            cache_dir: tmp.path().to_path_buf(),
            dump_timeout: Duration::from_secs(60),
            restore_timeout: Duration::from_secs(30),
        };
        let id = SnapshotId {
            intent_hash: "abc123".to_string(),
        };

        // No metadata yet
        let result = load_metadata(&config, &id).unwrap();
        assert!(result.is_none());

        let metadata = SnapshotMetadata {
            intent_hash: "abc123".to_string(),
            env_fingerprint: "fp-001".to_string(),
            entrypoint: "python".to_string(),
            argv: vec!["serve.py".to_string()],
            python_version: "3.11.5".to_string(),
            platform: "linux-x86_64".to_string(),
            created_at: Utc::now(),
            image_size_bytes: 1024,
            dump_duration_ms: 500,
            restore_count: 0,
            last_restored_at: None,
        };

        write_metadata(&config, &id, &metadata).unwrap();

        let loaded = load_metadata(&config, &id).unwrap().unwrap();
        assert_eq!(loaded.intent_hash, "abc123");
        assert_eq!(loaded.env_fingerprint, "fp-001");
        assert_eq!(loaded.entrypoint, "python");
        assert_eq!(loaded.argv, vec!["serve.py"]);
        assert_eq!(loaded.python_version, "3.11.5");
        assert_eq!(loaded.image_size_bytes, 1024);
        assert_eq!(loaded.dump_duration_ms, 500);
        assert_eq!(loaded.restore_count, 0);
        assert!(loaded.last_restored_at.is_none());
    }
}
