use std::path::PathBuf;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Configuration for snapshot operations.
pub struct SnapshotConfig {
    pub cache_dir: PathBuf,
    pub dump_timeout: Duration,
    pub restore_timeout: Duration,
}

/// Identifies a snapshot by its intent hash.
pub struct SnapshotId {
    pub intent_hash: String,
}

/// Persisted metadata for a snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub intent_hash: String,
    pub env_fingerprint: String,
    pub entrypoint: String,
    pub argv: Vec<String>,
    pub python_version: String,
    pub platform: String,
    pub created_at: DateTime<Utc>,
    pub image_size_bytes: u64,
    pub dump_duration_ms: u64,
    pub restore_count: u32,
    pub last_restored_at: Option<DateTime<Utc>>,
}

/// Result of a successful restore.
pub struct RestoredProcess {
    pub pid: u32,
    pub restore_duration_ms: u64,
}
