use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("criu binary not found in PATH")]
    CriuNotFound,

    #[error("insufficient permissions to run criu (requires root or CAP_SYS_PTRACE)")]
    InsufficientPermissions,

    #[error("snapshot not found: {intent_hash}")]
    NotFound { intent_hash: String },

    #[error("snapshot corrupted: {intent_hash}: {reason}")]
    Corrupted { intent_hash: String, reason: String },

    #[error("criu dump failed: {message}")]
    DumpFailed {
        message: String,
        log: Option<String>,
    },

    #[error("criu restore failed: {message}")]
    RestoreFailed {
        message: String,
        log: Option<String>,
    },

    #[error("criu dump timed out after {timeout:?}")]
    DumpTimeout { timeout: Duration },

    #[error("criu restore timed out after {timeout:?}")]
    RestoreTimeout { timeout: Duration },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Serde(#[from] serde_json::Error),
}
