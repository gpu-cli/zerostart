use std::path::Path;
use std::time::Instant;

use tokio::process::Command;
use tokio::time::timeout;

use crate::error::SnapshotError;
use crate::metadata::{images_dir, snapshot_dir};
use crate::types::{RestoredProcess, SnapshotConfig, SnapshotId};

/// Restore a process from a CRIU snapshot in detached mode.
///
/// The restored process runs in the background. Its PID is written to `pidfile`
/// and returned in `RestoredProcess`.
pub async fn restore_detached(
    config: &SnapshotConfig,
    id: &SnapshotId,
    pidfile: &Path,
) -> Result<RestoredProcess, SnapshotError> {
    let images = images_dir(config, id);
    let snap_dir = snapshot_dir(config, id);

    if !images.exists() {
        return Err(SnapshotError::NotFound {
            intent_hash: id.intent_hash.clone(),
        });
    }

    let start = Instant::now();

    let result = timeout(config.restore_timeout, async {
        Command::new("criu")
            .arg("restore")
            .arg("--images-dir")
            .arg(&images)
            .arg("--shell-job")
            .arg("--tcp-established")
            .arg("--ext-unix-sk")
            .arg("--file-locks")
            .arg("--restore-detached")
            .arg("--pidfile")
            .arg(pidfile)
            .arg("--log-file")
            .arg(snap_dir.join("restore.log"))
            .arg("-v4")
            .output()
            .await
    })
    .await;

    match result {
        Err(_) => Err(SnapshotError::RestoreTimeout {
            timeout: config.restore_timeout,
        }),
        Ok(output_result) => {
            let output = output_result?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let log_contents = std::fs::read_to_string(snap_dir.join("restore.log")).ok();
                return Err(SnapshotError::RestoreFailed {
                    message: stderr,
                    log: log_contents,
                });
            }

            let elapsed = start.elapsed();

            let pid_str =
                std::fs::read_to_string(pidfile).map_err(|e| SnapshotError::RestoreFailed {
                    message: format!("failed to read pidfile: {e}"),
                    log: None,
                })?;
            let pid: u32 = pid_str
                .trim()
                .parse()
                .map_err(|e| SnapshotError::RestoreFailed {
                    message: format!("invalid pid in pidfile: {e}"),
                    log: None,
                })?;

            Ok(RestoredProcess {
                pid,
                restore_duration_ms: elapsed.as_millis() as u64,
            })
        }
    }
}
