use std::time::Instant;

use tokio::process::Command;
use tokio::time::timeout;

use crate::error::SnapshotError;
use crate::metadata::{images_dir, snapshot_dir, write_metadata};
use crate::types::{SnapshotConfig, SnapshotId, SnapshotMetadata};

/// Dump a running process using CRIU with `--leave-running`.
///
/// The process continues running after the snapshot is taken.
/// On success, writes metadata and returns the updated metadata.
pub async fn dump_leave_running(
    config: &SnapshotConfig,
    id: &SnapshotId,
    pid: u32,
    mut metadata: SnapshotMetadata,
) -> Result<SnapshotMetadata, SnapshotError> {
    let images = images_dir(config, id);
    let snap_dir = snapshot_dir(config, id);

    std::fs::create_dir_all(&images)?;

    let start = Instant::now();

    let result = timeout(config.dump_timeout, async {
        Command::new("criu")
            .arg("dump")
            .arg("--tree")
            .arg(pid.to_string())
            .arg("--images-dir")
            .arg(&images)
            .arg("--leave-running")
            .arg("--shell-job")
            .arg("--tcp-established")
            .arg("--ext-unix-sk")
            .arg("--file-locks")
            .arg("--log-file")
            .arg(snap_dir.join("dump.log"))
            .arg("-v4")
            .output()
            .await
    })
    .await;

    match result {
        Err(_) => Err(SnapshotError::DumpTimeout {
            timeout: config.dump_timeout,
        }),
        Ok(output_result) => {
            let output = output_result?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let log_contents = std::fs::read_to_string(snap_dir.join("dump.log")).ok();
                return Err(SnapshotError::DumpFailed {
                    message: stderr,
                    log: log_contents,
                });
            }

            let elapsed = start.elapsed();
            metadata.dump_duration_ms = elapsed.as_millis() as u64;

            // Calculate image size
            let mut total_size: u64 = 0;
            if let Ok(entries) = std::fs::read_dir(&images) {
                for entry in entries.flatten() {
                    if let Ok(meta) = entry.metadata() {
                        total_size += meta.len();
                    }
                }
            }
            metadata.image_size_bytes = total_size;

            write_metadata(config, id, &metadata)?;

            Ok(metadata)
        }
    }
}
