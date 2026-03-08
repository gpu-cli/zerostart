use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};

use zs_snapshot::types::{SnapshotConfig, SnapshotId, SnapshotMetadata};
use zs_snapshot::{doctor, dump_leave_running, restore_detached};

#[derive(Parser)]
#[command(name = "zs-snapshot", about = "CRIU snapshot management for zerostart")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Check CRIU availability and environment
    Doctor,

    /// Dump a running process
    Dump {
        /// Intent hash identifying this snapshot
        #[arg(long)]
        intent_hash: String,

        /// PID of the process to snapshot
        #[arg(long)]
        pid: u32,

        /// Path to metadata JSON file
        #[arg(long)]
        metadata: PathBuf,

        /// Cache directory for snapshots
        #[arg(long, default_value = "~/.cache/zerostart")]
        cache_dir: PathBuf,

        /// Dump timeout in seconds
        #[arg(long, default_value = "120")]
        timeout_secs: u64,

        /// Pass --tcp-established to CRIU (for services with open sockets)
        #[arg(long, default_value = "false")]
        tcp_established: bool,

        /// Pass --ext-unix-sk to CRIU (for services with unix sockets)
        #[arg(long, default_value = "false")]
        ext_unix_sk: bool,
    },

    /// Restore a process from a snapshot
    Restore {
        /// Intent hash identifying the snapshot to restore
        #[arg(long)]
        intent_hash: String,

        /// Path where the restored PID will be written
        #[arg(long)]
        pidfile: PathBuf,

        /// Cache directory for snapshots
        #[arg(long, default_value = "~/.cache/zerostart")]
        cache_dir: PathBuf,

        /// Restore timeout in seconds
        #[arg(long, default_value = "60")]
        timeout_secs: u64,

        /// Pass --tcp-established to CRIU
        #[arg(long, default_value = "false")]
        tcp_established: bool,

        /// Pass --ext-unix-sk to CRIU
        #[arg(long, default_value = "false")]
        ext_unix_sk: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Doctor => {
            let report = doctor();
            let json = serde_json::to_string_pretty(&report)?;
            println!("{json}");
        }

        Commands::Dump {
            intent_hash,
            pid,
            metadata,
            cache_dir,
            timeout_secs,
            tcp_established,
            ext_unix_sk,
        } => {
            let cache_dir = expand_tilde(&cache_dir);
            let config = SnapshotConfig {
                cache_dir,
                dump_timeout: Duration::from_secs(timeout_secs),
                restore_timeout: Duration::from_secs(60),
                tcp_established,
                ext_unix_sk,
            };
            let id = SnapshotId { intent_hash };

            let meta_contents = std::fs::read_to_string(&metadata)?;
            let meta: SnapshotMetadata = serde_json::from_str(&meta_contents)?;

            let result = dump_leave_running(&config, &id, pid, meta).await?;
            let json = serde_json::to_string_pretty(&result)?;
            println!("{json}");
        }

        Commands::Restore {
            intent_hash,
            pidfile,
            cache_dir,
            timeout_secs,
            tcp_established,
            ext_unix_sk,
        } => {
            let cache_dir = expand_tilde(&cache_dir);
            let config = SnapshotConfig {
                cache_dir,
                dump_timeout: Duration::from_secs(120),
                restore_timeout: Duration::from_secs(timeout_secs),
                tcp_established,
                ext_unix_sk,
            };
            let id = SnapshotId { intent_hash };

            let result = restore_detached(&config, &id, &pidfile).await?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "pid": result.pid,
                    "restore_duration_ms": result.restore_duration_ms,
                }))?
            );
        }
    }

    Ok(())
}

fn expand_tilde(path: &std::path::Path) -> PathBuf {
    let s = path.to_string_lossy();
    if let Some(stripped) = s.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home).join(stripped);
    }
    path.to_path_buf()
}
