use zs_fast_wheel::daemon::{DaemonConfig, DaemonEngine};
use zs_fast_wheel::manifest::Manifest;
use zs_fast_wheel::pipeline;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "zs-fast-wheel", about = "Download + extract Python wheels at maximum speed")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// One-shot install: download and extract wheels
    Install {
        /// Wheel URLs or local .whl files to install
        #[arg(required = true)]
        wheels: Vec<String>,

        /// Target directory to extract into (site-packages equivalent)
        #[arg(short, long, default_value = "site-packages")]
        target: PathBuf,

        /// Number of parallel downloads / Range requests
        #[arg(short = 'j', long, default_value_t = 8)]
        parallel_downloads: usize,

        /// Number of parallel extraction threads
        #[arg(short = 'x', long, default_value_t = num_cpus())]
        extract_threads: usize,

        /// Use streaming mode: Range requests per entry, overlap download+decompress
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        stream: bool,

        /// Pre-allocate files with fallocate (reduces fragmentation)
        #[arg(long, default_value_t = true)]
        fallocate: bool,

        /// Skip fsync per file (single sync at end)
        #[arg(long, default_value_t = true)]
        batch_sync: bool,

        /// Benchmark mode: report detailed timings
        #[arg(long)]
        benchmark: bool,
    },

    /// Daemon mode: progressive per-package install from manifest
    Daemon {
        /// Path to manifest JSON file
        #[arg(long)]
        manifest: PathBuf,

        /// Number of parallel downloads
        #[arg(short = 'j', long, default_value_t = 8)]
        parallel_downloads: usize,

        /// Number of parallel extraction threads
        #[arg(short = 'x', long, default_value_t = num_cpus())]
        extract_threads: usize,
    },
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Install {
            wheels,
            target,
            parallel_downloads,
            extract_threads,
            stream,
            fallocate,
            batch_sync,
            benchmark,
        } => {
            std::fs::create_dir_all(&target)?;

            let config = pipeline::PipelineConfig {
                target,
                parallel_downloads,
                extract_threads,
                use_stream: stream,
                use_fallocate: fallocate,
                batch_sync,
                benchmark,
            };

            pipeline::run(wheels, config).await
        }
        Command::Daemon {
            manifest,
            parallel_downloads,
            extract_threads,
        } => {
            let manifest_data = Manifest::from_file(&manifest)?;

            let config = DaemonConfig {
                site_packages: manifest_data.site_packages.clone(),
                parallel_downloads,
                extract_threads,
            };

            let engine = DaemonEngine::new(manifest_data.wheels);

            let start = std::time::Instant::now();
            engine.run(&config).await?;
            let elapsed = start.elapsed();

            let (files, bytes) = engine.extract_stats();
            let (total, done, _, _, failed) = engine.stats().await;

            eprintln!();
            eprintln!("--- fast-wheel daemon summary ---");
            eprintln!("Wheels: {total} ({done} done, {failed} failed)");
            eprintln!(
                "Extracted: {files} files ({:.1} MB)",
                bytes as f64 / 1024.0 / 1024.0
            );
            eprintln!("Total time: {:.1}s", elapsed.as_secs_f64());

            Ok(())
        }
    }
}
