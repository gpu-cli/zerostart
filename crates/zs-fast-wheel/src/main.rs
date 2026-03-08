use zs_fast_wheel::daemon::{DaemonConfig, DaemonEngine};
use zs_fast_wheel::manifest::Manifest;
use zs_fast_wheel::pipeline;
use zs_fast_wheel::resolve;

use anyhow::{Context, Result};
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
    /// One-shot install: download and extract wheels by URL
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

        /// Use streaming mode
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        stream: bool,

        /// Pre-allocate files with fallocate
        #[arg(long, default_value_t = true)]
        fallocate: bool,

        /// Skip fsync per file
        #[arg(long, default_value_t = true)]
        batch_sync: bool,

        /// Benchmark mode
        #[arg(long)]
        benchmark: bool,
    },

    /// Daemon mode: install from a pre-resolved manifest JSON
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

    /// Warm: resolve requirements and start downloading immediately.
    ///
    /// Accepts requirements as text (stdin, --requirements, or --pyproject).
    /// No Python needed — uses uv for resolution and PyPI for wheel URLs.
    ///
    /// Examples:
    ///   echo "torch\ntransformers" | zs-fast-wheel warm --site-packages /opt/sp
    ///   zs-fast-wheel warm --requirements "torch>=2.0\ntransformers" --site-packages /opt/sp
    ///   zs-fast-wheel warm --pyproject "$(cat pyproject.toml)" --site-packages /opt/sp
    Warm {
        /// Requirements text (newline-separated, like requirements.txt)
        #[arg(long)]
        requirements: Option<String>,

        /// pyproject.toml content (extracts [project.dependencies])
        #[arg(long)]
        pyproject: Option<String>,

        /// Target site-packages directory
        #[arg(long)]
        site_packages: PathBuf,

        /// Python version for resolution (default: 3.11)
        #[arg(long, default_value = "3.11")]
        python_version: String,

        /// Platform for resolution (default: linux)
        #[arg(long, default_value = "linux")]
        platform: String,

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
            run_engine(manifest_data.wheels, manifest_data.site_packages, parallel_downloads, extract_threads).await
        }

        Command::Warm {
            requirements,
            pyproject,
            site_packages,
            python_version,
            platform,
            parallel_downloads,
            extract_threads,
        } => {
            // Get requirements text from --requirements, --pyproject, or stdin
            let req_text = if let Some(text) = requirements {
                text
            } else if let Some(toml_text) = pyproject {
                let deps = resolve::parse_pyproject_dependencies(&toml_text)?;
                deps.join("\n")
            } else {
                // Read from stdin
                use std::io::Read;
                let mut buf = String::new();
                std::io::stdin()
                    .read_to_string(&mut buf)
                    .context("failed to read requirements from stdin")?;
                buf
            };

            if req_text.trim().is_empty() {
                anyhow::bail!("no requirements provided (use --requirements, --pyproject, or pipe to stdin)");
            }

            eprintln!("Resolving requirements...");
            let wheels = resolve::resolve_requirements(&req_text, &python_version, &platform).await?;

            if wheels.is_empty() {
                anyhow::bail!("no wheels resolved from requirements");
            }

            eprintln!("Resolved {} wheels, starting download...", wheels.len());
            for w in &wheels {
                eprintln!("  {} ({:.1} MB)", w.distribution, w.size as f64 / 1024.0 / 1024.0);
            }

            run_engine(wheels, site_packages, parallel_downloads, extract_threads).await
        }
    }
}

async fn run_engine(
    wheels: Vec<zs_fast_wheel::manifest::WheelSpec>,
    site_packages: PathBuf,
    parallel_downloads: usize,
    extract_threads: usize,
) -> Result<()> {
    let config = DaemonConfig {
        site_packages,
        parallel_downloads,
        extract_threads,
    };

    let engine = DaemonEngine::new(wheels);

    let start = std::time::Instant::now();
    engine.run(&config).await?;
    let elapsed = start.elapsed();

    let (files, bytes) = engine.extract_stats();
    let (total, done, _, _, failed) = engine.stats();

    eprintln!();
    eprintln!("--- fast-wheel summary ---");
    eprintln!("Wheels: {total} ({done} done, {failed} failed)");
    eprintln!(
        "Extracted: {files} files ({:.1} MB)",
        bytes as f64 / 1024.0 / 1024.0
    );
    eprintln!("Total time: {:.1}s", elapsed.as_secs_f64());

    Ok(())
}
