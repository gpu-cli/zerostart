mod download;
mod extract;
mod pipeline;
mod streaming;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "fast-wheel", about = "Download + extract Python wheels at maximum speed")]
struct Cli {
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

    /// Use streaming mode: Range requests per entry, overlap download+decompress.
    /// Faster for multiple small wheels, slower for single large wheels.
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

    std::fs::create_dir_all(&cli.target)?;

    let config = pipeline::PipelineConfig {
        target: cli.target,
        parallel_downloads: cli.parallel_downloads,
        extract_threads: cli.extract_threads,
        use_stream: cli.stream,
        use_fallocate: cli.fallocate,
        batch_sync: cli.batch_sync,
        benchmark: cli.benchmark,
    };

    pipeline::run(cli.wheels, config).await
}
