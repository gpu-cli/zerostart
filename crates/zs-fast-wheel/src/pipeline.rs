use crate::download::{self, DownloadedWheel};
use crate::extract::{self, ExtractStats};
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct PipelineConfig {
    pub target: PathBuf,
    pub parallel_downloads: usize,
    pub extract_threads: usize,
    pub use_fallocate: bool,
    pub batch_sync: bool,
    pub benchmark: bool,
}

pub async fn run(wheels: Vec<String>, config: PipelineConfig) -> Result<()> {
    let total_start = std::time::Instant::now();
    let stats = Arc::new(ExtractStats::default());

    // Separate local files from URLs
    let mut urls = Vec::new();
    let mut local_wheels: Vec<DownloadedWheel> = Vec::new();

    for w in &wheels {
        if w.starts_with("http://") || w.starts_with("https://") {
            urls.push(w.clone());
        } else {
            let path = PathBuf::from(w);
            let size = std::fs::metadata(&path)
                .with_context(|| format!("wheel not found: {w}"))?
                .len();
            local_wheels.push(DownloadedWheel {
                path,
                url: w.clone(),
                size,
                download_ms: 0,
            });
        }
    }

    // Download whole files then extract (GET+pipeline)
    let (tx, mut rx) = mpsc::channel::<DownloadedWheel>(config.parallel_downloads * 2);

    for w in local_wheels {
        tx.send(w).await.ok();
    }

    let tmp_dir = tempfile::tempdir().context("failed to create temp dir")?;
    let tmp_path = tmp_dir.path().to_path_buf();

    let download_tx = tx;
    let download_handle = if !urls.is_empty() {
        let urls_clone = urls.clone();
        Some(tokio::spawn(async move {
            download::download_wheels(urls_clone, &tmp_path, config.parallel_downloads, download_tx)
                .await
        }))
    } else {
        drop(download_tx);
        None
    };

    let extract_threads = config.extract_threads;
    let use_fallocate = config.use_fallocate;
    let target = config.target.clone();
    let extract_stats = stats.clone();
    let benchmark = config.benchmark;

    let extract_handle = tokio::task::spawn_blocking(move || {
        let mut download_total_ms: u128 = 0;
        let mut extract_total_ms: u128 = 0;
        let mut wheel_count: usize = 0;
        let mut total_download_bytes: u64 = 0;

        while let Some(wheel) = rx.blocking_recv() {
            wheel_count += 1;
            download_total_ms = download_total_ms.max(wheel.download_ms);
            total_download_bytes += wheel.size;

            let extract_start = std::time::Instant::now();

            let short = wheel
                .path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();

            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} {prefix} extracting...")
                    .unwrap_or_else(|_| ProgressStyle::default_spinner()),
            );
            pb.set_prefix(short.clone());

            if let Err(e) = extract::extract_wheel(
                &wheel.path,
                &target,
                extract_threads,
                use_fallocate,
                &extract_stats,
            ) {
                pb.finish_with_message(format!("FAILED: {e}"));
                tracing::error!("extraction failed for {short}: {e}");
                continue;
            }

            let elapsed = extract_start.elapsed().as_millis();
            extract_total_ms += elapsed;
            let bytes = extract_stats.bytes_written.load(Ordering::Relaxed);

            pb.finish_with_message(format!("{short} extracted in {elapsed}ms"));

            if benchmark {
                let mb_per_s = if elapsed > 0 {
                    (bytes as f64 / 1024.0 / 1024.0) / (elapsed as f64 / 1000.0)
                } else {
                    0.0
                };
                eprintln!(
                    "  [{short}] download: {}ms, extract: {elapsed}ms ({mb_per_s:.0} MB/s)",
                    wheel.download_ms
                );
            }
        }

        (wheel_count, download_total_ms, extract_total_ms, total_download_bytes)
    });

    if let Some(handle) = download_handle {
        handle.await??;
    }

    let (wheel_count, _download_ms, extract_ms, download_bytes) = extract_handle.await?;

    if config.batch_sync {
        do_syncfs(&config.target);
    }

    let total_ms = total_start.elapsed().as_millis();
    let files = stats.files_written.load(Ordering::Relaxed);
    let bytes = stats.bytes_written.load(Ordering::Relaxed);
    let dirs = stats.dirs_created.load(Ordering::Relaxed);

    eprintln!();
    eprintln!("--- fast-wheel summary ---");
    eprintln!(
        "Wheels: {wheel_count} ({:.1} MB downloaded)",
        download_bytes as f64 / 1024.0 / 1024.0
    );
    eprintln!(
        "Extracted: {files} files, {dirs} dirs ({:.1} MB)",
        bytes as f64 / 1024.0 / 1024.0
    );
    eprintln!("Extract time: {extract_ms}ms");
    eprintln!("Total time: {total_ms}ms");

    if total_ms > 0 {
        let throughput = bytes as f64 / 1024.0 / 1024.0 / (total_ms as f64 / 1000.0);
        eprintln!("Throughput: {throughput:.0} MB/s");
    }

    Ok(())
}

#[allow(unused_variables)]
fn do_syncfs(target: &Path) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        if let Ok(f) = std::fs::File::open(target) {
            unsafe {
                nix::libc::syncfs(f.as_raw_fd());
            }
        }
    }
}

use std::path::Path;
