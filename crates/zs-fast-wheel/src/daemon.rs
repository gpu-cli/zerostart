use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

use crate::demand::DemandWatcher;
use crate::extract::{self, cleanup_stale_staging, ExtractStats};
use crate::manifest::Manifest;
use crate::queue::InstallQueue;
use crate::status::StatusDir;
use crate::streaming;

/// Streaming threshold: wheels above this size use Range request streaming
const STREAM_THRESHOLD: u64 = 50 * 1024 * 1024; // 50MB

/// Run the daemon: read manifest, download + extract wheels progressively,
/// watch demand file for priority changes.
pub async fn run(
    manifest_path: PathBuf,
    parallel_downloads: usize,
    extract_threads: usize,
) -> Result<()> {
    let start = Instant::now();
    let manifest = Manifest::from_file(&manifest_path)?;

    // Set up directories
    std::fs::create_dir_all(&manifest.site_packages)?;
    let status = StatusDir::create(&manifest.status_dir)?;
    status.mark_installing()?;

    // Clean up stale staging dirs from interrupted runs
    cleanup_stale_staging(&manifest.site_packages)?;

    // Skip already-done packages (resume support)
    let wheels: Vec<_> = manifest
        .wheels
        .into_iter()
        .filter(|w| !status.is_done(&w.distribution))
        .collect();

    if wheels.is_empty() {
        tracing::info!("all packages already installed");
        status.clear_installing()?;
        return Ok(());
    }

    let total_wheels = wheels.len();
    let queue = Arc::new(Mutex::new(InstallQueue::new(wheels)));
    let stats = Arc::new(ExtractStats::default());

    // HTTP client
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(parallel_downloads)
        .build()?;

    // Spawn demand watcher
    let demand_queue = queue.clone();
    let demand_path = status.demand_path();
    let demand_handle = tokio::spawn(async move {
        let mut watcher = DemandWatcher::new(demand_path);
        loop {
            let demands = watcher.poll();
            if !demands.is_empty() {
                let mut q = demand_queue.lock().await;
                for dist in &demands {
                    tracing::info!("demand signal: prioritizing {dist}");
                    q.prioritize(dist);
                }
            }
            // Check if queue is empty (all done)
            {
                let q = demand_queue.lock().await;
                if q.is_empty() {
                    break;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    });

    // Worker semaphore — reserve one slot for demand-responsive work
    // until demand arrives or queue is nearly drained
    let sem = Arc::new(tokio::sync::Semaphore::new(parallel_downloads));
    let mut handles = Vec::new();

    loop {
        // Get next wheel from queue
        let wheel = {
            let mut q = queue.lock().await;
            q.next()
        };

        let wheel = match wheel {
            Some(w) => w,
            None => break, // queue drained
        };

        let client = client.clone();
        let site_packages = manifest.site_packages.clone();
        let status_dir = manifest.status_dir.clone();
        let stats = stats.clone();
        let sem = sem.clone();
        let ext_threads = extract_threads;

        let handle = tokio::spawn(async move {
            let _permit = sem
                .acquire()
                .await
                .context("semaphore closed")?;

            let dist = wheel.distribution.clone();
            let wheel_start = Instant::now();

            tracing::info!("[{dist}] starting download ({} bytes)", wheel.size);

            let result = if wheel.size >= STREAM_THRESHOLD {
                // Large wheel: streaming Range requests
                streaming::stream_extract_wheel_atomic(
                    &client,
                    &wheel.url,
                    &site_packages,
                    &dist,
                    4, // parallel chunks per wheel
                    &stats,
                )
                .await
                .map(|_| ())
            } else {
                // Small/medium wheel: download whole file, then extract
                download_and_extract_atomic(
                    &client,
                    &wheel.url,
                    &site_packages,
                    &dist,
                    ext_threads,
                    &stats,
                )
                .await
            };

            let status = StatusDir::open(std::path::Path::new(&status_dir))?;
            match result {
                Ok(()) => {
                    let elapsed = wheel_start.elapsed();
                    tracing::info!(
                        "[{dist}] done in {:.1}s",
                        elapsed.as_secs_f64()
                    );
                    status.mark_done(&dist)?;
                }
                Err(e) => {
                    tracing::error!("[{dist}] failed: {e}");
                    status.mark_failed(&dist, &format!("{e:#}"))?;
                }
            }

            Ok::<_, anyhow::Error>(())
        });

        handles.push(handle);
    }

    // Wait for all in-flight downloads to complete
    for handle in handles {
        if let Err(e) = handle.await? {
            tracing::error!("worker error: {e}");
        }
    }

    // Signal completion
    // Mark any remaining in-progress items as done in the queue
    {
        let q = queue.lock().await;
        if !q.is_empty() {
            tracing::warn!("queue not empty at shutdown");
        }
    }

    status.clear_installing()?;

    // Wait for demand watcher to notice queue is empty and exit
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), demand_handle).await;

    let elapsed = start.elapsed();
    let files = stats.files_written.load(std::sync::atomic::Ordering::Relaxed);
    let bytes = stats.bytes_written.load(std::sync::atomic::Ordering::Relaxed);

    eprintln!();
    eprintln!("--- fast-wheel daemon summary ---");
    eprintln!("Wheels: {total_wheels}");
    eprintln!(
        "Extracted: {files} files ({:.1} MB)",
        bytes as f64 / 1024.0 / 1024.0
    );
    eprintln!("Total time: {:.1}s", elapsed.as_secs_f64());

    Ok(())
}

/// Download a whole wheel file, then extract atomically.
async fn download_and_extract_atomic(
    client: &reqwest::Client,
    url: &str,
    site_packages: &std::path::Path,
    pkg_name: &str,
    threads: usize,
    stats: &Arc<ExtractStats>,
) -> Result<()> {
    // Download to temp file
    let tmp_dir = tempfile::tempdir().context("failed to create temp dir")?;
    let filename = url.rsplit('/').next().unwrap_or("wheel.whl");
    let filename = urlencoding::decode(filename)
        .map(|s| s.into_owned())
        .unwrap_or_else(|_| filename.to_string());
    let dest = tmp_dir.path().join(&filename);

    let resp = client.get(url).send().await?.error_for_status()?;
    let bytes = resp.bytes().await?;
    tokio::fs::write(&dest, &bytes).await?;

    // Extract atomically (blocking — move to spawn_blocking)
    let site_packages = site_packages.to_path_buf();
    let pkg_name = pkg_name.to_string();
    let stats = stats.clone();

    tokio::task::spawn_blocking(move || {
        extract::extract_wheel_atomic(
            &dest,
            &site_packages,
            &pkg_name,
            threads,
            true, // use_fallocate
            &stats,
        )
    })
    .await?
}
