//! DaemonEngine: in-memory wheel installation engine.
//!
//! Core state is all in-memory — completion tracking, demand signaling,
//! and stats use Arc+Mutex+watch channels. No file-based IPC.
//!
//! Used by both the CLI and PyO3 bindings.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{watch, Mutex, Notify};

use crate::extract::{self, cleanup_stale_staging, ExtractStats};
use crate::manifest::WheelSpec;
use crate::queue::InstallQueue;
use crate::streaming;

/// Streaming threshold: wheels above this size use Range request streaming
const STREAM_THRESHOLD: u64 = 50 * 1024 * 1024; // 50MB

/// Configuration for the daemon engine.
pub struct DaemonConfig {
    pub site_packages: PathBuf,
    pub parallel_downloads: usize,
    pub extract_threads: usize,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            site_packages: PathBuf::from("site-packages"),
            parallel_downloads: 8,
            extract_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        }
    }
}

/// Per-distribution status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WheelStatus {
    Pending,
    InProgress,
    Done,
    Failed(String),
}

/// In-memory engine that downloads and extracts wheels progressively.
///
/// All state is in-memory. Use `signal_demand()` to prioritize a package,
/// `is_done()` / `wait_done()` to check completion, `wait_all()` to block
/// until everything finishes.
pub struct DaemonEngine {
    queue: Arc<Mutex<InstallQueue>>,
    /// Per-distribution completion channels
    completion_txs: Arc<Mutex<HashMap<String, watch::Sender<bool>>>>,
    completion_rxs: Arc<Mutex<HashMap<String, watch::Receiver<bool>>>>,
    /// Per-distribution failure messages
    failures: Arc<Mutex<HashMap<String, String>>>,
    /// Fires when all wheels are done (success or failure)
    all_done: Arc<Notify>,
    /// Extraction stats
    stats: Arc<ExtractStats>,
    /// Total wheel count
    total_wheels: usize,
    /// Distribution names (for iteration)
    distributions: Vec<String>,
}

impl DaemonEngine {
    /// Create a new engine from wheel specs. Does NOT start downloading yet.
    pub fn new(wheels: Vec<WheelSpec>) -> Self {
        let mut completion_txs = HashMap::new();
        let mut completion_rxs = HashMap::new();
        let distributions: Vec<String> = wheels.iter().map(|w| w.distribution.clone()).collect();

        for w in &wheels {
            let (tx, rx) = watch::channel(false);
            completion_txs.insert(w.distribution.clone(), tx);
            completion_rxs.insert(w.distribution.clone(), rx);
        }

        let total_wheels = wheels.len();

        Self {
            queue: Arc::new(Mutex::new(InstallQueue::new(wheels))),
            completion_txs: Arc::new(Mutex::new(completion_txs)),
            completion_rxs: Arc::new(Mutex::new(completion_rxs)),
            failures: Arc::new(Mutex::new(HashMap::new())),
            all_done: Arc::new(Notify::new()),
            stats: Arc::new(ExtractStats::default()),
            total_wheels,
            distributions,
        }
    }

    /// Start downloading and extracting all wheels. Returns when all are done.
    ///
    /// Call this from a tokio runtime (or spawn on a background thread).
    pub async fn run(&self, config: &DaemonConfig) -> Result<()> {
        let start = Instant::now();

        std::fs::create_dir_all(&config.site_packages)?;
        cleanup_stale_staging(&config.site_packages)?;

        if self.total_wheels == 0 {
            self.all_done.notify_waiters();
            return Ok(());
        }

        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(config.parallel_downloads)
            .build()?;

        let sem = Arc::new(tokio::sync::Semaphore::new(config.parallel_downloads));
        let mut handles = Vec::new();

        loop {
            let wheel = {
                let mut q = self.queue.lock().await;
                q.next()
            };

            let wheel = match wheel {
                Some(w) => w,
                None => break,
            };

            let client = client.clone();
            let site_packages = config.site_packages.clone();
            let stats = self.stats.clone();
            let sem = sem.clone();
            let ext_threads = config.extract_threads;
            let completion_txs = self.completion_txs.clone();
            let failures = self.failures.clone();
            let queue = self.queue.clone();
            let all_done = self.all_done.clone();

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.context("semaphore closed")?;

                let dist = wheel.distribution.clone();
                let wheel_start = Instant::now();

                tracing::info!("[{dist}] starting download ({} bytes)", wheel.size);

                let result = if wheel.size >= STREAM_THRESHOLD {
                    streaming::stream_extract_wheel_atomic(
                        &client,
                        &wheel.url,
                        &site_packages,
                        &dist,
                        4,
                        &stats,
                    )
                    .await
                    .map(|_| ())
                } else {
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

                match result {
                    Ok(()) => {
                        let elapsed = wheel_start.elapsed();
                        tracing::info!("[{dist}] done in {:.1}s", elapsed.as_secs_f64());

                        // Signal completion
                        let txs = completion_txs.lock().await;
                        if let Some(tx) = txs.get(&dist) {
                            let _ = tx.send(true);
                        }

                        let mut q = queue.lock().await;
                        q.mark_done(&dist);
                        if q.is_empty() {
                            all_done.notify_waiters();
                        }
                    }
                    Err(e) => {
                        let err_msg = format!("{e:#}");
                        tracing::error!("[{dist}] failed: {err_msg}");

                        failures.lock().await.insert(dist.clone(), err_msg);

                        let mut q = queue.lock().await;
                        q.mark_failed(&dist);
                        if q.is_empty() {
                            all_done.notify_waiters();
                        }
                    }
                }

                Ok::<_, anyhow::Error>(())
            });

            handles.push(handle);
        }

        for handle in handles {
            if let Err(e) = handle.await? {
                tracing::error!("worker error: {e}");
            }
        }

        // Final notification
        self.all_done.notify_waiters();

        let elapsed = start.elapsed();
        let files = self.stats.files_written.load(Ordering::Relaxed);
        let bytes = self.stats.bytes_written.load(Ordering::Relaxed);

        tracing::info!(
            "daemon complete: {} wheels, {} files ({:.1} MB) in {:.1}s",
            self.total_wheels,
            files,
            bytes as f64 / 1024.0 / 1024.0,
            elapsed.as_secs_f64()
        );

        Ok(())
    }

    /// Bump a distribution to the front of the download queue.
    pub async fn signal_demand(&self, distribution: &str) {
        let mut q = self.queue.lock().await;
        q.prioritize(distribution);
        tracing::info!("demand: prioritizing {distribution}");
    }

    /// Check if a distribution is done (non-blocking).
    pub async fn is_done(&self, distribution: &str) -> bool {
        let q = self.queue.lock().await;
        q.is_done(distribution)
    }

    /// Get the status of a distribution.
    pub async fn status(&self, distribution: &str) -> WheelStatus {
        let failures = self.failures.lock().await;
        if let Some(err) = failures.get(distribution) {
            return WheelStatus::Failed(err.clone());
        }
        drop(failures);

        let q = self.queue.lock().await;
        if q.is_done(distribution) {
            WheelStatus::Done
        } else if q.pending_count() > 0 || q.in_progress_count() > 0 {
            // Rough — could be pending or in_progress
            WheelStatus::InProgress
        } else {
            WheelStatus::Pending
        }
    }

    /// Wait for a specific distribution to complete. Returns Ok(true) on success,
    /// Ok(false) if failed, Err on timeout.
    pub async fn wait_done(&self, distribution: &str, timeout: Duration) -> Result<bool> {
        // Already done?
        {
            let q = self.queue.lock().await;
            if q.is_done(distribution) {
                return Ok(true);
            }
        }

        // Check failures
        {
            let failures = self.failures.lock().await;
            if failures.contains_key(distribution) {
                return Ok(false);
            }
        }

        let mut rx = {
            let rxs = self.completion_rxs.lock().await;
            match rxs.get(distribution) {
                Some(rx) => rx.clone(),
                None => return Ok(true), // not in our set
            }
        };

        match tokio::time::timeout(timeout, rx.wait_for(|done| *done)).await {
            Ok(Ok(_)) => Ok(true),
            Ok(Err(_)) => Ok(false), // channel closed
            Err(_) => anyhow::bail!("timed out waiting for {distribution}"),
        }
    }

    /// Wait for all wheels to complete.
    pub async fn wait_all(&self, timeout: Duration) -> Result<()> {
        // Already done?
        {
            let q = self.queue.lock().await;
            if q.is_empty() {
                return Ok(());
            }
        }

        match tokio::time::timeout(timeout, self.all_done.notified()).await {
            Ok(()) => Ok(()),
            Err(_) => anyhow::bail!("timed out waiting for all wheels"),
        }
    }

    /// Get summary stats: (total, done, pending, in_progress, failed)
    pub async fn stats(&self) -> (usize, usize, usize, usize, usize) {
        let q = self.queue.lock().await;
        let pending = q.pending_count();
        let in_progress = q.in_progress_count();
        let failed = self.failures.lock().await.len();
        let done = self.total_wheels.saturating_sub(pending + in_progress + failed);
        (self.total_wheels, done, pending, in_progress, failed)
    }

    /// Get extract stats (files written, bytes written).
    pub fn extract_stats(&self) -> (u64, u64) {
        let files = self.stats.files_written.load(Ordering::Relaxed);
        let bytes = self.stats.bytes_written.load(Ordering::Relaxed);
        (files, bytes)
    }

    /// List all distribution names.
    pub fn distributions(&self) -> &[String] {
        &self.distributions
    }

    /// Total wheel count.
    pub fn total_wheels(&self) -> usize {
        self.total_wheels
    }

    /// Get a clone of the all_done notifier.
    pub fn all_done_notify(&self) -> Arc<Notify> {
        self.all_done.clone()
    }
}

/// Download a whole wheel file, then extract atomically.
async fn download_and_extract_atomic(
    client: &reqwest::Client,
    url: &str,
    site_packages: &Path,
    pkg_name: &str,
    threads: usize,
    stats: &Arc<ExtractStats>,
) -> Result<()> {
    let tmp_dir = tempfile::tempdir().context("failed to create temp dir")?;
    let filename = url.rsplit('/').next().unwrap_or("wheel.whl");
    let filename = urlencoding::decode(filename)
        .map(|s| s.into_owned())
        .unwrap_or_else(|_| filename.to_string());
    let dest = tmp_dir.path().join(&filename);

    let resp = client.get(url).send().await?.error_for_status()?;
    let bytes = resp.bytes().await?;
    tokio::fs::write(&dest, &bytes).await?;

    let site_packages = site_packages.to_path_buf();
    let pkg_name = pkg_name.to_string();
    let stats = stats.clone();

    tokio::task::spawn_blocking(move || {
        extract::extract_wheel_atomic(&dest, &site_packages, &pkg_name, threads, true, &stats)
    })
    .await?
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_wheel(name: &str, size: u64) -> WheelSpec {
        WheelSpec {
            url: format!("https://example.com/{name}-1.0.whl"),
            distribution: name.to_string(),
            import_roots: vec![name.to_string()],
            size,
            hash: None,
        }
    }

    #[tokio::test]
    async fn test_engine_creation() {
        let wheels = vec![
            make_wheel("six", 12_000),
            make_wheel("torch", 900_000_000),
        ];
        let engine = DaemonEngine::new(wheels);
        assert_eq!(engine.total_wheels(), 2);
        assert_eq!(engine.distributions().len(), 2);
    }

    #[tokio::test]
    async fn test_signal_demand() {
        let wheels = vec![
            make_wheel("six", 12_000),
            make_wheel("numpy", 15_000_000),
            make_wheel("torch", 900_000_000),
        ];
        let engine = DaemonEngine::new(wheels);

        // Signal demand for torch — should move it to front
        engine.signal_demand("torch").await;

        let mut q = engine.queue.lock().await;
        let first = q.next();
        assert_eq!(first.as_ref().map(|w| w.distribution.as_str()), Some("torch"));
    }

    #[tokio::test]
    async fn test_stats_initial() {
        let wheels = vec![make_wheel("six", 12_000)];
        let engine = DaemonEngine::new(wheels);
        let (total, done, pending, in_progress, failed) = engine.stats().await;
        assert_eq!(total, 1);
        assert_eq!(done, 0);
        assert_eq!(pending, 1);
        assert_eq!(in_progress, 0);
        assert_eq!(failed, 0);
    }

    #[tokio::test]
    async fn test_empty_engine() {
        let engine = DaemonEngine::new(vec![]);
        assert_eq!(engine.total_wheels(), 0);

        let config = DaemonConfig {
            site_packages: PathBuf::from("/tmp/zs-test-empty-engine"),
            ..Default::default()
        };
        // Should complete immediately
        engine.run(&config).await.unwrap();
        let _ = std::fs::remove_dir_all("/tmp/zs-test-empty-engine");
    }
}
