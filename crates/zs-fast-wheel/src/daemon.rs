//! DaemonEngine: GET+pipeline wheel installation engine.
//!
//! Downloads full wheels via single GET requests (maximizing bandwidth),
//! then extracts each wheel immediately as it finishes downloading.
//! This pipelines download and extraction: while wheel N extracts,
//! wheels N+1..N+K are still downloading.
//!
//! Total time ≈ max(download_time, extract_time) instead of sum.
//!
//! All completion state is in-memory — Arc+Mutex+Condvar for cross-runtime
//! wake. Used by both the CLI and PyO3 bindings.

use anyhow::{Context, Result};
use futures::StreamExt;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::extract::{self, cleanup_stale_staging, ExtractStats};
use crate::manifest::WheelSpec;
use crate::queue::InstallQueue;

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
            parallel_downloads: 32,
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

/// Shared completion state — pure std, no tokio dependency.
struct CompletionState {
    done: HashSet<String>,
    failed: HashMap<String, String>,
    all_finished: bool,
}

/// A downloaded wheel ready for extraction.
struct DownloadedWheel {
    spec: WheelSpec,
    path: PathBuf,
}

/// In-memory engine that downloads and extracts wheels progressively.
///
/// Downloads full wheels via single GET (maximize bandwidth), extracts each
/// immediately as it finishes (pipeline download + extraction).
pub struct DaemonEngine {
    queue: Arc<Mutex<InstallQueue>>,
    /// Completion tracking — protected by Mutex + Condvar for cross-runtime wake
    completion: Arc<(Mutex<CompletionState>, Condvar)>,
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
        let distributions: Vec<String> = wheels.iter().map(|w| w.distribution.clone()).collect();
        let total_wheels = wheels.len();

        Self {
            queue: Arc::new(Mutex::new(InstallQueue::new(wheels))),
            completion: Arc::new((
                Mutex::new(CompletionState {
                    done: HashSet::new(),
                    failed: HashMap::new(),
                    all_finished: total_wheels == 0,
                }),
                Condvar::new(),
            )),
            stats: Arc::new(ExtractStats::default()),
            total_wheels,
            distributions,
        }
    }

    /// Start downloading and extracting all wheels. Returns when all are done.
    ///
    /// Architecture: GET+pipeline
    /// - Download workers (tokio tasks): single GET per wheel → temp file → channel
    /// - Extract worker (spawn_blocking): receives from channel → extract to site-packages
    /// - Backpressure: channel capacity limits temp disk usage
    pub async fn run(&self, config: &DaemonConfig) -> Result<()> {
        let start = Instant::now();

        std::fs::create_dir_all(&config.site_packages)?;
        cleanup_stale_staging(&config.site_packages)?;

        if self.total_wheels == 0 {
            return Ok(());
        }

        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(config.parallel_downloads)
            .build()?;

        // Channel: downloaded wheels flow from download workers → extract worker.
        // Small capacity (4) provides backpressure — if extraction is slow,
        // downloads pause rather than filling disk with temp files.
        let (tx, rx) = tokio::sync::mpsc::channel::<DownloadedWheel>(4);

        let tmp_dir = tempfile::tempdir().context("failed to create temp dir")?;
        let tmp_path = tmp_dir.path().to_path_buf();

        // === Download stage ===
        // Spawn one task per wheel, bounded by semaphore.
        // Each task: GET wheel → write to temp file → send through channel.
        let sem = Arc::new(tokio::sync::Semaphore::new(config.parallel_downloads));
        let mut download_handles = Vec::new();

        loop {
            let wheel = {
                let mut q = self.queue.lock().unwrap();
                q.next()
            };

            let wheel = match wheel {
                Some(w) => w,
                None => break,
            };

            let client = client.clone();
            let tx = tx.clone();
            let sem = sem.clone();
            let tmp_path = tmp_path.clone();

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.context("semaphore closed")?;

                let dist = wheel.distribution.clone();
                let dl_start = Instant::now();

                tracing::info!("[{dist}] downloading ({} bytes)", wheel.size);

                let result =
                    download_wheel_to_file(&client, &wheel.url, &tmp_path, &dist).await;

                match result {
                    Ok(path) => {
                        let elapsed = dl_start.elapsed();
                        tracing::info!(
                            "[{dist}] downloaded in {:.1}s",
                            elapsed.as_secs_f64()
                        );
                        // Send to extract worker — blocks if channel is full (backpressure)
                        tx.send(DownloadedWheel { spec: wheel, path })
                            .await
                            .ok();
                    }
                    Err(e) => {
                        tracing::error!("[{dist}] download failed: {e:#}");
                        // Return error info so we can mark it failed
                        return Err(e);
                    }
                }

                Ok::<String, anyhow::Error>(dist)
            });

            download_handles.push(handle);
        }

        // Drop our copy of tx so channel closes when all download tasks finish
        drop(tx);

        // === Extract stage ===
        // Single blocking loop: receives downloaded wheels, extracts each immediately.
        // Extraction uses all extract_threads for parallelism within a single wheel.
        let site_packages = config.site_packages.clone();
        let ext_threads = config.extract_threads;
        let stats = self.stats.clone();
        let completion = self.completion.clone();
        let queue = self.queue.clone();
        let total_wheels = self.total_wheels;

        let extract_handle = tokio::task::spawn_blocking(move || {
            let rx = rx;
            // blocking_recv in a loop — channel closes when all downloads finish
            let mut rx = rx;
            while let Some(downloaded) = rx.blocking_recv() {
                let dist = downloaded.spec.distribution.clone();
                let extract_start = Instant::now();

                let result = extract::extract_wheel_atomic(
                    &downloaded.path,
                    &site_packages,
                    &dist,
                    ext_threads,
                    true,
                    &stats,
                );

                let (lock, cvar) = &*completion;

                match result {
                    Ok(()) => {
                        let elapsed = extract_start.elapsed();
                        tracing::info!(
                            "[{dist}] extracted in {:.1}s",
                            elapsed.as_secs_f64()
                        );

                        {
                            let mut q = queue.lock().unwrap();
                            q.mark_done(&dist);
                        }

                        let mut state = lock.lock().unwrap();
                        state.done.insert(dist);
                        if state.done.len() + state.failed.len() >= total_wheels {
                            state.all_finished = true;
                        }
                        cvar.notify_all();
                    }
                    Err(e) => {
                        let err_msg = format!("{e:#}");
                        tracing::error!("[{dist}] extraction failed: {err_msg}");

                        {
                            let mut q = queue.lock().unwrap();
                            q.mark_failed(&dist);
                        }

                        let mut state = lock.lock().unwrap();
                        state.failed.insert(dist, err_msg);
                        if state.done.len() + state.failed.len() >= total_wheels {
                            state.all_finished = true;
                        }
                        cvar.notify_all();
                    }
                }

                // Clean up temp file
                let _ = std::fs::remove_file(&downloaded.path);
            }
        });

        // === Wait for download failures ===
        // Collect download errors and mark them as failed
        for handle in download_handles {
            match handle.await {
                Ok(Ok(_dist)) => {}
                Ok(Err(e)) => {
                    // Download failed — error already logged, but we need to mark completion
                    tracing::error!("download worker error: {e}");
                }
                Err(e) => {
                    tracing::error!("download task panicked: {e}");
                }
            }
        }

        // Wait for extract worker to finish
        extract_handle.await?;

        // Mark all finished
        {
            let (lock, cvar) = &*self.completion;
            let mut state = lock.lock().unwrap();
            state.all_finished = true;
            cvar.notify_all();
        }

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
    pub fn signal_demand(&self, distribution: &str) {
        let mut q = self.queue.lock().unwrap();
        q.prioritize(distribution);
        tracing::info!("demand: prioritizing {distribution}");
    }

    /// Check if a distribution is done (non-blocking).
    pub fn is_done(&self, distribution: &str) -> bool {
        let (lock, _) = &*self.completion;
        let state = lock.lock().unwrap();
        state.done.contains(distribution)
    }

    /// Get the status of a distribution.
    pub fn status(&self, distribution: &str) -> WheelStatus {
        let (lock, _) = &*self.completion;
        let state = lock.lock().unwrap();
        if let Some(err) = state.failed.get(distribution) {
            return WheelStatus::Failed(err.clone());
        }
        if state.done.contains(distribution) {
            return WheelStatus::Done;
        }
        drop(state);

        let q = self.queue.lock().unwrap();
        if q.pending_count() > 0 || q.in_progress_count() > 0 {
            WheelStatus::InProgress
        } else {
            WheelStatus::Pending
        }
    }

    /// Wait for a specific distribution to complete. Returns Ok(true) on success,
    /// Ok(false) if failed, Err on timeout.
    ///
    /// Pure std::sync — works from any thread/runtime.
    pub fn wait_done(&self, distribution: &str, timeout: Duration) -> Result<bool> {
        let (lock, cvar) = &*self.completion;
        let deadline = Instant::now() + timeout;

        let mut state = lock.lock().unwrap();
        loop {
            if state.done.contains(distribution) {
                return Ok(true);
            }
            if state.failed.contains_key(distribution) {
                return Ok(false);
            }
            if state.all_finished {
                return Ok(true);
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                anyhow::bail!("timed out waiting for {distribution}");
            }

            let (guard, timeout_result) = cvar.wait_timeout(state, remaining).unwrap();
            state = guard;
            if timeout_result.timed_out() {
                if state.done.contains(distribution) {
                    return Ok(true);
                }
                if state.failed.contains_key(distribution) {
                    return Ok(false);
                }
                anyhow::bail!("timed out waiting for {distribution}");
            }
        }
    }

    /// Wait for all wheels to complete.
    ///
    /// Pure std::sync — works from any thread/runtime.
    pub fn wait_all(&self, timeout: Duration) -> Result<()> {
        let (lock, cvar) = &*self.completion;
        let deadline = Instant::now() + timeout;

        let mut state = lock.lock().unwrap();
        loop {
            if state.all_finished {
                return Ok(());
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                anyhow::bail!("timed out waiting for all wheels");
            }

            let (guard, timeout_result) = cvar.wait_timeout(state, remaining).unwrap();
            state = guard;
            if timeout_result.timed_out() && !state.all_finished {
                anyhow::bail!("timed out waiting for all wheels");
            }
        }
    }

    /// Get summary stats: (total, done, pending, in_progress, failed)
    pub fn stats(&self) -> (usize, usize, usize, usize, usize) {
        let (lock, _) = &*self.completion;
        let state = lock.lock().unwrap();
        let done = state.done.len();
        let failed = state.failed.len();
        let pending_plus_in_progress = self.total_wheels.saturating_sub(done + failed);
        let q = self.queue.lock().unwrap();
        let pending = q.pending_count();
        let in_progress = pending_plus_in_progress.saturating_sub(pending);
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
}

/// Download a single wheel via GET to a temp file.
async fn download_wheel_to_file(
    client: &reqwest::Client,
    url: &str,
    tmp_dir: &Path,
    pkg_name: &str,
) -> Result<PathBuf> {
    let filename = url.rsplit('/').next().unwrap_or("wheel.whl");
    let filename = urlencoding::decode(filename)
        .map(|s| s.into_owned())
        .unwrap_or_else(|_| filename.to_string());
    // Use pkg_name prefix to avoid collisions
    let dest = tmp_dir.join(format!("{pkg_name}_{filename}"));

    let resp = client.get(url).send().await?.error_for_status()?;

    // Stream to file — don't buffer entire wheel in memory
    let mut stream = resp.bytes_stream();
    let file = tokio::fs::File::create(&dest).await?;
    let mut writer = tokio::io::BufWriter::with_capacity(1024 * 1024, file);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        tokio::io::AsyncWriteExt::write_all(&mut writer, &chunk).await?;
    }
    tokio::io::AsyncWriteExt::flush(&mut writer).await?;

    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_wheel(name: &str, size: u64) -> WheelSpec {
        WheelSpec {
            url: format!("https://example.com/{name}-1.0.whl"),
            distribution: name.to_string(),
            version: "1.0".to_string(),
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

        engine.signal_demand("torch");

        let mut q = engine.queue.lock().unwrap();
        let first = q.next();
        assert_eq!(first.as_ref().map(|w| w.distribution.as_str()), Some("torch"));
    }

    #[test]
    fn test_stats_initial() {
        let wheels = vec![make_wheel("six", 12_000)];
        let engine = DaemonEngine::new(wheels);
        let (total, done, pending, in_progress, failed) = engine.stats();
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
        engine.run(&config).await.unwrap();
        let _ = std::fs::remove_dir_all("/tmp/zs-test-empty-engine");
    }

    #[test]
    fn test_wait_done_not_in_set() {
        let engine = DaemonEngine::new(vec![]);
        let result = engine.wait_done("nonexistent", Duration::from_secs(1)).unwrap();
        assert!(result);
    }
}
