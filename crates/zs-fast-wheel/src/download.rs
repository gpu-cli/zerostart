use anyhow::{Context, Result};
use bytes::Bytes;
use futures::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::Client;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;

/// A downloaded wheel ready for extraction
pub struct DownloadedWheel {
    pub path: PathBuf,
    pub url: String,
    pub size: u64,
    pub download_ms: u128,
}

/// Download multiple wheels in parallel, sending each to the channel as it completes.
/// This allows extraction to begin before all downloads finish (pipelining).
pub async fn download_wheels(
    urls: Vec<String>,
    tmp_dir: &Path,
    parallel: usize,
    tx: mpsc::Sender<DownloadedWheel>,
) -> Result<()> {
    let client = Client::builder()
        .pool_max_idle_per_host(parallel)
        .build()?;

    let multi = Arc::new(MultiProgress::new());
    let style = ProgressStyle::default_bar()
        .template("{prefix:.cyan} [{bar:30}] {bytes}/{total_bytes} {bytes_per_sec}")
        .context("invalid progress template")?
        .progress_chars("=> ");

    // Process downloads with bounded concurrency
    let semaphore = Arc::new(tokio::sync::Semaphore::new(parallel));

    let mut handles = Vec::new();

    for url in urls {
        let client = client.clone();
        let tx = tx.clone();
        let tmp_dir = tmp_dir.to_path_buf();
        let multi = multi.clone();
        let style = style.clone();
        let sem = semaphore.clone();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await?;
            download_one(&client, &url, &tmp_dir, &multi, &style, &tx).await
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.await??;
    }

    Ok(())
}

async fn download_one(
    client: &Client,
    url: &str,
    tmp_dir: &Path,
    multi: &MultiProgress,
    style: &ProgressStyle,
    tx: &mpsc::Sender<DownloadedWheel>,
) -> Result<()> {
    let start = std::time::Instant::now();

    // Extract filename from URL
    let filename = url
        .rsplit('/')
        .next()
        .unwrap_or("wheel.whl");
    let filename = urlencoding::decode(filename)
        .map(|s| s.into_owned())
        .unwrap_or_else(|_| filename.to_string());

    let dest = tmp_dir.join(&filename);

    if url.starts_with("http://") || url.starts_with("https://") {
        let resp = client.get(url).send().await?.error_for_status()?;
        let total = resp.content_length().unwrap_or(0);

        let pb = multi.add(ProgressBar::new(total));
        pb.set_style(style.clone());
        pb.set_prefix(short_name(&filename));

        let mut stream = resp.bytes_stream();
        let file = tokio::fs::File::create(&dest).await?;
        let mut writer = tokio::io::BufWriter::with_capacity(1024 * 1024, file);

        while let Some(chunk) = stream.next().await {
            let chunk: Bytes = chunk?;
            pb.inc(chunk.len() as u64);
            tokio::io::AsyncWriteExt::write_all(&mut writer, &chunk).await?;
        }
        tokio::io::AsyncWriteExt::flush(&mut writer).await?;
        pb.finish();

        let size = std::fs::metadata(&dest)?.len();
        let elapsed = start.elapsed().as_millis();

        tx.send(DownloadedWheel {
            path: dest,
            url: url.to_string(),
            size,
            download_ms: elapsed,
        })
        .await
        .ok();
    } else {
        // Local file — just send it directly
        let size = std::fs::metadata(url)?.len();
        tx.send(DownloadedWheel {
            path: PathBuf::from(url),
            url: url.to_string(),
            size,
            download_ms: 0,
        })
        .await
        .ok();
    }

    Ok(())
}

fn short_name(filename: &str) -> String {
    // "torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl" -> "torch-2.6.0"
    let name = filename.strip_suffix(".whl").unwrap_or(filename);
    match name.find("-cp") {
        Some(idx) => name[..idx].to_string(),
        None => name.to_string(),
    }
}
