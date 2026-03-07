use anyhow::{bail, Context, Result};
use futures::StreamExt;
use reqwest::Client;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::extract::ExtractStats;

/// A parsed entry from the zip central directory
#[derive(Debug, Clone)]
struct ZipEntry {
    name: String,
    compressed_size: u64,
    uncompressed_size: u64,
    compression_method: u16,
    local_header_offset: u64,
    is_dir: bool,
}

/// A chunk is a contiguous byte range containing one or more zip entries
struct Chunk {
    range_start: u64,
    range_end: u64,
    entries: Vec<ZipEntry>,
}

const LOCAL_HEADER_FIXED_SIZE: u64 = 30;
const EOCD_SIGNATURE: u32 = 0x06054b50;
const EOCD64_SIGNATURE: u32 = 0x06064b50;
const EOCD64_LOCATOR_SIGNATURE: u32 = 0x07064b50;
const CD_SIGNATURE: u32 = 0x02014b50;

/// Large file threshold — entries above this get their own Range request
const LARGE_THRESHOLD: u64 = 1024 * 1024; // 1MB

/// Target chunk size for batching small files into single Range requests
const CHUNK_TARGET_BYTES: u64 = 8 * 1024 * 1024; // 8MB chunks

/// Stream-extract a wheel from a URL with overlapped download+decompress:
/// 1. Range request the central directory (small, fast)
/// 2. Parse entries to get file offsets and sizes
/// 3. Group small files into contiguous chunks (~8MB each)
/// 4. Large files get individual Range requests
/// 5. Download chunks in parallel, extract files from each chunk as it arrives
pub async fn stream_extract_wheel(
    client: &Client,
    url: &str,
    target: &Path,
    parallel: usize,
    stats: &Arc<ExtractStats>,
) -> Result<(u64, u64)> {
    // Step 1: HEAD to get content length
    let head = client.head(url).send().await?.error_for_status()?;
    let total_size = head
        .content_length()
        .context("server didn't return Content-Length")?;

    // Step 2: Fetch the End of Central Directory (last 64KB)
    let eocd_fetch_size: u64 = 65536.min(total_size);
    let eocd_start = total_size - eocd_fetch_size;
    let eocd_data = range_request(client, url, eocd_start, total_size - 1).await?;

    // Step 3: Parse EOCD to find central directory
    let (cd_offset, cd_size) = parse_eocd(&eocd_data, eocd_start)?;

    // Step 4: Fetch the central directory
    let cd_data = if cd_offset >= eocd_start {
        let local_offset = (cd_offset - eocd_start) as usize;
        eocd_data[local_offset..local_offset + cd_size as usize].to_vec()
    } else {
        range_request(client, url, cd_offset, cd_offset + cd_size - 1).await?
    };

    // Step 5: Parse all entries
    let entries = parse_central_directory(&cd_data)?;

    // Step 6: Create directories
    let mut dirs: Vec<String> = Vec::new();
    for entry in &entries {
        if entry.is_dir {
            dirs.push(entry.name.clone());
        } else {
            let dest = target.join(&entry.name);
            if let Some(parent) = dest.parent() {
                if let Ok(rel) = parent.strip_prefix(target) {
                    let s = rel.to_string_lossy().into_owned();
                    if !s.is_empty() {
                        dirs.push(s);
                    }
                }
            }
        }
    }
    dirs.sort();
    dirs.dedup();
    for dir in &dirs {
        fs::create_dir_all(target.join(dir))?;
        stats.dirs_created.fetch_add(1, Ordering::Relaxed);
    }

    // Step 7: Build chunks — group small files, isolate large files
    let file_entries: Vec<_> = entries.into_iter().filter(|e| !e.is_dir).collect();
    let chunks = build_chunks(&file_entries, total_size);

    // Step 8: Download + extract chunks in parallel
    let sem = Arc::new(tokio::sync::Semaphore::new(parallel));
    let mut handles = Vec::new();

    for chunk in chunks {
        let client = client.clone();
        let url = url.to_string();
        let target = target.to_path_buf();
        let stats = stats.clone();
        let sem = sem.clone();
        let chunk_bytes = chunk.range_end - chunk.range_start + 1;

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await?;

            // Download this byte range
            let data = range_request(&client, &url, chunk.range_start, chunk.range_end).await?;

            // Extract all entries from this chunk — CPU-bound, move to blocking
            let entries = chunk.entries;
            let chunk_start = chunk.range_start;
            tokio::task::spawn_blocking(move || {
                for entry in &entries {
                    if let Err(e) = extract_from_chunk(&data, chunk_start, entry, &target, &stats) {
                        tracing::error!("failed to extract {}: {e}", entry.name);
                    }
                }
                Ok::<_, anyhow::Error>(())
            })
            .await??;

            Ok::<u64, anyhow::Error>(chunk_bytes)
        });
        handles.push(handle);
    }

    let mut total_downloaded: u64 = eocd_fetch_size + cd_size;
    for handle in handles {
        total_downloaded += handle.await??;
    }

    Ok((total_downloaded, total_size))
}

/// Group entries into chunks for efficient Range requests.
/// Large files (>1MB) get their own chunk. Small files are batched
/// into contiguous byte ranges up to CHUNK_TARGET_BYTES.
fn build_chunks(entries: &[ZipEntry], total_size: u64) -> Vec<Chunk> {
    // Sort by file offset (entries are usually already sorted, but be safe)
    let mut sorted: Vec<&ZipEntry> = entries.iter().collect();
    sorted.sort_by_key(|e| e.local_header_offset);

    let mut chunks = Vec::new();
    let mut current_entries: Vec<ZipEntry> = Vec::new();
    let mut current_start: u64 = 0;
    let mut current_end: u64 = 0;

    for entry in sorted {
        let entry_start = entry.local_header_offset;
        let entry_end = entry_start
            + LOCAL_HEADER_FIXED_SIZE
            + entry.name.len() as u64
            + 256 // extra field allowance
            + entry.compressed_size;
        let entry_end = entry_end.min(total_size - 1);

        // Large files always get their own chunk
        if entry.compressed_size >= LARGE_THRESHOLD {
            // Flush current batch first
            if !current_entries.is_empty() {
                chunks.push(Chunk {
                    range_start: current_start,
                    range_end: current_end,
                    entries: std::mem::take(&mut current_entries),
                });
            }
            chunks.push(Chunk {
                range_start: entry_start,
                range_end: entry_end,
                entries: vec![entry.clone()],
            });
            continue;
        }

        // Check if this entry is contiguous with current batch
        if current_entries.is_empty() {
            current_start = entry_start;
            current_end = entry_end;
            current_entries.push(entry.clone());
        } else if entry_start <= current_end + 4096
            && (entry_end - current_start) < CHUNK_TARGET_BYTES
        {
            // Contiguous and within chunk size — add to batch
            current_end = current_end.max(entry_end);
            current_entries.push(entry.clone());
        } else {
            // Gap or chunk full — flush and start new batch
            chunks.push(Chunk {
                range_start: current_start,
                range_end: current_end,
                entries: std::mem::take(&mut current_entries),
            });
            current_start = entry_start;
            current_end = entry_end;
            current_entries.push(entry.clone());
        }
    }

    // Flush remaining
    if !current_entries.is_empty() {
        chunks.push(Chunk {
            range_start: current_start,
            range_end: current_end,
            entries: current_entries,
        });
    }

    chunks
}

/// Extract a single entry from a downloaded byte chunk
fn extract_from_chunk(
    chunk_data: &[u8],
    chunk_start: u64,
    entry: &ZipEntry,
    target: &Path,
    stats: &ExtractStats,
) -> Result<()> {
    // Find this entry's local header within the chunk
    let offset = (entry.local_header_offset - chunk_start) as usize;
    if offset + 30 > chunk_data.len() {
        bail!("entry {} offset out of bounds", entry.name);
    }

    let header = &chunk_data[offset..];
    let name_len = u16::from_le_bytes([header[26], header[27]]) as usize;
    let extra_len = u16::from_le_bytes([header[28], header[29]]) as usize;
    let data_offset = offset + 30 + name_len + extra_len;

    let end = data_offset + entry.compressed_size as usize;
    if end > chunk_data.len() {
        bail!(
            "insufficient data for {}: need {} have {}",
            entry.name,
            end,
            chunk_data.len()
        );
    }

    let compressed_data = &chunk_data[data_offset..end];
    let dest = target.join(&entry.name);

    let file = File::create(&dest)?;

    #[cfg(target_os = "linux")]
    if entry.uncompressed_size > 0 {
        use nix::fcntl::{fallocate, FallocateFlags};
        use std::os::unix::io::AsRawFd;
        let _ = fallocate(
            file.as_raw_fd(),
            FallocateFlags::empty(),
            0,
            entry.uncompressed_size as i64,
        );
    }

    let buf_size = if entry.uncompressed_size > 1024 * 1024 {
        1024 * 1024
    } else {
        256 * 1024
    };
    let mut writer = std::io::BufWriter::with_capacity(buf_size, file);

    match entry.compression_method {
        0 => {
            // STORED
            writer.write_all(compressed_data)?;
        }
        8 => {
            // DEFLATE
            let mut decoder = flate2::read::DeflateDecoder::new(compressed_data);
            let mut buf = vec![0u8; 1024 * 1024];
            loop {
                let n = std::io::Read::read(&mut decoder, &mut buf)?;
                if n == 0 {
                    break;
                }
                writer.write_all(&buf[..n])?;
            }
        }
        other => {
            bail!("unsupported compression method {} for {}", other, entry.name);
        }
    }

    writer.flush()?;

    stats.files_written.fetch_add(1, Ordering::Relaxed);
    stats
        .bytes_written
        .fetch_add(entry.uncompressed_size, Ordering::Relaxed);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let name = dest.to_string_lossy();
        if name.ends_with(".so") || name.contains("/bin/") || name.contains("/scripts/") {
            let _ = fs::set_permissions(&dest, fs::Permissions::from_mode(0o755));
        }
    }

    Ok(())
}

async fn range_request(client: &Client, url: &str, start: u64, end: u64) -> Result<Vec<u8>> {
    let resp = client
        .get(url)
        .header("Range", format!("bytes={start}-{end}"))
        .send()
        .await?;

    if !resp.status().is_success() {
        bail!(
            "Range request failed: {} for bytes={start}-{end}",
            resp.status()
        );
    }

    let mut data = Vec::with_capacity((end - start + 1) as usize);
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        data.extend_from_slice(&chunk?);
    }
    Ok(data)
}

fn parse_eocd(data: &[u8], data_file_offset: u64) -> Result<(u64, u64)> {
    if let Some(result) = try_parse_eocd64(data, data_file_offset) {
        return Ok(result);
    }

    for i in (0..data.len().saturating_sub(21)).rev() {
        let sig = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
        if sig == EOCD_SIGNATURE {
            let cd_size =
                u32::from_le_bytes([data[i + 12], data[i + 13], data[i + 14], data[i + 15]])
                    as u64;
            let cd_offset =
                u32::from_le_bytes([data[i + 16], data[i + 17], data[i + 18], data[i + 19]])
                    as u64;

            if cd_offset == 0xFFFFFFFF || cd_size == 0xFFFFFFFF {
                if let Some(result) = try_parse_eocd64(data, data_file_offset) {
                    return Ok(result);
                }
            }

            return Ok((cd_offset, cd_size));
        }
    }

    bail!("could not find End of Central Directory record")
}

fn try_parse_eocd64(data: &[u8], data_file_offset: u64) -> Option<(u64, u64)> {
    for i in (0..data.len().saturating_sub(19)).rev() {
        let sig = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
        if sig == EOCD64_LOCATOR_SIGNATURE {
            let eocd64_offset = u64::from_le_bytes([
                data[i + 8],
                data[i + 9],
                data[i + 10],
                data[i + 11],
                data[i + 12],
                data[i + 13],
                data[i + 14],
                data[i + 15],
            ]);

            if eocd64_offset >= data_file_offset {
                let local = (eocd64_offset - data_file_offset) as usize;
                if local + 56 <= data.len() {
                    let sig64 = u32::from_le_bytes([
                        data[local],
                        data[local + 1],
                        data[local + 2],
                        data[local + 3],
                    ]);
                    if sig64 == EOCD64_SIGNATURE {
                        let cd_size = u64::from_le_bytes([
                            data[local + 40],
                            data[local + 41],
                            data[local + 42],
                            data[local + 43],
                            data[local + 44],
                            data[local + 45],
                            data[local + 46],
                            data[local + 47],
                        ]);
                        let cd_offset = u64::from_le_bytes([
                            data[local + 48],
                            data[local + 49],
                            data[local + 50],
                            data[local + 51],
                            data[local + 52],
                            data[local + 53],
                            data[local + 54],
                            data[local + 55],
                        ]);
                        return Some((cd_offset, cd_size));
                    }
                }
            }
        }
    }
    None
}

fn parse_central_directory(data: &[u8]) -> Result<Vec<ZipEntry>> {
    let mut entries = Vec::new();
    let mut pos = 0;

    while pos + 46 <= data.len() {
        let sig = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        if sig != CD_SIGNATURE {
            break;
        }

        let compression = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
        let mut compressed_size = u32::from_le_bytes([
            data[pos + 20],
            data[pos + 21],
            data[pos + 22],
            data[pos + 23],
        ]) as u64;
        let mut uncompressed_size = u32::from_le_bytes([
            data[pos + 24],
            data[pos + 25],
            data[pos + 26],
            data[pos + 27],
        ]) as u64;
        let name_len = u16::from_le_bytes([data[pos + 28], data[pos + 29]]) as usize;
        let extra_len = u16::from_le_bytes([data[pos + 30], data[pos + 31]]) as usize;
        let comment_len = u16::from_le_bytes([data[pos + 32], data[pos + 33]]) as usize;
        let mut local_header_offset = u32::from_le_bytes([
            data[pos + 42],
            data[pos + 43],
            data[pos + 44],
            data[pos + 45],
        ]) as u64;

        let name_start = pos + 46;
        let name_end = name_start + name_len;
        if name_end > data.len() {
            break;
        }
        let name = String::from_utf8_lossy(&data[name_start..name_end]).into_owned();

        let extra_start = name_end;
        let extra_end = extra_start + extra_len;
        if extra_end <= data.len() {
            let extra = &data[extra_start..extra_end];
            parse_zip64_extra(
                extra,
                &mut uncompressed_size,
                &mut compressed_size,
                &mut local_header_offset,
            );
        }

        let is_dir = name.ends_with('/');

        entries.push(ZipEntry {
            name,
            compressed_size,
            uncompressed_size,
            compression_method: compression,
            local_header_offset,
            is_dir,
        });

        pos = extra_end + comment_len;
    }

    Ok(entries)
}

fn parse_zip64_extra(
    extra: &[u8],
    uncompressed_size: &mut u64,
    compressed_size: &mut u64,
    local_header_offset: &mut u64,
) {
    let mut epos = 0;
    while epos + 4 <= extra.len() {
        let tag = u16::from_le_bytes([extra[epos], extra[epos + 1]]);
        let size = u16::from_le_bytes([extra[epos + 2], extra[epos + 3]]) as usize;
        epos += 4;

        if tag == 0x0001 && epos + size <= extra.len() {
            let field = &extra[epos..epos + size];
            let mut fpos = 0;

            if *uncompressed_size == 0xFFFFFFFF && fpos + 8 <= field.len() {
                *uncompressed_size = u64::from_le_bytes([
                    field[fpos],
                    field[fpos + 1],
                    field[fpos + 2],
                    field[fpos + 3],
                    field[fpos + 4],
                    field[fpos + 5],
                    field[fpos + 6],
                    field[fpos + 7],
                ]);
                fpos += 8;
            }
            if *compressed_size == 0xFFFFFFFF && fpos + 8 <= field.len() {
                *compressed_size = u64::from_le_bytes([
                    field[fpos],
                    field[fpos + 1],
                    field[fpos + 2],
                    field[fpos + 3],
                    field[fpos + 4],
                    field[fpos + 5],
                    field[fpos + 6],
                    field[fpos + 7],
                ]);
                fpos += 8;
            }
            if *local_header_offset == 0xFFFFFFFF && fpos + 8 <= field.len() {
                *local_header_offset = u64::from_le_bytes([
                    field[fpos],
                    field[fpos + 1],
                    field[fpos + 2],
                    field[fpos + 3],
                    field[fpos + 4],
                    field[fpos + 5],
                    field[fpos + 6],
                    field[fpos + 7],
                ]);
            }
        }

        epos += size;
    }
}
