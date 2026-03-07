use anyhow::{Context, Result};
use memmap2::Mmap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Stats from extraction
#[derive(Debug, Default)]
pub struct ExtractStats {
    pub files_written: AtomicU64,
    pub bytes_written: AtomicU64,
    pub dirs_created: AtomicU64,
}

struct FileEntry {
    index: usize,
    name: String,
    uncompressed_size: u64,
    is_stored: bool,
}

/// Threshold for "large file" — files above this get their own thread
const LARGE_FILE_THRESHOLD: u64 = 1024 * 1024; // 1MB

/// Extract a .whl (zip) file to target directory.
///
/// Strategy: the bottleneck is decompressing large .so files (e.g., libtorch_cuda.so = 860MB).
/// These are inherently serial per-file (deflate is sequential).
///
/// Approach:
/// - Parse the central directory ONCE on the main thread
/// - Batch-create all directories
/// - Large files (>1MB): each gets its own rayon task (parallel decompression)
/// - Small files (<1MB): batched onto threads in chunks (avoid per-file overhead)
/// - STORED files: direct memcpy from mmap (zero decompression cost)
pub fn extract_wheel(
    wheel_path: &Path,
    target: &Path,
    threads: usize,
    use_fallocate: bool,
    stats: &Arc<ExtractStats>,
) -> Result<()> {
    let file = File::open(wheel_path)
        .with_context(|| format!("failed to open wheel: {}", wheel_path.display()))?;

    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to mmap wheel: {}", wheel_path.display()))?;

    #[cfg(target_os = "linux")]
    {
        use nix::sys::mman::{madvise, MmapAdvise};
        unsafe {
            let _ = madvise(
                std::ptr::NonNull::new(mmap.as_ptr() as *mut _).unwrap(),
                mmap.len(),
                MmapAdvise::MADV_SEQUENTIAL,
            );
        }
    }

    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(&mmap[..]))
        .with_context(|| format!("invalid zip: {}", wheel_path.display()))?;

    // Phase 1: Scan entries, collect dirs and categorize files
    let mut dirs = Vec::new();
    let mut large_files = Vec::new();
    let mut small_files = Vec::new();

    for i in 0..archive.len() {
        let entry = archive.by_index_raw(i)?;
        let name = entry.name().to_string();
        let is_dir = entry.is_dir();
        let size = entry.size();
        let is_stored = entry.compression() == zip::CompressionMethod::Stored;
        drop(entry);

        if is_dir {
            dirs.push(name);
            continue;
        }

        // Collect parent dirs
        let dest = target.join(&name);
        if let Some(parent) = dest.parent() {
            if let Ok(rel) = parent.strip_prefix(target) {
                let rel_str = rel.to_string_lossy().into_owned();
                if !rel_str.is_empty() {
                    dirs.push(rel_str);
                }
            }
        }

        let entry = FileEntry {
            index: i,
            name,
            uncompressed_size: size,
            is_stored,
        };

        if size >= LARGE_FILE_THRESHOLD {
            large_files.push(entry);
        } else {
            small_files.push(entry);
        }
    }

    // Phase 2: Batch create all directories
    dirs.sort();
    dirs.dedup();
    for dir in &dirs {
        let dest = target.join(dir);
        if !dest.exists() {
            fs::create_dir_all(&dest)?;
            stats.dirs_created.fetch_add(1, Ordering::Relaxed);
        }
    }

    // Phase 3: Extract files using rayon
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("failed to create thread pool")?;

    let mmap_ref: &[u8] = &mmap;
    let stats_ref = stats;

    pool.scope(|s| {
        // Each large file gets its own thread — these dominate runtime
        for entry in &large_files {
            let dest = target.join(&entry.name);
            let idx = entry.index;
            let size = entry.uncompressed_size;
            let stored = entry.is_stored;
            let name = &entry.name;

            s.spawn(move |_| {
                let result = if stored {
                    extract_stored(mmap_ref, idx, &dest, size, stats_ref)
                } else {
                    extract_compressed(mmap_ref, idx, &dest, size, use_fallocate, stats_ref)
                };
                if let Err(e) = result {
                    tracing::error!("failed to extract {name}: {e}");
                }
            });
        }

        // Small files: batch into chunks across threads
        // Each chunk shares one ZipArchive parse
        let chunk_size = (small_files.len() / threads).max(64);
        for chunk in small_files.chunks(chunk_size) {
            s.spawn(move |_| {
                let Ok(mut archive) =
                    zip::ZipArchive::new(std::io::Cursor::new(mmap_ref))
                else {
                    return;
                };

                for entry in chunk {
                    let dest = target.join(&entry.name);
                    let result = if entry.is_stored {
                        // For stored small files, extract via archive (simpler)
                        extract_from_archive(&mut archive, entry.index, &dest, entry.uncompressed_size, stats_ref)
                    } else {
                        extract_from_archive(&mut archive, entry.index, &dest, entry.uncompressed_size, stats_ref)
                    };
                    if let Err(e) = result {
                        tracing::error!("failed to extract {}: {e}", entry.name);
                    }
                }
            });
        }
    });

    Ok(())
}

/// Extract a STORED (uncompressed) file — direct memcpy from mmap
fn extract_stored(
    mmap: &[u8],
    index: usize,
    dest: &Path,
    expected_size: u64,
    stats: &ExtractStats,
) -> Result<()> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(mmap))?;
    let mut entry = archive.by_index(index)?;

    let file = File::create(dest)?;
    // 1MB buffer for large stored files — these are just memcpy from mmap
    let mut writer = std::io::BufWriter::with_capacity(1024 * 1024, file);
    std::io::copy(&mut entry, &mut writer)?;
    writer.flush()?;

    stats.files_written.fetch_add(1, Ordering::Relaxed);
    stats.bytes_written.fetch_add(expected_size, Ordering::Relaxed);
    set_permissions(dest);
    Ok(())
}

/// Extract a DEFLATED (compressed) file
fn extract_compressed(
    mmap: &[u8],
    index: usize,
    dest: &Path,
    expected_size: u64,
    use_fallocate: bool,
    stats: &ExtractStats,
) -> Result<()> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(mmap))?;
    let mut entry = archive.by_index(index)?;

    let file = File::create(dest)?;

    #[cfg(target_os = "linux")]
    if use_fallocate && expected_size > 0 {
        use nix::fcntl::{fallocate, FallocateFlags};
        use std::os::unix::io::AsRawFd;
        let _ = fallocate(file.as_raw_fd(), FallocateFlags::empty(), 0, expected_size as i64);
    }
    let _ = use_fallocate;

    // 1MB write buffer for large files
    let mut writer = std::io::BufWriter::with_capacity(1024 * 1024, file);

    // Read in 1MB chunks to keep things flowing
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = entry.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
    }
    writer.flush()?;

    stats.files_written.fetch_add(1, Ordering::Relaxed);
    stats.bytes_written.fetch_add(expected_size, Ordering::Relaxed);
    set_permissions(dest);
    Ok(())
}

/// Extract via a shared archive reference (for small files batched on one thread)
fn extract_from_archive(
    archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
    index: usize,
    dest: &Path,
    expected_size: u64,
    stats: &ExtractStats,
) -> Result<()> {
    let mut entry = archive.by_index(index)?;

    if let Some(parent) = dest.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(dest)?;
    let mut writer = std::io::BufWriter::with_capacity(256 * 1024, file);
    std::io::copy(&mut entry, &mut writer)?;
    writer.flush()?;

    stats.files_written.fetch_add(1, Ordering::Relaxed);
    stats.bytes_written.fetch_add(expected_size, Ordering::Relaxed);
    set_permissions(dest);
    Ok(())
}

#[cfg(unix)]
fn set_permissions(dest: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let name = dest.to_string_lossy();
    if name.ends_with(".so") || name.contains("/bin/") || name.contains("/scripts/") {
        let _ = fs::set_permissions(dest, fs::Permissions::from_mode(0o755));
    }
}

#[cfg(not(unix))]
fn set_permissions(_dest: &Path) {}
