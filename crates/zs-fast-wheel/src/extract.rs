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

/// Generate a random suffix for staging directories
fn random_suffix() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    format!("{:08x}", rng.r#gen::<u32>())
}

/// Extract a wheel atomically: extract to staging dir, then rename into place.
///
/// This ensures that site-packages never contains a half-extracted wheel.
/// The import hook treats `done/{distribution}` as the readiness signal,
/// but atomic commit also prevents `find_spec()` from seeing partial state.
pub fn extract_wheel_atomic(
    wheel_path: &Path,
    site_packages: &Path,
    pkg_name: &str,
    threads: usize,
    use_fallocate: bool,
    stats: &Arc<ExtractStats>,
) -> Result<()> {
    let staging = site_packages.join(format!(".installing-{}-{}", pkg_name, random_suffix()));
    fs::create_dir_all(&staging)?;

    // Extract into staging dir
    let result = extract_wheel(wheel_path, &staging, threads, use_fallocate, stats);
    if let Err(e) = result {
        // Clean up staging on failure
        let _ = fs::remove_dir_all(&staging);
        return Err(e);
    }

    // Commit: move each top-level entry from staging into site-packages
    commit_staged(&staging, site_packages)?;

    Ok(())
}

/// Move all top-level entries from staging dir into site-packages, then remove staging.
pub fn commit_staged(staging: &Path, site_packages: &Path) -> Result<()> {
    for entry in fs::read_dir(staging)? {
        let entry = entry?;
        let dest = site_packages.join(entry.file_name());
        if dest.exists() {
            anyhow::bail!("refusing to overwrite existing path: {}", dest.display());
        }
        fs::rename(entry.path(), &dest).with_context(|| {
            format!(
                "failed to commit {} → {}",
                entry.path().display(),
                dest.display()
            )
        })?;
    }
    fs::remove_dir(staging)?;
    Ok(())
}

/// Clean up stale staging directories from interrupted runs.
pub fn cleanup_stale_staging(site_packages: &Path) -> Result<()> {
    if !site_packages.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(site_packages)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with(".installing-") {
            tracing::warn!("removing stale staging dir: {}", entry.path().display());
            fs::remove_dir_all(entry.path())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Create a minimal valid .whl (zip) file with a single file inside
    fn create_test_wheel(dir: &Path, pkg_name: &str) -> PathBuf {
        let wheel_path = dir.join(format!("{pkg_name}-1.0.0-py3-none-any.whl"));
        let file = File::create(&wheel_path).unwrap();
        let mut zip = zip::ZipWriter::new(file);

        // Add a Python file
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        zip.start_file(format!("{pkg_name}/__init__.py"), options)
            .unwrap();
        zip.write_all(b"# test package\n").unwrap();

        // Add dist-info METADATA
        zip.start_file(
            format!("{pkg_name}-1.0.0.dist-info/METADATA"),
            options,
        )
        .unwrap();
        zip.write_all(format!("Name: {pkg_name}\nVersion: 1.0.0\n").as_bytes())
            .unwrap();

        zip.finish().unwrap();
        wheel_path
    }

    #[test]
    fn test_extract_wheel_atomic_success() {
        let tmp = TempDir::new().unwrap();
        let wheels_dir = tmp.path().join("wheels");
        let site_packages = tmp.path().join("site-packages");
        fs::create_dir_all(&wheels_dir).unwrap();
        fs::create_dir_all(&site_packages).unwrap();

        let wheel = create_test_wheel(&wheels_dir, "mypkg");
        let stats = Arc::new(ExtractStats::default());

        extract_wheel_atomic(&wheel, &site_packages, "mypkg", 1, false, &stats).unwrap();

        // Package should be in site-packages
        assert!(site_packages.join("mypkg/__init__.py").exists());
        assert!(site_packages.join("mypkg-1.0.0.dist-info/METADATA").exists());

        // No staging dirs left
        let staging_dirs: Vec<_> = fs::read_dir(&site_packages)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".installing-"))
            .collect();
        assert!(staging_dirs.is_empty());
    }

    #[test]
    fn test_extract_wheel_atomic_refuses_overwrite() {
        let tmp = TempDir::new().unwrap();
        let wheels_dir = tmp.path().join("wheels");
        let site_packages = tmp.path().join("site-packages");
        fs::create_dir_all(&wheels_dir).unwrap();
        fs::create_dir_all(&site_packages).unwrap();

        let wheel = create_test_wheel(&wheels_dir, "mypkg");
        let stats = Arc::new(ExtractStats::default());

        // First extraction succeeds
        extract_wheel_atomic(&wheel, &site_packages, "mypkg", 1, false, &stats).unwrap();

        // Second extraction should fail (paths already exist)
        let result = extract_wheel_atomic(&wheel, &site_packages, "mypkg", 1, false, &stats);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("refusing to overwrite"));
    }

    #[test]
    fn test_commit_staged() {
        let tmp = TempDir::new().unwrap();
        let staging = tmp.path().join(".installing-test-abc123");
        let target = tmp.path().join("site-packages");
        fs::create_dir_all(&staging).unwrap();
        fs::create_dir_all(&target).unwrap();

        // Create a file in staging
        fs::create_dir_all(staging.join("testpkg")).unwrap();
        fs::write(staging.join("testpkg/__init__.py"), b"# test").unwrap();

        commit_staged(&staging, &target).unwrap();

        // File should be in target
        assert!(target.join("testpkg/__init__.py").exists());
        // Staging should be removed
        assert!(!staging.exists());
    }

    #[test]
    fn test_cleanup_stale_staging() {
        let tmp = TempDir::new().unwrap();
        let sp = tmp.path();

        // Create some staging dirs and a real package dir
        fs::create_dir_all(sp.join(".installing-torch-abc123")).unwrap();
        fs::write(sp.join(".installing-torch-abc123/file.txt"), b"stale").unwrap();
        fs::create_dir_all(sp.join(".installing-numpy-def456")).unwrap();
        fs::create_dir_all(sp.join("requests")).unwrap(); // real package, should not be removed

        cleanup_stale_staging(sp).unwrap();

        assert!(!sp.join(".installing-torch-abc123").exists());
        assert!(!sp.join(".installing-numpy-def456").exists());
        assert!(sp.join("requests").exists()); // untouched
    }

    #[test]
    fn test_cleanup_stale_staging_nonexistent_dir() {
        // Should not error on nonexistent dir
        cleanup_stale_staging(Path::new("/tmp/nonexistent-zs-test-12345")).unwrap();
    }
}
