use zs_fast_wheel::daemon::{DaemonConfig, DaemonEngine};
use zs_fast_wheel::manifest::{Manifest, WheelSpec};
use zs_fast_wheel::pipeline;
use zs_fast_wheel::resolve;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "zerostart", about = "Fast Python package runner with progressive loading")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a Python package or script with fast installation
    ///
    /// Like uvx but with streaming wheel extraction for large packages.
    /// Checks cache in Rust for instant warm starts.
    /// Cold path: resolves via uv, installs small wheels via uv,
    /// streams large wheels via daemon — all in Rust, no Python orchestrator.
    ///
    /// Examples:
    ///   zerostart run comfyui
    ///   zerostart run -p torch comfyui -- --listen 0.0.0.0
    ///   zerostart run serve.py
    Run {
        /// Python script (.py) or package name to run
        target: String,

        /// Additional packages to install
        #[arg(short, long, action = clap::ArgAction::Append)]
        packages: Vec<String>,

        /// Requirements file
        #[arg(short, long)]
        requirements: Option<String>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Arguments passed to the target script/entry point
        #[arg(last = true)]
        target_args: Vec<String>,
    },

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
    ///   echo "torch\ntransformers" | zerostart warm --site-packages /opt/sp
    ///   zerostart warm --requirements "torch>=2.0\ntransformers" --site-packages /opt/sp
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

    /// Manage the zerostart cache
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },
}

#[derive(Subcommand)]
enum CacheAction {
    /// Show cache size and location
    Info,
    /// Remove all cached data (environments, shared wheels, resolution cache)
    Clean,
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Read a usize env var with a default. Used for tuning knobs.
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Parallel downloads (default 16). Set ZS_PARALLEL_DOWNLOADS to tune.
fn configured_parallel_downloads() -> usize {
    env_usize("ZS_PARALLEL_DOWNLOADS", 16)
}

/// Extraction threads (default num_cpus * 2). Set ZS_EXTRACT_THREADS to tune.
fn configured_extract_threads() -> usize {
    env_usize("ZS_EXTRACT_THREADS", num_cpus() * 2)
}

/// Compute the environment key (must match Python's _env_key).
fn env_key(requirements: &[String]) -> String {
    let mut sorted = requirements.to_vec();
    sorted.sort();

    let payload = serde_json::to_string(&sorted).unwrap_or_default();
    let hash = Sha256::digest(payload.as_bytes());
    hex::encode(hash)[..16].to_string()
}

/// Get the cache directory (matches Python's ENV_CACHE_DIR).
fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("ZEROSTART_CACHE") {
        PathBuf::from(dir)
    } else {
        dirs_or_home().join(".cache").join("zerostart")
    }
}

fn dirs_or_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}

/// Find Python interpreter — prefer uv's python, fall back to system.
fn find_python() -> Result<PathBuf> {
    if let Ok(output) = std::process::Command::new("uv")
        .args(["python", "find"])
        .output()
    {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Ok(PathBuf::from(path));
            }
        }
    }

    for name in &["python3", "python"] {
        if let Ok(output) = std::process::Command::new("which").arg(name).output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Ok(PathBuf::from(path));
                }
            }
        }
    }

    anyhow::bail!("no Python interpreter found (install via `uv python install` or add python3 to PATH)")
}

fn find_site_packages(env_dir: &std::path::Path) -> Option<PathBuf> {
    let lib_dir = env_dir.join("lib");
    if let Ok(entries) = std::fs::read_dir(&lib_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("python") {
                let sp = entry.path().join("site-packages");
                if sp.exists() {
                    return Some(sp);
                }
            }
        }
    }
    None
}

/// Exec Python to run the target (script or package entry point).
///
/// This replaces the current process — does not return on success.
fn exec_python(
    python: &std::path::Path,
    site_packages: &std::path::Path,
    target: &str,
    target_args: &[String],
) -> ! {
    let mut pythonpath = site_packages.to_string_lossy().to_string();
    if let Ok(existing) = std::env::var("PYTHONPATH") {
        pythonpath = format!("{pythonpath}:{existing}");
    }

    let sp_str = site_packages.display().to_string();
    let args_str = format!("{target_args:?}");

    // Inline entry point discovery — no dependency on zerostart Python package.
    // Discovers console_scripts from dist-info/entry_points.txt, or .data/scripts/ binaries.
    let script = format!(
        r#"
import sys, os, re, importlib, subprocess
from pathlib import Path
from configparser import ConfigParser

sp = {sp_repr}
target = {target_repr}
args = {args_repr}
sys.path.insert(0, sp)

if target.endswith('.py') or Path(target).is_file():
    sys.argv = [target] + args
    exec(compile(open(target).read(), target, 'exec'),
         {{'__name__': '__main__', '__file__': target}})
else:
    site = Path(sp)
    norm = re.sub(r'[-_.]+', '-', target).lower()

    # Find dist-info directory
    dist_info = None
    for d in site.glob('*.dist-info'):
        stem = d.name.removesuffix('.dist-info')
        m = re.match(r'^(.+?)-\d', stem)
        pkg = m.group(1) if m else stem
        if re.sub(r'[-_.]+', '-', pkg).lower() == norm:
            dist_info = d
            break

    # Try console_scripts from entry_points.txt
    ep = None
    if dist_info:
        ep_file = dist_info / 'entry_points.txt'
        if ep_file.exists():
            cp = ConfigParser()
            cp.read_string(ep_file.read_text())
            if cp.has_section('console_scripts'):
                items = list(cp.items('console_scripts'))
                if len(items) == 1:
                    ep = items[0]
                else:
                    for name, spec in items:
                        if re.sub(r'[-_.]+', '-', name).lower() == norm:
                            ep = (name, spec)
                            break
                    if not ep:
                        ep = items[0]

    if ep:
        name, spec = ep
        module_name, _, attr_name = spec.partition(':')
        sys.argv = [name] + args
        mod = importlib.import_module(module_name.strip())
        obj = mod
        for part in attr_name.strip().split('.'):
            obj = getattr(obj, part)
        obj()
    else:
        # Try .data/scripts/ binary (e.g. ruff)
        found = None
        for data_dir in site.glob('*.data'):
            dn = data_dir.name.removesuffix('.data') + '.dist-info'
            m2 = re.match(r'^(.+?)-\d', dn.removesuffix('.dist-info'))
            pkg2 = m2.group(1) if m2 else dn.removesuffix('.dist-info')
            if re.sub(r'[-_.]+', '-', pkg2).lower() != norm:
                continue
            scripts = data_dir / 'scripts'
            if scripts.is_dir():
                for f in scripts.iterdir():
                    if f.is_file() and os.access(f, os.X_OK):
                        found = f
                        if re.sub(r'[-_.]+', '-', f.name).lower() == norm:
                            break
            if found:
                break

        if found:
            r = subprocess.run([str(found)] + args)
            sys.exit(r.returncode)
        else:
            # Last resort: try importlib.metadata
            try:
                from importlib.metadata import distribution
                dist = distribution(target)
                for e in dist.entry_points:
                    if e.group == 'console_scripts':
                        module_name, _, attr_name = e.value.partition(':')
                        sys.argv = [e.name] + args
                        mod = importlib.import_module(module_name.strip())
                        obj = mod
                        for part in attr_name.strip().split('.'):
                            obj = getattr(obj, part)
                        obj()
                        sys.exit(0)
            except Exception:
                pass
            print(f"Error: no entry point found for '{{target}}'", file=sys.stderr)
            sys.exit(1)
"#,
        sp_repr = format!("'{}'", sp_str.replace('\'', "\\'")),
        target_repr = format!("'{}'", target.replace('\'', "\\'")),
        args_repr = args_str,
    );

    let err = std::process::Command::new(python)
        .env("PYTHONPATH", &pythonpath)
        .arg("-c")
        .arg(&script)
        .exec();

    eprintln!("exec failed: {err}");
    std::process::exit(1);
}

/// Create a venv using uv if it doesn't exist.
fn ensure_venv(venv: &std::path::Path, python: &std::path::Path) -> Result<()> {
    if venv.exists() {
        return Ok(());
    }

    let py_ver = resolve::detect_python_version(python)?;
    let output = std::process::Command::new("uv")
        .args(["venv", &venv.display().to_string(), "--python", &py_ver])
        .output()
        .context("failed to run uv venv")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("uv venv failed: {}", &stderr[..stderr.len().min(500)]);
    }

    Ok(())
}

/// Run `uv pip install` into a venv.
fn uv_install(venv: &std::path::Path, specs: &[String]) -> Result<()> {
    if specs.is_empty() {
        return Ok(());
    }

    let python = venv.join("bin").join("python");
    let mut args = vec![
        "pip".to_string(),
        "install".to_string(),
        "--no-deps".to_string(),
        "--python".to_string(),
        python.display().to_string(),
    ];
    args.extend(specs.iter().cloned());

    let output = std::process::Command::new("uv")
        .args(&args)
        .output()
        .context("failed to run uv pip install")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("uv pip install failed: {}", &stderr[..stderr.len().min(500)]);
    }

    Ok(())
}

/// Parse a requirements file into a list of package specifiers.
fn parse_requirements_file(path: &str) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read requirements file: {path}"))?;
    Ok(content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with('-'))
        .map(|l| l.to_string())
        .collect())
}

/// Shared wheel cache directory: `$ZEROSTART_CACHE/shared_wheels/{name}-{version}/`
///
/// CUDA libraries (nvidia-cuda-runtime-cu12, nvidia-cublas-cu12, etc.) are identical
/// across environments (torch, vllm, diffusers all share ~6GB of CUDA deps).
/// By caching extracted wheels and hardlinking them, we avoid re-downloading and
/// re-extracting the same wheels for every environment.
fn shared_wheel_cache_dir(spec: &WheelSpec) -> PathBuf {
    cache_dir()
        .join("shared_wheels")
        .join(format!("{}-{}", spec.distribution, spec.version))
}

/// Try to restore a wheel from the shared cache via hardlinks.
///
/// Returns true if the wheel was fully restored from cache.
fn restore_from_shared_cache(spec: &WheelSpec, site_packages: &Path) -> bool {
    let cache_path = shared_wheel_cache_dir(spec);
    let marker = cache_path.join(".complete");
    if !marker.exists() {
        return false;
    }

    if let Err(e) = hardlink_tree(&cache_path, site_packages) {
        tracing::warn!(
            "Failed to restore {} from shared cache: {e}",
            spec.distribution
        );
        return false;
    }

    true
}

/// Populate the shared cache from a freshly extracted wheel in site-packages.
///
/// Uses the RECORD file from dist-info to get the exact list of installed files.
/// Falls back to heuristic matching if RECORD is missing.
fn populate_shared_cache(spec: &WheelSpec, site_packages: &Path) {
    let cache_path = shared_wheel_cache_dir(spec);
    if cache_path.join(".complete").exists() {
        return; // already cached
    }

    // Use a staging dir for atomic population
    let staging = cache_path.with_extension("staging");
    let _ = std::fs::remove_dir_all(&staging); // clean stale staging
    if std::fs::create_dir_all(&staging).is_err() {
        return;
    }

    // Find dist-info directory for this package
    let norm = spec.distribution.replace('-', "_").to_lowercase();
    let dist_info = find_dist_info(site_packages, &norm);

    // Try RECORD-based populate first (ground truth)
    let populated = if let Some(ref di) = dist_info {
        populate_from_record(site_packages, di, &staging)
    } else {
        false
    };

    // Fallback: copy dist-info + known import roots
    if !populated {
        populate_heuristic(site_packages, &norm, &spec.import_roots, &staging);
    }

    // Atomic commit: rename staging → final
    if let Ok(()) = std::fs::create_dir_all(cache_path.parent().unwrap_or(Path::new("."))) {
        let _ = std::fs::rename(&staging, &cache_path);
        let _ = std::fs::File::create(cache_path.join(".complete"));
    } else {
        let _ = std::fs::remove_dir_all(&staging);
    }
}

/// Find the dist-info directory for a package in site-packages.
fn find_dist_info(site_packages: &Path, norm_name: &str) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(site_packages).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy().to_string();
        if name_str.ends_with(".dist-info") {
            let stem = name_str.trim_end_matches(".dist-info");
            let pkg = stem.split('-').next().unwrap_or(stem);
            if pkg.replace('-', "_").to_lowercase() == *norm_name {
                return Some(entry.path());
            }
        }
    }
    None
}

/// Populate cache from RECORD file — lists every file the wheel installed.
///
/// RECORD format: `path,hash,size` per line. Paths are relative to site-packages.
/// Returns true if RECORD was found and files were copied.
fn populate_from_record(site_packages: &Path, dist_info: &Path, staging: &Path) -> bool {
    let record_path = dist_info.join("RECORD");
    let content = match std::fs::read_to_string(&record_path) {
        Ok(c) => c,
        Err(_) => return false,
    };

    // Collect unique top-level dirs/files from RECORD
    let mut top_level_entries: std::collections::HashSet<String> = std::collections::HashSet::new();
    for line in content.lines() {
        let path = line.split(',').next().unwrap_or("").trim();
        if path.is_empty() {
            continue;
        }
        // Top-level entry is the first path component
        if let Some(top) = path.split('/').next() {
            top_level_entries.insert(top.to_string());
        }
    }

    if top_level_entries.is_empty() {
        return false;
    }

    let mut copied_any = false;
    for entry_name in &top_level_entries {
        let src = site_packages.join(entry_name);
        let dst = staging.join(entry_name);
        if src.is_dir() {
            if copy_dir_recursive(&src, &dst).is_ok() {
                copied_any = true;
            }
        } else if src.is_file() {
            if let Some(parent) = dst.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if std::fs::copy(&src, &dst).is_ok() {
                copied_any = true;
            }
        }
    }

    copied_any
}

/// Fallback: populate cache using heuristic name matching (dist-info + import roots).
fn populate_heuristic(
    site_packages: &Path,
    norm_name: &str,
    import_roots: &[String],
    staging: &Path,
) {
    if let Ok(entries) = std::fs::read_dir(site_packages) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy().to_string();

            let is_dist_info = name_str.ends_with(".dist-info") && {
                let stem = name_str.trim_end_matches(".dist-info");
                let pkg = stem.split('-').next().unwrap_or(stem);
                pkg.replace('-', "_").to_lowercase() == *norm_name
            };

            let is_data_dir = name_str.ends_with(".data") && {
                let stem = name_str.trim_end_matches(".data");
                let pkg = stem.split('-').next().unwrap_or(stem);
                pkg.replace('-', "_").to_lowercase() == *norm_name
            };

            let is_import_root = import_roots
                .iter()
                .any(|r| r == &name_str || name_str == format!("{norm_name}.py"));

            if is_dist_info || is_data_dir || is_import_root {
                let src = entry.path();
                let dst = staging.join(&name);
                if src.is_dir() {
                    let _ = copy_dir_recursive(&src, &dst);
                } else {
                    let _ = std::fs::copy(&src, &dst);
                }
            }
        }
    }
}

/// Recursively hardlink all files from src tree into dst.
fn hardlink_tree(src: &Path, dst: &Path) -> Result<()> {
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip .complete marker
        if name_str == ".complete" {
            continue;
        }

        let src_path = entry.path();
        let dst_path = dst.join(&name);

        if src_path.is_dir() {
            std::fs::create_dir_all(&dst_path)?;
            hardlink_tree(&src_path, &dst_path)?;
        } else if !dst_path.exists() {
            // Try hardlink first, fall back to copy (cross-device)
            if let Err(e) = std::fs::hard_link(&src_path, &dst_path) {
                tracing::debug!(
                    "Hardlink failed ({}), falling back to copy: {}",
                    e,
                    src_path.display()
                );
                std::fs::copy(&src_path, &dst_path)?;
            }
        }
    }
    Ok(())
}

/// Recursively copy a directory.
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            target,
            packages,
            requirements,
            verbose,
            target_args,
        } => {
            let python = find_python()?;

            // Build requirements list
            let mut reqs: Vec<String> = Vec::new();
            let is_script = target.ends_with(".py")
                || std::path::Path::new(&target).is_file();

            if !is_script {
                reqs.push(target.clone());
            }
            reqs.extend(packages.clone());

            if let Some(ref req_file) = requirements {
                reqs.extend(parse_requirements_file(req_file)?);
            }

            if reqs.is_empty() {
                // No deps — just exec the script directly
                exec_python(&python, std::path::Path::new("."), &target, &target_args);
            }

            // Check warm cache
            let key = env_key(&reqs);
            let env_dir = cache_dir().join("envs").join(&key);
            let complete_marker = env_dir.join(".complete");

            if complete_marker.exists() {
                if let Some(sp) = find_site_packages(&env_dir) {
                    if verbose {
                        eprintln!("Cache hit — skipping resolution");
                    }
                    exec_python(&python, &sp, &target, &target_args);
                }
            }

            // === COLD PATH — all in Rust ===
            if verbose {
                eprintln!("Cache miss — resolving...");
            }

            // Create venv
            ensure_venv(&env_dir, &python)?;
            let site_packages = find_site_packages(&env_dir)
                .context("could not find site-packages in venv")?;

            // Resolve via uv pip compile --format pylock.toml
            let py_ver = resolve::detect_python_version(&python)?;
            let platform = resolve::detect_platform().to_string();
            let plan = resolve::resolve_requirements(&reqs, &py_ver, &platform)?;

            if plan.all.is_empty() {
                eprintln!("No packages resolved");
                std::fs::File::create(&complete_marker).ok();
                exec_python(&python, &site_packages, &target, &target_args);
            }

            if verbose {
                eprintln!(
                    "Resolved: {} packages ({} sdist via uv, {} wheels via daemon)",
                    plan.all.len(),
                    plan.uv_specs.len(),
                    plan.daemon_wheels.len(),
                );
            }

            // Run uv sdist install + daemon streaming in parallel
            let uv_specs = plan.uv_specs.clone();
            let env_dir_clone = env_dir.clone();
            let uv_verbose = verbose;

            let uv_handle = if !uv_specs.is_empty() {
                if verbose {
                    eprintln!("Installing {} sdist-only packages via uv...", uv_specs.len());
                }
                Some(tokio::task::spawn_blocking(move || {
                    let result = uv_install(&env_dir_clone, &uv_specs);
                    if uv_verbose {
                        if let Err(ref e) = result {
                            eprintln!("Warning: uv sdist install failed: {e}");
                        }
                    }
                    result
                }))
            } else {
                None
            };

            if !plan.daemon_wheels.is_empty() {
                let no_shared_cache = std::env::var("ZS_NO_SHARED_CACHE").is_ok();

                // Check shared cache in parallel — restore cached wheels via hardlinks
                let (uncached_wheels, cached_count) = if no_shared_cache {
                    (plan.daemon_wheels.clone(), 0u32)
                } else {
                    let sp_for_cache = site_packages.clone();
                    let wheels_for_cache = plan.daemon_wheels.clone();
                    let cache_results: Vec<bool> = tokio::task::spawn_blocking(move || {
                        use rayon::prelude::*;
                        wheels_for_cache
                            .par_iter()
                            .map(|spec| restore_from_shared_cache(spec, &sp_for_cache))
                            .collect()
                    })
                    .await?;

                    let mut uncached = Vec::new();
                    let mut count = 0u32;

                    for (spec, was_cached) in plan.daemon_wheels.iter().zip(cache_results.iter()) {
                        if *was_cached {
                            count += 1;
                            if verbose {
                                eprintln!("  {} (shared cache hit)", spec.distribution);
                            }
                        } else {
                            uncached.push(spec.clone());
                        }
                    }
                    (uncached, count)
                };

                if verbose {
                    if cached_count > 0 {
                        eprintln!(
                            "Shared cache: {cached_count} wheels restored, {} to download",
                            uncached_wheels.len()
                        );
                    }
                    if !uncached_wheels.is_empty() {
                        eprintln!("Streaming {} packages via daemon...", uncached_wheels.len());
                        for w in &uncached_wheels {
                            eprintln!("  {} ({:.1} MB)", w.distribution, w.size as f64 / 1024.0 / 1024.0);
                        }
                    }
                }

                if !uncached_wheels.is_empty() {
                    let pd = configured_parallel_downloads();
                    let et = configured_extract_threads();
                    if verbose {
                        eprintln!("Config: parallel_downloads={pd}, extract_threads={et}");
                    }
                    let config = DaemonConfig {
                        site_packages: site_packages.clone(),
                        parallel_downloads: pd,
                        extract_threads: et,
                    };

                    let wheels_to_cache: Vec<WheelSpec> = uncached_wheels.clone();
                    let engine = DaemonEngine::new(uncached_wheels);
                    engine.run(&config).await?;

                    let (files, bytes) = engine.extract_stats();
                    if verbose {
                        eprintln!(
                            "Daemon: extracted {} files ({:.1} MB)",
                            files,
                            bytes as f64 / 1024.0 / 1024.0
                        );
                    }

                    // Populate shared cache for newly extracted wheels
                    // Skip if ZS_NO_SHARED_CACHE=1 or disk is tight (<2GB free)
                    if std::env::var("ZS_NO_SHARED_CACHE").is_ok() {
                        tracing::info!("Shared cache disabled via ZS_NO_SHARED_CACHE");
                    } else {
                        let sp_for_cache = site_packages.clone();
                        tokio::task::spawn_blocking(move || {
                            if let Ok(avail) = available_disk_mb(&sp_for_cache) {
                                if avail < 2048 {
                                    tracing::warn!(
                                        "Skipping shared cache — only {}MB free",
                                        avail
                                    );
                                    return;
                                }
                            }
                            for spec in &wheels_to_cache {
                                populate_shared_cache(spec, &sp_for_cache);
                            }
                        })
                        .await?;
                    }
                }
            }

            // Wait for uv small install to finish (if running)
            if let Some(handle) = uv_handle {
                handle.await??;
            }

            // Mark complete — no cache population needed.
            // Warm path uses our venv directly, doesn't need uv's cache.
            std::fs::File::create(&complete_marker).ok();

            if verbose {
                eprintln!("Environment ready — starting {target}");
            }

            exec_python(&python, &site_packages, &target, &target_args);
        }

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
            let req_text = if let Some(text) = requirements {
                text
            } else if let Some(toml_text) = pyproject {
                let deps = resolve::parse_pyproject_dependencies(&toml_text)?;
                deps.join("\n")
            } else {
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

            let reqs: Vec<String> = req_text
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect();

            eprintln!("Resolving requirements...");
            let plan = resolve::resolve_requirements(&reqs, &python_version, &platform)?;

            if plan.all.is_empty() {
                anyhow::bail!("no wheels resolved from requirements");
            }

            let wheels = plan.daemon_wheels;
            if wheels.is_empty() {
                eprintln!("All {} packages are small — use `uv pip install` directly", plan.all.len());
                return Ok(());
            }

            eprintln!("Resolved {} large wheels, starting download...", wheels.len());
            for w in &wheels {
                eprintln!("  {} ({:.1} MB)", w.distribution, w.size as f64 / 1024.0 / 1024.0);
            }

            run_engine(wheels.clone(), site_packages.clone(), parallel_downloads, extract_threads).await?;

            // Populate shared cache for future environments
            eprintln!("Populating shared cache...");
            for spec in &wheels {
                populate_shared_cache(spec, &site_packages);
            }
            eprintln!("Shared cache populated ({} wheels)", wheels.len());

            Ok(())
        }

        Command::Cache { action } => {
            let base = cache_dir();
            match action {
                CacheAction::Info => {
                    let envs = dir_size(&base.join("envs"));
                    let shared = dir_size(&base.join("shared_wheels"));
                    let pylock = dir_size(&base.join("pylock"));
                    let total = envs + shared + pylock;

                    eprintln!("Cache directory: {}", base.display());
                    eprintln!("  Environments:    {}", human_size(envs));
                    eprintln!("  Shared wheels:   {}", human_size(shared));
                    eprintln!("  Resolution:      {}", human_size(pylock));
                    eprintln!("  Total:           {}", human_size(total));
                    Ok(())
                }
                CacheAction::Clean => {
                    let size = dir_size(&base);
                    if base.exists() {
                        std::fs::remove_dir_all(&base)
                            .context("failed to remove cache directory")?;
                        eprintln!("Removed {} ({})", base.display(), human_size(size));
                    } else {
                        eprintln!("Cache directory does not exist: {}", base.display());
                    }
                    Ok(())
                }
            }
        }
    }
}

/// Check available disk space in MB for the filesystem containing `path`.
fn available_disk_mb(path: &Path) -> Result<u64> {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .context("invalid path for statvfs")?;
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
        if ret != 0 {
            anyhow::bail!("statvfs failed");
        }
        Ok((stat.f_bavail as u64 * stat.f_frsize as u64) / (1024 * 1024))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(u64::MAX) // assume unlimited on non-unix
    }
}

/// Recursively compute directory size in bytes.
fn dir_size(path: &Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let ft = entry.file_type().unwrap_or_else(|_| {
                std::fs::symlink_metadata(entry.path())
                    .map(|m| m.file_type())
                    .unwrap_or_else(|_| entry.file_type().unwrap())
            });
            if ft.is_dir() {
                total += dir_size(&entry.path());
            } else {
                total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    total
}

/// Format bytes as human-readable size.
fn human_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / 1024.0 / 1024.0 / 1024.0)
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / 1024.0 / 1024.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
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
