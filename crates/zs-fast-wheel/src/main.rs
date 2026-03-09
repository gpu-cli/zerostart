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
/// We identify the wheel's files by looking for its .dist-info directory,
/// then copy the top-level dirs that belong to it into the cache.
fn populate_shared_cache(spec: &WheelSpec, site_packages: &Path) {
    let cache_path = shared_wheel_cache_dir(spec);
    if cache_path.join(".complete").exists() {
        return; // already cached
    }

    if std::fs::create_dir_all(&cache_path).is_err() {
        return;
    }

    // Find this wheel's dist-info and import roots in site-packages
    let norm = spec.distribution.replace('-', "_").to_lowercase();
    if let Ok(entries) = std::fs::read_dir(site_packages) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy().to_string();

            // Match dist-info dir or import root dirs
            let is_dist_info = name_str.ends_with(".dist-info") && {
                let stem = name_str.trim_end_matches(".dist-info");
                let pkg = stem.split('-').next().unwrap_or(stem);
                pkg.replace('-', "_").to_lowercase() == norm
            };

            let is_data_dir = name_str.ends_with(".data") && {
                let stem = name_str.trim_end_matches(".data");
                let pkg = stem.split('-').next().unwrap_or(stem);
                pkg.replace('-', "_").to_lowercase() == norm
            };

            let is_import_root = spec
                .import_roots
                .iter()
                .any(|r| r == &name_str || name_str == format!("{norm}.py"));

            if is_dist_info || is_data_dir || is_import_root {
                let src = entry.path();
                let dst = cache_path.join(&name);
                if src.is_dir() {
                    let _ = copy_dir_recursive(&src, &dst);
                } else {
                    let _ = std::fs::copy(&src, &dst);
                }
            }
        }
    }

    // Mark cache as complete
    let _ = std::fs::File::create(cache_path.join(".complete"));
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
        } else {
            // Try hardlink first, fall back to copy (cross-device)
            if std::fs::hard_link(&src_path, &dst_path).is_err() {
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
                // Check shared cache in parallel — restore cached wheels via hardlinks
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

                let mut uncached_wheels = Vec::new();
                let mut cached_count = 0u32;

                for (spec, was_cached) in plan.daemon_wheels.iter().zip(cache_results.iter()) {
                    if *was_cached {
                        cached_count += 1;
                        if verbose {
                            eprintln!("  {} (shared cache hit)", spec.distribution);
                        }
                    } else {
                        uncached_wheels.push(spec.clone());
                    }
                }

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
                    let sp_for_cache = site_packages.clone();
                    tokio::task::spawn_blocking(move || {
                        for spec in &wheels_to_cache {
                            populate_shared_cache(spec, &sp_for_cache);
                        }
                    })
                    .await?;
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

            run_engine(wheels, site_packages, parallel_downloads, extract_threads).await
        }
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
