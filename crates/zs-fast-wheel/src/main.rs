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
    /// Fast parallel wheel installation with GET+pipeline architecture.
    /// Checks cache in Rust for instant warm starts.
    /// Cold path: resolves via uv, downloads full wheels via single GET,
    /// pipelines download+extraction — all in Rust, no Python orchestrator.
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

/// Parallel downloads (default 32). Set ZS_PARALLEL_DOWNLOADS to tune.
fn configured_parallel_downloads() -> usize {
    env_usize("ZS_PARALLEL_DOWNLOADS", 32)
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
    exec_python_with_cuda(python, site_packages, target, target_args, &[])
}

fn exec_python_with_cuda(
    python: &std::path::Path,
    site_packages: &std::path::Path,
    target: &str,
    target_args: &[String],
    cuda_dirs: &[PathBuf],
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

    let mut cmd = std::process::Command::new(python);
    cmd.env("PYTHONPATH", &pythonpath)
        .arg("-c")
        .arg(&script);

    if !cuda_dirs.is_empty() {
        cmd.env("LD_LIBRARY_PATH", cuda_ld_library_path(cuda_dirs));
    }

    let err = cmd.exec();

    eprintln!("exec failed: {err}");
    std::process::exit(1);
}

/// Spawn Python with progressive loading: imports block until their package is extracted.
///
/// The daemon runs in the background writing ready markers to `ready_dir`.
/// Python starts immediately with an inline lazy import hook that polls for markers.
/// Imports block only until their specific package's marker appears.
fn spawn_python_progressive(
    python: &std::path::Path,
    site_packages: &std::path::Path,
    target: &str,
    target_args: &[String],
    ready_dir: &std::path::Path,
    import_map: &std::collections::HashMap<String, String>,
    cuda_dirs: &[PathBuf],
) -> std::process::Child {
    let mut pythonpath = site_packages.to_string_lossy().to_string();
    if let Ok(existing) = std::env::var("PYTHONPATH") {
        pythonpath = format!("{pythonpath}:{existing}");
    }

    let sp_str = site_packages.display().to_string();
    let args_str = format!("{target_args:?}");
    let ready_dir_str = ready_dir.display().to_string();

    // Serialize import_map as Python dict literal
    let mut map_parts = Vec::new();
    for (k, v) in import_map {
        map_parts.push(format!("'{}':'{}'", k.replace('\'', "\\'"), v.replace('\'', "\\'")));
    }
    let import_map_py = format!("{{{}}}", map_parts.join(","));

    // Python bootstrap: lazy import hook + entry point discovery
    let script = format!(
        r#"
import sys, os, time, importlib, importlib.abc, importlib.util, re, subprocess
from pathlib import Path
from configparser import ConfigParser

# --- Progressive loading hook ---
class _ZSHook(importlib.abc.MetaPathFinder):
    def __init__(self, ready_dir, import_map):
        self._ready_dir = ready_dir
        self._import_map = import_map
        self._resolved = set()
        self._wait_times = {{}}

    def _can_import(self, name):
        idx = sys.meta_path.index(self)
        sys.meta_path.pop(idx)
        try:
            return importlib.util.find_spec(name) is not None
        except (ModuleNotFoundError, ValueError, ImportError):
            return False
        finally:
            sys.meta_path.insert(idx, self)

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split('.')[0]
        if top in self._resolved:
            return None
        if self._can_import(fullname):
            self._resolved.add(top)
            return None

        dist = self._import_map.get(top, top)
        all_done = os.path.join(self._ready_dir, '.all_done')
        marker = os.path.join(self._ready_dir, dist)

        # Unknown package not in our map — don't wait
        if dist == top and top not in self._import_map and self._import_map:
            norm = top.lower().replace('-', '_')
            known = any(v.lower().replace('-', '_') == norm for v in self._import_map.values())
            if not known:
                self._resolved.add(top)
                return None

        t0 = time.monotonic()
        wait = 0.01
        while time.monotonic() - t0 < 300:
            if os.path.exists(marker) or os.path.exists(all_done):
                importlib.invalidate_caches()
                sys.path_importer_cache.clear()
                elapsed = time.monotonic() - t0
                if elapsed > 0.1:
                    self._wait_times[top] = elapsed
                    print(f'  [zerostart] {{top}}: ready ({{elapsed:.1f}}s)', file=sys.stderr, flush=True)
                self._resolved.add(top)
                return None
            time.sleep(wait)
            wait = min(wait * 1.5, 0.2)

        self._resolved.add(top)
        return None

_zs_hook = _ZSHook({ready_dir_repr}, {import_map_repr})
sys.meta_path.insert(0, _zs_hook)

# --- Entry point discovery + execution ---
sp = {sp_repr}
target = {target_repr}
args = {args_repr}
sys.path.insert(0, sp)
# Remove system dist-packages to prevent conflicts
sys.path[:] = [p for p in sys.path if p == sp or 'dist-packages' not in p]

if target.endswith('.py') or Path(target).is_file():
    sys.argv = [target] + args
    exec(compile(open(target).read(), target, 'exec'),
         {{'__name__': '__main__', '__file__': target}})
else:
    site = Path(sp)
    norm = re.sub(r'[-_.]+', '-', target).lower()
    dist_info = None
    for d in site.glob('*.dist-info'):
        stem = d.name.removesuffix('.dist-info')
        m = re.match(r'^(.+?)-\d', stem)
        pkg = m.group(1) if m else stem
        if re.sub(r'[-_.]+', '-', pkg).lower() == norm:
            dist_info = d
            break
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

# Print wait report
if _zs_hook._wait_times:
    total = sum(_zs_hook._wait_times.values())
    print(f'  [zerostart] total import wait: {{total:.1f}}s', file=sys.stderr, flush=True)
"#,
        ready_dir_repr = format!("'{}'", ready_dir_str.replace('\'', "\\'")),
        import_map_repr = import_map_py,
        sp_repr = format!("'{}'", sp_str.replace('\'', "\\'")),
        target_repr = format!("'{}'", target.replace('\'', "\\'")),
        args_repr = args_str,
    );

    let mut cmd = std::process::Command::new(python);
    cmd.env("PYTHONPATH", &pythonpath)
        .arg("-c")
        .arg(&script);

    if !cuda_dirs.is_empty() {
        cmd.env("LD_LIBRARY_PATH", cuda_ld_library_path(cuda_dirs));
    }

    match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            eprintln!("Failed to spawn Python: {e}");
            std::process::exit(1);
        }
    }
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

/// Detect system CUDA installation.
///
/// Returns (lib_dirs, cuda_version) if system CUDA is found and version is sufficient.
/// Only returns lib dirs if the system CUDA version >= the required version (12.8 for
/// torch 2.10's cu128 wheels). This prevents ABI mismatches where system CUDA 12.4
/// libraries lack symbols required by torch compiled against CUDA 12.8.
fn detect_system_cuda() -> (Vec<PathBuf>, Option<(u32, u32)>) {
    // Disable with env var
    if std::env::var("ZS_NO_SYSTEM_CUDA").is_ok() {
        return (Vec::new(), None);
    }

    let mut lib_dirs = Vec::new();

    // Detect CUDA version
    let version = detect_cuda_version();

    // Check CUDA_HOME / CUDA_PATH first (most reliable)
    for var in &["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(cuda_home) = std::env::var(var) {
            for subdir in &["lib64", "lib"] {
                let path = PathBuf::from(&cuda_home).join(subdir);
                if has_lib(&path, "libcudart") && !lib_dirs.contains(&path) {
                    lib_dirs.push(path);
                }
            }
        }
    }

    // Check standard locations
    for dir in &[
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/usr/lib/x86_64-linux-gnu",
    ] {
        let path = PathBuf::from(dir);
        if has_lib(&path, "libcudart") && !lib_dirs.contains(&path) {
            lib_dirs.push(path);
        }
    }

    // Also check for cuDNN (separate from CUDA toolkit)
    for dir in &[
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
    ] {
        let path = PathBuf::from(dir);
        if has_lib(&path, "libcudnn") && !lib_dirs.contains(&path) {
            lib_dirs.push(path);
        }
    }

    (lib_dirs, version)
}

/// Detect the system CUDA toolkit version.
///
/// Tries (in order):
/// 1. /usr/local/cuda/version.json (newer CUDA installs)
/// 2. /usr/local/cuda/version.txt (older CUDA installs)
/// 3. nvcc --version
fn detect_cuda_version() -> Option<(u32, u32)> {
    // Try version.json first (CUDA 12.0+)
    if let Ok(content) = std::fs::read_to_string("/usr/local/cuda/version.json") {
        // Parse: {"cuda": {"name": "CUDA SDK", "version": "12.4.1"}}
        if let Some(ver) = parse_version_from_json(&content) {
            return Some(ver);
        }
    }

    // Try version.txt
    if let Ok(content) = std::fs::read_to_string("/usr/local/cuda/version.txt") {
        // Format: "CUDA Version 12.4.1"
        if let Some(ver) = parse_version_from_text(&content) {
            return Some(ver);
        }
    }

    // Try nvcc --version
    if let Ok(output) = std::process::Command::new("nvcc").arg("--version").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // "Cuda compilation tools, release 12.4, V12.4.131"
            if let Some(ver) = parse_version_from_text(&stdout) {
                return Some(ver);
            }
        }
    }

    None
}

/// Parse CUDA version (major, minor) from version.json content.
fn parse_version_from_json(content: &str) -> Option<(u32, u32)> {
    // Simple parsing — look for "version": "12.4.1"
    let version_key = "\"version\"";
    let idx = content.find(version_key)?;
    let rest = &content[idx + version_key.len()..];
    // Skip to the value
    let quote_start = rest.find('"')? + 1;
    let rest = &rest[quote_start..];
    let quote_end = rest.find('"')?;
    let version_str = &rest[..quote_end];
    parse_version_string(version_str)
}

/// Parse CUDA version (major, minor) from text containing "X.Y" or "X.Y.Z".
fn parse_version_from_text(content: &str) -> Option<(u32, u32)> {
    // Look for a pattern like "12.4" or "12.4.1"
    for word in content.split(|c: char| !c.is_ascii_digit() && c != '.') {
        if let Some(ver) = parse_version_string(word) {
            if ver.0 >= 10 {
                // CUDA versions are 10+
                return Some(ver);
            }
        }
    }
    None
}

/// Parse "X.Y" or "X.Y.Z" into (major, minor).
fn parse_version_string(s: &str) -> Option<(u32, u32)> {
    let mut parts = s.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    Some((major, minor))
}

/// Check if a directory contains a library with the given prefix.
fn has_lib(dir: &Path, prefix: &str) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(prefix) && name.contains(".so") {
            return true;
        }
    }
    false
}

/// Map nvidia wheel distribution names to the .so library they provide.
/// Only skip a wheel if the system actually has the corresponding library.
/// Some libraries (cusparselt, nvshmem) are NOT part of the standard CUDA toolkit.
fn nvidia_wheel_system_lib(distribution: &str) -> Option<&'static str> {
    match distribution {
        d if d.starts_with("nvidia-cublas-cu") => Some("libcublas"),
        d if d.starts_with("nvidia-cuda-cupti-cu") => Some("libcupti"),
        d if d.starts_with("nvidia-cuda-nvrtc-cu") => Some("libnvrtc"),
        d if d.starts_with("nvidia-cuda-runtime-cu") => Some("libcudart"),
        d if d.starts_with("nvidia-cudnn-cu") => Some("libcudnn"),
        d if d.starts_with("nvidia-cufft-cu") => Some("libcufft"),
        d if d.starts_with("nvidia-cufile-cu") => Some("libcufile"),
        d if d.starts_with("nvidia-curand-cu") => Some("libcurand"),
        d if d.starts_with("nvidia-cusolver-cu") => Some("libcusolver"),
        d if d.starts_with("nvidia-cusparse-cu") && !d.contains("cusparselt") => {
            Some("libcusparse")
        }
        d if d.starts_with("nvidia-cusparselt-cu") => Some("libcusparseLt"),
        d if d.starts_with("nvidia-nccl-cu") => Some("libnccl"),
        d if d.starts_with("nvidia-nvjitlink-cu") => Some("libnvjitlink"),
        d if d.starts_with("nvidia-nvshmem-cu") => Some("libnvshmem"),
        d if d.starts_with("nvidia-nvtx-cu") => Some("libnvToolsExt"),
        _ => None,
    }
}

/// Check if a specific library exists in any of the CUDA directories.
fn system_has_lib(cuda_dirs: &[PathBuf], lib_prefix: &str) -> bool {
    cuda_dirs.iter().any(|dir| has_lib(dir, lib_prefix))
}

/// Extract the CUDA minor version required by an nvidia wheel from its version.
///
/// nvidia-cublas-cu12 version "12.8.3.14" → needs CUDA 12.8
/// nvidia-cudnn-cu12 version "9.8.0.87" → cuDNN (separate versioning, always needs matching CUDA)
fn required_cuda_minor(spec: &WheelSpec) -> u32 {
    // For nvidia-cuda-* and nvidia-cu* packages, the version starts with "12.X"
    // where X is the CUDA minor version
    if let Some((major, minor)) = parse_version_string(&spec.version) {
        if major == 12 {
            return minor;
        }
    }
    // For cuDNN, NCCL, etc. with their own versioning, we can't tell from the
    // version string. Fall back to requiring the latest known CUDA minor.
    // torch 2.10 uses cu128, so require 12.8.
    8
}

/// Build LD_LIBRARY_PATH with system CUDA dirs prepended.
fn cuda_ld_library_path(cuda_dirs: &[PathBuf]) -> String {
    let mut parts: Vec<String> = cuda_dirs
        .iter()
        .map(|p| p.display().to_string())
        .collect();
    if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
        parts.push(existing);
    }
    parts.join(":")
}

/// Write stub .dist-info for a skipped nvidia wheel so importlib.metadata
/// can find it (prevents "package not found" from dependency checkers).
fn write_stub_dist_info(
    site_packages: &Path,
    distribution: &str,
    version: &str,
) {
    let dist_dir = site_packages.join(format!(
        "{}-{}.dist-info",
        distribution.replace('-', "_"),
        version
    ));
    if dist_dir.exists() {
        return;
    }
    if std::fs::create_dir_all(&dist_dir).is_err() {
        return;
    }

    // Minimal METADATA
    let metadata = format!(
        "Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n"
    );
    let _ = std::fs::write(dist_dir.join("METADATA"), metadata);

    // INSTALLER
    let _ = std::fs::write(dist_dir.join("INSTALLER"), "zerostart\n");

    // RECORD (empty)
    let _ = std::fs::write(dist_dir.join("RECORD"), "");
}

/// Parse a requirements file into a list of package specifiers.
fn parse_requirements_file(path: &str) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read requirements file: {path}"))?;
    Ok(parse_requirements_text(&content))
}

/// Parse requirements from text (requirements.txt format).
fn parse_requirements_text(content: &str) -> Vec<String> {
    content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with('-'))
        .map(|l| l.to_string())
        .collect()
}

/// Parse PEP 723 inline script metadata from a Python script.
///
/// Looks for:
///   # /// script
///   # dependencies = ["torch", "numpy"]
///   # ///
fn parse_pep723_deps(script_path: &std::path::Path) -> Vec<String> {
    let content = match std::fs::read_to_string(script_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    // Extract the script metadata block
    let mut in_block = false;
    let mut metadata = String::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "# /// script" {
            in_block = true;
            continue;
        }
        if in_block {
            if trimmed == "# ///" {
                break;
            }
            // Strip leading "# " prefix
            if let Some(rest) = trimmed.strip_prefix("# ") {
                metadata.push_str(rest);
            } else if let Some(rest) = trimmed.strip_prefix("#") {
                metadata.push_str(rest);
            }
            metadata.push('\n');
        }
    }

    if metadata.is_empty() {
        return Vec::new();
    }

    // Parse as TOML and extract dependencies
    if let Ok(table) = metadata.parse::<toml::Value>() {
        if let Some(deps) = table.get("dependencies").and_then(|d| d.as_array()) {
            return deps
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
        }
    }

    Vec::new()
}

/// Auto-detect dependencies from the script's directory.
///
/// Searches for pyproject.toml or requirements.txt in the script's directory
/// and parent directories (up to 3 levels).
fn auto_detect_deps(script_path: &std::path::Path) -> Vec<String> {
    let dir = script_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));

    // Walk up to 3 parent directories
    let mut search_dir = dir.to_path_buf();
    for _ in 0..4 {
        // Check pyproject.toml
        let pyproject = search_dir.join("pyproject.toml");
        if pyproject.exists() {
            if let Ok(content) = std::fs::read_to_string(&pyproject) {
                if let Ok(deps) = resolve::parse_pyproject_dependencies(&content) {
                    if !deps.is_empty() {
                        tracing::info!(
                            "Auto-detected deps from {}",
                            pyproject.display()
                        );
                        return deps;
                    }
                }
            }
        }

        // Check requirements.txt
        let reqs_txt = search_dir.join("requirements.txt");
        if reqs_txt.exists() {
            if let Ok(content) = std::fs::read_to_string(&reqs_txt) {
                let deps = parse_requirements_text(&content);
                if !deps.is_empty() {
                    tracing::info!(
                        "Auto-detected deps from {}",
                        reqs_txt.display()
                    );
                    return deps;
                }
            }
        }

        if !search_dir.pop() {
            break;
        }
    }

    Vec::new()
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

            // Build requirements list from auto-detection + explicit flags
            let mut reqs: Vec<String> = Vec::new();
            let target_path = std::path::Path::new(&target);
            let is_script = target.ends_with(".py") || target_path.is_file();

            if !is_script {
                // Target is a package name (e.g. "zerostart run torch")
                reqs.push(target.clone());
            } else {
                // Target is a script — auto-detect deps
                // Priority: PEP 723 inline metadata > pyproject.toml > requirements.txt
                let pep723 = parse_pep723_deps(target_path);
                if !pep723.is_empty() {
                    if verbose {
                        eprintln!("Found PEP 723 deps: {}", pep723.join(", "));
                    }
                    reqs.extend(pep723);
                } else {
                    let auto = auto_detect_deps(target_path);
                    if !auto.is_empty() {
                        if verbose {
                            eprintln!("Auto-detected deps: {}", auto.join(", "));
                        }
                        reqs.extend(auto);
                    }
                }
            }

            // Explicit flags are additive
            reqs.extend(packages.clone());

            if let Some(ref req_file) = requirements {
                reqs.extend(parse_requirements_file(req_file)?);
            }

            if reqs.is_empty() {
                // No deps found anywhere — just exec the script directly
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
                    // Detect system CUDA for LD_LIBRARY_PATH
                    let (cuda_dirs, _) = detect_system_cuda();
                    exec_python_with_cuda(&python, &sp, &target, &target_args, &cuda_dirs);
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

            // Detect system CUDA — skip nvidia-* wheels when system provides compatible libs
            let (cuda_dirs, cuda_version) = detect_system_cuda();
            let daemon_wheels = if cuda_dirs.is_empty() {
                plan.daemon_wheels.clone()
            } else {
                let sys_major = cuda_version.map(|v| v.0).unwrap_or(0);
                let sys_minor = cuda_version.map(|v| v.1).unwrap_or(0);

                if verbose {
                    if let Some((maj, min)) = cuda_version {
                        eprintln!("System CUDA {maj}.{min} detected");
                    } else {
                        eprintln!("System CUDA detected (version unknown — skipping nvidia wheel optimization)");
                    }
                }

                // Only proceed with skipping if we know the system CUDA version
                if cuda_version.is_none() {
                    plan.daemon_wheels.clone()
                } else {
                    let mut keep = Vec::new();
                    let mut skipped = Vec::new();
                    for spec in &plan.daemon_wheels {
                        if let Some(lib_prefix) = nvidia_wheel_system_lib(&spec.distribution) {
                            let needed_minor = required_cuda_minor(spec);
                            if sys_major >= 12
                                && sys_minor >= needed_minor
                                && system_has_lib(&cuda_dirs, lib_prefix)
                            {
                                skipped.push(spec.clone());
                                continue;
                            }
                            if verbose {
                                if sys_minor < needed_minor {
                                    eprintln!(
                                        "  {} — needs CUDA 12.{needed_minor}, system has 12.{sys_minor}",
                                        spec.distribution
                                    );
                                } else {
                                    eprintln!(
                                        "  {} — {lib_prefix}.so not found on system",
                                        spec.distribution
                                    );
                                }
                            }
                        }
                        keep.push(spec.clone());
                    }
                    if !skipped.is_empty() {
                        let skipped_mb: f64 =
                            skipped.iter().map(|s| s.size as f64).sum::<f64>() / 1024.0 / 1024.0;
                        eprintln!(
                            "System CUDA {sys_major}.{sys_minor} — skipping {} nvidia wheels ({:.0} MB)",
                            skipped.len(),
                            skipped_mb
                        );
                        // Write stub .dist-info so importlib.metadata finds them
                        for spec in &skipped {
                            write_stub_dist_info(&site_packages, &spec.distribution, &spec.version);
                        }
                    }
                    keep
                }
            };

            if verbose {
                eprintln!(
                    "Resolved: {} packages ({} sdist via uv, {} wheels via daemon)",
                    plan.all.len(),
                    plan.uv_specs.len(),
                    daemon_wheels.len(),
                );
            }

            // Run uv sdist install + daemon download+extract in parallel
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

            if !daemon_wheels.is_empty() {
                let no_shared_cache = std::env::var("ZS_NO_SHARED_CACHE").is_ok();

                // Check shared cache in parallel — restore cached wheels via hardlinks
                let (uncached_wheels, cached_count) = if no_shared_cache {
                    (daemon_wheels.clone(), 0u32)
                } else {
                    let sp_for_cache = site_packages.clone();
                    let wheels_for_cache = daemon_wheels.clone();
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

                    for (spec, was_cached) in daemon_wheels.iter().zip(cache_results.iter()) {
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

                    // === Progressive loading ===
                    // Start daemon in background, start Python immediately.
                    // Python imports block only until their specific package is extracted.
                    let ready_dir = tempfile::tempdir()
                        .context("failed to create ready dir")?
                        .keep();

                    // Build import map: import_name → distribution_name
                    let mut import_map = std::collections::HashMap::new();
                    for spec in &uncached_wheels {
                        for root in &spec.import_roots {
                            import_map.insert(root.clone(), spec.distribution.clone());
                        }
                    }
                    // Also add cached wheels to import map (they're already extracted)
                    for spec in &daemon_wheels {
                        for root in &spec.import_roots {
                            import_map.entry(root.clone())
                                .or_insert_with(|| spec.distribution.clone());
                        }
                    }

                    let config = DaemonConfig {
                        site_packages: site_packages.clone(),
                        parallel_downloads: pd,
                        extract_threads: et,
                        ready_dir: Some(ready_dir.clone()),
                    };

                    let wheels_to_cache: Vec<WheelSpec> = uncached_wheels.clone();
                    let engine = std::sync::Arc::new(DaemonEngine::new(uncached_wheels));

                    // Wait for uv small install to finish before starting Python
                    if let Some(handle) = uv_handle {
                        handle.await??;
                    }

                    // Spawn daemon in background
                    let engine_bg = engine.clone();
                    let daemon_handle = tokio::spawn(async move {
                        engine_bg.run(&config).await
                    });

                    eprintln!("Starting {target} (packages installing in background)...");

                    // Start Python immediately with progressive loading hook
                    let mut child = spawn_python_progressive(
                        &python,
                        &site_packages,
                        &target,
                        &target_args,
                        &ready_dir,
                        &import_map,
                        &cuda_dirs,
                    );

                    // Wait for Python to finish
                    let py_status = child.wait().unwrap_or_else(|e| {
                        eprintln!("Failed to wait for Python: {e}");
                        std::process::exit(1);
                    });

                    // Wait for daemon to finish
                    if let Err(e) = daemon_handle.await? {
                        eprintln!("Warning: daemon error: {e}");
                    }

                    let (files, bytes) = engine.extract_stats();
                    if verbose {
                        eprintln!(
                            "Daemon: extracted {} files ({:.1} MB)",
                            files,
                            bytes as f64 / 1024.0 / 1024.0
                        );
                    }

                    // Mark complete now that everything is extracted
                    std::fs::File::create(&complete_marker).ok();

                    // Populate shared cache in background (best-effort for future cold starts)
                    if std::env::var("ZS_NO_SHARED_CACHE").is_err() {
                        let sp_for_cache = site_packages.clone();
                        tokio::task::spawn_blocking(move || {
                            if let Ok(avail) = available_disk_mb(&sp_for_cache) {
                                if avail < 2048 {
                                    return;
                                }
                            }
                            for spec in &wheels_to_cache {
                                populate_shared_cache(spec, &sp_for_cache);
                            }
                        });
                    }

                    // Clean up ready dir
                    let _ = std::fs::remove_dir_all(&ready_dir);

                    std::process::exit(py_status.code().unwrap_or(1));
                }
            }

            // Wait for uv small install to finish (if running)
            if let Some(handle) = uv_handle {
                handle.await??;
            }

            // Mark complete
            std::fs::File::create(&complete_marker).ok();

            if verbose {
                eprintln!("Environment ready — starting {target}");
            }

            exec_python_with_cuda(&python, &site_packages, &target, &target_args, &cuda_dirs);
        }

        Command::Install {
            wheels,
            target,
            parallel_downloads,
            extract_threads,
            fallocate,
            batch_sync,
            benchmark,
        } => {
            std::fs::create_dir_all(&target)?;

            let config = pipeline::PipelineConfig {
                target,
                parallel_downloads,
                extract_threads,
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
        ready_dir: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pep723_basic() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("test.py");
        std::fs::write(
            &script,
            r#"# /// script
# dependencies = ["torch", "numpy>=1.24"]
# ///

import torch
"#,
        )
        .unwrap();

        let deps = parse_pep723_deps(&script);
        assert_eq!(deps, vec!["torch", "numpy>=1.24"]);
    }

    #[test]
    fn test_parse_pep723_no_block() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("test.py");
        std::fs::write(&script, "import torch\nprint('hello')\n").unwrap();

        let deps = parse_pep723_deps(&script);
        assert!(deps.is_empty());
    }

    #[test]
    fn test_parse_pep723_empty_deps() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("test.py");
        std::fs::write(
            &script,
            "# /// script\n# dependencies = []\n# ///\nimport os\n",
        )
        .unwrap();

        let deps = parse_pep723_deps(&script);
        assert!(deps.is_empty());
    }

    #[test]
    fn test_parse_pep723_multiline() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("test.py");
        std::fs::write(
            &script,
            r#"# /// script
# dependencies = [
#   "torch>=2.0",
#   "transformers",
#   "safetensors",
# ]
# ///

import torch
"#,
        )
        .unwrap();

        let deps = parse_pep723_deps(&script);
        assert_eq!(deps, vec!["torch>=2.0", "transformers", "safetensors"]);
    }

    #[test]
    fn test_parse_requirements_text() {
        let text = "torch>=2.0\n# comment\nnumpy\n\n-f https://url\nrequests\n";
        let deps = parse_requirements_text(text);
        assert_eq!(deps, vec!["torch>=2.0", "numpy", "requests"]);
    }

    #[test]
    fn test_auto_detect_pyproject() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("pyproject.toml"),
            "[project]\ndependencies = [\"numpy\", \"requests\"]\n",
        )
        .unwrap();
        let script = dir.path().join("app.py");
        std::fs::write(&script, "import numpy\n").unwrap();

        let deps = auto_detect_deps(&script);
        assert_eq!(deps, vec!["numpy", "requests"]);
    }

    #[test]
    fn test_auto_detect_requirements_txt() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("requirements.txt"),
            "numpy\nrequests\n",
        )
        .unwrap();
        let script = dir.path().join("app.py");
        std::fs::write(&script, "import numpy\n").unwrap();

        let deps = auto_detect_deps(&script);
        assert_eq!(deps, vec!["numpy", "requests"]);
    }

    #[test]
    fn test_auto_detect_prefers_pyproject_over_requirements() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("pyproject.toml"),
            "[project]\ndependencies = [\"from-pyproject\"]\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("requirements.txt"),
            "from-requirements\n",
        )
        .unwrap();
        let script = dir.path().join("app.py");
        std::fs::write(&script, "pass\n").unwrap();

        let deps = auto_detect_deps(&script);
        assert_eq!(deps, vec!["from-pyproject"]);
    }

    #[test]
    fn test_auto_detect_walks_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let subdir = dir.path().join("src");
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::write(dir.path().join("requirements.txt"), "numpy\n").unwrap();
        let script = subdir.join("app.py");
        std::fs::write(&script, "import numpy\n").unwrap();

        let deps = auto_detect_deps(&script);
        assert_eq!(deps, vec!["numpy"]);
    }

    #[test]
    fn test_auto_detect_no_deps_found() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("app.py");
        std::fs::write(&script, "print('hello')\n").unwrap();

        let deps = auto_detect_deps(&script);
        assert!(deps.is_empty());
    }
}
