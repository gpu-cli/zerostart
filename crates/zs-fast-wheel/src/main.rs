use zs_fast_wheel::daemon::{DaemonConfig, DaemonEngine};
use zs_fast_wheel::manifest::Manifest;
use zs_fast_wheel::pipeline;
use zs_fast_wheel::resolve;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
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
    /// Like uvx but with streaming wheel extraction and progressive loading.
    /// Checks cache in Rust for instant warm starts, falls back to Python
    /// orchestrator for cold installs.
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

        /// Cache directory
        #[arg(long, default_value = ".zerostart")]
        cache_dir: PathBuf,

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

/// Compute the input hash for cache lookup (must match Python's _input_hash).
fn input_hash(requirements: &[String], python_version: &str, platform: &str) -> String {
    let mut sorted = requirements.to_vec();
    sorted.sort();

    let payload = serde_json::json!({
        "requirements": sorted,
        "python_version": python_version,
        "platform": platform,
    });

    let bytes = serde_json::to_string(&payload).unwrap_or_default();
    let hash = Sha256::digest(bytes.as_bytes());
    hex::encode(hash)
}

/// Find Python interpreter — prefer uv's python, fall back to system.
fn find_python() -> Result<PathBuf> {
    // Try uv python first
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

    // Fall back to python3/python on PATH
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

/// Detect Python version from the interpreter binary path or by running it.
fn detect_python_version(python: &PathBuf) -> Result<String> {
    // Try to extract from binary name (e.g. "python3.11" → "3.11")
    if let Some(name) = python.file_name().and_then(|n| n.to_str()) {
        if let Some(ver) = name.strip_prefix("python") {
            if ver.contains('.') && ver.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                return Ok(ver.to_string());
            }
        }
    }

    // Fall back to running python --version
    let output = std::process::Command::new(python)
        .args(["--version"])
        .output()
        .context("failed to run Python for version detection")?;

    // Output: "Python 3.11.5"
    let version_str = String::from_utf8_lossy(&output.stdout);
    let version_str = version_str.trim();
    if let Some(ver) = version_str.strip_prefix("Python ") {
        let parts: Vec<&str> = ver.split('.').collect();
        if parts.len() >= 2 {
            return Ok(format!("{}.{}", parts[0], parts[1]));
        }
    }

    anyhow::bail!("could not parse Python version from: {version_str}");
}

/// Detect platform from OS at compile time + runtime check.
fn detect_platform() -> String {
    if cfg!(target_os = "linux") {
        "linux".to_string()
    } else if cfg!(target_os = "macos") {
        "macos".to_string()
    } else {
        std::env::consts::OS.to_string()
    }
}

/// Try the fast warm path: check input cache, exec Python directly if hit.
///
/// Returns Ok(()) if cache hit (exec replaces the process), or Err if miss.
fn try_warm_cache(
    cache_dir: &PathBuf,
    requirements: &[String],
    python_version: &str,
    platform: &str,
    python: &PathBuf,
    target: &str,
    target_args: &[String],
    verbose: bool,
) -> Result<(), ()> {
    let ih = input_hash(requirements, python_version, platform);
    let input_file = cache_dir.join("inputs").join(&ih);

    let env_key = match std::fs::read_to_string(&input_file) {
        Ok(key) => key.trim().to_string(),
        Err(_) => return Err(()),
    };

    let env_dir = cache_dir.join("envs").join(&env_key);
    if !env_dir.join(".complete").exists() {
        return Err(());
    }

    // Find site-packages
    let sp = find_site_packages(&env_dir);
    let site_packages = match sp {
        Some(sp) => sp,
        None => return Err(()),
    };

    if verbose {
        eprintln!("Cache hit — skipping resolution");
    }

    // Build PYTHONPATH: prepend our site-packages
    let mut pythonpath = site_packages.to_string_lossy().to_string();
    if let Ok(existing) = std::env::var("PYTHONPATH") {
        pythonpath = format!("{pythonpath}:{existing}");
    }

    // exec Python with the zerostart module for entry point discovery + run
    // This avoids the overhead of importing the full zerostart resolver
    let sp_str = site_packages.display().to_string();
    let args_str = format!("{target_args:?}");

    let script = format!(
        concat!(
            "import sys\n",
            "sys.path.insert(0, {sp_repr})\n",
            "target = {target_repr}\n",
            "args = {args_repr}\n",
            "if target.endswith('.py') or __import__('pathlib').Path(target).is_file():\n",
            "    sys.argv = [target] + args\n",
            "    exec(compile(open(target).read(), target, 'exec'), ",
            "{{'__name__': '__main__', '__file__': target}})\n",
            "else:\n",
            "    from zerostart.entrypoints import discover_entry_point, invoke_entry_point\n",
            "    ep = discover_entry_point(target, __import__('pathlib').Path({sp_repr}))\n",
            "    invoke_entry_point(ep, args)\n",
        ),
        sp_repr = format!("'{}'", sp_str.replace('\'', "\\'")),
        target_repr = format!("'{}'", target.replace('\'', "\\'")),
        args_repr = args_str,
    );

    let err = std::process::Command::new(python)
        .env("PYTHONPATH", &pythonpath)
        .arg("-c")
        .arg(&script)
        .exec();

    // exec() only returns on error
    eprintln!("exec failed: {err}");
    std::process::exit(1);
}

fn find_site_packages(env_dir: &PathBuf) -> Option<PathBuf> {
    let lib_dir = env_dir.join("lib");
    if !lib_dir.exists() {
        return None;
    }

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
            cache_dir,
            verbose,
            target_args,
        } => {
            let python = find_python()?;
            let python_version = detect_python_version(&python)?;
            let platform = detect_platform();

            // Build requirements list for cache lookup
            let mut reqs: Vec<String> = Vec::new();

            // In package mode, the target itself is a requirement
            let is_script = target.ends_with(".py")
                || std::path::Path::new(&target).is_file();

            if !is_script {
                reqs.push(target.clone());
            }
            reqs.extend(packages.clone());

            // If we have requirements to check, try warm cache
            if !reqs.is_empty() {
                // Try fast warm path (exec replaces process on hit)
                let _ = try_warm_cache(
                    &cache_dir,
                    &reqs,
                    &python_version,
                    &platform,
                    &python,
                    &target,
                    &target_args,
                    verbose,
                );
            }

            // Cold path: fall back to Python orchestrator
            if verbose {
                eprintln!("Cache miss — starting Python orchestrator");
            }

            let mut cmd = std::process::Command::new(&python);
            cmd.arg("-m").arg("zerostart.run");

            if verbose {
                cmd.arg("-v");
            }

            if let Some(req_file) = &requirements {
                cmd.arg("-r").arg(req_file);
            }

            for pkg in &packages {
                cmd.arg("-p").arg(pkg);
            }

            cmd.arg("--cache-dir").arg(&cache_dir);
            cmd.arg(&target);

            if !target_args.is_empty() {
                cmd.arg("--");
                cmd.args(&target_args);
            }

            // exec replaces the process
            let err = cmd.exec();
            anyhow::bail!("exec failed: {err}");
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
            // Get requirements text from --requirements, --pyproject, or stdin
            let req_text = if let Some(text) = requirements {
                text
            } else if let Some(toml_text) = pyproject {
                let deps = resolve::parse_pyproject_dependencies(&toml_text)?;
                deps.join("\n")
            } else {
                // Read from stdin
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

            eprintln!("Resolving requirements...");
            let wheels = resolve::resolve_requirements(&req_text, &python_version, &platform).await?;

            if wheels.is_empty() {
                anyhow::bail!("no wheels resolved from requirements");
            }

            eprintln!("Resolved {} wheels, starting download...", wheels.len());
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
