//! PyO3 bindings — thin wrapper around DaemonEngine.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::daemon::{DaemonConfig, DaemonEngine};
use crate::manifest::WheelSpec;

/// Python-facing handle to a running fast-wheel daemon.
///
/// All state is in-memory — no file-based IPC.
#[pyclass]
pub struct DaemonHandle {
    engine: Option<Arc<DaemonEngine>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

#[pymethods]
impl DaemonHandle {
    #[new]
    fn new() -> Self {
        Self {
            engine: None,
            thread: None,
        }
    }

    /// Start the daemon with the given wheel specs.
    #[pyo3(signature = (wheels, site_packages, parallel_downloads=32, extract_threads=0))]
    fn start(
        &mut self,
        wheels: Vec<Bound<'_, PyDict>>,
        site_packages: String,
        parallel_downloads: usize,
        extract_threads: usize,
    ) -> PyResult<()> {
        if self.engine.is_some() {
            return Err(PyRuntimeError::new_err("daemon already started"));
        }

        let wheel_specs: Vec<WheelSpec> = wheels
            .iter()
            .map(|d| parse_wheel_spec(d))
            .collect::<PyResult<Vec<_>>>()?;

        if wheel_specs.is_empty() {
            return Err(PyValueError::new_err("no wheels provided"));
        }

        let extract_threads = if extract_threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        } else {
            extract_threads
        };

        let config = DaemonConfig {
            site_packages: PathBuf::from(&site_packages),
            parallel_downloads,
            extract_threads,
        };

        let engine = Arc::new(DaemonEngine::new(wheel_specs));
        self.engine = Some(engine.clone());

        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(4.max(parallel_downloads))
                .build()
                .expect("failed to create tokio runtime");

            rt.block_on(async {
                if let Err(e) = engine.run(&config).await {
                    tracing::error!("daemon error: {e}");
                }
            });
        });
        self.thread = Some(handle);

        Ok(())
    }

    /// Signal that a distribution is needed urgently.
    fn signal_demand(&self, distribution: String) -> PyResult<()> {
        let engine = self.engine()?;
        engine.signal_demand(&distribution);
        Ok(())
    }

    /// Check if a distribution is done (non-blocking).
    fn is_done(&self, distribution: String) -> PyResult<bool> {
        let engine = self.engine()?;
        Ok(engine.is_done(&distribution))
    }

    /// Wait for a specific distribution to complete.
    #[pyo3(signature = (distribution, timeout_secs=60.0))]
    fn wait_done(
        &self,
        py: Python<'_>,
        distribution: String,
        timeout_secs: f64,
    ) -> PyResult<bool> {
        let engine = self.engine()?.clone();
        let timeout = std::time::Duration::from_secs_f64(timeout_secs);

        py.detach(move || {
            engine
                .wait_done(&distribution, timeout)
                .map_err(|e| PyTimeoutError::new_err(format!("{e}")))
        })
    }

    /// Wait for all wheels to complete.
    #[pyo3(signature = (timeout_secs=300.0))]
    fn wait_all(&self, py: Python<'_>, timeout_secs: f64) -> PyResult<()> {
        let engine = self.engine()?.clone();
        let timeout = std::time::Duration::from_secs_f64(timeout_secs);

        py.detach(move || {
            engine
                .wait_all(timeout)
                .map_err(|e| PyTimeoutError::new_err(format!("{e}")))
        })
    }

    /// Get stats: (total, done, pending, in_progress, failed)
    fn stats(&self) -> PyResult<(usize, usize, usize, usize, usize)> {
        let engine = self.engine()?;
        Ok(engine.stats())
    }

    /// Shut down the daemon. Blocks until done.
    fn shutdown(&mut self) -> PyResult<()> {
        if let Some(handle) = self.thread.take() {
            handle
                .join()
                .map_err(|_| PyRuntimeError::new_err("daemon thread panicked"))?;
        }
        self.engine = None;
        Ok(())
    }
}

impl DaemonHandle {
    fn engine(&self) -> PyResult<&Arc<DaemonEngine>> {
        self.engine
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("daemon not started"))
    }
}

fn parse_wheel_spec(d: &Bound<'_, PyDict>) -> PyResult<WheelSpec> {
    let url: String = d
        .get_item("url")?
        .ok_or_else(|| PyValueError::new_err("missing 'url'"))?
        .extract()?;
    let distribution: String = d
        .get_item("distribution")?
        .ok_or_else(|| PyValueError::new_err("missing 'distribution'"))?
        .extract()?;
    let size: u64 = match d.get_item("size")? {
        Some(v) => v.extract()?,
        None => 0,
    };
    let import_roots: Vec<String> = match d.get_item("import_roots")? {
        Some(v) => v.extract()?,
        None => Vec::new(),
    };
    let hash: Option<String> = match d.get_item("hash")? {
        Some(v) => {
            if v.is_none() {
                None
            } else {
                Some(v.extract()?)
            }
        }
        None => None,
    };

    Ok(WheelSpec {
        url,
        distribution,
        import_roots,
        size,
        hash,
    })
}

#[pymodule]
pub fn zs_fast_wheel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DaemonHandle>()?;
    Ok(())
}
