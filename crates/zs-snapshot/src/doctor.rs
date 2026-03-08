use serde::{Deserialize, Serialize};
use std::process::Command;

/// Report from running doctor checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    pub is_linux: bool,
    pub criu_found: bool,
    pub criu_version: Option<String>,
    pub has_privileges: bool,
}

/// Check whether CRIU is available and the environment supports snapshotting.
pub fn doctor() -> DoctorReport {
    let is_linux = cfg!(target_os = "linux");

    let (criu_found, criu_version) = check_criu();

    let has_privileges = if is_linux { check_privileges() } else { false };

    DoctorReport {
        is_linux,
        criu_found,
        criu_version,
        has_privileges,
    }
}

fn check_criu() -> (bool, Option<String>) {
    let output = Command::new("criu").arg("--version").output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            // criu --version prints to stdout on some versions, stderr on others
            let version_text = if stdout.contains("Version:") || stdout.contains("criu") {
                stdout.to_string()
            } else {
                stderr.to_string()
            };
            let version = version_text
                .lines()
                .find(|line| line.contains("Version:") || line.contains("version"))
                .map(|line| line.trim().to_string());
            (true, version)
        }
        Err(_) => (false, None),
    }
}

fn check_privileges() -> bool {
    // Check if running as root by invoking `id -u`
    let output = Command::new("id").arg("-u").output();
    match output {
        Ok(out) => {
            let uid_str = String::from_utf8_lossy(&out.stdout);
            uid_str.trim() == "0"
        }
        Err(_) => false,
    }
}
