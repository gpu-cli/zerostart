use std::fs;
use std::path::PathBuf;

/// Watches the demand file for priority signals from the import hook.
///
/// The import hook appends distribution names (one per line) to the demand file
/// when `import X` blocks. The daemon reads new lines and reprioritizes.
pub struct DemandWatcher {
    path: PathBuf,
    seen_lines: usize,
}

impl DemandWatcher {
    pub fn new(demand_path: PathBuf) -> Self {
        Self {
            path: demand_path,
            seen_lines: 0,
        }
    }

    /// Check for new demands. Returns newly demanded distribution IDs.
    pub fn poll(&mut self) -> Vec<String> {
        let content = match fs::read_to_string(&self.path) {
            Ok(c) => c,
            Err(_) => return vec![],
        };
        let lines: Vec<&str> = content.lines().collect();
        if lines.len() <= self.seen_lines {
            return vec![];
        }
        let new_lines: Vec<String> = lines[self.seen_lines..]
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| s.trim().to_string())
            .collect();
        self.seen_lines = lines.len();
        new_lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_poll_empty_file() {
        let tmp = TempDir::new().unwrap();
        let demand_path = tmp.path().join("demand");
        fs::write(&demand_path, "").unwrap();

        let mut watcher = DemandWatcher::new(demand_path);
        assert!(watcher.poll().is_empty());
    }

    #[test]
    fn test_poll_no_file() {
        let tmp = TempDir::new().unwrap();
        let demand_path = tmp.path().join("demand");
        // File doesn't exist yet

        let mut watcher = DemandWatcher::new(demand_path);
        assert!(watcher.poll().is_empty());
    }

    #[test]
    fn test_poll_new_demands() {
        let tmp = TempDir::new().unwrap();
        let demand_path = tmp.path().join("demand");

        let mut watcher = DemandWatcher::new(demand_path.clone());

        // First write
        fs::write(&demand_path, "torch\n").unwrap();
        let demands = watcher.poll();
        assert_eq!(demands, vec!["torch"]);

        // Second poll with no new data
        assert!(watcher.poll().is_empty());

        // Append more
        fs::write(&demand_path, "torch\nnumpy\ntransformers\n").unwrap();
        let demands = watcher.poll();
        assert_eq!(demands, vec!["numpy", "transformers"]);
    }

    #[test]
    fn test_poll_skips_blank_lines() {
        let tmp = TempDir::new().unwrap();
        let demand_path = tmp.path().join("demand");
        fs::write(&demand_path, "torch\n\n\nnumpy\n").unwrap();

        let mut watcher = DemandWatcher::new(demand_path);
        let demands = watcher.poll();
        assert_eq!(demands, vec!["torch", "numpy"]);
    }
}
