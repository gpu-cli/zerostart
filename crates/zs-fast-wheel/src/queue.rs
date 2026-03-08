use std::collections::{HashSet, VecDeque};

use crate::manifest::WheelSpec;

/// Priority queue for wheel installation scheduling.
///
/// Default order: small wheels first, large last.
/// Supports demand-driven reprioritization via `prioritize()`.
pub struct InstallQueue {
    /// Wheels not yet started, ordered by priority
    pending: VecDeque<WheelSpec>,
    /// Distribution names currently downloading/extracting
    in_progress: HashSet<String>,
    /// Distribution names that have completed
    done: HashSet<String>,
}

impl InstallQueue {
    /// Create a new queue from a list of wheel specs.
    /// Sorts by size ascending (small wheels first).
    pub fn new(mut wheels: Vec<WheelSpec>) -> Self {
        wheels.sort_by_key(|w| w.size);
        Self {
            pending: wheels.into(),
            in_progress: HashSet::new(),
            done: HashSet::new(),
        }
    }

    /// Bump a distribution to the front of the pending queue.
    /// No-op if already done or in progress.
    pub fn prioritize(&mut self, distribution: &str) {
        if self.done.contains(distribution) || self.in_progress.contains(distribution) {
            return;
        }
        if let Some(idx) = self
            .pending
            .iter()
            .position(|w| w.distribution == distribution)
        {
            if let Some(wheel) = self.pending.remove(idx) {
                self.pending.push_front(wheel);
            }
        }
    }

    /// Pop the next wheel to install from the front of the queue.
    pub fn next(&mut self) -> Option<WheelSpec> {
        let wheel = self.pending.pop_front()?;
        self.in_progress.insert(wheel.distribution.clone());
        Some(wheel)
    }

    /// Mark a distribution as completed.
    pub fn mark_done(&mut self, distribution: &str) {
        self.in_progress.remove(distribution);
        self.done.insert(distribution.to_string());
    }

    /// Mark a distribution as failed (removes from in_progress but not added to done).
    pub fn mark_failed(&mut self, distribution: &str) {
        self.in_progress.remove(distribution);
    }

    /// Check if all wheels are done (nothing pending or in progress).
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty() && self.in_progress.is_empty()
    }

    /// Number of wheels still pending.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Number of wheels currently in progress.
    pub fn in_progress_count(&self) -> usize {
        self.in_progress.len()
    }

    /// Check if a distribution is already done.
    pub fn is_done(&self, distribution: &str) -> bool {
        self.done.contains(distribution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_wheel(name: &str, size: u64) -> WheelSpec {
        WheelSpec {
            url: format!("https://example.com/{name}-1.0.whl"),
            distribution: name.to_string(),
            import_roots: vec![name.to_string()],
            size,
            hash: None,
        }
    }

    #[test]
    fn test_sorts_by_size_ascending() {
        let wheels = vec![
            make_wheel("torch", 900_000_000),
            make_wheel("six", 12_000),
            make_wheel("numpy", 15_000_000),
        ];
        let mut queue = InstallQueue::new(wheels);

        let first = queue.next().unwrap();
        assert_eq!(first.distribution, "six");
        let second = queue.next().unwrap();
        assert_eq!(second.distribution, "numpy");
        let third = queue.next().unwrap();
        assert_eq!(third.distribution, "torch");
        assert!(queue.next().is_none());
    }

    #[test]
    fn test_prioritize_moves_to_front() {
        let wheels = vec![
            make_wheel("six", 12_000),
            make_wheel("numpy", 15_000_000),
            make_wheel("torch", 900_000_000),
        ];
        let mut queue = InstallQueue::new(wheels);

        queue.prioritize("torch");
        let first = queue.next().unwrap();
        assert_eq!(first.distribution, "torch");
    }

    #[test]
    fn test_prioritize_noop_for_done() {
        let wheels = vec![
            make_wheel("six", 12_000),
            make_wheel("torch", 900_000_000),
        ];
        let mut queue = InstallQueue::new(wheels);

        let first = queue.next().unwrap();
        assert_eq!(first.distribution, "six");
        queue.mark_done("six");

        // Prioritizing a done package should be a no-op
        queue.prioritize("six");
        let second = queue.next().unwrap();
        assert_eq!(second.distribution, "torch");
    }

    #[test]
    fn test_prioritize_noop_for_in_progress() {
        let wheels = vec![
            make_wheel("six", 12_000),
            make_wheel("numpy", 15_000_000),
            make_wheel("torch", 900_000_000),
        ];
        let mut queue = InstallQueue::new(wheels);

        let first = queue.next().unwrap(); // six is now in_progress
        assert_eq!(first.distribution, "six");

        // Prioritizing an in-progress package should be a no-op
        queue.prioritize("six");
        let second = queue.next().unwrap();
        assert_eq!(second.distribution, "numpy"); // not six again
    }

    #[test]
    fn test_mark_done_lifecycle() {
        let wheels = vec![make_wheel("six", 12_000)];
        let mut queue = InstallQueue::new(wheels);

        assert!(!queue.is_empty());
        assert_eq!(queue.pending_count(), 1);

        let wheel = queue.next().unwrap();
        assert_eq!(queue.in_progress_count(), 1);
        assert_eq!(queue.pending_count(), 0);
        assert!(!queue.is_empty()); // still in progress

        queue.mark_done(&wheel.distribution);
        assert!(queue.is_empty());
        assert!(queue.is_done("six"));
    }

    #[test]
    fn test_mark_failed() {
        let wheels = vec![make_wheel("badpkg", 1000)];
        let mut queue = InstallQueue::new(wheels);

        let wheel = queue.next().unwrap();
        queue.mark_failed(&wheel.distribution);

        assert!(queue.is_empty());
        assert!(!queue.is_done("badpkg")); // failed, not done
    }

    #[test]
    fn test_prioritize_unknown_distribution() {
        let wheels = vec![make_wheel("six", 12_000)];
        let mut queue = InstallQueue::new(wheels);

        // Prioritizing a non-existent distribution should be a no-op
        queue.prioritize("nonexistent");
        let first = queue.next().unwrap();
        assert_eq!(first.distribution, "six");
    }
}
