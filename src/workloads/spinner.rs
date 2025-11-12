//! Spinner workload implementation.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Maximum number of CPUs we support for per-CPU tracking.
const MAX_CPUS: usize = 1024;

/// A (time_nanos, cpu_id) pair representing a migration event.
#[derive(Debug, Clone, Copy)]
pub struct MigrationEvent {
    pub time_nanos: u64,
    pub cpu_id: u32,
}

/// A workload that spins for a specified duration.
///
/// Note: This struct is used in SharedBox (shared memory), so it MUST NOT contain
/// any types that need Drop (like Vec, Mutex, Box, etc.). Use only fixed-size arrays
/// and primitive/atomic types.
pub struct Spinner {
    /// The ID of the CPU that the spinner last ran on.
    cpu_id: AtomicU32,
    /// The baseline instant from which we measure CPU change times.
    baseline: Instant,
    /// Per-CPU table of last migration times (indexed by CPU ID).
    /// u64::MAX means "never migrated to this CPU".
    last_migration_nanos: [AtomicU64; MAX_CPUS],
    /// Time from baseline to first observation (when control transferred to worker).
    initial_run_nanos: AtomicU64,
    /// Count of observed migrations.
    observed_migrations: AtomicU64,
}

impl Spinner {
    /// Default duration is 99 years.
    pub const DEFAULT_DURATION: Duration = Duration::from_secs(60 * 60 * 24 * 365 * 99);

    /// Create a new Spinner with the given baseline instant.
    ///
    /// # Arguments
    ///
    /// * `baseline` - The baseline instant from which to measure CPU change times
    pub fn new(baseline: Instant) -> Self {
        // Initialize per-CPU migration table with u64::MAX (never migrated)
        const INIT: AtomicU64 = AtomicU64::new(u64::MAX);
        let last_migration_nanos = [INIT; MAX_CPUS];

        Self {
            cpu_id: AtomicU32::new(0),
            baseline,
            last_migration_nanos,
            initial_run_nanos: AtomicU64::new(0),
            observed_migrations: AtomicU64::new(0),
        }
    }

    /// Get the CPU ID and the time we migrated to it.
    ///
    /// # Returns
    ///
    /// A tuple of (cpu_id, time_migrated_to_cpu_nanos). The time will be u64::MAX
    /// if we've never been observed on this CPU (shouldn't happen in practice).
    pub fn last_cpu(&self) -> (u32, u64) {
        let cpu_id = self.cpu_id.load(Ordering::Relaxed);
        let time_nanos = if (cpu_id as usize) < self.last_migration_nanos.len() {
            self.last_migration_nanos[cpu_id as usize].load(Ordering::Relaxed)
        } else {
            u64::MAX
        };
        (cpu_id, time_nanos)
    }

    /// Get the time from baseline to first observation.
    pub fn initial_run_nanos(&self) -> u64 {
        self.initial_run_nanos.load(Ordering::Relaxed)
    }

    /// Get the number of observed migrations.
    pub fn observed_migrations(&self) -> u64 {
        self.observed_migrations.load(Ordering::Relaxed)
    }

    /// Spin for the specified duration.
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration to spin for
    pub fn spin(&self, duration: Duration) {
        let start = std::time::Instant::now();
        let mut first_observation = true;
        let mut last_cpu = u32::MAX; // Start with invalid CPU to ensure first observation is recorded

        while start.elapsed() < duration {
            std::sync::atomic::compiler_fence(Ordering::SeqCst);
            if let Some(cpu) = Self::get_current_cpu() {
                let now = Instant::now();
                let nanos_since_baseline = now.duration_since(self.baseline).as_nanos() as u64;

                // Capture initial run time on first observation
                if first_observation {
                    self.initial_run_nanos.compare_exchange(
                        0,
                        nanos_since_baseline,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ).ok();

                    // Record the initial CPU we're running on
                    self.cpu_id.store(cpu, Ordering::Relaxed);
                    if (cpu as usize) < MAX_CPUS {
                        self.last_migration_nanos[cpu as usize]
                            .store(nanos_since_baseline, Ordering::Relaxed);
                    }
                    // Count initial placement as first "migration"
                    self.observed_migrations.store(1, Ordering::Relaxed);
                    last_cpu = cpu;
                    first_observation = false;
                }
                // Handle CPU migration
                else if cpu != last_cpu {
                    self.observed_migrations.fetch_add(1, Ordering::Relaxed);

                    // Update the migration table for this CPU
                    if (cpu as usize) < MAX_CPUS {
                        self.last_migration_nanos[cpu as usize]
                            .store(nanos_since_baseline, Ordering::Relaxed);
                    }

                    // Store the current CPU ID (after updating the table)
                    self.cpu_id.store(cpu, Ordering::Relaxed);
                    last_cpu = cpu;
                }
            }
        }
    }

    /// Get the migration history from the per-CPU tracking table.
    ///
    /// This collects all CPUs that have been visited and returns them sorted by time.
    /// Note: If a thread migrates back to a CPU it visited before, only the most recent
    /// visit is recorded. Therefore, the number of entries may be less than
    /// `observed_migrations()`.
    ///
    /// This should be called after the spinner has stopped spinning.
    ///
    /// # Returns
    ///
    /// A sorted vector of migration events (one per unique CPU visited).
    pub fn full_migration_history(&self) -> Vec<MigrationEvent> {
        let mut history = Vec::new();

        // Collect all entries from the last_migration_nanos table
        for (cpu_id, entry) in self.last_migration_nanos.iter().enumerate() {
            let time_nanos = entry.load(Ordering::Relaxed);
            if time_nanos != u64::MAX {
                history.push(MigrationEvent {
                    time_nanos,
                    cpu_id: cpu_id as u32,
                });
            }
        }

        // Sort by time
        history.sort_by_key(|e| e.time_nanos);

        history
    }

    /// Get the ID of the CPU that the current thread is running on.
    ///
    /// # Returns
    ///
    /// The ID of the CPU that the current thread is running on, or None if the
    /// information is not available.
    fn get_current_cpu() -> Option<u32> {
        let cpu = unsafe { libc::sched_getcpu() };
        if cpu >= 0 {
            Some(cpu as u32)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use std::time::Duration;

    #[test]
    fn test_spinner() -> Result<()> {
        let baseline = Instant::now();
        let spinner = Spinner::new(baseline);

        // Spin for a short duration.
        spinner.spin(Duration::from_millis(10));

        // Check that the CPU ID was updated.
        let (cpu_id, time_nanos) = spinner.last_cpu();
        println!("Last CPU: {}, migrated at {} ns", cpu_id, time_nanos);
        println!("Initial run: {} ns", spinner.initial_run_nanos());
        println!("Observed migrations: {}", spinner.observed_migrations());

        Ok(())
    }
}
