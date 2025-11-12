//! Spinner workload implementation.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Maximum number of CPUs we support for per-CPU tracking.
const MAX_CPUS: usize = 1024;

/// Maximum number of migration events we can log.
const MAX_LOG_ENTRIES: usize = 100000;

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
    /// Migration log: timestamps for each event.
    migration_log_times: [AtomicU64; MAX_LOG_ENTRIES],
    /// Migration log: CPU IDs for each event.
    migration_log_cpus: [AtomicU32; MAX_LOG_ENTRIES],
    /// Count of entries in the migration log.
    log_count: AtomicU64,
    /// Whether we've overflowed the log by dropping events.
    log_overflowed: AtomicBool,
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
        const INIT_U64: AtomicU64 = AtomicU64::new(u64::MAX);
        let last_migration_nanos = [INIT_U64; MAX_CPUS];

        // Initialize migration log arrays
        const INIT_LOG_TIME: AtomicU64 = AtomicU64::new(0);
        const INIT_LOG_CPU: AtomicU32 = AtomicU32::new(0);
        let migration_log_times = [INIT_LOG_TIME; MAX_LOG_ENTRIES];
        let migration_log_cpus = [INIT_LOG_CPU; MAX_LOG_ENTRIES];

        Self {
            cpu_id: AtomicU32::new(0),
            baseline,
            last_migration_nanos,
            initial_run_nanos: AtomicU64::new(0),
            observed_migrations: AtomicU64::new(0),
            migration_log_times,
            migration_log_cpus,
            log_count: AtomicU64::new(0),
            log_overflowed: AtomicBool::new(false),
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

                    // Append to migration log
                    self.append_to_log(cpu, nanos_since_baseline);

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

                    // Append to migration log
                    self.append_to_log(cpu, nanos_since_baseline);

                    last_cpu = cpu;
                }
            }
        }
    }

    /// Append a migration event to the log.
    ///
    /// # Arguments
    ///
    /// * `cpu` - The CPU ID
    /// * `time_nanos` - The timestamp in nanoseconds since baseline
    fn append_to_log(&self, cpu: u32, time_nanos: u64) {
        // Use fetch_add to atomically get the next index
        let index = self.log_count.fetch_add(1, Ordering::Relaxed);

        if index < MAX_LOG_ENTRIES as u64 {
            // We have space - write the entry
            self.migration_log_cpus[index as usize].store(cpu, Ordering::Relaxed);
            self.migration_log_times[index as usize].store(time_nanos, Ordering::Relaxed);
        } else {
            // Log is full - set overflow flag
            self.log_overflowed.store(true, Ordering::Relaxed);
        }
    }

    /// Get the complete migration history from the log.
    ///
    /// This collects all migration events that were recorded in the log, in chronological
    /// order. If the log overflowed, the returned history will be incomplete.
    ///
    /// This should be called after the spinner has stopped spinning.
    ///
    /// # Returns
    ///
    /// A tuple of (history_vec, observed_migrations_count):
    /// - history_vec: A sorted vector of migration events from the log
    /// - observed_migrations_count: The total number of migrations observed
    ///
    /// If the log did not overflow, the vector length will match observed_migrations_count.
    pub fn full_migration_history(&self) -> (Vec<MigrationEvent>, u64) {
        let mut history = Vec::new();
        let observed = self.observed_migrations.load(Ordering::Relaxed);
        let log_entries = self.log_count.load(Ordering::Relaxed);
        let overflowed = self.log_overflowed.load(Ordering::Relaxed);

        // Collect from the log (up to MAX_LOG_ENTRIES)
        let entries_to_read = std::cmp::min(log_entries, MAX_LOG_ENTRIES as u64);
        for i in 0..entries_to_read {
            let cpu_id = self.migration_log_cpus[i as usize].load(Ordering::Relaxed);
            let time_nanos = self.migration_log_times[i as usize].load(Ordering::Relaxed);
            history.push(MigrationEvent {
                cpu_id,
                time_nanos,
            });
        }

        // Sort by time (should already be mostly sorted, but ensure it)
        history.sort_by_key(|e| e.time_nanos);

        // If we didn't overflow, the vector length should exactly match observed migrations
        if !overflowed {
            assert_eq!(
                history.len() as u64,
                observed,
                "Migration history length mismatch: got {} events but observed {} migrations (no overflow)",
                history.len(),
                observed
            );
        }

        (history, observed)
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

        // Verify migration history
        let (history, observed) = spinner.full_migration_history();
        println!("Migration history: {} events logged, {} observed total", history.len(), observed);
        assert_eq!(history.len() as u64, observed, "All migrations should be logged");

        // Print the first few migration events
        for (i, event) in history.iter().take(5).enumerate() {
            println!("  Migration {}: CPU {} at {} ns", i, event.cpu_id, event.time_nanos);
        }

        Ok(())
    }
}
