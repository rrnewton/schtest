//! Tests for heavy IRQ workload scenarios.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Result;

use crate::test;
use crate::util::child::Child;
use crate::util::shared::{BumpAllocator, SharedBox};
use crate::util::system::{CPUMask, CPUSet, System};
use crate::workloads::spinner_utilization;

/// Test that creates one CPU hog per logical CPU to measure fairness.
///
/// This test pins one CPU hog to each logical CPU in the system, starts them
/// simultaneously, and measures the skew in work done (bogo_ops) to detect
/// unfairness that could be caused by IRQ load or other factors.
fn cpu_hogs_per_logical_cpu() -> Result<()> {
    let system = System::load()?;

    // Collect all hyperthreads (logical CPUs)
    let mut all_cpus = Vec::new();
    for core in system.cores() {
        for ht in core.hyperthreads() {
            all_cpus.push(ht.clone());
        }
    }

    let num_cpus = all_cpus.len();
    eprintln!("Found {} logical CPUs", num_cpus);

    if num_cpus == 0 {
        return Err(anyhow::anyhow!("No CPUs found"));
    }

    // Create shared memory for start signal and counters
    let allocator = BumpAllocator::new("irq_test", 1024 * 1024)?;
    let start_signal = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;

    // Allocate bogo_ops counters (one per CPU)
    let mut bogo_ops_counters = Vec::new();
    let mut scheduled_ns_counters = Vec::new();
    for _ in 0..num_cpus {
        bogo_ops_counters.push(SharedBox::new(allocator.clone(), AtomicU64::new(0))?);
        scheduled_ns_counters.push(SharedBox::new(allocator.clone(), AtomicU64::new(0))?);
    }

    // Launch CPU hogs, one pinned to each logical CPU
    let hog_duration = Duration::from_secs(3);
    eprintln!("\nLaunching {} CPU hogs (one per logical CPU) for {:?}...",
              num_cpus, hog_duration);

    let mut children = Vec::new();
    for (cpu_idx, cpu) in all_cpus.iter().enumerate() {
        let cpu_mask = CPUMask::new(cpu);
        let start_signal_clone = start_signal.clone();
        let bogo_ops_out = bogo_ops_counters[cpu_idx].clone();
        let scheduled_ns_out = scheduled_ns_counters[cpu_idx].clone();

        let child = Child::run(
            move || {
                // Pin to the specific CPU and run the hog
                cpu_mask.run(|| {
                    spinner_utilization::cpu_hog_workload(
                        hog_duration,
                        start_signal_clone,
                        scheduled_ns_out,
                        Some(bogo_ops_out),
                    );
                })?;
                Ok(())
            },
            None,
        )?;

        children.push((cpu.id(), child));
    }

    // Give hogs a moment to initialize
    std::thread::sleep(Duration::from_millis(100));

    // Signal all hogs to start simultaneously
    eprintln!("Signaling all hogs to START");
    start_signal.store(1, Ordering::Release);

    // Wait for all hogs to complete
    eprintln!("Waiting for hogs to complete...");
    for (cpu_id, mut child) in children {
        if let Some(result) = child.wait(true, false) {
            result.map_err(|e| anyhow::anyhow!("Hog on CPU {} failed: {}", cpu_id, e))?;
        }
    }
    eprintln!("All hogs completed successfully");

    // Collect results
    let mut bogo_ops_results: Vec<u64> = bogo_ops_counters
        .iter()
        .map(|counter| counter.load(Ordering::Acquire))
        .collect();

    let scheduled_ns_results: Vec<u64> = scheduled_ns_counters
        .iter()
        .map(|counter| counter.load(Ordering::Acquire))
        .collect();

    // Calculate statistics
    if bogo_ops_results.is_empty() {
        return Err(anyhow::anyhow!("No results collected"));
    }

    let min_bogo_ops = *bogo_ops_results.iter().min().unwrap();
    let max_bogo_ops = *bogo_ops_results.iter().max().unwrap();
    let sum_bogo_ops: u64 = bogo_ops_results.iter().sum();
    let avg_bogo_ops = sum_bogo_ops / num_cpus as u64;

    // Calculate p50 (median)
    bogo_ops_results.sort_unstable();
    let p50_bogo_ops = if num_cpus % 2 == 0 {
        (bogo_ops_results[num_cpus / 2 - 1] + bogo_ops_results[num_cpus / 2]) / 2
    } else {
        bogo_ops_results[num_cpus / 2]
    };

    // Calculate maximum skew (difference between min and max)
    let max_skew = max_bogo_ops - min_bogo_ops;
    let skew_pct = if max_bogo_ops > 0 {
        (max_skew as f64 / max_bogo_ops as f64) * 100.0
    } else {
        0.0
    };

    // Print detailed results
    eprintln!("\n=== Per-CPU Results ===");
    eprintln!("{:>6} {:>20} {:>20}", "CPU", "bogo_ops", "scheduled_ns");
    for (idx, cpu) in all_cpus.iter().enumerate() {
        eprintln!("{:>6} {:>20} {:>20}",
                  cpu.id(),
                  bogo_ops_counters[idx].load(Ordering::Acquire),
                  scheduled_ns_results[idx]);
    }

    eprintln!("\n=== Bogo Ops Statistics ===");
    eprintln!("Min:       {:>20}", min_bogo_ops);
    eprintln!("Avg:       {:>20}", avg_bogo_ops);
    eprintln!("P50:       {:>20}", p50_bogo_ops);
    eprintln!("Max:       {:>20}", max_bogo_ops);
    eprintln!("Max Skew:  {:>20} ({:.2}%)", max_skew, skew_pct);

    Ok(())
}

test!("cpu_hogs_per_logical_cpu", cpu_hogs_per_logical_cpu);
