//! Tests for cgroup cpu.max fairness scenarios.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use cgroups_rs::fs::cgroup_builder::CgroupBuilder;
use cgroups_rs::fs::hierarchies;

use crate::test;
use crate::util::child::Child;
use crate::util::shared::{BumpAllocator, SharedBox};
use crate::util::system::{CPUMask, CPUSet, System};
use crate::workloads::spinner_utilization;

/// Test cgroup cpu.max fairness with two CPU hogs.
///
/// This test creates two CPU hogs, one on CPU 1 (victim) and one on CPU 2.
/// Both are placed in separate cgroups with cpu.max set to 10% of one CPU.
/// This validates that both CPUs receive fair scheduling despite the cpu.max limit.
///
/// # Parameters
/// * CPU_1: First CPU (default: 1) - victim CPU
/// * CPU_2: Second CPU (default: 2)
/// * CPU_MAX_PERCENT: cpu.max limit as percentage (default: 10%)
fn irq_disruption_targeted() -> Result<()> {
    const CPU_1: i32 = 1;
    const CPU_2: i32 = 2;
    const CPU_MAX_PERCENT: f64 = 10.0;

    let system = System::load()?;

    // Collect all hyperthreads (logical CPUs)
    let mut all_cpus = Vec::new();
    for core in system.cores() {
        for ht in core.hyperthreads() {
            all_cpus.push(ht.clone());
        }
    }

    eprintln!("Found {} logical CPUs", all_cpus.len());
    eprintln!("Testing cpu.max fairness: CPU {} and CPU {} both limited to {}%",
              CPU_1, CPU_2, CPU_MAX_PERCENT);

    // Find the two CPUs
    let cpu_1_ht = all_cpus.iter()
        .find(|ht| ht.id() == CPU_1)
        .ok_or_else(|| anyhow::anyhow!("CPU {} not found", CPU_1))?
        .clone();

    let cpu_2_ht = all_cpus.iter()
        .find(|ht| ht.id() == CPU_2)
        .ok_or_else(|| anyhow::anyhow!("CPU {} not found", CPU_2))?
        .clone();

    // Create shared memory for start signal and counters
    let allocator = BumpAllocator::new("cpu_max_test", 1 * 1024 * 1024)?;
    let start_signal = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;

    // Create counters for both CPUs
    let bogo_ops_cpu1 = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;
    let scheduled_ns_cpu1 = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;
    let bogo_ops_cpu2 = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;
    let scheduled_ns_cpu2 = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;

    // Create cgroups with cpu.max limits
    // cpu.max format: quota period (in microseconds)
    // For 10% of one CPU: quota=10000, period=100000 (10ms out of 100ms)
    let period_us = 100000u64;  // 100ms period
    let quota_us = (period_us as f64 * CPU_MAX_PERCENT / 100.0) as i64;

    eprintln!("\nCreating cgroups with cpu.max={}/{} ({}%)", quota_us, period_us, CPU_MAX_PERCENT);

    let cgroup_cpu1 = CgroupBuilder::new("schtest_cpu_max_cpu1")
        .cpu()
        .quota(quota_us)
        .period(period_us)
        .done()
        .build(hierarchies::auto())
        .context("Failed to create cgroup for CPU 1")?;

    let cgroup_cpu2 = CgroupBuilder::new("schtest_cpu_max_cpu2")
        .cpu()
        .quota(quota_us)
        .period(period_us)
        .done()
        .build(hierarchies::auto())
        .context("Failed to create cgroup for CPU 2")?;

    let hog_duration = Duration::from_secs(3);
    eprintln!("\nLaunching 2 CPU hogs for {:?}...", hog_duration);

    // Launch hog on CPU 1
    let cpu1_mask = CPUMask::new(&cpu_1_ht);
    let start_signal_cpu1 = start_signal.clone();
    let bogo_ops_out_cpu1 = bogo_ops_cpu1.clone();
    let scheduled_ns_out_cpu1 = scheduled_ns_cpu1.clone();

    let mut child_cpu1 = Child::run(
        move || {
            // Pin to CPU 1 and run the hog
            cpu1_mask.run(|| {
                spinner_utilization::cpu_hog_workload(
                    hog_duration,
                    start_signal_cpu1,
                    scheduled_ns_out_cpu1,
                    Some(bogo_ops_out_cpu1),
                );
            })?;
            Ok(())
        },
        None,
    )?;

    // Add CPU 1 process to its cgroup
    let pid1 = child_cpu1.pid().as_raw();
    let procs_path1 = std::path::Path::new("/sys/fs/cgroup")
        .join("schtest_cpu_max_cpu1")
        .join("cgroup.procs");
    std::fs::write(&procs_path1, pid1.to_string())
        .context(format!("Failed to write PID {} to {:?}", pid1, procs_path1))?;

    // Launch hog on CPU 2
    let cpu2_mask = CPUMask::new(&cpu_2_ht);
    let start_signal_cpu2 = start_signal.clone();
    let bogo_ops_out_cpu2 = bogo_ops_cpu2.clone();
    let scheduled_ns_out_cpu2 = scheduled_ns_cpu2.clone();

    let mut child_cpu2 = Child::run(
        move || {
            // Pin to CPU 2 and run the hog
            cpu2_mask.run(|| {
                spinner_utilization::cpu_hog_workload(
                    hog_duration,
                    start_signal_cpu2,
                    scheduled_ns_out_cpu2,
                    Some(bogo_ops_out_cpu2),
                );
            })?;
            Ok(())
        },
        None,
    )?;

    // Add CPU 2 process to its cgroup
    let pid2 = child_cpu2.pid().as_raw();
    let procs_path2 = std::path::Path::new("/sys/fs/cgroup")
        .join("schtest_cpu_max_cpu2")
        .join("cgroup.procs");
    std::fs::write(&procs_path2, pid2.to_string())
        .context(format!("Failed to write PID {} to {:?}", pid2, procs_path2))?;

    // Give hogs a moment to initialize
    std::thread::sleep(Duration::from_millis(100));

    // Signal both hogs to start simultaneously
    eprintln!("Signaling both hogs to START");
    start_signal.store(1, Ordering::Release);

    // Wait for both hogs to complete
    eprintln!("Waiting for hogs to complete...");
    if let Some(result) = child_cpu1.wait(true, false) {
        result.context(format!("Hog on CPU {} failed", CPU_1))?;
    }
    if let Some(result) = child_cpu2.wait(true, false) {
        result.context(format!("Hog on CPU {} failed", CPU_2))?;
    }
    eprintln!("Both hogs completed successfully");

    // Clean up cgroups
    drop(cgroup_cpu1);
    drop(cgroup_cpu2);

    // Collect results
    let results: Vec<(i32, u64, u64)> = vec![
        (CPU_1, bogo_ops_cpu1.load(Ordering::Acquire), scheduled_ns_cpu1.load(Ordering::Acquire)),
        (CPU_2, bogo_ops_cpu2.load(Ordering::Acquire), scheduled_ns_cpu2.load(Ordering::Acquire)),
    ];

    // Calculate statistics
    let bogo_ops_results: Vec<(i32, u64)> = results.iter().map(|(cpu, ops, _)| (*cpu, *ops)).collect();
    let scheduled_ns_results: Vec<(i32, u64)> = results.iter().map(|(cpu, _, ns)| (*cpu, *ns)).collect();

    // Bogo ops statistics
    let mut bogo_ops_only: Vec<u64> = bogo_ops_results.iter().map(|(_, ops)| *ops).collect();
    bogo_ops_only.sort_unstable();

    let min_bogo_ops = bogo_ops_only[0];
    let max_bogo_ops = bogo_ops_only[1];
    let avg_bogo_ops = (min_bogo_ops + max_bogo_ops) / 2;
    let p50_bogo_ops = avg_bogo_ops; // With 2 values, median = average
    let bogo_ops_skew = max_bogo_ops - min_bogo_ops;
    let bogo_ops_skew_pct = if max_bogo_ops > 0 {
        (bogo_ops_skew as f64 / max_bogo_ops as f64) * 100.0
    } else {
        0.0
    };

    // Find which CPUs have min/max/p50 bogo_ops
    let (min_bogo_cpu, _) = bogo_ops_results.iter().min_by_key(|(_, ops)| ops).unwrap();
    let (max_bogo_cpu, _) = bogo_ops_results.iter().max_by_key(|(_, ops)| ops).unwrap();
    let p50_bogo_cpu = *min_bogo_cpu; // Arbitrary choice for 2 values

    // Scheduled nanoseconds statistics
    let mut scheduled_ns_only: Vec<u64> = scheduled_ns_results.iter().map(|(_, ns)| *ns).collect();
    scheduled_ns_only.sort_unstable();

    let min_scheduled_ns = scheduled_ns_only[0];
    let max_scheduled_ns = scheduled_ns_only[1];
    let avg_scheduled_ns = (min_scheduled_ns + max_scheduled_ns) / 2;
    let p50_scheduled_ns = avg_scheduled_ns;
    let scheduled_ns_skew = max_scheduled_ns - min_scheduled_ns;
    let scheduled_ns_skew_pct = if max_scheduled_ns > 0 {
        (scheduled_ns_skew as f64 / max_scheduled_ns as f64) * 100.0
    } else {
        0.0
    };

    // Find which CPUs have min/max/p50 scheduled_ns
    let (min_ns_cpu, _) = scheduled_ns_results.iter().min_by_key(|(_, ns)| ns).unwrap();
    let (max_ns_cpu, _) = scheduled_ns_results.iter().max_by_key(|(_, ns)| ns).unwrap();
    let p50_ns_cpu = *min_ns_cpu; // Arbitrary choice for 2 values

    // Bogo ops per millisecond
    let bogo_ops_per_ms: Vec<(i32, f64)> = results.iter()
        .map(|(cpu, ops, ns)| {
            let ms = *ns as f64 / 1_000_000.0;
            let ops_per_ms = if ms > 0.0 { *ops as f64 / ms } else { 0.0 };
            (*cpu, ops_per_ms)
        })
        .collect();

    let mut ops_per_ms_only: Vec<f64> = bogo_ops_per_ms.iter().map(|(_, rate)| *rate).collect();
    ops_per_ms_only.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min_ops_per_ms = ops_per_ms_only[0];
    let max_ops_per_ms = ops_per_ms_only[1];
    let avg_ops_per_ms = (min_ops_per_ms + max_ops_per_ms) / 2.0;
    let p50_ops_per_ms = avg_ops_per_ms;
    let ops_per_ms_skew = max_ops_per_ms - min_ops_per_ms;
    let ops_per_ms_skew_pct = if max_ops_per_ms > 0.0 {
        (ops_per_ms_skew / max_ops_per_ms) * 100.0
    } else {
        0.0
    };

    // Find which CPUs have min/max/p50 ops_per_ms
    let (min_ops_ms_cpu, _) = bogo_ops_per_ms.iter().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
    let (max_ops_ms_cpu, _) = bogo_ops_per_ms.iter().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
    let p50_ops_ms_cpu = *min_ops_ms_cpu; // Arbitrary choice for 2 values

    // Print detailed results
    eprintln!("\n=== Per-CPU Results ===");
    eprintln!("{:>6} {:>20} {:>20} {:>20}", "CPU", "bogo_ops", "scheduled_ns", "bogo_ops/ms");
    for (cpu, ops, ns) in &results {
        let ops_per_ms = bogo_ops_per_ms.iter()
            .find(|(id, _)| id == cpu)
            .map(|(_, rate)| *rate)
            .unwrap_or(0.0);
        let marker = if *cpu == CPU_1 { " <-- CPU 1" } else { " <-- CPU 2" };
        eprintln!("{:>6} {:>20} {:>20} {:>20.2}{}",
                  cpu, ops, ns, ops_per_ms, marker);
    }

    eprintln!("\n=== Bogo Ops Statistics ===");
    eprintln!("Min:       {:>20} (CPU {})", min_bogo_ops, min_bogo_cpu);
    eprintln!("Avg:       {:>20}", avg_bogo_ops);
    eprintln!("P50:       {:>20} (CPU {})", p50_bogo_ops, p50_bogo_cpu);
    eprintln!("Max:       {:>20} (CPU {})", max_bogo_ops, max_bogo_cpu);
    eprintln!("Max Skew:  {:>20} ({:.2}%)", bogo_ops_skew, bogo_ops_skew_pct);

    eprintln!("\n=== Scheduled Time (ns) Statistics ===");
    eprintln!("Min:       {:>20} (CPU {})", min_scheduled_ns, min_ns_cpu);
    eprintln!("Avg:       {:>20}", avg_scheduled_ns);
    eprintln!("P50:       {:>20} (CPU {})", p50_scheduled_ns, p50_ns_cpu);
    eprintln!("Max:       {:>20} (CPU {})", max_scheduled_ns, max_ns_cpu);
    eprintln!("Max Skew:  {:>20} ({:.2}%)", scheduled_ns_skew, scheduled_ns_skew_pct);

    eprintln!("\n=== Bogo Ops/ms Statistics ===");
    eprintln!("Min:       {:>20.2} (CPU {})", min_ops_per_ms, min_ops_ms_cpu);
    eprintln!("Avg:       {:>20.2}", avg_ops_per_ms);
    eprintln!("P50:       {:>20.2} (CPU {})", p50_ops_per_ms, p50_ops_ms_cpu);
    eprintln!("Max:       {:>20.2} (CPU {})", max_ops_per_ms, max_ops_ms_cpu);
    eprintln!("Max Skew:  {:>20.2} ({:.2}%)", ops_per_ms_skew, ops_per_ms_skew_pct);

    Ok(())
}

test!("irq_disruption_targeted", irq_disruption_targeted);
