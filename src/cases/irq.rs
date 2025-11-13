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
fn irq_disruption() -> Result<()> {
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

test!("irq_disruption", irq_disruption);

/// Test with targeted IRQ disruption on a specific CPU.
///
/// This test creates CPU hogs on all logical CPUs except CPU 0, and adds
/// a waker thread on CPU 0 that sends high-frequency wakeups to a receiver
/// thread on the victim CPU. This simulates heavy IRQ load on that CPU.
///
/// # Parameters
/// * Victim CPU: 1 (configurable via const)
/// * Waker CPU: 0 (runs the waker thread)
/// * IRQ rate: 10000 Hz (configurable via const)
fn irq_disruption_targeted() -> Result<()> {
    const VICTIM_CPU: i32 = 1;
    const WAKER_CPU: i32 = 0;
    const IRQ_HZ: u64 = 10000;
    const PERFORM_IRQ_DISRUPTION: bool = true;

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
    eprintln!("Waker CPU: {}, Victim CPU: {}, IRQ rate: {} Hz", WAKER_CPU, VICTIM_CPU, IRQ_HZ);

    if num_cpus < 2 {
        return Err(anyhow::anyhow!("Need at least 2 CPUs for this test"));
    }

    // Find victim and waker CPUs
    let victim_hyperthread = all_cpus.iter()
        .find(|ht| ht.id() == VICTIM_CPU)
        .ok_or_else(|| anyhow::anyhow!("Victim CPU {} not found", VICTIM_CPU))?
        .clone();

    let waker_hyperthread = all_cpus.iter()
        .find(|ht| ht.id() == WAKER_CPU)
        .ok_or_else(|| anyhow::anyhow!("Waker CPU {} not found", WAKER_CPU))?
        .clone();

    // Create shared memory for start signal and counters
    let allocator = BumpAllocator::new("irq_test", 2 * 1024 * 1024)?;
    let start_signal = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;
    let stop_signal = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;
    let wakeup_signal = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;
    let wakeup_count = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;

    // Filter CPUs for hogs: exclude CPU 0 (waker)
    let hog_cpus: Vec<_> = all_cpus.iter()
        .filter(|ht| ht.id() != WAKER_CPU)
        .cloned()
        .collect();

    let num_hogs = hog_cpus.len();

    // Allocate bogo_ops counters (one per hog)
    let mut bogo_ops_counters = Vec::new();
    let mut scheduled_ns_counters = Vec::new();
    for _ in 0..num_hogs {
        bogo_ops_counters.push(SharedBox::new(allocator.clone(), AtomicU64::new(0))?);
        scheduled_ns_counters.push(SharedBox::new(allocator.clone(), AtomicU64::new(0))?);
    }

    // Launch receiver thread on victim CPU
    eprintln!("\nLaunching receiver thread on CPU {}...", VICTIM_CPU);
    let receiver_mask = CPUMask::new(&victim_hyperthread);
    let receiver_wakeup_signal = wakeup_signal.clone();
    let receiver_stop_signal = stop_signal.clone();

    let receiver_child = Child::run(
        move || {
            receiver_mask.run(|| {
                // Wait for wakeups and immediately go back to sleep
                loop {
                    // Wait for wakeup signal
                    while receiver_wakeup_signal.load(Ordering::Acquire) == 0 {
                        std::hint::spin_loop();
                    }

                    // Clear the signal
                    receiver_wakeup_signal.store(0, Ordering::Release);

                    // Check if we should stop
                    if receiver_stop_signal.load(Ordering::Acquire) != 0 {
                        break;
                    }
                }
            })?;
            Ok(())
        },
        None,
    )?;

    // Launch waker thread on CPU 0
    eprintln!("Launching waker thread on CPU {} at {} Hz...", WAKER_CPU, IRQ_HZ);
    let waker_mask = CPUMask::new(&waker_hyperthread);
    let waker_start_signal = start_signal.clone();
    let waker_stop_signal = stop_signal.clone();
    let waker_wakeup_signal = wakeup_signal.clone();
    let waker_count = wakeup_count.clone();

    let waker_child = Child::run(
        move || {
            waker_mask.run(|| {
                use std::time::Instant;

                // Wait for start signal
                while waker_start_signal.load(Ordering::Acquire) == 0 {
                    std::hint::spin_loop();
                }

                let start_time = Instant::now();
                let target_interval = Duration::from_nanos(1_000_000_000 / IRQ_HZ);
                let mut wakeups_sent = 0u64;

                loop {
                    // Check if we should stop
                    if waker_stop_signal.load(Ordering::Acquire) != 0 {
                        break;
                    }

                    // Calculate when the next wakeup should happen
                    let target_time = start_time + target_interval * (wakeups_sent as u32);

                    // Spin until it's time
                    while Instant::now() < target_time {
                        std::hint::spin_loop();
                    }

                    // Send wakeup
                    waker_wakeup_signal.store(1, Ordering::Release);
                    wakeups_sent += 1;

                    // Record count every 1000 wakeups to avoid overhead
                    if wakeups_sent % 1000 == 0 {
                        waker_count.store(wakeups_sent, Ordering::Release);
                    }
                }

                // Final count
                waker_count.store(wakeups_sent, Ordering::Release);
            })?;
            Ok(())
        },
        None,
    )?;

    // Launch CPU hogs (excluding CPU 0)
    let hog_duration = Duration::from_secs(3);
    eprintln!("\nLaunching {} CPU hogs for {:?}...", num_hogs, hog_duration);

    let mut children = Vec::new();
    for (hog_idx, cpu) in hog_cpus.iter().enumerate() {
        let cpu_mask = CPUMask::new(cpu);
        let start_signal_clone = start_signal.clone();
        let bogo_ops_out = bogo_ops_counters[hog_idx].clone();
        let scheduled_ns_out = scheduled_ns_counters[hog_idx].clone();

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

    // Give all threads a moment to initialize
    std::thread::sleep(Duration::from_millis(100));

    // Signal all threads to start simultaneously
    eprintln!("Signaling all threads to START");
    start_signal.store(1, Ordering::Release);

    // Wait for hogs to complete
    eprintln!("Waiting for hogs to complete...");
    for (cpu_id, mut child) in children {
        if let Some(result) = child.wait(true, false) {
            result.map_err(|e| anyhow::anyhow!("Hog on CPU {} failed: {}", cpu_id, e))?;
        }
    }
    eprintln!("All hogs completed successfully");

    // Stop waker and receiver
    eprintln!("Stopping waker and receiver threads...");
    stop_signal.store(1, Ordering::Release);

    // Give them a moment to see the stop signal
    std::thread::sleep(Duration::from_millis(100));

    // Clean up threads
    drop(waker_child);
    drop(receiver_child);

    let total_wakeups = wakeup_count.load(Ordering::Acquire);
    eprintln!("Waker sent {} wakeups total", total_wakeups);

    // Collect results
    let bogo_ops_results: Vec<(i32, u64)> = hog_cpus.iter()
        .enumerate()
        .map(|(idx, cpu)| (cpu.id(), bogo_ops_counters[idx].load(Ordering::Acquire)))
        .collect();

    let scheduled_ns_results: Vec<u64> = scheduled_ns_counters
        .iter()
        .map(|counter| counter.load(Ordering::Acquire))
        .collect();

    // Calculate statistics
    if bogo_ops_results.is_empty() {
        return Err(anyhow::anyhow!("No results collected"));
    }

    // Bogo ops statistics
    let mut bogo_ops_only: Vec<u64> = bogo_ops_results.iter().map(|(_, ops)| *ops).collect();
    bogo_ops_only.sort_unstable();

    let min_bogo_ops = *bogo_ops_only.first().unwrap();
    let max_bogo_ops = *bogo_ops_only.last().unwrap();
    let sum_bogo_ops: u64 = bogo_ops_only.iter().sum();
    let avg_bogo_ops = sum_bogo_ops / num_hogs as u64;

    let p50_bogo_ops = if num_hogs % 2 == 0 {
        (bogo_ops_only[num_hogs / 2 - 1] + bogo_ops_only[num_hogs / 2]) / 2
    } else {
        bogo_ops_only[num_hogs / 2]
    };

    let bogo_ops_skew = max_bogo_ops - min_bogo_ops;
    let bogo_ops_skew_pct = if max_bogo_ops > 0 {
        (bogo_ops_skew as f64 / max_bogo_ops as f64) * 100.0
    } else {
        0.0
    };

    // Scheduled nanoseconds statistics
    let mut scheduled_ns_only = scheduled_ns_results.clone();
    scheduled_ns_only.sort_unstable();

    let min_scheduled_ns = *scheduled_ns_only.first().unwrap();
    let max_scheduled_ns = *scheduled_ns_only.last().unwrap();
    let sum_scheduled_ns: u64 = scheduled_ns_only.iter().sum();
    let avg_scheduled_ns = sum_scheduled_ns / num_hogs as u64;

    let p50_scheduled_ns = if num_hogs % 2 == 0 {
        (scheduled_ns_only[num_hogs / 2 - 1] + scheduled_ns_only[num_hogs / 2]) / 2
    } else {
        scheduled_ns_only[num_hogs / 2]
    };

    let scheduled_ns_skew = max_scheduled_ns - min_scheduled_ns;
    let scheduled_ns_skew_pct = if max_scheduled_ns > 0 {
        (scheduled_ns_skew as f64 / max_scheduled_ns as f64) * 100.0
    } else {
        0.0
    };

    // Bogo ops per millisecond
    let bogo_ops_per_ms: Vec<(i32, f64)> = bogo_ops_results.iter()
        .zip(scheduled_ns_results.iter())
        .map(|((cpu_id, ops), ns)| {
            let ms = *ns as f64 / 1_000_000.0;
            let ops_per_ms = if ms > 0.0 { *ops as f64 / ms } else { 0.0 };
            (*cpu_id, ops_per_ms)
        })
        .collect();

    let mut ops_per_ms_only: Vec<f64> = bogo_ops_per_ms.iter().map(|(_, rate)| *rate).collect();
    ops_per_ms_only.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min_ops_per_ms = *ops_per_ms_only.first().unwrap();
    let max_ops_per_ms = *ops_per_ms_only.last().unwrap();
    let sum_ops_per_ms: f64 = ops_per_ms_only.iter().sum();
    let avg_ops_per_ms = sum_ops_per_ms / num_hogs as f64;

    let p50_ops_per_ms = if num_hogs % 2 == 0 {
        (ops_per_ms_only[num_hogs / 2 - 1] + ops_per_ms_only[num_hogs / 2]) / 2.0
    } else {
        ops_per_ms_only[num_hogs / 2]
    };

    let ops_per_ms_skew = max_ops_per_ms - min_ops_per_ms;
    let ops_per_ms_skew_pct = if max_ops_per_ms > 0.0 {
        (ops_per_ms_skew / max_ops_per_ms) * 100.0
    } else {
        0.0
    };

    // Find which CPU has minimum bogo_ops
    let (min_cpu, _) = bogo_ops_results.iter()
        .min_by_key(|(_, ops)| ops)
        .unwrap();

    // Print detailed results
    eprintln!("\n=== Per-CPU Results ===");
    eprintln!("{:>6} {:>20} {:>20} {:>20}", "CPU", "bogo_ops", "scheduled_ns", "bogo_ops/ms");
    for (idx, cpu) in hog_cpus.iter().enumerate() {
        let marker = if cpu.id() == VICTIM_CPU { " <-- VICTIM" } else { "" };
        let ops_per_ms = bogo_ops_per_ms.iter()
            .find(|(id, _)| *id == cpu.id())
            .map(|(_, rate)| *rate)
            .unwrap_or(0.0);
        eprintln!("{:>6} {:>20} {:>20} {:>20.2}{}",
                  cpu.id(),
                  bogo_ops_counters[idx].load(Ordering::Acquire),
                  scheduled_ns_results[idx],
                  ops_per_ms,
                  marker);
    }

    eprintln!("\n=== Bogo Ops Statistics ===");
    eprintln!("Min:       {:>20} (CPU {})", min_bogo_ops, min_cpu);
    eprintln!("Avg:       {:>20}", avg_bogo_ops);
    eprintln!("P50:       {:>20}", p50_bogo_ops);
    eprintln!("Max:       {:>20}", max_bogo_ops);
    eprintln!("Max Skew:  {:>20} ({:.2}%)", bogo_ops_skew, bogo_ops_skew_pct);

    eprintln!("\n=== Scheduled Time (ns) Statistics ===");
    eprintln!("Min:       {:>20}", min_scheduled_ns);
    eprintln!("Avg:       {:>20}", avg_scheduled_ns);
    eprintln!("P50:       {:>20}", p50_scheduled_ns);
    eprintln!("Max:       {:>20}", max_scheduled_ns);
    eprintln!("Max Skew:  {:>20} ({:.2}%)", scheduled_ns_skew, scheduled_ns_skew_pct);

    eprintln!("\n=== Bogo Ops/ms Statistics ===");
    eprintln!("Min:       {:>20.2}", min_ops_per_ms);
    eprintln!("Avg:       {:>20.2}", avg_ops_per_ms);
    eprintln!("P50:       {:>20.2}", p50_ops_per_ms);
    eprintln!("Max:       {:>20.2}", max_ops_per_ms);
    eprintln!("Max Skew:  {:>20.2} ({:.2}%)", ops_per_ms_skew, ops_per_ms_skew_pct);

    // Assert that the victim CPU has the minimum bogo_ops
    if *min_cpu != VICTIM_CPU {
        return Err(anyhow::anyhow!(
            "Expected victim CPU {} to have minimum bogo_ops, but CPU {} had minimum",
            VICTIM_CPU,
            min_cpu
        ));
    }

    eprintln!("\n✓ Assertion passed: Victim CPU {} has minimum bogo_ops", VICTIM_CPU);

    Ok(())
}

test!("irq_disruption_targeted", irq_disruption_targeted);
