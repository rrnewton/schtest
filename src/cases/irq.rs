//! Tests for IRQ disruption with cgroup cpu.max fairness.

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

/// Test IRQ disruption impact on cgroup cpu.max fairness.
///
/// This test creates two CPU hogs on CPU 1 and CPU 2, both limited to 10% cpu.max.
/// Additionally, a waker thread on CPU 0 sends cross-core wakeup IPIs to a receiver
/// thread on CPU 1 (victim) at IRQ_HZ frequency. This simulates heavy IRQ load.
///
/// # Parameters
/// * CPU_1: First CPU (default: 1) - victim CPU with IRQ load
/// * CPU_2: Second CPU (default: 2) - control CPU without IRQ load
/// * WAKER_CPU: CPU that runs the waker thread (default: 0)
/// * CPU_MAX_PERCENT: cpu.max limit as percentage (default: 10%)
/// * IRQ_HZ: Wakeup IPIs per second (default: 10000)
fn irq_disruption_targeted() -> Result<()> {
    const CPU_1: i32 = 1;
    const CPU_2: i32 = 2;
    const WAKER_CPU: i32 = 0;
    const CPU_MAX_PERCENT: f64 = 10.0;
    const IRQ_HZ: u64 = 10000;

    let system = System::load()?;

    // Collect all hyperthreads (logical CPUs)
    let mut all_cpus = Vec::new();
    for core in system.cores() {
        for ht in core.hyperthreads() {
            all_cpus.push(ht.clone());
        }
    }

    eprintln!("Found {} logical CPUs", all_cpus.len());
    eprintln!("Testing IRQ disruption impact on cpu.max fairness:");
    eprintln!("  CPU {} and CPU {} both limited to {}%", CPU_1, CPU_2, CPU_MAX_PERCENT);
    eprintln!("  Waker on CPU {} sending IPIs to CPU {} at {} Hz", WAKER_CPU, CPU_1, IRQ_HZ);

    // Find the CPUs
    let cpu_1_ht = all_cpus.iter()
        .find(|ht| ht.id() == CPU_1)
        .ok_or_else(|| anyhow::anyhow!("CPU {} not found", CPU_1))?
        .clone();

    let cpu_2_ht = all_cpus.iter()
        .find(|ht| ht.id() == CPU_2)
        .ok_or_else(|| anyhow::anyhow!("CPU {} not found", CPU_2))?
        .clone();

    let waker_ht = all_cpus.iter()
        .find(|ht| ht.id() == WAKER_CPU)
        .ok_or_else(|| anyhow::anyhow!("CPU {} not found", WAKER_CPU))?
        .clone();

    // Create shared memory for start signal and counters
    let allocator = BumpAllocator::new("cpu_max_test", 2 * 1024 * 1024)?;
    let start_signal = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;
    let stop_signal = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;
    let futex_word = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;
    let wakeup_count = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;

    // Create counters for both CPUs
    let bogo_ops_cpu1 = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;
    let scheduled_ns_cpu1 = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;
    let bogo_ops_cpu2 = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;
    let scheduled_ns_cpu2 = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;

    // Create cgroups with cpu.max limits
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

    // Launch receiver thread on CPU 1 (victim) - this will receive the IPI wakeups
    eprintln!("\nLaunching IPI receiver thread on CPU {} (victim)...", CPU_1);
    let receiver_mask = CPUMask::new(&cpu_1_ht);
    let receiver_futex = futex_word.clone();
    let receiver_stop = stop_signal.clone();

    let receiver_child = Child::run(
        move || {
            receiver_mask.run(|| {
                let futex_ptr = receiver_futex.as_ptr() as *mut u32;

                loop {
                    // Check if we should stop
                    if receiver_stop.load(Ordering::Acquire) != 0 {
                        break;
                    }

                    // futex_wait: blocks in kernel until woken by futex_wake
                    // This generates actual cross-core IPI wakeups
                    unsafe {
                        let ret = libc::syscall(
                            libc::SYS_futex,
                            futex_ptr,
                            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                            0u32,  // expected value
                            std::ptr::null::<libc::timespec>(),  // no timeout
                            std::ptr::null::<u32>(),  // uaddr2 (unused)
                            0u32   // val3 (unused)
                        );

                        // EAGAIN means value changed, EINTR means interrupted, both are fine
                        if ret == -1 {
                            let errno = *libc::__errno_location();
                            if errno != libc::EAGAIN && errno != libc::EINTR {
                                eprintln!("futex_wait failed: errno={}", errno);
                            }
                        }
                    }
                }
            })?;
            Ok(())
        },
        None,
    )?;

    // Launch waker thread on CPU 0 - sends IPIs to receiver
    eprintln!("Launching IPI waker thread on CPU {} at {} Hz...", WAKER_CPU, IRQ_HZ);
    let waker_mask = CPUMask::new(&waker_ht);
    let waker_start = start_signal.clone();
    let waker_stop = stop_signal.clone();
    let waker_futex = futex_word.clone();
    let waker_count_shared = wakeup_count.clone();

    let waker_child = Child::run(
        move || {
            waker_mask.run(|| {
                use std::time::Instant;

                // Wait for start signal
                while waker_start.load(Ordering::Acquire) == 0 {
                    std::hint::spin_loop();
                }

                let start_time = Instant::now();
                let target_interval = Duration::from_nanos(1_000_000_000 / IRQ_HZ);
                let mut wakeups_sent = 0u64;
                let futex_ptr = waker_futex.as_ptr() as *mut u32;

                loop {
                    // Check if we should stop
                    if waker_stop.load(Ordering::Acquire) != 0 {
                        break;
                    }

                    // Calculate when the next wakeup should happen
                    let target_time = start_time + target_interval * (wakeups_sent as u32);

                    // Spin until it's time
                    while Instant::now() < target_time {
                        std::hint::spin_loop();
                    }

                    // Increment futex word and wake the receiver
                    // This generates a cross-core IPI to wake the blocked thread
                    waker_futex.fetch_add(1, Ordering::Release);

                    unsafe {
                        libc::syscall(
                            libc::SYS_futex,
                            futex_ptr,
                            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                            1i32,  // wake 1 waiter
                            std::ptr::null::<libc::timespec>(),
                            std::ptr::null::<u32>(),
                            0u32
                        );
                    }

                    wakeups_sent += 1;

                    // Record count every 1000 wakeups to avoid overhead
                    if wakeups_sent % 1000 == 0 {
                        waker_count_shared.store(wakeups_sent, Ordering::Release);
                    }
                }

                // Final count
                waker_count_shared.store(wakeups_sent, Ordering::Release);
            })?;
            Ok(())
        },
        None,
    )?;

    let hog_duration = Duration::from_secs(3);
    eprintln!("\nLaunching 2 CPU hogs for {:?}...", hog_duration);

    // Launch hog on CPU 1 (victim - receives IPIs)
    let mut child_cpu1 = launch_cgroup_hog(
        CPU_1,
        &cpu_1_ht,
        "schtest_cpu_max_cpu1",
        hog_duration,
        start_signal.clone(),
        bogo_ops_cpu1.clone(),
        scheduled_ns_cpu1.clone(),
    )?;

    // Launch hog on CPU 2 (control - no IPI load)
    let mut child_cpu2 = launch_cgroup_hog(
        CPU_2,
        &cpu_2_ht,
        "schtest_cpu_max_cpu2",
        hog_duration,
        start_signal.clone(),
        bogo_ops_cpu2.clone(),
        scheduled_ns_cpu2.clone(),
    )?;

    // Give all threads a moment to initialize
    std::thread::sleep(Duration::from_millis(100));

    // Signal all threads to start simultaneously
    eprintln!("Signaling all threads to START");
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

    // Stop waker and receiver
    eprintln!("Stopping waker and receiver threads...");
    stop_signal.store(1, Ordering::Release);

    // Wake the receiver one last time so it can see the stop signal
    futex_word.fetch_add(1, Ordering::Release);
    unsafe {
        let futex_ptr = futex_word.as_ptr() as *mut u32;
        libc::syscall(
            libc::SYS_futex,
            futex_ptr,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            1i32,
            std::ptr::null::<libc::timespec>(),
            std::ptr::null::<u32>(),
            0u32
        );
    }

    std::thread::sleep(Duration::from_millis(100));
    drop(waker_child);
    drop(receiver_child);

    let total_wakeups = wakeup_count.load(Ordering::Acquire);
    eprintln!("Waker sent {} IPI wakeups total", total_wakeups);

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
    let p50_bogo_ops = avg_bogo_ops;
    let bogo_ops_skew = max_bogo_ops - min_bogo_ops;
    let bogo_ops_skew_pct = if max_bogo_ops > 0 {
        (bogo_ops_skew as f64 / max_bogo_ops as f64) * 100.0
    } else {
        0.0
    };

    let (min_bogo_cpu, _) = bogo_ops_results.iter().min_by_key(|(_, ops)| ops).unwrap();
    let (max_bogo_cpu, _) = bogo_ops_results.iter().max_by_key(|(_, ops)| ops).unwrap();
    let p50_bogo_cpu = *min_bogo_cpu;

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

    let (min_ns_cpu, _) = scheduled_ns_results.iter().min_by_key(|(_, ns)| ns).unwrap();
    let (max_ns_cpu, _) = scheduled_ns_results.iter().max_by_key(|(_, ns)| ns).unwrap();
    let p50_ns_cpu = *min_ns_cpu;

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

    let (min_ops_ms_cpu, _) = bogo_ops_per_ms.iter().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
    let (max_ops_ms_cpu, _) = bogo_ops_per_ms.iter().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
    let p50_ops_ms_cpu = *min_ops_ms_cpu;

    // Print detailed results
    eprintln!("\n=== Per-CPU Results ===");
    eprintln!("{:>6} {:>20} {:>20} {:>20}", "CPU", "bogo_ops", "scheduled_ns", "bogo_ops/ms");
    for (cpu, ops, ns) in &results {
        let ops_per_ms = bogo_ops_per_ms.iter()
            .find(|(id, _)| id == cpu)
            .map(|(_, rate)| *rate)
            .unwrap_or(0.0);
        let marker = if *cpu == CPU_1 { " <-- VICTIM (IPI load)" } else { " <-- CONTROL" };
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

    // Assert that CPU 1 (victim) has lower bogo_ops than CPU 2 (control)
    if bogo_ops_results.iter().find(|(cpu, _)| *cpu == CPU_1).unwrap().1 >=
       bogo_ops_results.iter().find(|(cpu, _)| *cpu == CPU_2).unwrap().1 {
        eprintln!("\n⚠ WARNING: Victim CPU {} did not have lower bogo_ops than control CPU {}",
                  CPU_1, CPU_2);
    } else {
        eprintln!("\n✓ Victim CPU {} has lower bogo_ops than control CPU {} (IPI impact detected)",
                  CPU_1, CPU_2);
    }

    Ok(())
}

/// Helper function to launch a CPU hog and add it to a cgroup
fn launch_cgroup_hog(
    _cpu_id: i32,
    cpu_ht: &crate::util::system::Hyperthread,
    cgroup_name: &str,
    hog_duration: Duration,
    start_signal: SharedBox<AtomicU32>,
    bogo_ops_out: SharedBox<AtomicU64>,
    scheduled_ns_out: SharedBox<AtomicU64>,
) -> Result<Child> {
    let cpu_mask = CPUMask::new(cpu_ht);

    let mut child = Child::run(
        move || {
            cpu_mask.run(|| {
                spinner_utilization::cpu_hog_workload(
                    hog_duration,
                    start_signal,
                    scheduled_ns_out,
                    Some(bogo_ops_out),
                );
            })?;
            Ok(())
        },
        None,
    )?;

    // Add process to its cgroup
    let pid = child.pid().as_raw();
    let procs_path = std::path::Path::new("/sys/fs/cgroup")
        .join(cgroup_name)
        .join("cgroup.procs");
    std::fs::write(&procs_path, pid.to_string())
        .context(format!("Failed to write PID {} to {:?}", pid, procs_path))?;

    Ok(child)
}

test!("irq_disruption_targeted", irq_disruption_targeted);
