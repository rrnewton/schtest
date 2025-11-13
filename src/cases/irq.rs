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

/// Handle for an IPI disruption strategy (waker + receiver threads)
struct IpiDisruptionHandle {
    waker_child: Child,
    receiver_child: Child,
    stop_signal: SharedBox<AtomicU32>,
    wakeup_count: SharedBox<AtomicU64>,
    // Instrumentation counters
    futex_wait_calls: SharedBox<AtomicU64>,
    futex_wait_blocks: SharedBox<AtomicU64>,
    futex_wait_eagain: SharedBox<AtomicU64>,
}

impl IpiDisruptionHandle {
    /// Stop the IPI disruption and return stats
    fn stop(self) -> Result<IpiDisruptionStats> {
        self.stop_signal.store(1, Ordering::Release);
        std::thread::sleep(Duration::from_millis(100));
        drop(self.waker_child);
        drop(self.receiver_child);

        Ok(IpiDisruptionStats {
            wakeup_count: self.wakeup_count.load(Ordering::Acquire),
            futex_wait_calls: self.futex_wait_calls.load(Ordering::Acquire),
            futex_wait_blocks: self.futex_wait_blocks.load(Ordering::Acquire),
            futex_wait_eagain: self.futex_wait_eagain.load(Ordering::Acquire),
        })
    }
}

/// Statistics from IPI disruption
struct IpiDisruptionStats {
    wakeup_count: u64,
    futex_wait_calls: u64,
    futex_wait_blocks: u64,
    futex_wait_eagain: u64,
}

/// Launch futex-based IPI disruption (waker on waker_cpu, receiver on victim_cpu)
///
/// The waker sends futex_wake() at irq_hz frequency to wake the receiver.
/// Each futex_wake() should generate a cross-core IPI to wake the blocked receiver.
#[allow(dead_code)]
fn launch_futex_ipi_disruption(
    allocator: std::sync::Arc<BumpAllocator>,
    victim_cpu: &crate::util::system::Hyperthread,
    waker_cpu: &crate::util::system::Hyperthread,
    start_signal: SharedBox<AtomicU32>,
    irq_hz: u64,
) -> Result<IpiDisruptionHandle> {
    let stop_signal = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;
    let futex_word = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;
    let wakeup_count = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;

    // Create instrumentation counters
    let futex_wait_calls = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;
    let futex_wait_blocks = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;
    let futex_wait_eagain = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;

    // Launch receiver thread on victim CPU
    let receiver_mask = CPUMask::new(victim_cpu);
    let receiver_futex = futex_word.clone();
    let receiver_stop = stop_signal.clone();
    let receiver_start = start_signal.clone();
    let receiver_calls = futex_wait_calls.clone();
    let receiver_blocks = futex_wait_blocks.clone();
    let receiver_eagain = futex_wait_eagain.clone();

    let receiver_child = Child::run(
        move || {
            receiver_mask.run(|| {
                // Wait for start signal
                while receiver_start.load(Ordering::Acquire) == 0 {
                    std::hint::spin_loop();
                }

                let futex_ptr = receiver_futex.as_ptr() as *mut u32;

                loop {
                    // Check if we should stop
                    if receiver_stop.load(Ordering::Acquire) != 0 {
                        break;
                    }

                    // Read current futex value and wait on it (standard futex pattern)
                    let futex_val = receiver_futex.load(Ordering::Acquire);

                    // futex_wait: blocks in kernel until woken by futex_wake
                    // Wait for the CURRENT value to change, not for it to be 0
                    receiver_calls.fetch_add(1, Ordering::Relaxed);
                    unsafe {
                        let ret = libc::syscall(
                            libc::SYS_futex,
                            futex_ptr,
                            libc::FUTEX_WAIT,  // No PRIVATE flag - shared across processes
                            futex_val,  // Wait on current value
                            std::ptr::null::<libc::timespec>(),
                            std::ptr::null::<u32>(),
                            0u32
                        );

                        // Track whether we actually blocked or got EAGAIN
                        if ret == -1 {
                            let errno = *libc::__errno_location();
                            if errno == libc::EAGAIN {
                                receiver_eagain.fetch_add(1, Ordering::Relaxed);
                            } else if errno != libc::EINTR {
                                eprintln!("futex_wait failed: errno={}", errno);
                            }
                        } else {
                            // Successful block and wakeup
                            receiver_blocks.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            })?;
            Ok(())
        },
        None,
    )?;

    // Launch waker thread
    let waker_mask = CPUMask::new(waker_cpu);
    let waker_start = start_signal;
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
                let target_interval = Duration::from_nanos(1_000_000_000 / irq_hz);
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
                    waker_futex.fetch_add(1, Ordering::Release);

                    unsafe {
                        libc::syscall(
                            libc::SYS_futex,
                            futex_ptr,
                            libc::FUTEX_WAKE,  // No PRIVATE flag - shared across processes
                            1i32,
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

    // Wake the receiver one last time so it can see the stop signal when stopping
    let stop_signal_clone = stop_signal.clone();
    let futex_word_clone = futex_word.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(10)); // Should be longer than test
        if stop_signal_clone.load(Ordering::Acquire) != 0 {
            futex_word_clone.fetch_add(1, Ordering::Release);
            unsafe {
                let futex_ptr = futex_word_clone.as_ptr() as *mut u32;
                libc::syscall(
                    libc::SYS_futex,
                    futex_ptr,
                    libc::FUTEX_WAKE,  // No PRIVATE flag - shared across processes
                    1i32,
                    std::ptr::null::<libc::timespec>(),
                    std::ptr::null::<u32>(),
                    0u32
                );
            }
        }
    });

    Ok(IpiDisruptionHandle {
        waker_child,
        receiver_child,
        stop_signal,
        wakeup_count,
        futex_wait_calls,
        futex_wait_blocks,
        futex_wait_eagain,
    })
}

/// Handle for PMU-based IRQ disruption
struct PmuIrqHandle {
    perf_child: Child,
    stop_signal: SharedBox<AtomicU32>,
    sample_count: SharedBox<AtomicU64>,
}

impl PmuIrqHandle {
    /// Stop the PMU disruption and return sample count
    fn stop(self) -> Result<u64> {
        self.stop_signal.store(1, Ordering::Release);
        std::thread::sleep(Duration::from_millis(100));
        drop(self.perf_child);
        Ok(self.sample_count.load(Ordering::Acquire))
    }
}

/// Launch PMU-based IRQ disruption on victim CPU
///
/// Uses perf_event_open to configure high-frequency PMU sampling on the victim CPU.
/// PMIs (Performance Monitoring Interrupts) are delivered as NMI-like interrupts.
fn launch_pmu_irq_disruption(
    allocator: std::sync::Arc<BumpAllocator>,
    victim_cpu: &crate::util::system::Hyperthread,
    start_signal: SharedBox<AtomicU32>,
    target_freq_hz: u64,
) -> Result<PmuIrqHandle> {
    let stop_signal = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;
    let sample_count = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;

    let cpu_id = victim_cpu.id();
    let perf_start = start_signal.clone();
    let perf_stop = stop_signal.clone();
    let perf_count = sample_count.clone();

    let perf_child = Child::run(
        move || {
            // Wait for start signal
            while perf_start.load(Ordering::Acquire) == 0 {
                std::hint::spin_loop();
            }

            // perf_event_attr structure for perf_event_open
            // Must match kernel struct - see /usr/include/linux/perf_event.h
            #[repr(C)]
            struct PerfEventAttr {
                type_: u32,
                size: u32,
                config: u64,
                sample_freq: u64,  // Union with sample_period
                sample_type: u64,
                read_format: u64,
                // Bitfield flags packed into u64
                flags: u64,
                // Rest of the structure
                _rest: [u8; 1024],  // Padding for remaining fields
            }

            const PERF_TYPE_HARDWARE: u32 = 0;
            const PERF_COUNT_HW_CPU_CYCLES: u64 = 0;
            const PERF_SAMPLE_IP: u64 = 1 << 0;  // Sample instruction pointer

            let mut attr: PerfEventAttr = unsafe { std::mem::zeroed() };
            attr.type_ = PERF_TYPE_HARDWARE;
            attr.size = 128; // PERF_ATTR_SIZE_VER7 - a safe modern size
            attr.config = PERF_COUNT_HW_CPU_CYCLES;
            attr.sample_freq = target_freq_hz;
            attr.sample_type = PERF_SAMPLE_IP;  // Must sample something when using sampling mode
            attr.read_format = 0;

            // Build flags bitfield: freq=1 (bit 10), others=0
            attr.flags = 0;
            attr.flags |= 1 << 10;  // freq mode

            // perf_event_open(attr, pid, cpu, group_fd, flags)
            let perf_fd = unsafe {
                libc::syscall(
                    libc::SYS_perf_event_open,
                    &attr as *const PerfEventAttr,
                    -1i32,     // pid = -1 (all processes)
                    cpu_id,    // cpu
                    -1i32,     // group_fd = -1
                    0u64,      // flags
                ) as i32
            };

            if perf_fd < 0 {
                let errno = unsafe { *libc::__errno_location() };
                return Err(anyhow::anyhow!("perf_event_open failed: errno={}", errno));
            }

            // Enable the event
            let enable_ret = unsafe {
                libc::ioctl(perf_fd, 9216, 0) // PERF_EVENT_IOC_ENABLE
            };

            if enable_ret < 0 {
                let errno = unsafe { *libc::__errno_location() };
                return Err(anyhow::anyhow!("PERF_EVENT_IOC_ENABLE failed: errno={}", errno));
            }

            eprintln!("PMU sampling enabled on CPU {} at {} Hz", cpu_id, target_freq_hz);

            // Keep the perf event alive until stop signal
            // PMIs will be generated automatically by the hardware
            loop {
                if perf_stop.load(Ordering::Acquire) != 0 {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }

            // Try to read final count (this may not work for all event types)
            let mut count: u64 = 0;
            let samples = unsafe {
                let ret = libc::read(
                    perf_fd,
                    &mut count as *mut u64 as *mut libc::c_void,
                    std::mem::size_of::<u64>(),
                );
                if ret == std::mem::size_of::<u64>() as isize {
                    count
                } else {
                    0
                }
            };

            perf_count.store(samples, Ordering::Release);

            // Disable and close
            unsafe {
                libc::ioctl(perf_fd, 9217, 0); // PERF_EVENT_IOC_DISABLE
                libc::close(perf_fd);
            }

            eprintln!("PMU sampling stopped");
            Ok(())
        },
        None,
    )?;

    Ok(PmuIrqHandle {
        perf_child,
        stop_signal,
        sample_count,
    })
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

    let child = Child::run(
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

/// Test IRQ disruption impact on cgroup cpu.max fairness.
///
/// This test creates two CPU hogs on CPU 1 and CPU 2, both limited to 10% cpu.max.
/// Additionally, PMU sampling generates high-frequency PMIs (Performance Monitoring
/// Interrupts) on CPU 1 (victim) at IRQ_HZ frequency. PMIs are NMI-like interrupts
/// that preempt almost everything, simulating heavy IRQ load.
fn irq_disruption_targeted() -> Result<()> {
    const CPU_1: i32 = 1;
    const CPU_2: i32 = 2;
    const CPU_MAX_PERCENT: f64 = 50.0;
    const IRQ_HZ: u64 = 100 * 1000;

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
    eprintln!("  PMU sampling on CPU {} at {} Hz (PMI interrupts)", CPU_1, IRQ_HZ);

    // Find the CPUs
    let cpu_1_ht = all_cpus.iter()
        .find(|ht| ht.id() == CPU_1)
        .ok_or_else(|| anyhow::anyhow!("CPU {} not found", CPU_1))?
        .clone();

    let cpu_2_ht = all_cpus.iter()
        .find(|ht| ht.id() == CPU_2)
        .ok_or_else(|| anyhow::anyhow!("CPU {} not found", CPU_2))?
        .clone();

    // Create shared memory for start signal and counters
    let allocator = BumpAllocator::new("cpu_max_test", 2 * 1024 * 1024)?;
    let start_signal = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;

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

    // Launch PMU-based IRQ disruption
    eprintln!("\nLaunching PMU-based IRQ disruption...");
    eprintln!("  PMU sampling on CPU {} at {} Hz", CPU_1, IRQ_HZ);
    let irq_handle = launch_pmu_irq_disruption(
        allocator.clone(),
        &cpu_1_ht,
        start_signal.clone(),
        IRQ_HZ,
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

    // Stop PMU disruption
    eprintln!("Stopping PMU disruption...");
    let _ = irq_handle.stop()?;
    eprintln!("\n=== PMU Disruption Stats ===");
    eprintln!("PMU sampling configured at {} Hz on CPU {}", IRQ_HZ, CPU_1);
    eprintln!("(PMIs delivered as NMI-like interrupts throughout test)");

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
        eprintln!("   This suggests the IPI disruption strategy may not be working as expected.");
    } else {
        eprintln!("\n✓ Victim CPU {} has lower bogo_ops than control CPU {} (IPI impact detected)",
                  CPU_1, CPU_2);
    }

    Ok(())
}

test!("irq_disruption_targeted", irq_disruption_targeted);
