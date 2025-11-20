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

use std::collections::HashMap;

/// Per-CPU interrupt counts parsed from /proc/interrupts
#[derive(Debug, Clone)]
struct InterruptSnapshot {
    /// Total interrupt count per CPU (sum of all interrupt types)
    per_cpu_total: HashMap<usize, u64>,
    /// Rescheduling interrupts (RES) - IPIs for rescheduling
    per_cpu_reschedule: HashMap<usize, u64>,
    /// Function call interrupts (CAL) - IPIs for function calls
    per_cpu_function_call: HashMap<usize, u64>,
    /// TLB shootdown interrupts (TLB) - IPIs for TLB invalidation
    per_cpu_tlb: HashMap<usize, u64>,
    /// Number of CPUs detected
    num_cpus: usize,
}

impl InterruptSnapshot {
    /// Parse /proc/interrupts and sum all interrupt types per CPU
    fn capture() -> Result<Self> {
        let data = std::fs::read_to_string("/proc/interrupts")
            .context("Failed to read /proc/interrupts")?;

        let mut lines = data.lines();

        // First line contains CPU headers: "CPU0  CPU1  CPU2  ..."
        let header = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty /proc/interrupts"))?;
        let num_cpus = header.split_whitespace().count();

        let mut per_cpu_total: HashMap<usize, u64> = HashMap::new();
        let mut per_cpu_reschedule: HashMap<usize, u64> = HashMap::new();
        let mut per_cpu_function_call: HashMap<usize, u64> = HashMap::new();
        let mut per_cpu_tlb: HashMap<usize, u64> = HashMap::new();

        for cpu in 0..num_cpus {
            per_cpu_total.insert(cpu, 0);
            per_cpu_reschedule.insert(cpu, 0);
            per_cpu_function_call.insert(cpu, 0);
            per_cpu_tlb.insert(cpu, 0);
        }

        // Parse each interrupt line
        for line in lines {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            // Identify interrupt type
            let is_reschedule = line.contains("Rescheduling interrupts");
            let is_function_call = line.contains("Function call interrupts");
            let is_tlb = line.contains("TLB shootdowns");

            // Skip the interrupt number/name (first column)
            // Next num_cpus columns are the counts per CPU
            for (cpu_idx, count_str) in parts.iter().skip(1).take(num_cpus).enumerate() {
                if let Ok(count) = count_str.parse::<u64>() {
                    *per_cpu_total.entry(cpu_idx).or_insert(0) += count;

                    if is_reschedule {
                        *per_cpu_reschedule.entry(cpu_idx).or_insert(0) += count;
                    } else if is_function_call {
                        *per_cpu_function_call.entry(cpu_idx).or_insert(0) += count;
                    } else if is_tlb {
                        *per_cpu_tlb.entry(cpu_idx).or_insert(0) += count;
                    }
                }
            }
        }

        Ok(InterruptSnapshot {
            per_cpu_total,
            per_cpu_reschedule,
            per_cpu_function_call,
            per_cpu_tlb,
            num_cpus,
        })
    }

    /// Calculate delta from another snapshot (self - other)
    fn delta(&self, other: &InterruptSnapshot) -> InterruptDelta {
        let mut total = HashMap::new();
        let mut reschedule = HashMap::new();
        let mut function_call = HashMap::new();
        let mut tlb = HashMap::new();

        for cpu in 0..self.num_cpus {
            total.insert(
                cpu,
                *self.per_cpu_total.get(&cpu).unwrap_or(&0) as i64
                    - *other.per_cpu_total.get(&cpu).unwrap_or(&0) as i64,
            );
            reschedule.insert(
                cpu,
                *self.per_cpu_reschedule.get(&cpu).unwrap_or(&0) as i64
                    - *other.per_cpu_reschedule.get(&cpu).unwrap_or(&0) as i64,
            );
            function_call.insert(
                cpu,
                *self.per_cpu_function_call.get(&cpu).unwrap_or(&0) as i64
                    - *other.per_cpu_function_call.get(&cpu).unwrap_or(&0) as i64,
            );
            tlb.insert(
                cpu,
                *self.per_cpu_tlb.get(&cpu).unwrap_or(&0) as i64
                    - *other.per_cpu_tlb.get(&cpu).unwrap_or(&0) as i64,
            );
        }

        InterruptDelta {
            total,
            reschedule,
            function_call,
            tlb,
        }
    }
}

/// Delta of interrupt counts between two snapshots
#[derive(Debug)]
struct InterruptDelta {
    total: HashMap<usize, i64>,
    reschedule: HashMap<usize, i64>,
    function_call: HashMap<usize, i64>,
    tlb: HashMap<usize, i64>,
}

/// Different methods for generating IRQ load on the victim CPU
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IrqDisruptionMode {
    /// No IRQ disruption - baseline measurement
    None,
    /// Use futex_wait/futex_wake to generate cross-core IPI wakeups
    Futex,
    /// Use perf_event_open to generate PMI (Performance Monitoring Interrupt) sampling
    Pmu,
    /// Use timerfd for high-frequency timer interrupts
    Timer,
    /// Use all methods simultaneously for maximum IRQ pressure
    Combined,
}

/// Unified handle for any IRQ disruption strategy
enum IrqDisruptionHandle {
    None,
    Futex(FutexIpiDisruptionHandle),
    Pmu(PmuIrqHandle),
    Timer(TimerIrqHandle),
    Combined {
        futex: FutexIpiDisruptionHandle,
        pmu: PmuIrqHandle,
        timer: TimerIrqHandle,
    },
}

impl IrqDisruptionHandle {
    fn stop(self) -> Result<IrqDisruptionStats> {
        match self {
            IrqDisruptionHandle::None => Ok(IrqDisruptionStats::None),
            IrqDisruptionHandle::Futex(handle) => handle.stop(),
            IrqDisruptionHandle::Pmu(handle) => {
                let _ = handle.stop()?;
                Ok(IrqDisruptionStats::Pmu)
            }
            IrqDisruptionHandle::Timer(handle) => {
                let timer_wakeups = handle.stop()?;
                Ok(IrqDisruptionStats::Timer { timer_wakeups })
            }
            IrqDisruptionHandle::Combined { futex, pmu, timer } => {
                let futex_stats = futex.stop()?;
                let _ = pmu.stop()?;
                let timer_wakeups = timer.stop()?;

                match futex_stats {
                    IrqDisruptionStats::Futex {
                        wakeup_count,
                        futex_wait_calls,
                        futex_wait_blocks,
                        futex_wait_eagain,
                    } => Ok(IrqDisruptionStats::Combined {
                        wakeup_count,
                        futex_wait_calls,
                        futex_wait_blocks,
                        futex_wait_eagain,
                        timer_wakeups,
                    }),
                    _ => unreachable!(),
                }
            }
        }
    }
}

/// Statistics from IRQ disruption
enum IrqDisruptionStats {
    None,
    Futex {
        wakeup_count: u64,
        futex_wait_calls: u64,
        futex_wait_blocks: u64,
        futex_wait_eagain: u64,
    },
    Pmu,
    Timer {
        timer_wakeups: u64,
    },
    Combined {
        wakeup_count: u64,
        futex_wait_calls: u64,
        futex_wait_blocks: u64,
        futex_wait_eagain: u64,
        timer_wakeups: u64,
    },
}

/// Handle for futex-based IPI disruption strategy
struct FutexIpiDisruptionHandle {
    waker_child: Child,
    receiver_child: Child,
    stop_signal: SharedBox<AtomicU32>,
    wakeup_count: SharedBox<AtomicU64>,
    // Instrumentation counters
    futex_wait_calls: SharedBox<AtomicU64>,
    futex_wait_blocks: SharedBox<AtomicU64>,
    futex_wait_eagain: SharedBox<AtomicU64>,
}

impl FutexIpiDisruptionHandle {
    /// Stop the IPI disruption and return stats
    fn stop(self) -> Result<IrqDisruptionStats> {
        self.stop_signal.store(1, Ordering::Release);
        std::thread::sleep(Duration::from_millis(100));
        drop(self.waker_child);
        drop(self.receiver_child);

        Ok(IrqDisruptionStats::Futex {
            wakeup_count: self.wakeup_count.load(Ordering::Acquire),
            futex_wait_calls: self.futex_wait_calls.load(Ordering::Acquire),
            futex_wait_blocks: self.futex_wait_blocks.load(Ordering::Acquire),
            futex_wait_eagain: self.futex_wait_eagain.load(Ordering::Acquire),
        })
    }
}

/// Launch futex-based IPI disruption (waker on waker_cpu, receiver on victim_cpu)
///
/// The waker sends futex_wake() at irq_hz frequency to wake the receiver.
/// Each futex_wake() should generate a cross-core IPI to wake the blocked receiver.
fn launch_futex_ipi_disruption(
    allocator: std::sync::Arc<BumpAllocator>,
    victim_cpu: &crate::util::system::Hyperthread,
    waker_cpu: &crate::util::system::Hyperthread,
    start_signal: SharedBox<AtomicU32>,
    irq_hz: u64,
) -> Result<FutexIpiDisruptionHandle> {
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
                            libc::FUTEX_WAIT, // No PRIVATE flag - shared across processes
                            futex_val,        // Wait on current value
                            std::ptr::null::<libc::timespec>(),
                            std::ptr::null::<u32>(),
                            0u32,
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
                            libc::FUTEX_WAKE, // No PRIVATE flag - shared across processes
                            1i32,
                            std::ptr::null::<libc::timespec>(),
                            std::ptr::null::<u32>(),
                            0u32,
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
                    libc::FUTEX_WAKE, // No PRIVATE flag - shared across processes
                    1i32,
                    std::ptr::null::<libc::timespec>(),
                    std::ptr::null::<u32>(),
                    0u32,
                );
            }
        }
    });

    Ok(FutexIpiDisruptionHandle {
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

/// Handle for timer-based IRQ disruption
struct TimerIrqHandle {
    timer_child: Child,
    stop_signal: SharedBox<AtomicU32>,
    wakeup_count: SharedBox<AtomicU64>,
}

impl TimerIrqHandle {
    /// Stop the timer disruption and return wakeup count
    fn stop(self) -> Result<u64> {
        self.stop_signal.store(1, Ordering::Release);
        std::thread::sleep(Duration::from_millis(100));
        drop(self.timer_child);
        Ok(self.wakeup_count.load(Ordering::Acquire))
    }
}

/// Launch timer-based IRQ disruption on victim CPU
///
/// Uses setitimer() with SIGALRM to generate high-frequency timer interrupts.
/// The signal handler does minimal work (just increments a counter), so most
/// time is wasted in kernel interrupt context, not userspace.
fn launch_timer_irq_disruption(
    allocator: std::sync::Arc<BumpAllocator>,
    victim_cpu: &crate::util::system::Hyperthread,
    start_signal: SharedBox<AtomicU32>,
    timer_hz: u64,
) -> Result<TimerIrqHandle> {
    let stop_signal = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;
    let wakeup_count = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;

    let cpu_mask = CPUMask::new(victim_cpu);
    let cpu_id = victim_cpu.id(); // Capture CPU ID before moving into closure
    let timer_start = start_signal.clone();
    let timer_stop = stop_signal.clone();
    let timer_count = wakeup_count.clone();

    let timer_child = Child::run(
        move || {
            cpu_mask.run(|| {
                // Wait for start signal
                while timer_start.load(Ordering::Acquire) == 0 {
                    std::hint::spin_loop();
                }

                // Set up signal handler for SIGALRM that does minimal work
                // The handler just increments the counter - most time is wasted in kernel
                static SIGNAL_COUNT: AtomicU64 = AtomicU64::new(0);

                extern "C" fn timer_signal_handler(_sig: i32) {
                    // Minimal work - just count the signal
                    SIGNAL_COUNT.fetch_add(1, Ordering::Relaxed);
                }

                // Install signal handler
                let sa = libc::sigaction {
                    sa_sigaction: timer_signal_handler as usize,
                    sa_mask: unsafe { std::mem::zeroed() },
                    sa_flags: libc::SA_RESTART,
                    sa_restorer: None,
                };

                let ret = unsafe { libc::sigaction(libc::SIGALRM, &sa, std::ptr::null_mut()) };

                if ret < 0 {
                    let errno = unsafe { *libc::__errno_location() };
                    panic!("sigaction failed: errno={}", errno);
                }

                // Calculate timer interval in nanoseconds
                let interval_ns = 1_000_000_000 / timer_hz;
                let interval_sec = interval_ns / 1_000_000_000;
                let interval_nsec = interval_ns % 1_000_000_000;

                // Create POSIX timer using timer_create
                let mut timer_id: libc::timer_t = std::ptr::null_mut();
                let mut sev: libc::sigevent = unsafe { std::mem::zeroed() };
                sev.sigev_notify = libc::SIGEV_SIGNAL;
                sev.sigev_signo = libc::SIGALRM;
                sev.sigev_value.sival_ptr = std::ptr::null_mut();

                let ret =
                    unsafe { libc::timer_create(libc::CLOCK_MONOTONIC, &mut sev, &mut timer_id) };

                if ret < 0 {
                    let errno = unsafe { *libc::__errno_location() };
                    panic!("timer_create failed: errno={}", errno);
                }

                // Set up timer to fire at timer_hz frequency
                let timer_spec = libc::itimerspec {
                    it_interval: libc::timespec {
                        tv_sec: interval_sec as i64,
                        tv_nsec: interval_nsec as i64,
                    },
                    it_value: libc::timespec {
                        tv_sec: 0,
                        tv_nsec: interval_nsec as i64, // First expiry
                    },
                };

                let ret =
                    unsafe { libc::timer_settime(timer_id, 0, &timer_spec, std::ptr::null_mut()) };

                if ret < 0 {
                    let errno = unsafe { *libc::__errno_location() };
                    panic!("timer_settime failed: errno={}", errno);
                }

                eprintln!(
                    "Timer interrupts enabled on CPU {} at {} Hz (using SIGALRM)",
                    cpu_id, timer_hz
                );

                // Use pause() to block indefinitely - timer signals will interrupt this
                // Each signal increments SIGNAL_COUNT then returns from pause()
                // We spin in userspace doing minimal work, just blocking and handling signals
                loop {
                    if timer_stop.load(Ordering::Acquire) != 0 {
                        break;
                    }

                    // Update shared counter
                    let count = SIGNAL_COUNT.load(Ordering::Relaxed);
                    timer_count.store(count, Ordering::Release);

                    // Use nanosleep for a very short time (10us) to let signals interrupt
                    // The SA_RESTART flag will cause nanosleep to restart after each signal
                    // This creates a tight loop that gets interrupted by timer signals
                    let sleep_spec = libc::timespec {
                        tv_sec: 0,
                        tv_nsec: 10_000, // 10 microseconds
                    };
                    unsafe {
                        libc::nanosleep(&sleep_spec, std::ptr::null_mut());
                    }
                }

                // Disable and delete timer
                unsafe {
                    libc::timer_delete(timer_id);
                }

                // Store final count
                let final_count = SIGNAL_COUNT.load(Ordering::Relaxed);
                timer_count.store(final_count, Ordering::Release);

                eprintln!(
                    "Timer interrupts stopped, {} signals delivered",
                    final_count
                );
            })?;
            Ok(())
        },
        None,
    )?;

    Ok(TimerIrqHandle {
        timer_child,
        stop_signal,
        wakeup_count,
    })
}

/// Determine disruption mode from environment, defaulting to Combined.
/// Set SCHTEST_IRQ_MODE=none|futex|pmu|timer|combined to select mode.
fn get_disruption_mode() -> IrqDisruptionMode {
    match std::env::var("SCHTEST_IRQ_MODE") {
        Ok(val) if val.eq_ignore_ascii_case("none") => IrqDisruptionMode::None,
        Ok(val) if val.eq_ignore_ascii_case("futex") => IrqDisruptionMode::Futex,
        Ok(val) if val.eq_ignore_ascii_case("pmu") => IrqDisruptionMode::Pmu,
        Ok(val) if val.eq_ignore_ascii_case("timer") => IrqDisruptionMode::Timer,
        Ok(val) if val.eq_ignore_ascii_case("combined") => IrqDisruptionMode::Combined,
        _ => IrqDisruptionMode::Combined,
    }
}

/// Read the kernel's maximum allowed perf sample rate
fn get_max_perf_sample_rate() -> Result<u64> {
    let rate_str = std::fs::read_to_string("/proc/sys/kernel/perf_event_max_sample_rate")
        .context("Failed to read perf_event_max_sample_rate")?;
    rate_str
        .trim()
        .parse::<u64>()
        .context("Failed to parse perf_event_max_sample_rate")
}

/// Detect and report scheduler information
fn log_scheduler_info() {
    fn read_sysfs(path: &str) -> Option<String> {
        std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_string())
    }

    eprintln!("\nScheduler Information:");

    // Check if sched_ext is active by looking for /sys/kernel/sched_ext
    let sched_ext_path = "/sys/kernel/sched_ext";
    let sched_ext_active = std::path::Path::new(sched_ext_path).exists();

    if sched_ext_active {
        eprintln!("  sched_ext:                    ACTIVE");

        // Try to read the current scheduler name
        if let Some(scheduler) = read_sysfs("/sys/kernel/sched_ext/root/ops") {
            eprintln!("  Current scheduler:            {}", scheduler);
        } else if let Some(scheduler) = read_sysfs("/sys/kernel/sched_ext/state") {
            eprintln!("  sched_ext state:              {}", scheduler);
        }
    } else {
        eprintln!("  sched_ext:                    NOT ACTIVE");
        eprintln!("  Scheduler:                    CFS (default)");
    }

    // Check scheduler features
    if let Some(features) = read_sysfs("/sys/kernel/debug/sched/features") {
        eprintln!(
            "  Scheduler features:           {}",
            if features.len() > 60 {
                &features[..60]
            } else {
                &features
            }
        );
    }
}

/// Log relevant perf sysctls to aid debugging PMU sampling behavior.
fn log_perf_sysctls() {
    fn read_sysctl(path: &str) -> Option<String> {
        std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_string())
    }
    let paranoid = read_sysctl("/proc/sys/kernel/perf_event_paranoid");
    let max_rate = read_sysctl("/proc/sys/kernel/perf_event_max_sample_rate");
    eprintln!("\nPerf sysctls:");
    match paranoid {
        Some(v) => eprintln!("  perf_event_paranoid           = {}", v),
        None => eprintln!("  perf_event_paranoid           = <unavailable>"),
    }
    match max_rate {
        Some(v) => eprintln!("  perf_event_max_sample_rate    = {}", v),
        None => eprintln!("  perf_event_max_sample_rate    = <unavailable>"),
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
    // Check kernel's maximum allowed sample rate and cap our request
    let max_kernel_rate = get_max_perf_sample_rate()?;
    let actual_freq_hz = target_freq_hz.min(max_kernel_rate);

    if actual_freq_hz < target_freq_hz {
        eprintln!(
            "WARNING: Requested sample rate {} Hz exceeds kernel limit {} Hz",
            target_freq_hz, max_kernel_rate
        );
        eprintln!("         Using capped rate: {} Hz", actual_freq_hz);
    }

    let stop_signal = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;
    let sample_count = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;

    let cpu_id = victim_cpu.id();
    // Pin the perf child to the victim CPU so PMIs/NMIs are delivered on that core.
    let perf_mask = CPUMask::new(victim_cpu);
    let perf_start = start_signal.clone();
    let perf_stop = stop_signal.clone();
    let perf_count = sample_count.clone();

    let perf_child = Child::run(
        move || {
            perf_mask.run(|| {
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
                    sample_freq: u64, // Union with sample_period
                    sample_type: u64,
                    read_format: u64,
                    // Bitfield flags packed into u64
                    flags: u64,
                    // Rest of the structure
                    _rest: [u8; 1024], // Padding for remaining fields
                }

                const PERF_TYPE_HARDWARE: u32 = 0;
                const PERF_COUNT_HW_CPU_CYCLES: u64 = 0;
                const PERF_SAMPLE_IP: u64 = 1 << 0; // Sample instruction pointer

                let mut attr: PerfEventAttr = unsafe { std::mem::zeroed() };
                attr.type_ = PERF_TYPE_HARDWARE;
                attr.size = 128; // PERF_ATTR_SIZE_VER7 - a safe modern size
                attr.config = PERF_COUNT_HW_CPU_CYCLES;
                attr.sample_freq = actual_freq_hz;
                attr.sample_type = PERF_SAMPLE_IP; // Must sample something when using sampling mode
                attr.read_format = 0;

                // Build flags bitfield: freq=1 (bit 10), others=0
                attr.flags = 0;
                attr.flags |= 1 << 10; // freq mode

                // perf_event_open(attr, pid, cpu, group_fd, flags)
                let perf_fd = unsafe {
                    libc::syscall(
                        libc::SYS_perf_event_open,
                        &attr as *const PerfEventAttr,
                        -1i32,  // pid = -1 (all processes)
                        cpu_id, // cpu
                        -1i32,  // group_fd = -1
                        0u64,   // flags
                    ) as i32
                };

                if perf_fd < 0 {
                    let errno = unsafe { *libc::__errno_location() };
                    panic!(
                        "perf_event_open failed on CPU {} with {} Hz: errno={}",
                        cpu_id, actual_freq_hz, errno
                    );
                }

                // Enable the event
                let enable_ret = unsafe {
                    libc::ioctl(perf_fd, 9216, 0) // PERF_EVENT_IOC_ENABLE
                };

                if enable_ret < 0 {
                    let errno = unsafe { *libc::__errno_location() };
                    panic!(
                        "PERF_EVENT_IOC_ENABLE failed on CPU {}: errno={}",
                        cpu_id, errno
                    );
                }

                eprintln!(
                    "PMU sampling enabled on CPU {} at {} Hz",
                    cpu_id, actual_freq_hz
                );

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
            })?;
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
    const WAKER_CPU: i32 = 0; // Only used for Futex mode
    const CPU_MAX_PERCENT: f64 = 50.0;
    const IRQ_HZ: u64 = 100 * 1000;
    let disruption_mode: IrqDisruptionMode = get_disruption_mode();

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
    eprintln!(
        "  CPU {} and CPU {} both limited to {}%",
        CPU_1, CPU_2, CPU_MAX_PERCENT
    );
    match disruption_mode {
        IrqDisruptionMode::None => {
            eprintln!("  NO IRQ disruption - baseline measurement");
        }
        IrqDisruptionMode::Futex => {
            eprintln!(
                "  Futex-based IPI disruption: Waker on CPU {} -> Receiver on CPU {} at {} Hz",
                WAKER_CPU, CPU_1, IRQ_HZ
            );
        }
        IrqDisruptionMode::Pmu => {
            eprintln!(
                "  PMU sampling on CPU {} at {} Hz (PMI interrupts)",
                CPU_1, IRQ_HZ
            );
        }
        IrqDisruptionMode::Timer => {
            eprintln!("  Timer interrupts on CPU {} at {} Hz", CPU_1, IRQ_HZ);
        }
        IrqDisruptionMode::Combined => {
            eprintln!("  COMBINED mode: PMU sampling + Futex IPI + Timer interrupts");
            eprintln!("    PMU: CPU {} at {} Hz (PMI interrupts)", CPU_1, IRQ_HZ);
            eprintln!(
                "    Futex: Waker on CPU {} -> Receiver on CPU {} at {} Hz",
                WAKER_CPU, CPU_1, IRQ_HZ
            );
            eprintln!("    Timer: CPU {} at {} Hz", CPU_1, IRQ_HZ);
        }
    }

    // Log scheduler and perf information
    log_scheduler_info();
    log_perf_sysctls();

    // Find the CPUs
    let cpu_1_ht = all_cpus
        .iter()
        .find(|ht| ht.id() == CPU_1)
        .ok_or_else(|| anyhow::anyhow!("CPU {} not found", CPU_1))?
        .clone();

    let cpu_2_ht = all_cpus
        .iter()
        .find(|ht| ht.id() == CPU_2)
        .ok_or_else(|| anyhow::anyhow!("CPU {} not found", CPU_2))?
        .clone();

    let waker_ht = all_cpus
        .iter()
        .find(|ht| ht.id() == WAKER_CPU)
        .ok_or_else(|| anyhow::anyhow!("CPU {} not found", WAKER_CPU))?
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
    let period_us = 100000u64; // 100ms period
    let quota_us = (period_us as f64 * CPU_MAX_PERCENT / 100.0) as i64;

    eprintln!(
        "\nCreating cgroups with cpu.max={}/{} ({}%)",
        quota_us, period_us, CPU_MAX_PERCENT
    );

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

    // Capture interrupt counts before the test
    eprintln!("\nCapturing interrupt baseline...");
    let interrupts_before = InterruptSnapshot::capture()?;

    // Launch IRQ disruption based on selected mode
    let irq_handle = match disruption_mode {
        IrqDisruptionMode::None => {
            eprintln!("\nSkipping IRQ disruption (baseline mode)");
            IrqDisruptionHandle::None
        }
        IrqDisruptionMode::Futex => {
            eprintln!("\nLaunching IRQ disruption...");
            eprintln!(
                "  Futex mode: Receiver on CPU {}, Waker on CPU {} at {} Hz",
                CPU_1, WAKER_CPU, IRQ_HZ
            );
            let handle = launch_futex_ipi_disruption(
                allocator.clone(),
                &cpu_1_ht,
                &waker_ht,
                start_signal.clone(),
                IRQ_HZ,
            )?;
            IrqDisruptionHandle::Futex(handle)
        }
        IrqDisruptionMode::Pmu => {
            eprintln!("\nLaunching IRQ disruption...");
            eprintln!("  PMU mode: Sampling on CPU {} at {} Hz", CPU_1, IRQ_HZ);
            let handle = launch_pmu_irq_disruption(
                allocator.clone(),
                &cpu_1_ht,
                start_signal.clone(),
                IRQ_HZ,
            )?;
            IrqDisruptionHandle::Pmu(handle)
        }
        IrqDisruptionMode::Combined => {
            eprintln!("\nLaunching IRQ disruption...");
            eprintln!("  Combined mode: Launching both PMU and Futex...");
            let pmu_handle = launch_pmu_irq_disruption(
                allocator.clone(),
                &cpu_1_ht,
                start_signal.clone(),
                IRQ_HZ,
            )?;
            let futex_handle = launch_futex_ipi_disruption(
                allocator.clone(),
                &cpu_1_ht,
                &waker_ht,
                start_signal.clone(),
                IRQ_HZ,
            )?;
            let timer_handle = launch_timer_irq_disruption(
                allocator.clone(),
                &cpu_1_ht,
                start_signal.clone(),
                IRQ_HZ,
            )?;
            IrqDisruptionHandle::Combined {
                futex: futex_handle,
                pmu: pmu_handle,
                timer: timer_handle,
            }
        }
        IrqDisruptionMode::Timer => {
            eprintln!("\nLaunching IRQ disruption...");
            eprintln!("  Timer mode: timerfd on CPU {} at {} Hz", CPU_1, IRQ_HZ);
            let handle = launch_timer_irq_disruption(
                allocator.clone(),
                &cpu_1_ht,
                start_signal.clone(),
                IRQ_HZ,
            )?;
            IrqDisruptionHandle::Timer(handle)
        }
    };

    let hog_duration = Duration::from_secs(10); // TODO: make a CLI flag.
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

    // Stop IRQ disruption
    eprintln!("Stopping IRQ disruption...");
    let stats = irq_handle.stop()?;

    // Capture interrupt counts after the test
    let interrupts_after = InterruptSnapshot::capture()?;
    let interrupt_delta = interrupts_after.delta(&interrupts_before);

    // Report stats based on mode
    match stats {
        IrqDisruptionStats::None => {
            eprintln!("\n=== No IRQ Disruption (Baseline) ===");
            eprintln!("No artificial IRQ load applied");
        }
        IrqDisruptionStats::Futex {
            wakeup_count,
            futex_wait_calls,
            futex_wait_blocks,
            futex_wait_eagain,
        } => {
            eprintln!("\n=== Futex IPI Disruption Stats ===");
            eprintln!("Waker sent {} wakeups total", wakeup_count);
            eprintln!("Receiver futex_wait calls:  {}", futex_wait_calls);
            eprintln!("  Actual blocks:             {}", futex_wait_blocks);
            eprintln!("  EAGAIN returns:            {}", futex_wait_eagain);

            let block_pct = if futex_wait_calls > 0 {
                (futex_wait_blocks as f64 / futex_wait_calls as f64) * 100.0
            } else {
                0.0
            };
            eprintln!("  Block rate:                {:.2}%", block_pct);
        }
        IrqDisruptionStats::Pmu => {
            eprintln!("\n=== PMU Disruption Stats ===");
            eprintln!("PMU sampling configured at {} Hz on CPU {}", IRQ_HZ, CPU_1);
            eprintln!("(PMIs delivered as NMI-like interrupts throughout test)");
        }
        IrqDisruptionStats::Timer { timer_wakeups } => {
            eprintln!("\n=== Timer Disruption Stats ===");
            eprintln!(
                "Timer: {} wakeups total at {} Hz on CPU {}",
                timer_wakeups, IRQ_HZ, CPU_1
            );
        }
        IrqDisruptionStats::Combined {
            wakeup_count,
            futex_wait_calls,
            futex_wait_blocks,
            futex_wait_eagain,
            timer_wakeups,
        } => {
            eprintln!("\n=== COMBINED Disruption Stats ===");
            eprintln!("PMU: Sampling configured at {} Hz on CPU {}", IRQ_HZ, CPU_1);
            eprintln!("     (PMIs delivered as NMI-like interrupts throughout test)");
            eprintln!("\nFutex: Waker sent {} wakeups total", wakeup_count);
            eprintln!("  Receiver futex_wait calls:  {}", futex_wait_calls);
            eprintln!("    Actual blocks:             {}", futex_wait_blocks);
            eprintln!("    EAGAIN returns:            {}", futex_wait_eagain);

            let block_pct = if futex_wait_calls > 0 {
                (futex_wait_blocks as f64 / futex_wait_calls as f64) * 100.0
            } else {
                0.0
            };
            eprintln!("    Block rate:                {:.2}%", block_pct);
            eprintln!("\nTimer: {} wakeups total", timer_wakeups);
        }
    }

    // Report interrupt deltas per CPU
    eprintln!("\n=== Interrupt Counts (Delta During Test) ===");
    eprintln!(
        "{:>6} {:>15} {:>15} {:>15} {:>15}",
        "CPU", "Total", "Reschedule", "Func Call", "TLB"
    );

    // Collect key CPUs
    let key_cpus = vec![
        (WAKER_CPU as usize, "WAKER"),
        (CPU_1 as usize, "VICTIM"),
        (CPU_2 as usize, "CONTROL"),
    ];

    for (cpu, label) in &key_cpus {
        let total = *interrupt_delta.total.get(cpu).unwrap_or(&0);
        let reschedule = *interrupt_delta.reschedule.get(cpu).unwrap_or(&0);
        let function_call = *interrupt_delta.function_call.get(cpu).unwrap_or(&0);
        let tlb = *interrupt_delta.tlb.get(cpu).unwrap_or(&0);

        eprintln!(
            "{:>6} {:>15} {:>15} {:>15} {:>15}  <-- {}",
            cpu, total, reschedule, function_call, tlb, label
        );
    }

    // Calculate IPI totals (reschedule + function_call + tlb)
    let victim_total = *interrupt_delta.total.get(&(CPU_1 as usize)).unwrap_or(&0);
    let victim_ipis = *interrupt_delta
        .reschedule
        .get(&(CPU_1 as usize))
        .unwrap_or(&0)
        + *interrupt_delta
            .function_call
            .get(&(CPU_1 as usize))
            .unwrap_or(&0)
        + *interrupt_delta.tlb.get(&(CPU_1 as usize)).unwrap_or(&0);

    let control_total = *interrupt_delta.total.get(&(CPU_2 as usize)).unwrap_or(&0);
    let control_ipis = *interrupt_delta
        .reschedule
        .get(&(CPU_2 as usize))
        .unwrap_or(&0)
        + *interrupt_delta
            .function_call
            .get(&(CPU_2 as usize))
            .unwrap_or(&0)
        + *interrupt_delta.tlb.get(&(CPU_2 as usize)).unwrap_or(&0);

    eprintln!("\nInterrupt Summary:");
    eprintln!(
        "  Victim CPU {} total interrupts: {} (IPIs: {})",
        CPU_1, victim_total, victim_ipis
    );
    eprintln!(
        "  Control CPU {} total interrupts: {} (IPIs: {})",
        CPU_2, control_total, control_ipis
    );

    if victim_ipis > control_ipis * 2 {
        eprintln!(
            "  ✓ Victim CPU has {}x more IPIs than control",
            victim_ipis / control_ipis.max(1)
        );
    } else {
        eprintln!(
            "  ⚠ Victim CPU IPI rate similar to control (ratio: {:.2}x)",
            victim_ipis as f64 / control_ipis.max(1) as f64
        );
    }

    // Clean up cgroups
    drop(cgroup_cpu1);
    drop(cgroup_cpu2);

    // Collect results
    let results: Vec<(i32, u64, u64)> = vec![
        (
            CPU_1,
            bogo_ops_cpu1.load(Ordering::Acquire),
            scheduled_ns_cpu1.load(Ordering::Acquire),
        ),
        (
            CPU_2,
            bogo_ops_cpu2.load(Ordering::Acquire),
            scheduled_ns_cpu2.load(Ordering::Acquire),
        ),
    ];

    // Calculate statistics
    let bogo_ops_results: Vec<(i32, u64)> =
        results.iter().map(|(cpu, ops, _)| (*cpu, *ops)).collect();
    let scheduled_ns_results: Vec<(i32, u64)> =
        results.iter().map(|(cpu, _, ns)| (*cpu, *ns)).collect();

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

    let (min_ns_cpu, _) = scheduled_ns_results
        .iter()
        .min_by_key(|(_, ns)| ns)
        .unwrap();
    let (max_ns_cpu, _) = scheduled_ns_results
        .iter()
        .max_by_key(|(_, ns)| ns)
        .unwrap();
    let p50_ns_cpu = *min_ns_cpu;

    // Bogo ops per millisecond
    let bogo_ops_per_ms: Vec<(i32, f64)> = results
        .iter()
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

    let (min_ops_ms_cpu, _) = bogo_ops_per_ms
        .iter()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let (max_ops_ms_cpu, _) = bogo_ops_per_ms
        .iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let p50_ops_ms_cpu = *min_ops_ms_cpu;

    // Print detailed results
    eprintln!("\n=== Per-CPU Results ===");
    eprintln!(
        "{:>6} {:>20} {:>20} {:>20}",
        "CPU", "bogo_ops", "scheduled_ns", "bogo_ops/ms"
    );
    for (cpu, ops, ns) in &results {
        let ops_per_ms = bogo_ops_per_ms
            .iter()
            .find(|(id, _)| id == cpu)
            .map(|(_, rate)| *rate)
            .unwrap_or(0.0);
        let marker = if *cpu == CPU_1 {
            " <-- VICTIM (IPI load)"
        } else {
            " <-- CONTROL"
        };
        eprintln!(
            "{:>6} {:>20} {:>20} {:>20.2}{}",
            cpu, ops, ns, ops_per_ms, marker
        );
    }

    eprintln!("\n=== Bogo Ops Statistics ===");
    eprintln!("Min:       {:>20} (CPU {})", min_bogo_ops, min_bogo_cpu);
    eprintln!("Avg:       {:>20}", avg_bogo_ops);
    eprintln!("P50:       {:>20} (CPU {})", p50_bogo_ops, p50_bogo_cpu);
    eprintln!("Max:       {:>20} (CPU {})", max_bogo_ops, max_bogo_cpu);
    eprintln!(
        "Max Skew:  {:>20} ({:.2}%)",
        bogo_ops_skew, bogo_ops_skew_pct
    );

    eprintln!("\n=== Scheduled Time (ns) Statistics ===");
    eprintln!("Min:       {:>20} (CPU {})", min_scheduled_ns, min_ns_cpu);
    eprintln!("Avg:       {:>20}", avg_scheduled_ns);
    eprintln!("P50:       {:>20} (CPU {})", p50_scheduled_ns, p50_ns_cpu);
    eprintln!("Max:       {:>20} (CPU {})", max_scheduled_ns, max_ns_cpu);
    eprintln!(
        "Max Skew:  {:>20} ({:.2}%)",
        scheduled_ns_skew, scheduled_ns_skew_pct
    );

    eprintln!("\n=== Bogo Ops/ms Statistics ===");
    eprintln!(
        "Min:       {:>20.2} (CPU {})",
        min_ops_per_ms, min_ops_ms_cpu
    );
    eprintln!("Avg:       {:>20.2}", avg_ops_per_ms);
    eprintln!(
        "P50:       {:>20.2} (CPU {})",
        p50_ops_per_ms, p50_ops_ms_cpu
    );
    eprintln!(
        "Max:       {:>20.2} (CPU {})",
        max_ops_per_ms, max_ops_ms_cpu
    );
    eprintln!(
        "Max Skew:  {:>20.2} ({:.2}%)",
        ops_per_ms_skew, ops_per_ms_skew_pct
    );

    // Assert that CPU 1 (victim) has lower bogo_ops than CPU 2 (control)
    if bogo_ops_results
        .iter()
        .find(|(cpu, _)| *cpu == CPU_1)
        .unwrap()
        .1
        >= bogo_ops_results
            .iter()
            .find(|(cpu, _)| *cpu == CPU_2)
            .unwrap()
            .1
    {
        eprintln!(
            "\n⚠ WARNING: Victim CPU {} did not have lower bogo_ops than control CPU {}",
            CPU_1, CPU_2
        );
        eprintln!("   This suggests the IPI disruption strategy may not be working as expected.");
    } else {
        eprintln!(
            "\n✓ Victim CPU {} has lower bogo_ops than control CPU {} (IPI impact detected)",
            CPU_1, CPU_2
        );
    }

    Ok(())
}

test!("irq_disruption_targeted", irq_disruption_targeted);
