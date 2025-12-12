//! Tests for IRQ-induced task migration.
//!
//! This test measures whether the scheduler migrates a task away from IRQ-heavy
//! CPUs to a quiet control CPU. Unlike irq_accounting which compares throughput
//! of two pinned workers, this test:
//!
//! 1. Victimizes ALL physical cores except one control CPU with timer IRQs
//! 2. Starts a single worker pinned to a victim CPU
//! 3. Unpins the worker at test start
//! 4. Measures whether the scheduler migrates the worker to the control CPU
//!
//! The idea is that by stressing all cores but one, the scheduler has only one
//! good choice for where to migrate the task.
//!
//! # Environment Variables
//!
//! - `SCHTEST_IRQ_DURATION`: Test duration in seconds. Defaults to `10`.
//! - `SCHTEST_IRQ_RESERVE_TRACING_CORE`: CPU ID to reserve from IRQ storm for tracing
//!   (e.g., for wprof). When set, this CPU will not receive IRQ disruption, allowing
//!   a tracer to run on it without interference.

use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};

use crate::test;
use crate::util::child::Child;
use crate::util::shared::{BumpAllocator, SharedBox};
use crate::util::system::{CPUMask, CPUSet, System};
use crate::workloads::spinner_utilization;

use super::irq_common::{
    get_reserved_tracing_core, get_test_duration, launch_timer_irq_disruption, log_scheduler_info,
    InterruptSnapshot, TimerIrqHandle, DEFAULT_IRQ_HZ, NUM_TIMERS,
};

/// Test whether the scheduler migrates a task from IRQ-heavy CPUs to a quiet CPU.
///
/// This test:
/// 1. Picks one CPU as the "control" (no IRQ load)
/// 2. Launches timer IRQ disruption on ALL other physical cores
/// 3. Starts a spinner on one of the victim CPUs, then unpins it
/// 4. Measures which CPU the task ends up on
fn irq_migration_test() -> Result<()> {
    let system = System::load()?;

    // Collect all physical cores (we'll victimize all but one)
    let cores: Vec<_> = system.cores().into_iter().collect();
    if cores.len() < 2 {
        anyhow::bail!("Need at least 2 physical cores for this test");
    }

    // Pick control core (last physical core) and its first hyperthread
    let control_core = &cores[cores.len() - 1];
    let control_cpu = control_core.hyperthreads().first().unwrap().clone();
    let control_cpu_id = control_cpu.id();

    // Pick initial victim (first physical core's first hyperthread)
    let initial_victim_core = &cores[0];
    let initial_victim_cpu = initial_victim_core.hyperthreads().first().unwrap().clone();
    let initial_victim_cpu_id = initial_victim_cpu.id();

    // Check if a CPU should be reserved for tracing (e.g., for wprof)
    let reserved_tracing_cpu = get_reserved_tracing_core();

    // Collect all victim CPUs (one hyperthread per physical core, excluding control core)
    let mut victim_cpus = Vec::new();
    for core in &cores {
        // Skip the control core entirely
        let first_ht = core.hyperthreads().first().unwrap();
        if first_ht.id() == control_cpu_id {
            continue;
        }
        // Skip the reserved tracing core if set
        if let Some(reserved_cpu) = reserved_tracing_cpu {
            if first_ht.id() == reserved_cpu as i32 {
                continue;
            }
        }
        // Only use the first hyperthread of each physical core
        victim_cpus.push(first_ht.clone());
    }

    eprintln!("=== IRQ Migration Test ===");
    eprintln!("Physical cores: {}", cores.len());
    eprintln!("Control CPU: {} (no IRQ load)", control_cpu_id);
    if let Some(reserved_cpu) = reserved_tracing_cpu {
        eprintln!("Reserved tracing CPU: {} (no IRQ load)", reserved_cpu);
    }
    eprintln!(
        "Victim CPUs: {} (one per physical core, excluding control{})",
        victim_cpus.len(),
        if reserved_tracing_cpu.is_some() { " and tracing" } else { "" }
    );
    eprintln!("Initial worker CPU: {} (will be unpinned)", initial_victim_cpu_id);
    eprintln!(
        "IRQ load: {} Hz x {} timers = {} Hz per victim CPU",
        DEFAULT_IRQ_HZ,
        NUM_TIMERS,
        DEFAULT_IRQ_HZ * NUM_TIMERS as u64
    );

    log_scheduler_info();

    // Create shared memory
    let allocator = BumpAllocator::new("irq_migration_test", 4 * 1024 * 1024)?;
    let start_signal = SharedBox::new(allocator.clone(), AtomicU32::new(0))?;

    // Worker output counters
    let scheduled_ns = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;
    let bogo_ops = SharedBox::new(allocator.clone(), AtomicU64::new(0))?;
    let final_cpu = SharedBox::new(allocator.clone(), AtomicI32::new(-1))?;

    // Capture interrupt baseline
    eprintln!("\nCapturing interrupt baseline...");
    let interrupts_before = InterruptSnapshot::capture()?;

    // Launch timer IRQ disruption on ALL victim CPUs
    eprintln!("\nLaunching timer IRQ disruption on {} victim CPUs...", victim_cpus.len());
    let mut irq_handles: Vec<TimerIrqHandle> = Vec::new();

    for victim_cpu in &victim_cpus {
        let handle = launch_timer_irq_disruption(
            allocator.clone(),
            victim_cpu,
            start_signal.clone(),
            DEFAULT_IRQ_HZ,
        )?;
        irq_handles.push(handle);
    }

    let hog_duration = get_test_duration();
    eprintln!("\nLaunching worker on CPU {} (will unpin after start)...", initial_victim_cpu_id);

    // Launch worker pinned to initial victim CPU
    let worker_start = start_signal.clone();
    let worker_scheduled = scheduled_ns.clone();
    let worker_bogo = bogo_ops.clone();
    let worker_final_cpu = final_cpu.clone();
    let initial_mask = CPUMask::new(&initial_victim_cpu);

    let mut worker = Child::run(
        move || {
            // Start pinned to initial victim
            initial_mask.run(|| {
                // Wait for start signal
                while worker_start.load(Ordering::Acquire) == 0 {
                    std::hint::spin_loop();
                }

                // Unpin ourselves - allow migration to any CPU
                unsafe {
                    let mut mask: libc::cpu_set_t = std::mem::zeroed();
                    // Set all CPUs in the mask
                    for i in 0..libc::CPU_SETSIZE as usize {
                        libc::CPU_SET(i, &mut mask);
                    }
                    let ret = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mask);
                    if ret != 0 {
                        eprintln!("Warning: sched_setaffinity failed");
                    }
                }

                eprintln!("Worker unpinned, starting workload...");

                // Run the spinner workload (no bogo ops tracking needed, just scheduled time)
                spinner_utilization::cpu_hog_workload(
                    "migration_worker",
                    hog_duration,
                    worker_start.clone(),
                    worker_scheduled,
                    Some(worker_bogo),
                );

                // Record which CPU we ended up on
                let cpu = unsafe { libc::sched_getcpu() };
                worker_final_cpu.store(cpu, Ordering::Release);
            })?;
            Ok(())
        },
        None,
    )?;

    // Give everything a moment to initialize
    std::thread::sleep(Duration::from_millis(100));

    // Signal start
    eprintln!("Signaling START (worker will unpin and run for {:?})", hog_duration);
    start_signal.store(1, Ordering::Release);

    // Wait for worker to complete
    if let Some(result) = worker.wait(true, false) {
        result.context("Worker failed")?;
    }
    eprintln!("Worker completed");

    // Stop all IRQ disruption
    eprintln!("Stopping IRQ disruption on {} CPUs...", irq_handles.len());
    let mut total_timer_wakeups = 0u64;
    for handle in irq_handles {
        total_timer_wakeups += handle.stop()?;
    }

    // Capture interrupt counts after
    let interrupts_after = InterruptSnapshot::capture()?;
    let interrupt_delta = interrupts_after.delta(&interrupts_before);

    // Results
    let worker_final_cpu_id = final_cpu.load(Ordering::Acquire);
    let worker_scheduled_ns = scheduled_ns.load(Ordering::Acquire);
    let worker_bogo_ops = bogo_ops.load(Ordering::Acquire);

    eprintln!("\n=== Results ===");
    eprintln!("Total timer wakeups across all victim CPUs: {}", total_timer_wakeups);
    eprintln!("Worker scheduled time: {:.3} seconds", worker_scheduled_ns as f64 / 1e9);
    eprintln!("Worker bogo ops: {}", worker_bogo_ops);
    eprintln!();
    eprintln!("Initial CPU: {}", initial_victim_cpu_id);
    eprintln!("Final CPU:   {}", worker_final_cpu_id);

    let migrated = worker_final_cpu_id != initial_victim_cpu_id;
    let migrated_to_control = worker_final_cpu_id == control_cpu_id;

    if migrated_to_control {
        eprintln!("✓ SUCCESS: Worker migrated from victim CPU {} to control CPU {}",
                  initial_victim_cpu_id, control_cpu_id);
    } else if migrated {
        eprintln!("⚠ PARTIAL: Worker migrated from CPU {} to CPU {} (not the control CPU {})",
                  initial_victim_cpu_id, worker_final_cpu_id, control_cpu_id);
    } else {
        eprintln!("✗ FAILED: Worker stayed on victim CPU {} (expected migration to control CPU {})",
                  initial_victim_cpu_id, control_cpu_id);
    }

    // Print interrupt summary for key CPUs
    eprintln!("\n=== Interrupt Counts (Delta) ===");
    eprintln!(
        "{:>6} {:>15} {:>15}  {}",
        "CPU", "Total", "Reschedule", "Role"
    );

    // Show initial victim, final CPU, and control
    let interesting_cpus = vec![
        (initial_victim_cpu_id as usize, "INITIAL"),
        (worker_final_cpu_id as usize, "FINAL"),
        (control_cpu_id as usize, "CONTROL"),
    ];

    for (cpu, label) in &interesting_cpus {
        let total = *interrupt_delta.total.get(cpu).unwrap_or(&0);
        let reschedule = *interrupt_delta.reschedule.get(cpu).unwrap_or(&0);
        eprintln!("{:>6} {:>15} {:>15}  {}", cpu, total, reschedule, label);
    }

    // Summary stats
    let control_interrupts = *interrupt_delta.total.get(&(control_cpu_id as usize)).unwrap_or(&0);
    let victim_avg_interrupts: i64 = if !victim_cpus.is_empty() {
        let sum: i64 = victim_cpus
            .iter()
            .map(|cpu| *interrupt_delta.total.get(&(cpu.id() as usize)).unwrap_or(&0))
            .sum();
        sum / victim_cpus.len() as i64
    } else {
        0
    };

    eprintln!("\nInterrupt Summary:");
    eprintln!("  Control CPU {} interrupts: {}", control_cpu_id, control_interrupts);
    eprintln!("  Victim CPUs avg interrupts: {}", victim_avg_interrupts);

    if victim_avg_interrupts > control_interrupts * 10 {
        eprintln!("  ✓ Victim CPUs have ~{}x more interrupts than control",
                  victim_avg_interrupts / control_interrupts.max(1));
    }

    Ok(())
}

test!("irq_migration", irq_migration_test);
