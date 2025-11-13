//! Tests for heavy IRQ workload scenarios.

use std::time::Duration;

use anyhow::Result;

use crate::test;
use crate::util::system::{CPUMask, CPUSet, System};
use crate::workloads::context::Context;
use crate::workloads::spinner::Spinner;
use crate::workloads::spinner_utilization;

use crate::process;

/// Test that loads certain threads with heavy IRQ work.
///
/// This test creates processes that simulate heavy interrupt handling work
/// to validate scheduler behavior under IRQ load.
fn heavy_irq_load() -> Result<()> {
    let mut ctx = Context::create()?;
    let system = System::load()?;
    let mask = CPUMask::new(
        system
            .cores()
            .first()
            .unwrap()
            .hyperthreads()
            .first()
            .unwrap(),
    );

    // TODO: Implement IRQ simulation workload
    // - Create threads that simulate heavy IRQ handling
    // - Measure impact on scheduler fairness/latency

    process!(&mut ctx, None, (mask), move |mut get_iters| {
        mask.run(|| {
            let spinner = Spinner::default();
            loop {
                let iters = get_iters();
                // Placeholder: spin for some time to simulate IRQ work
                spinner.spin(Duration::from_millis(iters as u64));
            }
        })
    });

    ctx.start(1);
    ctx.wait()?;
    Ok(())
}

test!("heavy_irq_load", heavy_irq_load);
