---
title: Implement CPU sequestering for benchmark_mode (isolate background processes to CPU0)
status: open
priority: 2
issue_type: task
created_at: 2025-11-11T18:47:58.746901052+00:00
updated_at: 2025-11-11T18:47:58.746901052+00:00
---

# Description

## Overview
Implement step (2) of benchmark_mode.py: sequester ALL other running processes on CPU0 so CPU1..N can be used for pure benchmarking.

## Context
This is part of the benchmark_mode.py script that prepares the system for reliable microbenchmarking. Step (1) (CPU frequency locking) has been implemented. This issue tracks step (2).

## Implementation Plan

### Goals
- Move all background/system processes to CPU0
- Leave CPU1..N completely free for benchmark workloads
- Ensure processes stay pinned to CPU0 throughout benchmarking
- Properly restore process affinity on cleanup

### Approach
1. **Enumerate all running processes**
   - Use /proc to iterate over all PIDs
   - Identify system processes, kernel threads, and user processes
   - Skip processes that cannot be moved (e.g., per-CPU kernel threads)

2. **Set CPU affinity**
   - Use sched_setaffinity() (via Python's os.sched_setaffinity()) to pin processes to CPU0
   - Handle processes that may have restricted affinity
   - Consider using cgroups as an alternative/additional mechanism (cpusets)

3. **Track original affinities**
   - Store original CPU affinity masks for each process before modification
   - Ensure cleanup can restore original state
   - Handle new processes that spawn during benchmarking

4. **Handle edge cases**
   - Some kernel threads cannot be moved (will fail with permission errors)
   - IRQ handling and softirqs may need separate treatment (look into /proc/irq/*/smp_affinity)
   - Consider impact on systemd and other critical system services

5. **Verification**
   - After setup, verify that no processes have affinity including CPU1..N
   - Provide diagnostic output showing what was moved
   - Consider ongoing monitoring/enforcement during benchmarking

### Technical Considerations
- Need root/sudo privileges for sched_setaffinity on other users' processes
- Some processes may immediately reset their affinity
- May need to handle container/namespace isolation
- Consider using cgroups cpuset controller for more robust isolation

### Testing
- Verify on idle system
- Verify with various background loads
- Ensure cleanup restores system to normal state
- Test behavior with processes spawning during benchmarking

### Integration
- Add to benchmark_mode.py after CPU frequency locking
- Ensure it's included in cleanup handler
- May want a flag to enable/disable this feature independently
