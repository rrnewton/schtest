#!/usr/bin/env python3
"""
Additional analysis of the CPU scheduling experiment results
"""

import pandas as pd

def analyze_results() -> None:
    """Provide detailed analysis of the experiment results."""

    # Load results from latest symlink
    df = pd.read_csv('results/latest/experiment_results.csv')
    
    print("# CPU Scheduling Experiment Analysis\n")

    # Basic stats
    print("## Experiment Overview\n")
    print(f"- **System**: {df.iloc[0]['num_cores']} physical cores")
    print(f"- **Duration**: 10 seconds per experiment")
    print(f"- **Configurations tested**: {len(df)}\n")

    # Check if we have sufficient data for analysis
    if len(df) < 2:
        print("⚠️  **Limited Dataset**: Only one configuration tested. Run a full experiment for comprehensive analysis.\n")
        print("## Single Configuration Results\n")
        row = df.iloc[0]
        print(f"- **Workload**: {row['workload']}")
        print(f"- **Pinning**: {row['pinning']}")
        print(f"- **Scheduler**: {row['scheduler']}")
        if row['bogo_cpu'] > 0:
            print(f"- **CPU Performance**: {row['bogo_cpu']:,} bogo ops ({row['bogo_cpu_persec']:,.1f} ops/sec)")
        if row['bogo_mem'] > 0:
            print(f"- **Memory Performance**: {row['bogo_mem']:,} bogo ops ({row['bogo_mem_persec']:,.1f} ops/sec)")
        print(f"- **Instructions**: {row['instructions']:,.0f}")
        print(f"- **Cycles**: {row['cycles']:,.0f}")
        if row['cycles'] > 0:
            print(f"- **IPC**: {row['instructions']/row['cycles']:.2f}")
        
        return

    # Normalization factors (only if we have multiple configs)
    max_cpu = df['bogo_cpu_persec'].max()
    max_mem = df['bogo_mem_persec'].max()

    print("## Performance Baselines (100%)\n")
    if max_cpu > 0:
        best_cpu_row = df.loc[df['bogo_cpu_persec'].idxmax()]
        print(f"- **CPU**: {max_cpu:,.0f} bogo ops/sec (achieved with: {best_cpu_row['workload']}/{best_cpu_row['pinning']}/{best_cpu_row['scheduler']})")
    if max_mem > 0:
        best_mem_row = df.loc[df['bogo_mem_persec'].idxmax()]
        print(f"- **MEM**: {max_mem:,.0f} bogo ops/sec (achieved with: {best_mem_row['workload']}/{best_mem_row['pinning']}/{best_mem_row['scheduler']})")
    print()

    # Calculate normalized metrics
    df['cpu_norm'] = df['bogo_cpu_persec'] / max_cpu * 100 if max_cpu > 0 else 0
    df['mem_norm'] = df['bogo_mem_persec'] / max_mem * 100 if max_mem > 0 else 0
    df['combined'] = df['cpu_norm'] + df['mem_norm']

    print("## Performance Analysis by Workload Type\n")

    # Analyze each workload type
    for workload in ['both', 'cpu', 'mem']:
        subset = df[df['workload'] == workload]
        if len(subset) == 0:
            continue
            
        print(f"### {workload.upper()} Workload Results\n")
        
        if len(subset) > 1:
            best_idx = subset['combined'].idxmax()
            best = subset.loc[best_idx]
            print(f"**Best Configuration**: {best['pinning']}/{best['scheduler']}")
            print(f"- CPU: {best['cpu_norm']:.1f}% ({best['bogo_cpu_persec']:,.0f} ops/sec)")
            print(f"- MEM: {best['mem_norm']:.1f}% ({best['bogo_mem_persec']:,.0f} ops/sec)")
            print(f"- Combined: {best['combined']:.1f}%\n")

            print("**All Configurations**:\n")
            print("| Pinning | Scheduler | CPU % | MEM % | Combined % |")
            print("|---------|-----------|-------|-------|------------|")
            for _, row in subset.iterrows():
                print(f"| {row['pinning']} | {row['scheduler']} | {row['cpu_norm']:.1f} | {row['mem_norm']:.1f} | {row['combined']:.1f} |")
            print()
        else:
            row = subset.iloc[0]
            print(f"**Configuration**: {row['pinning']}/{row['scheduler']}")
            print(f"- CPU: {row['cpu_norm']:.1f}% ({row['bogo_cpu_persec']:,.0f} ops/sec)")
            print(f"- MEM: {row['mem_norm']:.1f}% ({row['bogo_mem_persec']:,.0f} ops/sec)")
            print(f"- Combined: {row['combined']:.1f}%\n")

    # Scheduler comparison if multiple schedulers tested
    schedulers = df['scheduler'].unique()
    if len(schedulers) > 1:
        print("## Scheduler Comparison\n")
        print("| Scheduler | Avg CPU % | Avg MEM % | Avg Combined % |")
        print("|-----------|-----------|-----------|----------------|")
        for sched in schedulers:
            sched_data = df[df['scheduler'] == sched]
            avg_cpu = sched_data['cpu_norm'].mean()
            avg_mem = sched_data['mem_norm'].mean()
            avg_combined = sched_data['combined'].mean()
            print(f"| {sched} | {avg_cpu:.1f} | {avg_mem:.1f} | {avg_combined:.1f} |")
        print()

    # Summary insights
    print("## Key Insights\n")
    
    if len(df) > 1:
        best_overall = df.loc[df['combined'].idxmax()]
        print(f"1. **Best Overall Performance**: {best_overall['workload']}/{best_overall['pinning']}/{best_overall['scheduler']} ({best_overall['combined']:.1f}%)")
        
        if 'both' in df['workload'].values:
            both_best = df[df['workload'] == 'both']['combined'].max()
            print(f"2. **Best Mixed Workload**: {both_best:.1f}% combined throughput")
        
        # Pinning strategy analysis
        pinning_perf = df.groupby('pinning')['combined'].mean().sort_values(ascending=False)
        best_pinning = pinning_perf.index[0]
        print(f"3. **Best Pinning Strategy**: {best_pinning} (avg {pinning_perf.iloc[0]:.1f}%)")

    print(f"\n---")
    print(f"*Analysis generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    print(f"\n" + "="*60)
    print("PERFORMANCE ANALYSIS BY WORKLOAD TYPE")
    print("="*60)

    # Analysis by workload
    for workload in ['both', 'cpu', 'mem']:
        subset = df[df['workload'] == workload].copy()
        print(f"\n{workload.upper()} Workload Results:")
        print("-" * 30)

        for _, row in subset.iterrows():
            pinning = row['pinning']
            cpu_pct = row['cpu_norm']
            mem_pct = row['mem_norm']
            combined = row['combined']

            print(f"  {pinning:>8}: CPU={cpu_pct:5.1f}%, MEM={mem_pct:5.1f}%, Combined={combined:5.1f}%")

        # Find best and worst for this workload
        best_idx = subset['combined'].idxmax()
        worst_idx = subset['combined'].idxmin()
        best = subset.loc[best_idx]
        worst = subset.loc[worst_idx]

        improvement = (best['combined'] - worst['combined']) / worst['combined'] * 100
        print(f"    → Best: {best['pinning']} ({best['combined']:.1f}%)")
        print(f"    → Worst: {worst['pinning']} ({worst['combined']:.1f}%)")
        print(f"    → Improvement: {improvement:.1f}% better")

    print(f"\n" + "="*60)
    print("ANALYSIS BY PINNING STRATEGY")
    print("="*60)

    for pinning in ['none', 'spread', 'half']:
        subset = df[df['pinning'] == pinning].copy()
        print(f"\n{pinning.upper()} Pinning Strategy:")
        print("-" * 30)

        avg_combined = subset['combined'].mean()
        print(f"  Average combined performance: {avg_combined:.1f}%")

        for _, row in subset.iterrows():
            workload = row['workload']
            combined = row['combined']
            print(f"    {workload:>5}: {combined:5.1f}%")

    print(f"\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    # Key insights
    print("\n1. SINGLE WORKLOAD PERFORMANCE:")
    cpu_only = df[df['workload'] == 'cpu']
    mem_only = df[df['workload'] == 'mem']

    print(f"   CPU-only workload:")
    for _, row in cpu_only.iterrows():
        print(f"     {row['pinning']:>8}: {row['cpu_norm']:5.1f}% ({row['bogo_cpu_persec']:6.0f} ops/sec)")

    print(f"   MEM-only workload:")
    for _, row in mem_only.iterrows():
        print(f"     {row['pinning']:>8}: {row['mem_norm']:5.1f}% ({row['bogo_mem_persec']:6.0f} ops/sec)")

    print("\n2. COMBINED WORKLOAD PERFORMANCE:")
    both_workload = df[df['workload'] == 'both']
    for _, row in both_workload.iterrows():
        cpu_eff = row['cpu_norm'] / 100  # Efficiency vs single workload
        mem_eff = row['mem_norm'] / 100
        print(f"     {row['pinning']:>8}: {row['combined']:5.1f}% total (CPU:{cpu_eff:.1%}, MEM:{mem_eff:.1%} of peak)")

    print("\n3. PINNING STRATEGY EFFECTIVENESS:")

    # Compare pinning strategies
    pinning_avg = df.groupby('pinning')['combined'].mean()
    best_pinning = pinning_avg.idxmax()
    worst_pinning = pinning_avg.idxmin()

    print(f"   Best overall: {best_pinning} ({pinning_avg[best_pinning]:.1f}% avg)")
    print(f"   Worst overall: {worst_pinning} ({pinning_avg[worst_pinning]:.1f}% avg)")

    # Cache analysis
    print(f"\n4. CACHE BEHAVIOR:")
    df['cache_miss_rate'] = df['cache_misses'] / df['cache_refs'] * 100
    df['ipc'] = df['instructions'] / df['cycles']  # Instructions per cycle

    print("   Cache miss rates by configuration:")
    for _, row in df.iterrows():
        print(f"     {row['workload']:>4}/{row['pinning']:>6}: {row['cache_miss_rate']:5.2f}% miss rate, IPC={row['ipc']:.2f}")

    print(f"\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    print("\n1. For MIXED CPU+MEMORY workloads:")
    both_best = df[df['workload'] == 'both']['combined'].idxmax()
    both_best_config = df.loc[both_best]
    print(f"   → Use '{both_best_config['pinning']}' pinning ({both_best_config['combined']:.1f}% performance)")

    print("\n2. For CPU-INTENSIVE workloads:")
    cpu_best = df[df['workload'] == 'cpu']['cpu_norm'].idxmax()
    cpu_best_config = df.loc[cpu_best]
    print(f"   → Use '{cpu_best_config['pinning']}' pinning ({cpu_best_config['cpu_norm']:.1f}% performance)")

    print("\n3. For MEMORY-INTENSIVE workloads:")
    mem_best = df[df['workload'] == 'mem']['mem_norm'].idxmax()
    mem_best_config = df.loc[mem_best]
    print(f"   → Use '{mem_best_config['pinning']}' pinning ({mem_best_config['mem_norm']:.1f}% performance)")

    print(f"\n4. Performance Impact of Thread Pinning:")
    none_avg = df[df['pinning'] == 'none']['combined'].mean()
    spread_avg = df[df['pinning'] == 'spread']['combined'].mean()
    half_avg = df[df['pinning'] == 'half']['combined'].mean()

    print(f"   → 'none' (no pinning): {none_avg:.1f}% average performance")
    print(f"   → 'spread' (optimal): {spread_avg:.1f}% average performance")
    print(f"   → 'half' (suboptimal): {half_avg:.1f}% average performance")

    spread_vs_none = (spread_avg - none_avg) / none_avg * 100
    spread_vs_half = (spread_avg - half_avg) / half_avg * 100

    print(f"   → Spread pinning is {spread_vs_none:+.1f}% vs no pinning")
    print(f"   → Spread pinning is {spread_vs_half:+.1f}% vs half pinning")

if __name__ == "__main__":
    analyze_results()