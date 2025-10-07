#!/usr/bin/env python3
"""
Additional analysis of the CPU scheduling experiment results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_results():
    """Provide detailed analysis of the experiment results."""
    
    # Load results
    df = pd.read_csv('results/experiment_results.csv')
    
    print("=" * 80)
    print("DETAILED ANALYSIS OF CPU SCHEDULING EXPERIMENT RESULTS")
    print("=" * 80)
    
    # Basic stats
    print(f"\nExperiment Overview:")
    print(f"- System: {df.iloc[0]['num_cores']} physical cores")
    print(f"- Duration: 10 seconds per experiment")
    print(f"- Configurations tested: {len(df)}")
    
    # Normalization factors
    max_cpu = df['bogo_cpu_persec'].max()
    max_mem = df['bogo_mem_persec'].max()
    
    print(f"\nPerformance Baselines (100%):")
    print(f"- CPU: {max_cpu:,.0f} bogo ops/sec (achieved with: {df.loc[df['bogo_cpu_persec'].idxmax(), 'workload']}/{df.loc[df['bogo_cpu_persec'].idxmax(), 'pinning']})")
    print(f"- MEM: {max_mem:,.0f} bogo ops/sec (achieved with: {df.loc[df['bogo_mem_persec'].idxmax(), 'workload']}/{df.loc[df['bogo_mem_persec'].idxmax(), 'pinning']})")
    
    # Calculate normalized metrics
    df['cpu_norm'] = df['bogo_cpu_persec'] / max_cpu * 100
    df['mem_norm'] = df['bogo_mem_persec'] / max_mem * 100
    df['combined'] = df['cpu_norm'] + df['mem_norm']
    
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