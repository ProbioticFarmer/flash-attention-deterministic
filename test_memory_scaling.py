#!/usr/bin/env python3
"""
Test memory scaling at different workload sizes to understand realistic overhead.

Target baselines:
- 330MB (10x original test)
- 3.3GB (100x original test)
"""

import torch
from flash_attn import flash_attn_func, set_deterministic_mode
import gc

def measure_memory(config, deterministic=False):
    """Measure peak memory for a given configuration."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    set_deterministic_mode(enabled=deterministic, split_size=512 if deterministic else None)

    batch, seqlen, heads, headdim = config

    # Correct layout: (batch, seqlen, heads, headdim)
    q = torch.randn(batch, seqlen, heads, headdim, dtype=torch.float16, device='cuda')
    k = torch.randn(batch, seqlen, heads, headdim, dtype=torch.float16, device='cuda')
    v = torch.randn(batch, seqlen, heads, headdim, dtype=torch.float16, device='cuda')

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated() / (1024**2)

    out = flash_attn_func(q, k, v, causal=False)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024**2)

    del q, k, v, out
    torch.cuda.empty_cache()

    return peak - before

def calculate_input_size(batch, seqlen, heads, headdim):
    """Calculate input tensor size in MB."""
    # 3 tensors (q, k, v) × batch × seqlen × heads × headdim × 2 bytes (fp16)
    return 3 * batch * seqlen * heads * headdim * 2 / (1024**2)

# Test configurations
configs = [
    # Original test (B=4, L=2048, H=32, D=64)
    ("Baseline", (4, 2048, 32, 64)),

    # 10x memory: increase batch and seqlen
    ("10x - Batch", (16, 4096, 32, 64)),      # ~330MB inputs

    # 100x memory: large batch + long sequence
    ("100x - Large", (32, 8192, 32, 64)),     # ~3.2GB inputs

    # Alternative configs at similar memory levels
    ("10x - Heads", (8, 4096, 64, 64)),       # More heads
    ("100x - Heads", (16, 8192, 64, 64)),     # Even more heads
]

print("="*80)
print("MEMORY SCALING TEST")
print("="*80)
print()

results = []

for name, config in configs:
    batch, seqlen, heads, headdim = config
    input_size = calculate_input_size(*config)

    print(f"\n{name}: B={batch}, L={seqlen}, H={heads}, D={headdim}")
    print(f"Input size: {input_size:.1f} MB")
    print("-" * 60)

    try:
        mem_std = measure_memory(config, deterministic=False)
        mem_det = measure_memory(config, deterministic=True)

        overhead_mb = mem_det - mem_std
        overhead_pct = ((mem_det / mem_std) - 1) * 100

        print(f"  Standard:      {mem_std:8.1f} MB")
        print(f"  Deterministic: {mem_det:8.1f} MB")
        print(f"  Overhead:      {overhead_mb:8.1f} MB ({overhead_pct:+.1f}%)")

        # Calculate overhead relative to input size
        overhead_vs_input = overhead_mb / input_size
        print(f"  Overhead/Input: {overhead_vs_input:.2f}x")

        results.append({
            'name': name,
            'config': config,
            'input_mb': input_size,
            'std_mb': mem_std,
            'det_mb': mem_det,
            'overhead_mb': overhead_mb,
            'overhead_pct': overhead_pct,
            'overhead_vs_input': overhead_vs_input
        })

    except RuntimeError as e:
        print(f"  ❌ FAILED: {e}")
        results.append({
            'name': name,
            'error': str(e)
        })

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print()
print(f"{'Config':<15} {'Input':<12} {'Std':<12} {'Det':<12} {'Overhead':<15} {'OH/Input':<10}")
print("-" * 80)

for r in results:
    if 'error' in r:
        print(f"{r['name']:<15} {'FAILED':<12}")
    else:
        print(f"{r['name']:<15} "
              f"{r['input_mb']:>8.1f} MB  "
              f"{r['std_mb']:>8.1f} MB  "
              f"{r['det_mb']:>8.1f} MB  "
              f"{r['overhead_mb']:>8.1f} MB ({r['overhead_pct']:+5.1f}%)  "
              f"{r['overhead_vs_input']:>4.2f}x")

print()
print("="*80)
print("KEY INSIGHTS")
print("="*80)

# Find successful results
successful = [r for r in results if 'error' not in r]

if len(successful) > 1:
    baseline = successful[0]

    print(f"\nBaseline (33MB input):")
    print(f"  Overhead: {baseline['overhead_mb']:.1f} MB ({baseline['overhead_pct']:.1f}%)")
    print(f"  OH/Input: {baseline['overhead_vs_input']:.2f}x")

    for r in successful[1:]:
        print(f"\n{r['name']} ({r['input_mb']:.0f}MB input):")
        print(f"  Overhead: {r['overhead_mb']:.1f} MB ({r['overhead_pct']:.1f}%)")
        print(f"  OH/Input: {r['overhead_vs_input']:.2f}x")

        # Compare percentage overhead change
        pct_change = r['overhead_pct'] - baseline['overhead_pct']
        print(f"  vs Baseline: {pct_change:+.1f} percentage points")

print()
