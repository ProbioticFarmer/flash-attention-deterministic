"""
Paste this into a Colab cell to test memory scaling.
Run after the library is already installed from the overnight build.
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
    return 3 * batch * seqlen * heads * headdim * 2 / (1024**2)

# Test configurations - carefully sized to avoid OOM on A100 80GB
configs = [
    # Original (33MB input)
    ("Baseline", (4, 2048, 32, 64)),

    # ~10x input size (330MB)
    ("10x-Batch", (16, 4096, 32, 64)),

    # ~100x input size (3.3GB) - May need to reduce if OOM
    ("100x-Large", (32, 8192, 32, 64)),

    # Alternative large configs
    ("Medium-1", (8, 8192, 32, 64)),     # ~400MB input
    ("Medium-2", (16, 8192, 32, 64)),    # ~800MB input
]

print("="*80)
print("MEMORY SCALING TEST - A100")
print("="*80)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
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
        print(f"  Standard:      {mem_std:8.1f} MB")

        mem_det = measure_memory(config, deterministic=True)
        print(f"  Deterministic: {mem_det:8.1f} MB")

        overhead_mb = mem_det - mem_std
        overhead_pct = ((mem_det / mem_std) - 1) * 100

        print(f"  Overhead:      {overhead_mb:8.1f} MB ({overhead_pct:+.1f}%)")
        print(f"  OH/Input:      {overhead_mb/input_size:.2f}x of input size")

        results.append({
            'name': name,
            'config': config,
            'input_mb': input_size,
            'std_mb': mem_std,
            'det_mb': mem_det,
            'overhead_mb': overhead_mb,
            'overhead_pct': overhead_pct,
        })

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  ❌ OOM - Skipping larger configs")
            break
        else:
            print(f"  ❌ ERROR: {e}")

# Summary Table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print()
print(f"{'Config':<12} {'Input MB':<10} {'Std MB':<10} {'Det MB':<10} {'OH MB':<10} {'OH %':<8} {'OH/In':<8}")
print("-" * 80)

for r in results:
    print(f"{r['name']:<12} "
          f"{r['input_mb']:>8.1f}  "
          f"{r['std_mb']:>8.1f}  "
          f"{r['det_mb']:>8.1f}  "
          f"{r['overhead_mb']:>8.1f}  "
          f"{r['overhead_pct']:>6.1f}%  "
          f"{r['overhead_mb']/r['input_mb']:>5.2f}x")

# Analysis
if len(results) > 1:
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    baseline = results[0]
    print(f"\nBaseline overhead: {baseline['overhead_mb']:.1f}MB ({baseline['overhead_pct']:.1f}%)")

    print("\nAs workload increases:")
    for r in results[1:]:
        ratio = r['input_mb'] / baseline['input_mb']
        print(f"\n  {r['name']} ({ratio:.1f}x input size):")
        print(f"    Overhead: {r['overhead_mb']:.1f}MB ({r['overhead_pct']:.1f}%)")
        print(f"    Change: {r['overhead_pct'] - baseline['overhead_pct']:+.1f} percentage points")

        # Check if overhead % is decreasing
        if r['overhead_pct'] < baseline['overhead_pct']:
            print(f"    ✅ Overhead % DECREASED - kernel overhead more significant at scale")
        else:
            print(f"    ⚠️  Overhead % same or increased")

print()
