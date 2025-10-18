"""
Debug the CUDA error at long sequences (L=8192).
Run this in Colab to get detailed error info.
"""

import torch
from flash_attn import flash_attn_func, set_deterministic_mode
import os

# Enable CUDA error checking
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("Testing long sequence with CUDA_LAUNCH_BLOCKING=1")
print("This will show exactly which operation causes the error")
print()

# Test progressively longer sequences
configs = [
    ("Safe", (8, 4096, 32, 64)),   # Known to work
    ("Edge", (8, 6144, 32, 64)),   # Between working and broken
    ("Break", (8, 8192, 32, 64)),  # Known to fail
]

for name, config in configs:
    batch, seqlen, heads, headdim = config

    print(f"\n{'='*60}")
    print(f"Testing {name}: B={batch}, L={seqlen}, H={heads}, D={headdim}")
    print(f"{'='*60}")

    try:
        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Enable deterministic mode
        set_deterministic_mode(enabled=True, split_size=512)

        # Create tensors
        print(f"Creating tensors...")
        q = torch.randn(batch, seqlen, heads, headdim, dtype=torch.float16, device='cuda')
        k = torch.randn(batch, seqlen, heads, headdim, dtype=torch.float16, device='cuda')
        v = torch.randn(batch, seqlen, heads, headdim, dtype=torch.float16, device='cuda')
        print(f"✓ Tensors created")

        # Run standard mode first
        print(f"Testing standard mode...")
        set_deterministic_mode(enabled=False)
        out_std = flash_attn_func(q, k, v, causal=False)
        print(f"✓ Standard mode works: output shape {out_std.shape}")
        del out_std

        # Now try deterministic mode
        print(f"Testing deterministic mode...")
        set_deterministic_mode(enabled=True, split_size=512)
        out_det = flash_attn_func(q, k, v, causal=False)
        print(f"✓ Deterministic mode works: output shape {out_det.shape}")

        # Success!
        print(f"\n✅ {name} PASSED")

        del q, k, v, out_det
        torch.cuda.empty_cache()

    except RuntimeError as e:
        print(f"\n❌ {name} FAILED")
        print(f"Error: {e}")
        print()

        # Try to get more info
        if 'illegal memory access' in str(e):
            print("ILLEGAL MEMORY ACCESS - likely indexing bug in kernel")
            print()
            print("Possible causes:")
            print("1. num_splits calculation produces invalid split indices")
            print("2. Accumulator buffer size mismatch")
            print("3. Kernel assumes max sequence length")

        break  # Stop on first failure

print(f"\n{'='*60}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*60}")
