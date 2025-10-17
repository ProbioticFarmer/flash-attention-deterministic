# Deterministic Flash Attention

A modified Flash Attention implementation providing deterministic forward pass with perfect batch invariance.

## What This Solves

Standard Flash Attention is non-deterministic - running the same input with different batch sizes produces different results due to floating-point accumulation order. This breaks reproducibility for:
- Post-training (RLHF, alignment, fine-tuning)
- A/B testing of training configurations
- Debugging training instabilities
- Compliance requirements for model auditing

**This implementation guarantees bit-exact results regardless of batch size.**

## Features

✅ **Perfect Batch Invariance** - `batch_size=8` gives identical results to `batch_size=4+4`
✅ **Bit-Exact Reproducibility** - Same inputs always produce identical outputs
✅ **Simple API** - Single `deterministic=True` parameter
⚠️ **Performance** - Currently under verification (see [Status](#status))

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ProbioticFarmer/flash-attention-deterministic.git
cd flash-attention-deterministic

# Build from source
pip install -e . --no-build-isolation
```

See [INSTALLATION.md](INSTALLATION.md) for detailed build instructions and troubleshooting.

### Usage

```python
from flash_attn import flash_attn_func

# Standard (non-deterministic)
output = flash_attn_func(q, k, v, causal=False, deterministic=False)

# Deterministic (batch-invariant)
output = flash_attn_func(q, k, v, causal=False, deterministic=True)
```

### Verification

```python
# Verify installation
python validate_determinism.py
```

Expected output:
```
✓ Batch Invariance Test: PASSED
✓ Determinism Test: PASSED
✓ Performance Test: <results>
```

## How It Works

The implementation modifies the Flash Attention forward pass to:
1. Process attention in deterministic splits
2. Accumulate results in float32 accumulators
3. Combine splits using parallel PyTorch operations (logsumexp, exp, sum)

Key changes:
- `setup.py`: Added `-DFLASH_ATTENTION_DETERMINISTIC` compile flag
- `csrc/flash_attn/flash_api.cpp`: Deterministic mode handling
- `csrc/flash_attn/src/flash.h`: Split accumulation logic

## Status

**Current Status:** ✅ Implementation complete, verification in progress

The deterministic code path compiles and executes correctly. We are currently verifying:
- Memory overhead characteristics
- Performance impact across different workloads
- Optimal split size configuration

Performance results will be updated once Colab benchmarking completes.

## Technical Details

**Deterministic guarantees:**
- Batch invariance: O(B=8) ≡ O(B=4) ⊕ O(B=4) (bit-exact)
- Reproducibility: Multiple runs produce identical outputs
- Accumulation: Float32 for numerical stability

**Implementation approach:**
- Uses existing Flash Attention split kernel infrastructure
- Parallel reduction (not single-threaded)
- Configurable via runtime parameter

## Requirements

- CUDA 11.7+
- PyTorch 2.0+
- Flash Attention compatible GPU (sm_80+, sm_90)

## Contributing

Issues and PRs welcome! This implementation went through several debugging iterations - contributions to improve performance or documentation are appreciated.

## License

BSD License (same as original Flash Attention)

## Acknowledgments

Based on [Flash Attention](https://github.com/Dao-AILab/flash-attention) by Tri Dao.

---

**Development Note:** This implementation required persistent debugging through several wrong turns. Thanks to Claude for the troubleshooting sessions!
