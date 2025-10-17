# Installation Guide

## Prerequisites

- Python 3.9+
- CUDA 11.7+ (12.x recommended)
- PyTorch 2.0+
- NVIDIA GPU with compute capability sm_80 or sm_90 (A100, H100, etc.)
- 16+ CPU cores recommended for faster compilation

## Local Installation

### 1. Clone Repository

```bash
git clone https://github.com/ProbioticFarmer/flash-attention-deterministic.git
cd flash-attention-deterministic
```

### 2. Install Dependencies

```bash
pip install torch ninja packaging wheel einops
```

### 3. Build from Source

```bash
# Install with parallel compilation (much faster)
MAX_JOBS=16 pip install -e . --no-build-isolation
```

**Build time:** 5-10 minutes with 16 parallel jobs

### 4. Verify Installation

```bash
python validate_determinism.py
```

Expected output:
```
✓ Batch Invariance Test: PASSED
✓ Determinism Test: PASSED
✓ Performance Test: <timing results>
```

## Google Colab Installation

**Important:** Colab's Google Drive has file operation limits. Always build on Colab's local filesystem, not Drive.

### Step 1: Upload to Drive

Upload the repository zip to your Google Drive.

### Step 2: Extract to Local Filesystem

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy from Drive to local Colab filesystem (critical!)
!cp -r /content/drive/MyDrive/flash-attention-deterministic /content/

# Navigate to local copy
%cd /content/flash-attention-deterministic
```

### Step 3: Build

```python
# Install dependencies
!pip install ninja

# Build with parallel jobs
!MAX_JOBS=16 pip install -e . --no-build-isolation -v 2>&1 | tee build.log

# Verify flag was used
!grep -c "DFLASH_ATTENTION_DETERMINISTIC" build.log
# Should show 100+ occurrences
```

### Step 4: Restart Runtime

**Critical:** After building, go to `Runtime > Restart runtime` to unload the old library from memory.

### Step 5: Verify

```python
# After restart
import torch
from flash_attn import flash_attn_func

# Quick test
q = torch.randn(2, 16, 1024, 64, dtype=torch.float16, device='cuda')
k = torch.randn(2, 16, 1024, 64, dtype=torch.float16, device='cuda')
v = torch.randn(2, 16, 1024, 64, dtype=torch.float16, device='cuda')

out = flash_attn_func(q, k, v, causal=False, deterministic=True)
print("✅ Deterministic mode working!")
```

## Troubleshooting

### Build Fails: "cutlass/numeric_types.h: No such file or directory"

**Problem:** Git submodules not initialized

**Solution:**
```bash
git submodule update --init --recursive
```

### Build Fails: "FLASH_ATTENTION_DETERMINISTIC not defined"

**Problem:** Compile flag missing from setup.py

**Solution:**
Check line 256 of `setup.py` contains:
```python
"-DFLASH_ATTENTION_DETERMINISTIC",  # Enable deterministic mode support
```

### Colab: "A google drive error has occurred"

**Problem:** Building on Drive filesystem (too many file operations)

**Solution:**
Always copy to `/content/` first (local SSD), never build directly in `/content/drive/`

### Build is Very Slow

**Problem:** Not using parallel compilation

**Solution:**
```bash
# Install ninja
pip install ninja

# Use MAX_JOBS
MAX_JOBS=16 pip install -e . --no-build-isolation
```

### Runtime: "deterministic flag not recognized"

**Problem:** Library wasn't built with deterministic support

**Solution:**
1. Check build log: `grep DFLASH_ATTENTION_DETERMINISTIC build.log`
2. Should show 100+ occurrences
3. If 0, the flag wasn't passed - check setup.py
4. Rebuild and restart runtime

### Memory Test Fails (No Increase)

**Problem:** Old library still in memory, or flag not compiled in

**Solution:**
1. Restart runtime (Colab) or Python session (local)
2. Check build log confirmed flag was used
3. Re-verify with `validate_determinism.py`

## Verification Checklist

After installation, verify:

- [ ] Build completed without errors
- [ ] `grep DFLASH_ATTENTION_DETERMINISTIC build.log` shows 100+ results
- [ ] Runtime restarted (Colab) or Python session fresh (local)
- [ ] `validate_determinism.py` shows all tests PASSED
- [ ] Memory usage increases significantly with `deterministic=True`

## Build Configuration

### Compile Flags

The key deterministic flag is added in `setup.py`:
```python
nvcc_flags = [
    "-O3",
    "-std=c++17",
    # ... other flags ...
    "-DFLASH_ATTENTION_DETERMINISTIC",  # This enables deterministic mode
]
```

### GPU Architectures

Default build supports:
- sm_80 (A100)
- sm_90 (H100)

To build for other architectures, modify `setup.py` line 234.

## Performance Tips

- Use `MAX_JOBS=16` for faster compilation (adjust based on CPU cores)
- Install `ninja` build system for significant speedup
- On Colab, always extract to `/content/` (local SSD) before building
- Expect 5-10 minute build time with optimizations

## Getting Help

If installation fails:
1. Check troubleshooting section above
2. Verify prerequisites are met
3. Check build.log for specific errors
4. Open an issue with:
   - Error message
   - Output of `grep DFLASH_ATTENTION_DETERMINISTIC build.log`
   - GPU model and CUDA version
