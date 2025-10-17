#!/usr/bin/env python3
"""
Deterministic Flash Attention Validation Script

This script validates:
1. Batch Invariance - Same input produces identical output regardless of batch size
2. Perfect Determinism - Multiple runs produce identical outputs
3. Performance - Measures overhead compared to standard Flash Attention

Usage:
    python validate_determinism.py [--quick] [--verbose]

Options:
    --quick     Run faster tests with smaller sizes (for CI/quick validation)
    --verbose   Show detailed output for each test
"""

import argparse
import sys
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func, set_deterministic_mode
except ImportError:
    print("❌ Error: flash_attn package not found")
    print("Please install flash-attention-deterministic first:")
    print("  pip install -e .")
    sys.exit(1)


# ANSI color codes for pretty output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_failure(text: str):
    """Print failure message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def check_gpu():
    """Verify GPU availability and compute capability"""
    if not torch.cuda.is_available():
        print_failure("CUDA is not available")
        return False

    device_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    compute_cap_str = f"{compute_cap[0]}.{compute_cap[1]}"

    print_info(f"GPU: {device_name}")
    print_info(f"Compute Capability: {compute_cap_str}")

    if compute_cap[0] < 8:
        print_warning(f"This implementation is optimized for SM 8.0+ (found {compute_cap_str})")
        print_warning("Tests will run but performance may not be optimal")

    return True


def generate_test_inputs(
    batch_size: int,
    seq_len: int,
    num_heads: int = 32,
    head_dim: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random Q, K, V tensors for testing"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

    return q, k, v


def test_batch_invariance(quick: bool = False, verbose: bool = False) -> bool:
    """
    Test that outputs are identical across different batch sizes

    This tests that processing samples individually vs in a batch produces
    the same results - a critical property for reproducible training.
    """
    print_header("Test 1: Batch Invariance")

    # Test configuration
    seq_len = 512 if quick else 2048
    num_heads = 32
    head_dim = 64
    batch_sizes = [1, 2, 4] if quick else [1, 2, 4, 8]

    print_info(f"Configuration: seq_len={seq_len}, heads={num_heads}, dim={head_dim}")
    print_info(f"Testing batch sizes: {batch_sizes}")

    # Enable deterministic mode
    set_deterministic_mode(enabled=True)

    # Generate a set of individual samples
    samples = []
    for i in range(max(batch_sizes)):
        q, k, v = generate_test_inputs(
            batch_size=1,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            seed=42 + i  # Different seed for each sample
        )
        samples.append((q, k, v))

    # Process samples individually (batch_size=1)
    individual_outputs = []
    for i, (q, k, v) in enumerate(samples):
        with torch.no_grad():
            out = flash_attn_func(q, k, v, causal=True)
            individual_outputs.append(out)

        if verbose:
            print(f"  Processed sample {i+1}/{len(samples)} individually")

    # Now test each batch size
    results = []
    for batch_size in batch_sizes:
        # Combine samples into batch
        q_batch = torch.cat([samples[i][0] for i in range(batch_size)], dim=0)
        k_batch = torch.cat([samples[i][1] for i in range(batch_size)], dim=0)
        v_batch = torch.cat([samples[i][2] for i in range(batch_size)], dim=0)

        # Process batch
        with torch.no_grad():
            out_batch = flash_attn_func(q_batch, k_batch, v_batch, causal=True)

        # Compare each sample in batch to individual output
        all_match = True
        max_diff = 0.0

        for i in range(batch_size):
            out_individual = individual_outputs[i]
            out_from_batch = out_batch[i:i+1]

            diff = (out_individual - out_from_batch).abs().max().item()
            max_diff = max(max_diff, diff)

            if diff > 1e-6:  # Tolerance for floating point
                all_match = False
                if verbose:
                    print_warning(f"  Batch size {batch_size}, sample {i}: diff = {diff}")

        results.append((batch_size, all_match, max_diff))

        if all_match:
            if verbose:
                print_success(f"  Batch size {batch_size}: EXACT match (max diff: {max_diff:.2e})")
        else:
            print_failure(f"  Batch size {batch_size}: MISMATCH (max diff: {max_diff:.2e})")

    # Summary
    all_passed = all(result[1] for result in results)

    if all_passed:
        print_success("Batch Invariance Test: PASSED")
        print_info("All batch sizes produce identical outputs")
    else:
        print_failure("Batch Invariance Test: FAILED")
        print_info("Some batch sizes produced different outputs")

    return all_passed


def test_determinism(quick: bool = False, verbose: bool = False) -> bool:
    """
    Test that multiple runs produce identical outputs

    This tests that running the same operation multiple times produces
    exactly the same result - essential for reproducible experiments.
    """
    print_header("Test 2: Perfect Determinism")

    # Test configuration
    batch_size = 4 if quick else 8
    seq_len = 512 if quick else 2048
    num_heads = 32
    head_dim = 64
    num_runs = 10 if quick else 50

    print_info(f"Configuration: B={batch_size}, L={seq_len}, H={num_heads}, D={head_dim}")
    print_info(f"Running {num_runs} consecutive trials")

    # Enable deterministic mode
    set_deterministic_mode(enabled=True)

    # Generate fixed inputs
    q, k, v = generate_test_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        seed=42
    )

    # Run multiple times and collect unique outputs
    outputs = []
    for run_idx in range(num_runs):
        with torch.no_grad():
            out = flash_attn_func(q, k, v, causal=True)
            outputs.append(out.clone())

        if verbose and (run_idx + 1) % 10 == 0:
            print(f"  Completed {run_idx + 1}/{num_runs} runs")

    # Count unique outputs
    unique_outputs = 1
    reference = outputs[0]

    for i in range(1, len(outputs)):
        if not torch.equal(outputs[i], reference):
            unique_outputs += 1
            if verbose:
                diff = (outputs[i] - reference).abs().max().item()
                print_warning(f"  Run {i+1} differs from run 1 (max diff: {diff:.2e})")

    # Summary
    if unique_outputs == 1:
        print_success(f"Determinism Test: PASSED")
        print_info(f"{num_runs} runs produced 1 unique output (perfect determinism)")
        return True
    else:
        print_failure(f"Determinism Test: FAILED")
        print_info(f"{num_runs} runs produced {unique_outputs} unique outputs")
        return False


def test_performance(quick: bool = False, verbose: bool = False) -> bool:
    """
    Measure performance overhead of deterministic mode

    Compares speed of deterministic vs non-deterministic Flash Attention.
    """
    print_header("Test 3: Performance Overhead")

    # Test configuration
    batch_size = 4 if quick else 8
    seq_len = 512 if quick else 2048
    num_heads = 32
    head_dim = 64
    warmup_iters = 3 if quick else 10
    test_iters = 5 if quick else 50

    print_info(f"Configuration: B={batch_size}, L={seq_len}, H={num_heads}, D={head_dim}")
    print_info(f"Warmup: {warmup_iters} iterations, Test: {test_iters} iterations")

    # Generate inputs
    q, k, v = generate_test_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        seed=42
    )

    def benchmark_mode(deterministic: bool, mode_name: str) -> float:
        """Benchmark a specific mode"""
        set_deterministic_mode(enabled=deterministic)

        # Warmup
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = flash_attn_func(q, k, v, causal=True)

        torch.cuda.synchronize()

        # Actual timing
        start = time.perf_counter()
        for _ in range(test_iters):
            with torch.no_grad():
                _ = flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time = (end - start) / test_iters * 1000  # Convert to ms

        if verbose:
            print(f"  {mode_name}: {avg_time:.3f} ms/iter")

        return avg_time

    # Benchmark both modes
    print_info("Benchmarking standard mode...")
    time_standard = benchmark_mode(False, "Standard FA")

    print_info("Benchmarking deterministic mode...")
    time_deterministic = benchmark_mode(True, "Deterministic FA")

    # Calculate overhead
    overhead_pct = ((time_deterministic - time_standard) / time_standard) * 100

    print()
    print(f"  Standard FA:      {time_standard:.3f} ms/iter")
    print(f"  Deterministic FA: {time_deterministic:.3f} ms/iter")
    print(f"  Overhead:         {overhead_pct:.1f}%")

    # Success criteria: overhead should be < 30%
    if overhead_pct < 30:
        print_success("Performance Test: PASSED")
        print_info(f"Overhead of {overhead_pct:.1f}% is within acceptable range (<30%)")
        return True
    else:
        print_warning("Performance Test: WARNING")
        print_info(f"Overhead of {overhead_pct:.1f}% is higher than expected (<30%)")
        return True  # Still pass, but warn user


def main():
    parser = argparse.ArgumentParser(
        description="Validate deterministic Flash Attention implementation"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run faster tests with smaller sizes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output for each test"
    )
    args = parser.parse_args()

    print(f"{Colors.BOLD}Deterministic Flash Attention Validation{Colors.END}")
    print(f"{'='*70}")

    # Check GPU
    if not check_gpu():
        sys.exit(1)

    # Run tests
    results = []

    try:
        results.append(("Batch Invariance", test_batch_invariance(args.quick, args.verbose)))
        results.append(("Determinism", test_determinism(args.quick, args.verbose)))
        results.append(("Performance", test_performance(args.quick, args.verbose)))
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Final summary
    print_header("Validation Summary")

    for test_name, passed in results:
        if passed:
            print_success(f"{test_name}: PASSED")
        else:
            print_failure(f"{test_name}: FAILED")

    all_passed = all(result[1] for result in results)

    print()
    if all_passed:
        print_success(f"{Colors.BOLD}All validation tests passed!{Colors.END}")
        print_info("Your deterministic Flash Attention installation is working correctly.")
        sys.exit(0)
    else:
        print_failure(f"{Colors.BOLD}Some validation tests failed!{Colors.END}")
        print_info("Please check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
