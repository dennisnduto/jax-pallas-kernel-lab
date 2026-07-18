"""Simple, synchronization-correct local benchmark harness."""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp

from .matmul import blocked_matmul


def _time(callable_, iterations: int) -> list[float]:
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        callable_().block_until_ready()
        durations.append(time.perf_counter() - start)
    return durations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--block", type=int, default=32)
    args = parser.parse_args()
    if args.size % args.block:
        raise SystemExit("--size must be divisible by --block")

    left_key, right_key = jax.random.split(jax.random.key(3))
    left = jax.random.normal(left_key, (args.size, args.size), dtype=jnp.float32)
    right = jax.random.normal(right_key, (args.size, args.size), dtype=jnp.float32)

    kernel = lambda: blocked_matmul(
        left,
        right,
        block_m=args.block,
        block_n=args.block,
    )
    reference = jax.jit(lambda a, b: a @ b)

    kernel().block_until_ready()
    reference(left, right).block_until_ready()
    kernel_times = _time(kernel, args.iterations)
    reference_times = _time(lambda: reference(left, right), args.iterations)

    print(f"Backend: {jax.default_backend()}")
    print(f"Shape: {args.size}x{args.size}; iterations: {args.iterations}")
    print(f"Pallas median: {statistics.median(kernel_times) * 1000:.3f} ms")
    print(f"JAX matmul median: {statistics.median(reference_times) * 1000:.3f} ms")
    if jax.default_backend() == "cpu":
        print("Note: CPU uses Pallas interpret mode; these timings are not accelerator results.")


if __name__ == "__main__":
    main()
