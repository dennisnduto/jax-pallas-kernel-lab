"""Validate both kernels against ordinary JAX references."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .fused import fused_silu_gate, fused_silu_gate_reference
from .matmul import blocked_matmul


def main() -> None:
    left_key, right_key, x_key, gate_key, bias_key = jax.random.split(jax.random.key(0), 5)
    left = jax.random.normal(left_key, (64, 48), dtype=jnp.float32)
    right = jax.random.normal(right_key, (48, 32), dtype=jnp.float32)
    actual_matmul = blocked_matmul(left, right, block_m=16, block_n=16)
    reference_matmul = left @ right

    x = jax.random.normal(x_key, (32, 64), dtype=jnp.float32)
    gate = jax.random.normal(gate_key, x.shape, dtype=jnp.float32)
    bias = 0.1 * jax.random.normal(bias_key, x.shape, dtype=jnp.float32)
    actual_fused = fused_silu_gate(x, gate, bias)
    reference_fused = fused_silu_gate_reference(x, gate, bias)

    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(
        "blocked_matmul max error:",
        float(jnp.max(jnp.abs(actual_matmul - reference_matmul))),
    )
    print(
        "fused_silu_gate max error:",
        float(jnp.max(jnp.abs(actual_fused - reference_fused))),
    )


if __name__ == "__main__":
    main()
