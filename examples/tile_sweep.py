"""Validate several tile sizes without publishing machine-specific timings."""

import jax
import jax.numpy as jnp

from jax_pallas_kernels import blocked_matmul

left_key, right_key = jax.random.split(jax.random.key(9))
left = jax.random.normal(left_key, (128, 64), dtype=jnp.float32)
right = jax.random.normal(right_key, (64, 128), dtype=jnp.float32)
reference = left @ right

for tile in (8, 16, 32, 64):
    result = blocked_matmul(left, right, block_m=tile, block_n=tile, interpret=True)
    error = jnp.max(jnp.abs(result - reference))
    print(f"tile={tile:2d} max_error={float(error):.8e}")
