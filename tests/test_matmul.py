import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pallas_kernels import blocked_matmul


def test_blocked_matmul_matches_reference_in_interpret_mode():
    left_key, right_key = jax.random.split(jax.random.key(0))
    left = jax.random.normal(left_key, (64, 48), dtype=jnp.float32)
    right = jax.random.normal(right_key, (48, 32), dtype=jnp.float32)
    actual = blocked_matmul(left, right, block_m=16, block_n=16, interpret=True)
    expected = left @ right
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-5)


def test_blocked_matmul_composes_with_jit_in_interpret_mode():
    left = jnp.arange(32 * 16, dtype=jnp.float32).reshape(32, 16) / 100.0
    right = jnp.arange(16 * 24, dtype=jnp.float32).reshape(16, 24) / 100.0
    compiled = jax.jit(
        lambda a, b: blocked_matmul(a, b, block_m=8, block_n=8, interpret=True)
    )
    np.testing.assert_allclose(compiled(left, right), left @ right, rtol=2e-5, atol=3e-5)


def test_shape_validation_is_clear():
    with pytest.raises(ValueError, match="contracting dimensions"):
        blocked_matmul(jnp.ones((8, 7)), jnp.ones((6, 8)), block_m=4, block_n=4)
    with pytest.raises(ValueError, match="must be divisible"):
        blocked_matmul(jnp.ones((10, 8)), jnp.ones((8, 10)), block_m=4, block_n=4)


@pytest.mark.parametrize(("block_m", "block_n"), [(0, 4), (4, 0), (-1, 4), (4, -1)])
def test_block_sizes_must_be_positive(block_m, block_n):
    with pytest.raises(ValueError, match="block sizes must be positive"):
        blocked_matmul(
            jnp.ones((8, 8)),
            jnp.ones((8, 8)),
            block_m=block_m,
            block_n=block_n,
        )
