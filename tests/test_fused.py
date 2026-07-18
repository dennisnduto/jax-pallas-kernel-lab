import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pallas_kernels import fused_silu_gate, fused_silu_gate_reference


def test_fused_kernel_matches_reference_in_interpret_mode():
    x_key, gate_key, bias_key = jax.random.split(jax.random.key(4), 3)
    x = jax.random.normal(x_key, (16, 32), dtype=jnp.float32)
    gate = jax.random.normal(gate_key, x.shape, dtype=jnp.float32)
    bias = 0.1 * jax.random.normal(bias_key, x.shape, dtype=jnp.float32)
    actual = fused_silu_gate(x, gate, bias, interpret=True)
    expected = fused_silu_gate_reference(x, gate, bias)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


def test_fused_kernel_validates_shape_and_dtype():
    with pytest.raises(ValueError, match="equal shapes"):
        fused_silu_gate(jnp.ones((2, 3)), jnp.ones((2, 4)), jnp.ones((2, 3)))
    with pytest.raises(TypeError, match="same dtype"):
        fused_silu_gate(
            jnp.ones((2, 3), dtype=jnp.float32),
            jnp.ones((2, 3), dtype=jnp.float16),
            jnp.ones((2, 3), dtype=jnp.float32),
        )
