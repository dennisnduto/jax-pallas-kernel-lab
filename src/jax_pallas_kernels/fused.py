"""A fused bias, SiLU, and gate kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def fused_silu_gate_reference(
    x: jax.Array,
    gate: jax.Array,
    bias: jax.Array,
) -> jax.Array:
    value = x + bias
    return jax.nn.silu(value) * gate


def _fused_silu_gate_kernel(x_ref, gate_ref, bias_ref, output_ref):
    value = x_ref[...] + bias_ref[...]
    output_ref[...] = (value * jax.nn.sigmoid(value)) * gate_ref[...]


def fused_silu_gate(
    x: jax.Array,
    gate: jax.Array,
    bias: jax.Array,
    *,
    interpret: bool | None = None,
) -> jax.Array:
    """Launch one fused Pallas kernel over equally shaped arrays."""

    if x.shape != gate.shape or x.shape != bias.shape:
        raise ValueError(
            f"x, gate, and bias must have equal shapes; got {x.shape}, {gate.shape}, {bias.shape}"
        )
    if x.dtype != gate.dtype or x.dtype != bias.dtype:
        raise TypeError("x, gate, and bias must have the same dtype")
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError("fused_silu_gate requires floating-point arrays")

    use_interpret = jax.default_backend() == "cpu" if interpret is None else interpret
    return pl.pallas_call(
        _fused_silu_gate_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=use_interpret,
        name="portfolio_fused_silu_gate",
        metadata={"project": "jax-pallas-kernel-lab", "kernel": "fused_silu_gate"},
    )(x, gate, bias)
