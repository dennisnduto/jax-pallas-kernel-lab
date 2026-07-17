"""A BlockSpec-tiled matrix multiplication kernel."""

from __future__ import annotations

import jax
from jax.experimental import pallas as pl


def _matmul_kernel(left_ref, right_ref, output_ref):
    output_ref[...] = left_ref[...] @ right_ref[...]


def blocked_matmul(
    left: jax.Array,
    right: jax.Array,
    *,
    block_m: int = 32,
    block_n: int = 32,
    interpret: bool | None = None,
) -> jax.Array:
    """Compute a tiled matrix multiplication with one grid program per output block."""

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("blocked_matmul requires two rank-2 arrays")
    rows, contraction = left.shape
    right_contraction, columns = right.shape
    if contraction != right_contraction:
        raise ValueError(
            f"contracting dimensions differ: {contraction} versus {right_contraction}"
        )
    if left.dtype != right.dtype:
        raise TypeError(f"left/right dtypes differ: {left.dtype} versus {right.dtype}")
    if rows % block_m or columns % block_n:
        raise ValueError(
            f"shape {(rows, columns)} must be divisible by tile {(block_m, block_n)}"
        )
    if block_m < 1 or block_n < 1:
        raise ValueError("block sizes must be positive")

    use_interpret = jax.default_backend() == "cpu" if interpret is None else interpret
    call = pl.pallas_call(
        _matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, columns), left.dtype),
        grid=(rows // block_m, columns // block_n),
        in_specs=[
            pl.BlockSpec((block_m, contraction), lambda i, j: (i, 0)),
            pl.BlockSpec((contraction, block_n), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        interpret=use_interpret,
        name="portfolio_blocked_matmul",
        metadata={"project": "jax-pallas-kernel-lab", "kernel": "blocked_matmul"},
    )
    return call(left, right)
