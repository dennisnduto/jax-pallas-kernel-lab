# JAX Pallas Kernel Lab

A hardware-aware portfolio repository containing custom kernels written with JAX Pallas.

## Kernels

### Blocked matrix multiplication

`blocked_matmul` assigns each grid program one output tile. `BlockSpec` maps that program to a row
block of the left operand, a column block of the right operand, and a disjoint output block.

### Fused SiLU gate

`fused_silu_gate` performs bias addition, SiLU activation, and elementwise gating in one Pallas
kernel invocation. The separate reference function is used for numerical validation.

## CPU and accelerator modes

Pallas kernels target GPU/TPU hardware. JAX also provides `interpret=True`, which executes a kernel
through a JAX interpretation path and is the only supported CPU debugging route. Tests use
interpret mode so CI remains hardware-independent. On a supported accelerator, pass
`interpret=False` to exercise the hardware backend.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m jax_pallas_kernels.demo
```

## Benchmark correctly

Accelerator dispatch is asynchronous. `benchmark.py` calls `.block_until_ready()` before reading a
wall-clock result and separates warm-up compilation from steady-state execution. It reports
measurements only from the machine on which it is run; no benchmark numbers are committed here.

```bash
python -m jax_pallas_kernels.benchmark --size 128 --iterations 5
# On a supported accelerator, increase the size after the first successful run.
```

## Limitations

- Matrix dimensions must be divisible by the selected tile sizes in this concise implementation.
- Backend-specific tuning is intentionally left explicit rather than hidden behind heuristics.
- Pallas is experimental and its API may change; the project pins JAX to the `0.9.x` range.
- CPU interpret mode validates semantics, not accelerator performance.

This repository is an independent portfolio project and is not an upstream JAX contribution.
