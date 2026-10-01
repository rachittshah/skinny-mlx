# skinny-mlx

**Flat-cost multi-token verification for MLX on Apple Silicon** — the missing piece that makes
speculative decoding pay off on a Mac.

Single-token decode in MLX is already near the memory-bandwidth wall. The only way to multiply
local decode speed is to emit several tokens per weight read (speculative decoding), and on
Metal that keeps failing because verifying 8 tokens costs ~2.5× one token. This repo fixes that
cost and builds toward **FlatSpec** (`docs/flatspec-design.md`): flat verify + the model's own
native MTP draft head + replay-free rollback for hybrid (Gated DeltaNet) models.

## Results so far (M4 Max 32-core GPU, Qwen3.5-4B 4-bit, mlx 0.32.1)

Full-model decode step, pipelined, best of 3 after GPU warm-up (`bench/step_cost_curve.py`):

| tokens per step | stock MLX | skinny-mlx | step speedup | cost vs 1 token |
|---:|---:|---:|---:|---:|
| 1 | 8.70 ms | (same path) | — | 1.00× |
| 6 | 16.86 ms | 13.18 ms | 1.28× | 1.94× → 1.52× |
| 8 | 21.31 ms | **14.80 ms** | **1.44×** | **2.45× → 1.70×** |
| 16 | 35.85 ms | 29.22 ms | 1.23× | 4.12× → 3.36× |

The model's own 249 quantized matmuls alone (`bench/model_matmuls.py`, bf16 both sides):
**1.82× at M=8, 1.52× at M=16**. M=8 logits match stock MLX (100% argmax agreement,
max rel diff 1.3e-2 — bf16 rounding level).

## How

- `skinny_mlx/qmm.py` — 4-bit quantized matmul for 2–16 tokens: 8×8 `simdgroup_matrix` MMA,
  split-K across simdgroups, weights relaid **once at load** into MMA-fragment lane order (each
  simdgroup streams one coalesced 256 B block per group), scale/bias applied once per group,
  nibbles unpacked with the `0x6400` half trick.
- `skinny_mlx/patch.py` — drop-in patch for any mlx_lm model with 4-bit/group-64 weights:
  routes 6–16-token steps to the kernel and fuses same-input projections (249 → 129 matmuls).

```python
from mlx_lm import load
from skinny_mlx.patch import patch_model
model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
patch_model(model)
```

## Reproduce

```bash
uv sync
uv run python -m skinny_mlx.qmm            # kernel correctness + synthetic bench
uv run python -m skinny_mlx.patch          # logit equivalence on Qwen3.5-4B
uv run python -m bench.model_matmuls       # real-shape matmuls, MLX vs skinny
uv run python -m bench.step_cost_curve     # full-model step cost
```

Close GPU-heavy apps first — background GPU load swings results 2–3×.

## Docs

- `docs/roofline-and-plan.md` — measured hardware roofline and why kernels alone cap at ~1.3×
- `docs/lit-2026.md` — 2026 literature on Mac decode, speculation, hybrid rollback
- `docs/flatspec-design.md` — the full design, status, and projections

## Status

Prototype. Keeps both weight layouts resident (2× weight memory). Apple M1–M4 tuned on M4 Max.
Next: native MTP drafting, pruned-vocab draft head, replay-free GDN rollback, end-to-end tok/s.

MIT licensed.
