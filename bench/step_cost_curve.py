"""Full-model pipelined step cost for M tokens/step = speculative-verify cost curve.

Steady-state cost per step of M tokens, mirroring mlx_lm's async_eval generate loop, for:
  mlx      stock mlx_lm model
  +fuse    horizontal projection fusion, MLX kernels for every M
  +skinny  fusion + skinny qmm for M in SKINNY_RANGE
Each cell is the best of 3 windows of 30 steps, GPU pre-warmed (DVFS).

Run: uv run python -m bench.step_cost_curve [--no-fuse]   (--no-fuse: skinny without projection fusion)
"""

import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from bench._util import gpu_warm
from skinny_mlx import patch as skinny_patch

model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
ctx = mx.array([[1000 + i % 5000 for i in range(512)]])
Ms = [1, 2, 4, 6, 8, 12, 16]


def step(y, cache, M):
    logits = model(y, cache=cache)
    return mx.argmax(logits[:, -M:, :], axis=-1).astype(mx.int32)  # true sequential chain


def measure(M: int, N: int = 30) -> float:
    best = float("inf")
    for _ in range(3):
        cache = make_prompt_cache(model)
        mx.eval(model(ctx, cache=cache))
        y = step(mx.array([[42] * M]), cache, M)
        mx.async_eval(y)
        for _ in range(3):
            nxt = step(y, cache, M); mx.async_eval(nxt); mx.eval(y); y = nxt
        gpu_warm(0.2)
        t = time.perf_counter()
        for _ in range(N):
            nxt = step(y, cache, M); mx.async_eval(nxt); mx.eval(y); y = nxt
        mx.eval(y)
        best = min(best, (time.perf_counter() - t) / N)
    return best


FUSE = "--no-fuse" not in sys.argv
res = {"mlx": {M: measure(M) for M in Ms}}
print("patch:", skinny_patch.patch_model(model, fuse=FUSE))
ON = skinny_patch.SKINNY_RANGE
skinny_patch.SKINNY_RANGE = (99, 99)
res["+fuse" if FUSE else "patched, mlx kernels"] = {M: measure(M) for M in Ms}
skinny_patch.SKINNY_RANGE = ON
res["+skinny"] = {M: measure(M) for M in Ms}

base = res["mlx"][1]
print(f"{'':4s}" + "".join(f"{k:>24s}" for k in res) + f"{'speedup':>10s}")
for M in Ms:
    cells = "".join(f"{r[M] * 1e3:8.2f} ms {r[M] / base:4.2f}x {M / r[M]:4.0f}/s" for r in res.values())
    print(f"M={M:<2d}" + cells + f"{res['mlx'][M] / res['+skinny'][M]:9.2f}x")
