"""Full-model pipelined step cost for M tokens/step = speculative-verify cost curve.

Steady-state pipelined cost per step of M tokens (mirrors mlx_lm's async_eval generate loop).
With --skinny, baseline MLX and skinny_qmm are measured interleaved in one process (best of 3
each) so background GPU load hits both equally.

Run: uv run python -m bench.step_cost_curve [--skinny]
"""

import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
ctx = mx.array([[1000 + i % 5000 for i in range(512)]])
SKINNY = "--skinny" in sys.argv
if SKINNY:
    from skinny_mlx import patch as skinny_patch

    print("patched", skinny_patch.patch_model(model), "modules with skinny_qmm")
    ON = skinny_patch.SKINNY_RANGE


def step(y, cache, M):
    logits = model(y, cache=cache)
    return mx.argmax(logits[:, -M:, :], axis=-1).astype(mx.int32)  # true sequential chain


def measure(M: int, N: int = 30) -> float:
    cache = make_prompt_cache(model)
    mx.eval(model(ctx, cache=cache))
    y = step(mx.array([[42] * M]), cache, M)
    mx.async_eval(y)
    for _ in range(5):
        nxt = step(y, cache, M); mx.async_eval(nxt); mx.eval(y); y = nxt
    t = time.perf_counter()
    for _ in range(N):
        nxt = step(y, cache, M); mx.async_eval(nxt); mx.eval(y); y = nxt
    mx.eval(y)
    return (time.perf_counter() - t) / N


base = None
for M in [1, 2, 3, 4, 6, 8, 12, 16]:
    mlx_t, sk_t = [], []
    for _ in range(3 if SKINNY else 1):
        if SKINNY:
            skinny_patch.SKINNY_RANGE = (99, 99)
        mlx_t.append(measure(M))
        if SKINNY:
            skinny_patch.SKINNY_RANGE = ON
            sk_t.append(measure(M))
    a = min(mlx_t)
    base = base or a
    line = f"M={M:2d}: mlx {a * 1e3:6.2f} ms ({a / base:4.2f}x, {M / a:5.0f} tok/s)"
    if SKINNY:
        b = min(sk_t)
        line += f" | skinny {b * 1e3:6.2f} ms ({b / base:4.2f}x, {M / b:5.0f} tok/s) | step speedup {a / b:4.2f}x"
    print(line)
