"""Is MLX decode GPU-bound or host-bound? Split one step into graph-build (CPU, lazy) and execution.

mlx_lm pipelines step t+1's graph build with step t's GPU work, so steady-state step time is
~max(build, gpu). If build >= gpu, a faster kernel buys nothing until the host side shrinks.

Run: uv run python -m bench.host_vs_gpu [--skinny]
"""

import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from bench._util import gpu_warm

model, _ = load("mlx-community/Qwen3.5-4B-MLX-4bit")
if "--skinny" in sys.argv:
    from skinny_mlx.patch import patch_model

    patch_model(model)
ctx = mx.array([[1000 + i % 5000 for i in range(512)]])
print(f"{'M':>3} {'build ms':>9} {'gpu ms':>8} {'serial ms':>10}")
for M in [1, 4, 8, 16]:
    cache = make_prompt_cache(model)
    mx.eval(model(ctx, cache=cache))
    x = mx.array([[42] * M])
    builds, gpus = [], []
    for i in range(12):
        if i % 4 == 0:
            gpu_warm(0.2)
        t0 = time.perf_counter()
        y = model(x, cache=cache)          # lazy: pure host-side graph construction
        t1 = time.perf_counter()
        mx.eval(y)                         # GPU execution (+ command encoding)
        t2 = time.perf_counter()
        if i >= 2:
            builds.append(t1 - t0); gpus.append(t2 - t1)
    b, g = sorted(builds)[len(builds) // 2], sorted(gpus)[len(gpus) // 2]
    print(f"{M:3d} {b * 1e3:9.2f} {g * 1e3:8.2f} {(b + g) * 1e3:10.2f}")
