"""Time exactly the model's own quantized matmuls (real shapes + weights), MLX vs skinny, per M.

Both sides get the model's real activation dtype (bf16; MLX is ~2x slower on mismatched fp16).
Separates "is the kernel fast on the real shapes?" from "is the rest of the step the problem?".

Run: uv run python -m bench.model_matmuls
"""

from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from bench._util import best_time
from skinny_mlx.qmm import repack_for_skinny, skinny_qmm

model, _ = load("mlx-community/Qwen3.5-4B-MLX-4bit")
mods = [(n.split(".")[-1], m) for n, m in model.named_modules() if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding))]
W = [(name, m.weight, m.scales, m.biases, repack_for_skinny(m.weight)) for name, m in mods]
mx.eval([w[4] for w in W])
nbytes = sum(w.nbytes + s.nbytes + b.nbytes for _, w, s, b, _ in W)
print(f"{len(W)} matmuls, {nbytes / 1e9:.2f} GB")

for M in [1, 4, 8, 16]:
    xs = {}
    for _, w, *_ in W:
        k = w.shape[1] * 8
        xs.setdefault(k, mx.random.normal([M, k]).astype(W[0][2].dtype))  # model activation dtype (bf16)
    mx.eval(list(xs.values()))
    per = defaultdict(lambda: [0.0, 0.0])

    def run_mlx(sel=None):
        mx.eval([mx.quantized_matmul(xs[w.shape[1] * 8], w, s, b, transpose=True) for n, w, s, b, _ in W if sel in (None, n)])

    def run_sk(sel=None):
        mx.eval([skinny_qmm(xs[w.shape[1] * 8], r, s, b) for n, w, s, b, r in W if sel in (None, n)])

    a, b = best_time(run_mlx), best_time(run_sk)
    print(f"M={M:2d}: mlx {a * 1e3:6.2f} ms ({nbytes / a / 1e9:3.0f} GB/s) | skinny {b * 1e3:6.2f} ms ({nbytes / b / 1e9:3.0f} GB/s) | {a / b:4.2f}x")
    if M in (8, 16):
        for name in sorted({n for n, *_ in W}):
            pa, pb = best_time(lambda: run_mlx(name), reps=3), best_time(lambda: run_sk(name), reps=3)
            print(f"     {name:14s} mlx {pa * 1e3:6.2f}  skinny {pb * 1e3:6.2f}  {pa / pb:4.2f}x")
