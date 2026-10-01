"""Where does a multi-token decode step spend time? Per-sublayer-type cost at several M.

Each sublayer type is timed in isolation with no host syncs between calls: 100 chained calls
in one lazy graph, evaluated once, then scaled by how many layers of that type the model has. GPU is pre-warmed (DVFS).
Mixers run without a cache (fresh state / no KV history), so attention is a lower bound.

Run: uv run python -m bench.profile_step [--skinny]
"""

import sys
import time

import mlx.core as mx
from mlx_lm import load

from bench._util import best_time

model, _ = load("mlx-community/Qwen3.5-4B-MLX-4bit")
if "--skinny" in sys.argv:
    from skinny_mlx.patch import patch_model

    patch_model(model)

layers = model.layers
gdn = next(l for l in layers if getattr(l, "linear_attn", None) is not None)
att = next(l for l in layers if getattr(l, "self_attn", None) is not None)
n_gdn = sum(getattr(l, "linear_attn", None) is not None for l in layers)
n_att = len(layers) - n_gdn
emb = model.language_model.model.embed_tokens if hasattr(model, "language_model") else model.model.embed_tokens
d = emb.weight.shape[1] * 8

kinds = {
    "gdn mixer": (lambda x: gdn.linear_attn(x), n_gdn),
    "full attention": (lambda x: att.self_attn(x, mask="causal" if x.shape[1] > 1 else None), n_att),
    "mlp": (lambda x: gdn.mlp(x), len(layers)),
    "norms": (lambda x: gdn.input_layernorm(x), 2 * len(layers)),
    "lm head": (lambda x: emb.as_linear(x)[..., :d], 1),
}


def cost(fn, M: int, chain: int = 100) -> float:
    x0 = mx.random.normal([1, M, d]).astype(mx.bfloat16) * 0.1
    mx.eval(x0)

    def run():
        x = x0
        for _ in range(chain):
            x = x + 0.001 * fn(x)  # data dependency serializes calls like a real layer stack
        mx.eval(x)

    return best_time(run) / chain

Ms = [1, 2, 4, 8, 16]
rows = {k: [cost(fn, M) * n * 1e3 for M in Ms] for k, (fn, n) in kinds.items()}
print(f"{'ms/step (scaled)':18s}" + "".join(f"{'M=' + str(m):>9s}" for m in Ms))
for k, vals in rows.items():
    print(f"{k:18s}" + "".join(f"{v:9.2f}" for v in vals))
print(f"{'total':18s}" + "".join(f"{sum(r[i] for r in rows.values()):9.2f}" for i in range(len(Ms))))
