"""Kill test 3 for Fast-Weight Drafting: is reversing the gated delta rule numerically usable?

CPU only (mx.cpu; no Metal kernels). Captures real per-token gates (g, beta) and keys/values of every
GDN layer of Qwen3.5-4B on one prompt, then for windows of R "rejected" tokens:
  forward  S_{t+1} = g_t * S_t (I - b_t k_t k_t^T) + b_t v_t k_t^T          (fp32)
  reverse  S_t     = (S_{t+1} - b_t v_t k_t^T)(I + b_t/(1-b_t) k_t k_t^T) / g_t
and reports the relative reconstruction error of the pre-window state, plus how often the
conditioning factor g*(1-b) is small enough to force a replay fallback.

Run: uv run python -m bench.reversal_check [n_layers]   (default 8: first 8 decoder layers = 6 GDN layers)
"""

import sys

import mlx.core as mx
import numpy as np

N_LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 8  # MLX CPU qmm is slow; truncate the trunk


def _apply(S, k, v, g, b):
    """One gated delta step on [H,Dv,Dk] state: g*S(I - b k k^T) + b v k^T."""
    S = g[:, None, None] * S
    Sk = np.einsum("hvk,hk->hv", S, k)
    return S + b[:, None, None] * np.einsum("hv,hk->hvk", v - Sk, k)


def _unapply(S, k, v, g, b):
    """Exact inverse via Sherman-Morrison: (S - b v k^T)(I + b/(1-b) k k^T) / g."""
    Sp = S - b[:, None, None] * np.einsum("hv,hk->hvk", v, k)
    Spk = np.einsum("hvk,hk->hv", Sp, k)
    Sp = Sp + (b / (1 - b))[:, None, None] * np.einsum("hv,hk->hvk", Spk, k)
    return Sp / g[:, None, None]
from mlx_lm import load
from mlx_lm.models import qwen3_5
from mlx_lm.models.gated_delta import compute_g

mx.set_default_device(mx.cpu)
CAPT = []
_orig = qwen3_5.gated_delta_update


def _capture(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
    g = mx.exp(compute_g(A_log, a, dt_bias))  # decay factor in (0, 1]
    CAPT.append(dict(k=np.array(k.astype(mx.float32)), v=np.array(v.astype(mx.float32)),
                     g=np.array(g.astype(mx.float32)), b=np.array(mx.sigmoid(b).astype(mx.float32))))
    return _orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=False)


def main() -> None:
    qwen3_5.gated_delta_update = _capture
    model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    inner = model.language_model.model
    inner.layers = inner.layers[:N_LAYERS]  # first N layers keep real activations for the GDN layers they hold
    text = "Explain how a CPU cache hierarchy works, including the MESI coherence protocol."
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                  enable_thinking=False)
    mx.eval(inner(mx.array([ids])))
    print(f"captured {len(CAPT)} GDN layers x {len(ids)} tokens")

    all_gb = np.concatenate([(c["g"][0] * (1 - c["b"][0])).ravel() for c in CAPT])
    for thr in (1e-1, 1e-2, 1e-3):
        print(f"fraction of (token, head) with g*(1-b) < {thr:g}: {np.mean(all_gb < thr):.3%}")

    rng = np.random.default_rng(0)
    for R in (1, 4, 8, 15):
        errs, unsafe = [], 0
        for c in CAPT:
            k, v, g, b = c["k"][0], c["v"][0], c["g"][0], c["b"][0]  # [T,Hk,Dk], [T,Hv,Dv], [T,Hv]
            T, Hk, Dk = k.shape
            Hv, Dv = v.shape[1], v.shape[2]
            rep = Hv // Hk
            k = np.repeat(k, rep, axis=1)
            start = int(rng.integers(8, T - R))
            S = np.zeros((Hv, Dv, Dk), np.float32)
            for t in range(start):  # warm the state with real history
                S = _apply(S, k[t], v[t], g[t], b[t])
            S0 = S.copy()
            for t in range(start, start + R):  # apply R tokens that verify will reject
                S = _apply(S, k[t], v[t], g[t], b[t])
            for t in reversed(range(start, start + R)):  # un-apply them
                unsafe += int(np.any(g[t] * (1 - b[t]) < 1e-3))
                S = _unapply(S, k[t], v[t], g[t], b[t])
            errs.append(np.linalg.norm(S - S0) / max(np.linalg.norm(S0), 1e-12))
        e = np.array(errs)
        print(f"R={R:2d} rejected: rel err median {np.median(e):.2e}  p95 {np.percentile(e, 95):.2e}  "
              f"max {e.max():.2e} | layers needing fallback {unsafe}/{len(CAPT) * R} token-steps")


if __name__ == "__main__":
    main()
