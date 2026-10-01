"""Drop-in patch: route 3..16-token quantized matmuls of an mlx_lm model through skinny_qmm.

M = 1..2 keeps MLX's qmv (already ~97% of bandwidth); M = 3..16 (speculative verify, small
batches) uses the MMA kernel on lane-order repacked weights. Prototype keeps both weight
layouts resident (2x weight memory) — a production version would add an M=1 path on the
repacked layout and drop the original.

Run:  uv run python -m skinny_mlx.patch   (logit equivalence check vs unpatched model)
"""

import mlx.core as mx
import mlx.nn as nn

from skinny_mlx.qmm import repack_for_skinny, skinny_qmm

SKINNY_RANGE = (3, 16)


def _qmm(x: mx.array, w, s, b, rw, group_size: int, bits: int) -> mx.array:
    k = x.shape[-1]
    m = x.size // k
    if SKINNY_RANGE[0] <= m <= SKINNY_RANGE[1]:
        return skinny_qmm(x.reshape(m, k), rw, s, b).reshape(*x.shape[:-1], -1)
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=group_size, bits=bits)


class SkinnyQuantizedLinear(nn.Module):
    def __init__(self, ql: nn.QuantizedLinear):
        super().__init__()
        self.group_size, self.bits = ql.group_size, ql.bits
        self.weight, self.scales, self.biases = ql.weight, ql.scales, ql.biases
        self.rweight = repack_for_skinny(ql.weight)
        if "bias" in ql:
            self.bias = ql.bias

    def __call__(self, x: mx.array) -> mx.array:
        y = _qmm(x, self.weight, self.scales, self.biases, self.rweight, self.group_size, self.bits)
        return y + self.bias if "bias" in self else y


def _patch_tied_head(emb: nn.QuantizedEmbedding) -> None:
    rw = repack_for_skinny(emb.weight)
    mx.eval(rw)
    emb.as_linear = lambda x: _qmm(x, emb.weight, emb.scales, emb.biases, rw, emb.group_size, emb.bits)


def patch_model(model: nn.Module) -> int:
    """Swap every 4-bit/group-64 QuantizedLinear (and a tied quantized LM head). Returns modules patched."""
    swaps = [(n, SkinnyQuantizedLinear(m)) for n, m in model.named_modules()
             if isinstance(m, nn.QuantizedLinear) and m.bits == 4 and m.group_size == 64]
    model.update_modules(_tree(swaps))
    mx.eval(model.parameters())
    n = len(swaps)
    for _, m in model.named_modules():
        if isinstance(m, nn.QuantizedEmbedding) and m.bits == 4 and m.group_size == 64:
            _patch_tied_head(m)
            n += 1
    return n


def _tree(pairs):
    """[('a.b.0.c', mod)] -> nested dict/list tree for nn.Module.update_modules."""
    from mlx.utils import tree_unflatten
    return tree_unflatten(pairs)


def main() -> None:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    ids = mx.array([tok.encode("The quick brown fox jumps over the lazy dog. In computer science, a cache is")])
    draft = mx.array([tok.encode(" a hardware or software component that stores data")[:8]])
    def logits_after_prefix():
        cache = make_prompt_cache(model)
        model(ids, cache=cache)
        return model(draft, cache=cache).astype(mx.float32)   # M = 8 verify step
    ref = logits_after_prefix(); mx.eval(ref)
    print("dtype", ref.dtype, "patched modules:", patch_model(model))
    got = logits_after_prefix(); mx.eval(got)
    rel = (mx.abs(ref - got).max() / mx.abs(ref).max()).item()
    agree = (mx.argmax(ref, -1) == mx.argmax(got, -1)).mean().item()
    print(f"M=8 verify logits: max rel diff {rel:.2e}, argmax agreement {agree:.0%}")


if __name__ == "__main__":
    main()
