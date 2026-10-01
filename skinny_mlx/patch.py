"""Drop-in patch for mlx_lm models: skinny qmm for 6..16-token steps + horizontal projection fusion.

1. Skinny routing: M = 1..5 keeps MLX's qmv/qmm (qmv is ~97% of bandwidth at M=1); M = 6..16
   (speculative verify, small batches) uses the MMA kernel on lane-order repacked weights.
2. Horizontal fusion: projections that read the same input (GDN qkv/z/b/a, attention q/k/v,
   MLP gate/up) are concatenated along N at load time and run as ONE matmul; siblings return
   slices of the cached result (matched by input identity, so the model code is untouched).
   Cuts Qwen3.5-4B from 249 to ~129 matmul dispatches per step and removes the N=32
   launch-latency-bound in_proj_a/b calls. Helps the M=1 MLX path too.

Prototype keeps both weight layouts resident (2x weight memory); a production version would
add an M=1 path on the repacked layout and drop the original.

Run:  uv run python -m skinny_mlx.patch   (M=8 verify logits vs unpatched model and an fp32 reference)
"""

import mlx.core as mx
import mlx.nn as nn

from skinny_mlx.qmm import repack_for_skinny, skinny_qmm

SKINNY_RANGE = (6, 16)
FUSE_GROUPS = [
    ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),  # Gated DeltaNet mixer
    ("q_proj", "k_proj", "v_proj"),                          # full attention
    ("gate_proj", "up_proj"),                                # MLP
]


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


class _FusedGroup:
    """Concatenated weights of same-input projections + a one-entry result cache."""

    def __init__(self, qls: list[nn.QuantizedLinear]):
        self.group_size, self.bits = qls[0].group_size, qls[0].bits
        self.w = mx.concatenate([q.weight for q in qls])
        self.s = mx.concatenate([q.scales for q in qls])
        self.b = mx.concatenate([q.biases for q in qls])
        self.rw = repack_for_skinny(self.w)
        sizes = [q.weight.shape[0] for q in qls]
        self.bounds = [(sum(sizes[:i]), sum(sizes[: i + 1])) for i in range(len(sizes))]
        self.x = self.y = None

    def output(self, x: mx.array) -> mx.array:
        if self.x is not x:  # strong ref held, so identity can't be recycled
            self.x, self.y = x, _qmm(x, self.w, self.s, self.b, self.rw, self.group_size, self.bits)
        return self.y


class FusedSliceLinear(nn.Module):
    def __init__(self, group: _FusedGroup, idx: int):
        super().__init__()
        self._group, (self._lo, self._hi) = group, group.bounds[idx]

    def __call__(self, x: mx.array) -> mx.array:
        return self._group.output(x)[..., self._lo : self._hi]


def _ok(m) -> bool:
    return isinstance(m, nn.QuantizedLinear) and m.bits == 4 and m.group_size == 64 and "bias" not in m


def _patch_tied_head(emb: nn.QuantizedEmbedding) -> None:
    rw = repack_for_skinny(emb.weight)
    mx.eval(rw)
    emb.as_linear = lambda x: _qmm(x, emb.weight, emb.scales, emb.biases, rw, emb.group_size, emb.bits)


def patch_model(model: nn.Module, fuse: bool = True) -> dict:
    """Patch in place. Returns counts of fused groups, single swaps, and patched heads."""
    stats = {"fused_groups": 0, "single": 0, "heads": 0}
    fused_ids = set()
    if fuse:
        for _, parent in model.named_modules():
            children = dict(parent.children()) if isinstance(parent, nn.Module) else {}
            for names in FUSE_GROUPS:
                if all(_ok(children.get(n)) for n in names):
                    group = _FusedGroup([children[n] for n in names])
                    mx.eval(group.rw, group.w, group.s, group.b)
                    for i, n in enumerate(names):
                        fused_ids.add(id(children[n]))
                        setattr(parent, n, FusedSliceLinear(group, i))
                    stats["fused_groups"] += 1
    for _, parent in model.named_modules():
        for n, child in dict(parent.children()).items():
            if _ok(child) and id(child) not in fused_ids:
                setattr(parent, n, SkinnyQuantizedLinear(child))
                stats["single"] += 1
    mx.eval(model.parameters())
    for _, m in model.named_modules():
        if isinstance(m, nn.QuantizedEmbedding) and m.bits == 4 and m.group_size == 64:
            _patch_tied_head(m)
            stats["heads"] += 1
    return stats


def main() -> None:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    ids = mx.array([tok.encode("The quick brown fox jumps over the lazy dog. In computer science, a cache is")])
    draft = mx.array([tok.encode(" a hardware or software component that stores data")[:8]])

    def verify_logits():
        cache = make_prompt_cache(model)
        model(ids, cache=cache)
        return model(draft, cache=cache).astype(mx.float32)  # M = 8 verify step

    ref = verify_logits()
    mx.eval(ref)
    print("patch:", patch_model(model))
    got = verify_logits()
    mx.eval(got)
    rel = (mx.abs(ref - got).max() / mx.abs(ref).max()).item()
    top_ref = mx.argmax(ref, -1)
    agree = (top_ref == mx.argmax(got, -1)).mean().item()
    # where argmax differs, is it a near-tie? (logit gap between the two candidates under the reference)
    gap = mx.max(ref, -1) - mx.take_along_axis(ref, mx.argmax(got, -1)[..., None], -1)[..., 0]
    print(f"M=8 verify logits: max rel diff {rel:.2e}, argmax agreement {agree:.0%}, "
          f"max ref-logit gap at disagreements {gap.max().item():.3f}")


if __name__ == "__main__":
    main()
