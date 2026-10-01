"""Native multi-token-prediction (MTP) draft head for Qwen3.5, which mlx_lm strips at conversion.

Qwen3.5 checkpoints ship `mtp.*`: two RMSNorms, an fc fusing [norm(embed(next token)),
norm(target hidden)] -> hidden, one full-attention decoder layer, and a final norm, scored by the
tied embedding head. We fetch only those tensors from the original HF checkpoint with HTTP range
requests (~240 MB instead of the ~5 GB shards), reuse mlx_lm's own full-attention DecoderLayer,
and apply the same +1 zero-centered norm shift mlx_lm's sanitize applies to the trunk.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import requests
from huggingface_hub import hf_hub_download, hf_hub_url
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer

CACHE_DIR = Path.home() / ".cache" / "skinny-mlx"
_DT = {"BF16": (np.uint16, mx.bfloat16), "F16": (np.float16, None), "F32": (np.float32, None)}


def fetch_mtp_weights(repo: str = "Qwen/Qwen3.5-4B") -> dict[str, mx.array]:
    """Range-fetch the mtp.* tensors from a sharded safetensors checkpoint (cached locally)."""
    out_path = CACHE_DIR / (repo.replace("/", "--") + "-mtp.safetensors")
    if out_path.exists():
        return mx.load(str(out_path))
    index = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))["weight_map"]
    weights = {}
    for shard in sorted({v for k, v in index.items() if k.startswith("mtp.")}):
        url = hf_hub_url(repo, shard)
        n = int.from_bytes(requests.get(url, headers={"Range": "bytes=0-7"}).content, "little")
        header = requests.get(url, headers={"Range": f"bytes=8-{7 + n}"}).json()
        names = [k for k in header if k.startswith("mtp.")]
        lo = min(header[k]["data_offsets"][0] for k in names)
        hi = max(header[k]["data_offsets"][1] for k in names)
        blob = requests.get(url, headers={"Range": f"bytes={8 + n + lo}-{8 + n + hi - 1}"}).content
        assert len(blob) == hi - lo, (shard, len(blob), hi - lo)
        for k in names:
            a, b = header[k]["data_offsets"]
            np_dt, mx_view = _DT[header[k]["dtype"]]
            arr = mx.array(np.frombuffer(blob[a - lo : b - lo], dtype=np_dt).reshape(header[k]["shape"]))
            weights[k] = arr.view(mx_view) if mx_view is not None else arr
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_path), weights)
    return weights


class MTPHead(nn.Module):
    """h_next = layer(fc([norm_e(embed(tok_{t+1})), norm_h(h_t)])); logits(norm(h_next)) predicts tok_{t+2}."""

    def __init__(self, args):
        super().__init__()
        H, eps = args.hidden_size, args.rms_norm_eps
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=eps)
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layer = DecoderLayer(args, layer_idx=args.full_attention_interval - 1)  # full attention
        assert not self.layer.is_linear
        self.norm = nn.RMSNorm(H, eps=eps)

    def __call__(self, emb: mx.array, hidden: mx.array, cache: KVCache) -> mx.array:
        x = self.fc(mx.concatenate([self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(hidden)], -1))
        mask = "causal" if x.shape[1] > 1 else None
        return self.norm(self.layer(x, mask=mask, cache=cache))


def load_mtp_head(model: nn.Module, repo: str = "Qwen/Qwen3.5-4B", bits: int | None = 4) -> MTPHead:
    """Build the head for an mlx_lm Qwen3.5 model; optionally quantize it like the trunk."""
    args = model.language_model.args
    head = MTPHead(args)
    w = {}
    for k, v in fetch_mtp_weights(repo).items():
        k = k.removeprefix("mtp.").replace("layers.0.", "layer.")
        if v.ndim == 1:  # every 1-D mtp tensor is a zero-centered RMSNorm gamma (incl. pre_fc_norm_*)
            v = v + 1.0
        w[k] = v
    head.load_weights(list(w.items()))
    if bits:
        nn.quantize(head, group_size=64, bits=bits)
    mx.eval(head.parameters())
    return head


def teacher_forced_accuracy(model, head, ids: list[int], prenorm: bool) -> float:
    """Top-1 agreement of the MTP head's t+2 prediction with the target's own greedy tokens."""
    inner = model.language_model.model
    norm = inner.norm
    if prenorm:
        inner.norm = lambda x: x
    h = inner(mx.array([ids]))
    inner.norm = norm
    out = head(inner.embed_tokens(mx.array([ids[1:]])), h[:, :-1], KVCache())
    pred = mx.argmax(inner.embed_tokens.as_linear(out), -1)[0, :-1]
    return (pred == mx.array(ids[2:])).mean().item()


def main() -> None:
    """Sanity check: teacher-forced top-1 of the MTP head's t+2 prediction vs the target's greedy tokens."""
    from mlx_lm import load

    from skinny_mlx.spec import greedy

    model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    prompt = tok.apply_chat_template([{"role": "user", "content": "Explain how a CPU cache hierarchy works."}],
                                     add_generation_prompt=True, enable_thinking=False)
    ids = prompt + greedy(model, prompt, 160)
    for bits in (None, 8, 4):
        head = load_mtp_head(model, bits=bits)
        acc = teacher_forced_accuracy(model, head, ids, prenorm=False)
        print(f"MTP head {'bf16' if bits is None else f'{bits}-bit'}: teacher-forced top-1 (t+2) {acc:.1%}")


if __name__ == "__main__":
    main()
