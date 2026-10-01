"""Greedy-exact speculative decoding for mlx_lm hybrid models (Qwen3.5) with pluggable drafters.

Each step feeds [last emitted token] + K draft tokens (M = K+1) in one forward, accepts the
longest draft prefix matching the target's own argmax, emits one bonus token, and rolls the
hybrid cache back to the accepted prefix (skinny_mlx.rollback). Output equals plain greedy
decoding up to fp near-ties between batched and single-token kernels (checked in main()).

Drafters implement begin / draft / update and see the target's final hidden states, so a
native MTP head (skinny_mlx.mtp) and model-free prompt lookup share one loop.

Run:  uv run python -m skinny_mlx.spec [--skinny]
"""

import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, make_prompt_cache

from skinny_mlx.rollback import HybridRollback


def forward(model: nn.Module, ids, cache) -> tuple[mx.array, mx.array]:
    """(final normed hidden [1,T,H], logits [1,T,V]) for a tied-embedding mlx_lm Qwen3.5 model.
    ids may be a list or a lazy mx.array (lets draft tokens flow into verify without a host sync)."""
    inner = model.language_model.model
    h = inner(ids[None] if isinstance(ids, mx.array) else mx.array([ids]), cache)
    return h, inner.embed_tokens.as_linear(h)


class PromptLookupDrafter:
    """Model-free: propose what followed the latest earlier occurrence of the trailing n-gram."""

    def __init__(self, ngram: int = 3):
        self.ngram = ngram

    def begin(self, prompt, h, first):
        self.ctx = list(prompt) + [first]

    def draft(self, k: int) -> mx.array:
        ctx = self.ctx
        for n in range(self.ngram, 0, -1):
            tail = ctx[-n:]
            for start in range(len(ctx) - n - 1, -1, -1):
                if ctx[start : start + n] == tail:
                    return mx.array(ctx[start + n : start + n + k], dtype=mx.int32)
        return mx.array([], dtype=mx.int32)

    def update(self, h_kept, new_tokens):
        self.ctx += new_tokens


class MTPDrafter:
    """Chains the model's native MTP head K times; keeps its own KV cache in sync with accepted tokens."""

    def __init__(self, model: nn.Module, head: nn.Module, vocab: int | None = 32768):
        """vocab: score drafts over token ids [0, vocab) only. BPE ids are roughly frequency-ordered,
        so a prefix keeps most mass at a fraction of the tied head's 248k-row read; verify still
        uses the full head, so output is unchanged — only acceptance can drop."""
        self.embed = model.language_model.model.embed_tokens
        self.head = head
        e = self.embed
        self._head_w = (e.weight[:vocab], e.scales[:vocab], e.biases[:vocab]) if vocab else None
        if self._head_w:
            mx.eval(self._head_w)

    def _logits(self, h: mx.array) -> mx.array:
        if self._head_w is None:
            return self.embed.as_linear(h)
        w, s, b = self._head_w
        return mx.quantized_matmul(h, w, s, b, transpose=True, group_size=self.embed.group_size, bits=self.embed.bits)

    def _step(self, tokens: mx.array, hidden: mx.array) -> mx.array:
        return self.head(self.embed(tokens), hidden, self.cache)

    def begin(self, prompt, h, first):
        self.cache = KVCache()
        self.h_last = self._step(mx.array([list(prompt[1:]) + [first]]), h)[:, -1:]
        self.n_spec = 0

    def draft(self, k: int) -> mx.array:
        toks, h = [], self.h_last
        for i in range(k):  # lazy: the chain feeds verify directly, no host sync in between
            t = mx.argmax(self._logits(h)[:, -1], axis=-1).astype(mx.int32)
            toks.append(t)
            if i < k - 1:
                h = self._step(t[None], h)
        self.n_spec = k - 1
        return mx.concatenate(toks)

    def update(self, h_kept, new_tokens):
        self.cache.trim(self.n_spec)  # drop the speculative chain entries
        self.h_last = self._step(mx.array([new_tokens]), h_kept)[:, -1:]
        self.n_spec = 0


def greedy(model: nn.Module, prompt: list[int], max_tokens: int) -> list[int]:
    cache = make_prompt_cache(model)
    y = mx.argmax(model(mx.array([prompt]), cache=cache)[0, -1])
    out = []
    for _ in range(max_tokens):
        mx.async_eval(y)
        nxt = mx.argmax(model(y.reshape(1, 1), cache=cache)[0, -1])
        out.append(y.item())
        y = nxt
    return out


def speculative(model: nn.Module, prompt: list[int], max_tokens: int, drafter, k: int,
                stats: dict | None = None, profile: bool = False) -> list[int]:
    """profile=True syncs at phase boundaries and adds per-phase ms/step to stats (slower overall)."""
    phase = {"draft": 0.0, "verify": 0.0, "rollback": 0.0, "drafter update": 0.0}
    clock = [time.perf_counter()]

    def mark(name, *arrays):
        if profile:
            mx.eval(*arrays)
            now = time.perf_counter()
            phase[name] += now - clock[0]
            clock[0] = now

    cache = make_prompt_cache(model)
    rb = HybridRollback.for_model(model)
    h, logits = forward(model, prompt, cache)
    out = [mx.argmax(logits[0, -1]).item()]
    drafter.begin(prompt, h, out[0])
    steps = 0
    while len(out) < max_tokens:
        if profile:
            clock[0] = time.perf_counter()
        draft_arr = drafter.draft(k)
        mark("draft", draft_arr)
        x_arr = mx.concatenate([mx.array([out[-1]], dtype=mx.int32), draft_arr])
        rb.record()
        h, logits = forward(model, x_arr, cache)
        pred_arr = mx.argmax(logits[0], axis=-1)
        mx.eval(pred_arr, draft_arr)  # the step's one host sync
        pred, draft = pred_arr.tolist(), draft_arr.tolist()
        x = [out[-1]] + draft
        mark("verify")
        n = 0
        while n < len(draft) and draft[n] == pred[n]:
            n += 1
        rb.rollback(cache, fed=len(x), keep=n + 1)
        mark("rollback", *[c for layer_cache in cache for c in getattr(layer_cache, "cache", [])])
        new = draft[:n] + [pred[n]]
        drafter.update(h[:, : n + 1], x[1 : n + 1] + [pred[n]])
        mark("drafter update", *([drafter.h_last] if hasattr(drafter, "h_last") else []))
        out += new
        steps += 1
    if stats is not None:
        stats.update(steps=steps, tokens_per_step=len(out) / max(steps, 1))
        if profile:
            stats["ms_per_step"] = {k: round(v / max(steps, 1) * 1e3, 2) for k, v in phase.items()}
    return out[:max_tokens]


PROMPTS = {
    "code-rewrite": "Rewrite this Python class adding type hints and docstrings, keep all logic:\n\n"
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n"
    "        self.items.append(item)\n    def pop(self):\n        if not self.items:\n"
    "            raise IndexError('pop from empty stack')\n        return self.items.pop()\n"
    "    def peek(self):\n        if not self.items:\n            raise IndexError('peek from empty stack')\n"
    "        return self.items[-1]\n    def __len__(self):\n        return len(self.items)\n",
    "explain": "Explain how a CPU cache hierarchy works, including L1, L2, L3 and cache coherence.",
}


def main() -> None:
    """Interleaved end-to-end bench: every config runs once per round, best of ROUNDS kept,
    so background CPU/GPU load hits all configs alike. Exactness checked against greedy."""
    from mlx_lm import load

    model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    if "--skinny" in sys.argv:
        from skinny_mlx.patch import patch_model

        patch_model(model)
    from skinny_mlx.mtp import load_mtp_head

    head = load_mtp_head(model)
    N, ROUNDS = 200, 5
    configs = [("greedy", None, 0)] + [("mtp32k", MTPDrafter(model, head), k) for k in (1, 2, 3, 4, 5, 6)]
    for name, text in PROMPTS.items():
        prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                         enable_thinking=False)
        greedy(model, prompt, 8)  # warm-up
        ref = greedy(model, prompt, N)
        best, stats = {}, {}
        for _ in range(ROUNDS):
            for cname, drafter, k in configs:
                st = {}
                t = time.perf_counter()
                got = greedy(model, prompt, N) if drafter is None else speculative(model, prompt, N, drafter, k, st)
                dt = time.perf_counter() - t
                key = f"{cname} K={k}" if drafter else cname
                best[key] = min(best.get(key, 1e9), dt)
                stats[key] = (st.get("tokens_per_step", 1.0), next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), N))
        base = best["greedy"]
        print(f"[{name}] best of {ROUNDS}, {N} tokens")
        for key, dt in best.items():
            tps, same = stats[key]
            print(f"  {key:12s} {N / dt:6.1f} tok/s ({base / dt:4.2f}x) | {tps:.2f} tok/step | identical first {same}/{N}")


if __name__ == "__main__":
    main()
