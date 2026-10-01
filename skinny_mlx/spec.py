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


def forward(model: nn.Module, ids: list[int], cache) -> tuple[mx.array, mx.array]:
    """(final normed hidden [1,T,H], logits [1,T,V]) for a tied-embedding mlx_lm Qwen3.5 model."""
    inner = model.language_model.model
    h = inner(mx.array([ids]), cache)
    return h, inner.embed_tokens.as_linear(h)


class PromptLookupDrafter:
    """Model-free: propose what followed the latest earlier occurrence of the trailing n-gram."""

    def __init__(self, ngram: int = 3):
        self.ngram = ngram

    def begin(self, prompt, h, first):
        self.ctx = list(prompt) + [first]

    def draft(self, k: int) -> list[int]:
        ctx = self.ctx
        for n in range(self.ngram, 0, -1):
            tail = ctx[-n:]
            for start in range(len(ctx) - n - 1, -1, -1):
                if ctx[start : start + n] == tail:
                    return ctx[start + n : start + n + k]
        return []

    def update(self, h_kept, new_tokens):
        self.ctx += new_tokens


class MTPDrafter:
    """Chains the model's native MTP head K times; keeps its own KV cache in sync with accepted tokens."""

    def __init__(self, model: nn.Module, head: nn.Module):
        self.embed = model.language_model.model.embed_tokens
        self.head = head

    def _step(self, tokens: mx.array, hidden: mx.array) -> mx.array:
        return self.head(self.embed(tokens), hidden, self.cache)

    def begin(self, prompt, h, first):
        self.cache = KVCache()
        self.h_last = self._step(mx.array([list(prompt[1:]) + [first]]), h)[:, -1:]
        self.n_spec = 0

    def draft(self, k: int) -> list[int]:
        toks, h = [], self.h_last
        for i in range(k):  # built lazily: one host sync for the whole chain
            t = mx.argmax(self.embed.as_linear(h)[:, -1], axis=-1)
            toks.append(t)
            if i < k - 1:
                h = self._step(t[None], h)
        self.n_spec = k - 1
        return mx.concatenate(toks).tolist()

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
                stats: dict | None = None) -> list[int]:
    cache = make_prompt_cache(model)
    rb = HybridRollback.for_model(model)
    h, logits = forward(model, prompt, cache)
    out = [mx.argmax(logits[0, -1]).item()]
    drafter.begin(prompt, h, out[0])
    steps = 0
    while len(out) < max_tokens:
        draft = drafter.draft(k)
        x = [out[-1]] + draft
        rb.record()
        h, logits = forward(model, x, cache)
        pred = mx.argmax(logits[0], axis=-1).tolist()
        n = 0
        while n < len(draft) and draft[n] == pred[n]:
            n += 1
        rb.rollback(cache, fed=len(x), keep=n + 1)
        new = draft[:n] + [pred[n]]
        drafter.update(h[:, : n + 1], x[1 : n + 1] + [pred[n]])
        out += new
        steps += 1
    if stats is not None:
        stats.update(steps=steps, tokens_per_step=len(out) / max(steps, 1))
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
    from mlx_lm import load

    model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    if "--skinny" in sys.argv:
        from skinny_mlx.patch import patch_model

        patch_model(model)
    from skinny_mlx.mtp import load_mtp_head

    head = load_mtp_head(model)
    N = 200
    for name, text in PROMPTS.items():
        prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                         enable_thinking=False)
        greedy(model, prompt, 8)  # warm-up
        t = time.perf_counter(); ref = greedy(model, prompt, N); tg = time.perf_counter() - t
        print(f"[{name}] greedy {N / tg:6.1f} tok/s")
        runs = [("lookup", PromptLookupDrafter(), 3)] + [("mtp", MTPDrafter(model, head), k) for k in (1, 2, 3, 4, 6)]
        for dname, drafter, k in runs:
            st = {}
            t = time.perf_counter(); got = speculative(model, prompt, N, drafter, k, st); ts = time.perf_counter() - t
            same = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), N)
            margin = ""
            if same < N:  # near-tie? top-2 gap of a fresh full-prefix forward at the divergence point
                lg = model(mx.array([prompt + ref[:same]]))[0, -1].astype(mx.float32)
                top = mx.sort(lg)[-2:].tolist()
                margin = f" (top1-top2 gap there {top[1] - top[0]:.3f})"
            print(f"  {dname:6s} K={k}: {N / ts:6.1f} tok/s ({tg / ts:4.2f}x) | "
                  f"{st['tokens_per_step']:.2f} tok/step | identical first {same}/{N}{margin}")


if __name__ == "__main__":
    main()
