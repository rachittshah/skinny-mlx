"""Greedy-exact speculative decoding for mlx_lm hybrid models (Qwen3.5) with pluggable drafters.

Each step feeds [last emitted token] + K draft tokens (M = K+1) in one forward, accepts the
longest draft prefix matching the target's own argmax, emits one bonus token, and rolls the
hybrid cache back to the accepted prefix (skinny_mlx.rollback). Output equals plain greedy
decoding up to fp near-ties between batched and single-token kernels.

Run:  uv run python -m skinny_mlx.spec    (exactness + speed vs plain greedy, prompt-lookup drafter)
"""

import time
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache

from skinny_mlx.rollback import HybridRollback

Drafter = Callable[[list[int], int], list[int]]


def prompt_lookup_drafter(ngram: int = 3) -> Drafter:
    """Model-free drafter: find the latest earlier occurrence of the trailing n-gram, propose what followed."""

    def draft(ctx: list[int], k: int) -> list[int]:
        for n in range(ngram, 0, -1):
            tail = ctx[-n:]
            for start in range(len(ctx) - n - 1, -1, -1):
                if ctx[start : start + n] == tail:
                    return ctx[start + n : start + n + k]
        return []

    return draft


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


def speculative(model: nn.Module, prompt: list[int], max_tokens: int, drafter: Drafter, k: int,
                stats: dict | None = None) -> list[int]:
    cache = make_prompt_cache(model)
    rb = HybridRollback(model)
    out = [mx.argmax(model(mx.array([prompt]), cache=cache)[0, -1]).item()]
    steps = accepted = 0
    while len(out) < max_tokens:
        draft = drafter(prompt + out, k)
        x = [out[-1]] + draft
        rb.record()
        pred = mx.argmax(model(mx.array([x]), cache=cache)[0], axis=-1).tolist()
        n = 0
        while n < len(draft) and draft[n] == pred[n]:
            n += 1
        rb.rollback(cache, fed=len(x), keep=n + 1)
        out += draft[:n] + [pred[n]]
        steps += 1
        accepted += n
    if stats is not None:
        stats.update(steps=steps, tokens_per_step=len(out) / max(steps, 1), accepted_drafts=accepted)
    return out[:max_tokens]


def main() -> None:
    import sys

    from mlx_lm import load

    model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    if "--skinny" in sys.argv:
        from skinny_mlx.patch import patch_model

        patch_model(model)
    msgs = [{"role": "user", "content": "Rewrite this Python class adding type hints and docstrings, keep all logic:\n\n"
             "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n"
             "        self.items.append(item)\n    def pop(self):\n        if not self.items:\n"
             "            raise IndexError('pop from empty stack')\n        return self.items.pop()\n"
             "    def peek(self):\n        if not self.items:\n            raise IndexError('peek from empty stack')\n"
             "        return self.items[-1]\n    def __len__(self):\n        return len(self.items)\n"}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    N = 200
    greedy(model, prompt, 8)  # warm-up / compile
    t = time.perf_counter(); ref = greedy(model, prompt, N); tg = time.perf_counter() - t
    for k in [3]:
        st = {}
        t = time.perf_counter(); got = speculative(model, prompt, N, prompt_lookup_drafter(), k, st); ts = time.perf_counter() - t
        same = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), N)
        margin = ""
        if same < N:  # near-tie? top-2 gap of a fresh full-prefix forward at the divergence point
            lg = model(mx.array([prompt + ref[:same]]))[0, -1].astype(mx.float32)
            top = mx.sort(lg)[-2:].tolist()
            margin = f" (ref top1-top2 logit gap there: {top[1] - top[0]:.3f})"
        print(f"K={k}: greedy {N / tg:6.1f} tok/s | spec {N / ts:6.1f} tok/s ({tg / ts:4.2f}x) | "
              f"{st['tokens_per_step']:.2f} tok/step | identical first {same}/{N} tokens{margin}")


if __name__ == "__main__":
    main()
