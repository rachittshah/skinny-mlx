"""Replay-free rollback for hybrid (Gated DeltaNet + attention) caches in mlx_lm.

Speculative verify feeds M tokens; only a prefix of n is accepted. Attention KV caches just trim.
GDN layers carry a recurrent state and a conv window that can't be trimmed. Instead of storing a
state per token (B*T*H*Dv*Dk per layer, mlx-lm #1730) or re-running the whole model on the
accepted prefix, we record each GDN layer's recurrence inputs during verify (q,k,v,a,b — a few
KB per token) plus its pre-verify state, and on rollback rerun ONLY the recurrence kernel over
the accepted n tokens. MLX arrays are immutable, so "snapshotting" the pre-verify state is just
keeping a reference. Conv state is re-sliced from the recorded conv input.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5

_orig_gdu = qwen3_5.gated_delta_update


class HybridRollback:
    @classmethod
    def for_model(cls, model: nn.Module) -> "HybridRollback":
        """One instance per model: hooks are installed once, not stacked per generation."""
        if getattr(model, "_skinny_rollback", None) is None:
            model._skinny_rollback = cls(model)
        return model._skinny_rollback

    def __init__(self, model: nn.Module):
        self.layers = model.layers
        self.gdn_idx = [i for i, l in enumerate(self.layers) if getattr(l, "is_linear", False)]
        self.recording = False
        self.calls: list[tuple] = []
        self.convs: list[mx.array] = []
        qwen3_5.gated_delta_update = self._gdu
        for i in self.gdn_idx:
            conv = self.layers[i].linear_attn.conv1d
            cls = type(conv)
            conv.__class__ = type(cls.__name__ + "Rec", (cls,), {"__call__": self._conv_wrapper(cls.__call__)})

    def _gdu(self, q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
        if self.recording:
            self.calls.append((q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel))
        return _orig_gdu(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)

    def _conv_wrapper(self, fn):
        rb = self

        def call(mod, x):
            if rb.recording:
                rb.convs.append(x)
            return fn(mod, x)

        return call

    def record(self) -> None:
        """Call right before the verify forward."""
        self.calls, self.convs, self.recording = [], [], True

    def rollback(self, cache: list, fed: int, keep: int) -> None:
        """After a verify forward of `fed` tokens, keep only the first `keep` of them in `cache`."""
        self.recording = False
        if keep == fed:
            return
        assert 1 <= keep < fed and len(self.calls) == len(self.gdn_idx) == len(self.convs)
        for j, li in enumerate(self.gdn_idx):
            q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel = self.calls[j]
            _, st = _orig_gdu(q[:, :keep], k[:, :keep], v[:, :keep], a[:, :keep], b[:, :keep],
                              A_log, dt_bias, state, None if mask is None else mask[:, :keep], use_kernel)
            cache[li][1] = st
            n_keep = self.convs[j].shape[1] - fed  # conv window rows (kernel_size - 1)
            cache[li][0] = mx.contiguous(self.convs[j][:, keep : keep + n_keep])
        for li, layer in enumerate(self.layers):
            if li not in self.gdn_idx:
                cache[li].trim(fed - keep)
