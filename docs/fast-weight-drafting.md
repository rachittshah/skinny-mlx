# Fast-Weight Drafting: drafting by reading the target's own recurrent memory

**Status: design (no GPU used). Not yet implemented or measured.** Numbers marked *measured* come from
earlier runs in this repo on an M4 Max; everything else is a model or a hypothesis with a kill test.

## The one-line idea

On hybrid models (Qwen3.5/3.6/3.8, Qwen3-Next: 24 of 32 layers are Gated DeltaNet), the target already
keeps its entire long-range context as a **25 MB fast-weight memory** (`S_ℓ ∈ R^{32×128×128}` per GDN
layer). Every existing drafter ignores it and rebuilds context on its own (own KV over past features).
A Fast-Weight Drafter (FWD) instead **queries the target's live states** — `r = S_ℓ · q` — so its
context cost is O(1) in sequence length, its memory is exactly the target's, and its reads are
cache-sized.

## Why this is the right object (computed from the model config)

| quantity (Qwen3.5-4B) | value |
|---|---:|
| GDN layers / heads / head dims | 24 / 32 value heads / 128×128 |
| recurrent state, all GDN layers | **50.3 MB fp32, 25.2 MB bf16** |
| MACs to read all 24 states for one draft row | **12.6 M** |
| MACs for one target forward row | ~3.5 G (≈280× more) |
| weight bytes per target step (*measured*) | 2.4 GB → 8.7 ms |

The states are written by the target with the delta rule, which is trained as an associative memory:
`S_t = g_t·S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, read as `o = S·q`. Querying it with the right key
retrieves what the target "remembers" about the context — the same information an EAGLE/DFlash drafter
spends a context-length-proportional attention pass to reconstruct from its own feature cache.

## Mechanism

```
target verify pass (unchanged)  ──► accepted prefix, last hidden h_t, live states {S_ℓ} (already in memory)
                                         │ zero-copy (unified memory), no snapshot needed (MLX arrays immutable)
FWD block drafter (K positions, one pass, block-parallel like DFlash):
   x_i = embed(mask or prev draft) + proj(h_t)
   for each FWD layer (2–4 layers):
       local:  causal self-attention over the ≤16 draft positions only        (tiny, context-free)
       memory: r_i = Σ_{ℓ∈L_sel} S_ℓ · (W_q^ℓ x_i)                             (reads target fast weights)
       optional copy-on-write: S̃_ℓ ← delta-update(S_ℓ, k(x_i), v(x_i)) for i<j  (draft tokens "write" too)
       MLP
   logits_i = target head over a pruned vocab (verify still uses the full head → exact)
```

- **No drafter context state at all** → nothing to roll back on rejection, nothing to prefill, no
  long-context degradation (dflash-mlx reports Qwen3.5-9B 4.37× @1k → 2.22× @8k with a KV-based drafter).
- **Copy-on-write writes** let draft position j see draft tokens < j through the same memory the target
  will use, without touching the target's states (discarded each cycle).
- **Weights shared with the target** (embeddings, pruned head); FWD's own weights are a few small layers.

## Cost model (measured costs + stated assumptions)

Throughput = τ / (C_draft + C_verify(K)). Measured on M4 Max, Qwen3.5-4B 4-bit, skinny-mlx verify:
C_verify(8) = 14.8 ms (1.70× one step), C_verify(16) ≈ 29 ms. llama.cpp baseline 74.6 tok/s.
τ uses per-position acceptance α (expected run Σα^i, +1 bonus).

| drafter | C_draft | ctx 1k: α → τ(K=8) | tok/s | × llama.cpp | ctx 8k |
|---|---:|---|---:|---:|---|
| DFlash bf16 (1.27 GB) | ~4.5 ms | 0.87 → 5.5 | 285 | 3.8× | decays (published 2.2× on 9B) |
| DFlash 4-bit | ~1.3 ms | 0.86 → 5.4 | 335 | 4.5× | decays |
| **FWD 4-bit (≈0.1 GB weights + 25 MB states)** | **~0.4 ms** | 0.85 → 5.3 (hyp.) | **348** | **4.7×** | **flat (hyp.)** |
| FWD, α=0.90 (hyp.: exact target memory) | ~0.4 ms | 0.90 → 6.1 | 400 | 5.4× | flat |

The short-context gain over a 4-bit DFlash is modest (draft cost is already small); the claim is the
**long-context column** and the possibility that reading the target's exact memory raises α.

## Supporting inventions (same design, independently useful)

**1. Reversible gated-delta rollback.** The GDN update is exactly invertible: with ‖k‖=1 (Qwen3.5
RMS-normalizes k) and β∈(0,1), Sherman–Morrison gives `(I − βkkᵀ)⁻¹ = I + (β/(1−β))kkᵀ`, so
`S_{t−1} = (S_t − β v kᵀ)(I + (β/(1−β)) kkᵀ) / g_t`. Rejected draft rows can be *un-applied* from the
post-verify state in O(rejected) time with O(1) extra memory — no per-token snapshots (#1730) and no
replay of accepted rows (tape-replay). Per cycle choose min(replay accepted, unroll rejected). Guard:
fall back to replay when `g(1−β)` is small (division amplifies rounding).

**2. Knee-budgeted verify shapes.** Treat the verify pass as a fixed-cost bus of R rows, where R is
the *measured* knee of this device's cost curve (≈8 on M4 with skinny-mlx, larger on M5 TensorOps),
and choose block depth per cycle by maximizing E[accepted]/C(rows) from the drafter's per-position
confidences. Extra rows under the knee go to the next-best alternatives as batch rows; each carries a
copy of the 25 MB state, so this works on hybrids without tree-recurrence kernels.

## Prior art and what is new

| work | what it reads to draft | context cost | relation |
|---|---|---|---|
| EAGLE-1/2/3 | target hidden features (own KV over them) | O(ctx) | different memory |
| DFlash / DDTree | target multi-layer hidden features (own KV) | O(ctx) | block-parallel drafting reused here |
| GLIDE (2024) | **target's KV cache** via cross-attention | O(ctx) | closest: reuse target memory — for attention models |
| Mamba drafters (2025) | its **own** SSM state | O(1) | O(1), but not the target's memory |
| mlx-lm #1730, TreeWY, Tree-Scan | target GDN states, for **rollback / tree verify** | — | treat state as a cost, not a drafting signal |
| **FWD (this)** | **target's live GDN fast-weight states** | **O(1)** | hybrid-native analog of GLIDE, cache-sized |

I found no prior use of a target's linear-attention / delta-rule state as the drafter's context memory,
nor of Sherman–Morrison reversal of the delta rule for speculative rollback, in the 2026 sweep
(`lit-2026.md`). That is a literature-search claim, not a proof — re-check before publishing.

## Kill criteria (cheap tests first)

1. **Probe (no training of a full drafter):** fit a linear map from `{S_ℓ·W_q x}` reads + `h_t` to
   the target's token at t+2…t+4 on recorded states. If FWD-style reads add < 2 points of top-1 over
   `h_t` alone, the memory isn't carrying usable draft signal → stop.
2. **Drafter:** train a 2-layer FWD (distillation on target greedy outputs). Kill if α@1k < 0.80, or
   α@8k − α@1k < (DFlash's α@8k − α@1k), i.e. no long-context advantage.
3. **Reversal stability:** on recorded (g, β) traces, measure fraction of tokens with `g(1−β) < 1e-3`
   and reconstruction error of `S_{t−1}`. Kill reversal if fallback is needed on > 20% of cycles.
4. **System:** end-to-end on `evals/` (paired vs llama.cpp, lossless gate) at 1k and 8k context.
