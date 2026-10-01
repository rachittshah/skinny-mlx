# FlatSpec: making MLX the fastest local decoder on Apple Silicon

**Thesis.** Single-token decode is already within ~1.3× of the memory-bandwidth wall in MLX, so a
faster *kernel* alone can't deliver multiples. The only way to multiply local decode speed is to
produce several tokens per weight read — speculative decoding — and on Metal that has failed
everywhere (mlx-lm, llama.cpp, EAGLE-3 ports; see `lit-2026.md`) for one reason: **verifying K
tokens costs ~K/3–K/2 tokens, not ~1**. FlatSpec removes that cost and then spends the freed
budget on the model's own native draft head.

## Components

| # | component | status | evidence |
|---|---|---|---|
| F1 | **Fragment-native skinny qmm** — 8×8 simdgroup MMA, split-K, weights relaid at load into MMA lane order (one coalesced 256 B block per simdgroup per group), scale/bias folded out of the inner loop, nibble pairs via the `0x6400` half trick | built | M=8: 1.82× MLX on the model's real matmuls (bf16, fair); flat cost M=1..8 |
| F2 | **Horizontal projection fusion** — same-input projections (GDN qkv/z/b/a, attn q/k/v, MLP gate/up) concatenated at load: 249 → 129 dispatches | built | needed for F1 on N=32 launch-bound layers |
| F3 | **Native MTP drafting** — Qwen3.5 ships a 1-layer MTP head (`mtp.fc` + one attention layer) that mlx-lm strips; chain it K times | next | head verified present in `Qwen/Qwen3.5-4B` |
| F4 | **Pruned-vocab draft head** — draft steps score only the top-V' frequent tokens (skinny/qmv over a row subset of the tied head); verify still uses the full head, so output is exact | next | full head = 0.32 GB/token at 4-bit ≈ 1 ms per draft step |
| F5 | **Replay-free hybrid rollback** — during verify, keep each GDN layer's recurrence inputs (q,k,v,g,β; KBs) and conv inputs; on partial accept, rerun only the recurrence kernel over the accepted prefix from the pre-verify state | next | avoids #1730's per-token state snapshots and any full-model replay |
| F6 | Greedy-exact acceptance first; rejection sampling (#1709-style) for T>0 | next | — |

## Measured so far (M4 Max 32-core GPU, Qwen3.5-4B 4-bit, `bench/step_cost_curve.py`)

| M tokens/step | stock MLX | FlatSpec F1+F2 | step speedup |
|---:|---:|---:|---:|
| 1 | 8.70 ms | 9.14 ms* | — |
| 6 | 16.86 ms | 13.18 ms | 1.28× |
| 8 | 21.31 ms (2.45× M=1) | **14.80 ms (1.70× M=1)** | **1.44×** |
| 16 | 35.85 ms | 29.22 ms | 1.23× |

\* M≤5 runs the identical MLX path; the difference is run-to-run noise (~5%).

## Projection (explicitly not yet measured end to end)

Throughput = τ / (verify step + K draft steps), τ = mean tokens emitted per step.
Draft step with F4 ≈ one decoder layer + pruned head ≈ 0.4 ms (estimate).

| config | verify (M=8) | drafts (K=7) | τ | tok/s | vs MLX 115 | vs llama.cpp 75 |
|---|---:|---:|---:|---:|---:|---:|
| stock MLX + MTP (today's kernels) | 21.3 | 2.8 | 4 | 166 | 1.4× | 2.2× |
| **F1–F5, kernel as measured** | 14.8 | 2.8 | 4 | 227 | 2.0× | 3.0× |
| F1–F5, τ=5.5 (code / structured) | 14.8 | 2.8 | 5.5 | 313 | 2.7× | 4.2× |
| F1–F5, kernel at synthetic-stack efficiency (~290 GB/s) | ~11 | 2.8 | 5.5 | 400 | 3.5× | 5.3× |

τ for a chained 1-layer MTP head on Qwen3.5-4B is the main unknown; MTPLX reports 100/98/94%
per-position acceptance at depth 3 on a 27B sibling (claimed).

## Remaining kernel headroom (F1)

- Real-shape efficiency 200–215 GB/s vs 292 GB/s on a uniform synthetic stack: small-N and
  very-wide (248k-row head) shapes need their own (RF, SK) tiles.
- Ablation (`bench/` history): removing MMA or dequant changes nothing → load-pattern bound;
  the lane-order relayout was the largest single win (+16%).
- Tried and reverted: fusing cast/pad/Σx into the kernel (scalar activation loads cost more than
  the dispatches they removed — those are pipelined in one command buffer).
- M5: same structure maps onto Metal 4 TensorOps (per-core neural accelerators), which would
  push the compute crossover from M≈8 to well beyond 16.
