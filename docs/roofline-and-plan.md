# A Mac-native inference kernel: where 4–8× can (and can't) come from

**Question:** can a custom Metal inference kernel be 4–8× faster for local Mac inference?
**Answer (measured):** not by kernel efficiency alone — single-stream decode has only
**~1.3× headroom over MLX** (1.8× over llama.cpp) before hitting the memory-bandwidth wall.
4–8× is reachable only by *changing how many tokens each weight read produces*:
a **skinny quantized GEMM** that makes verifying 8–16 tokens cost ~1 token, driving
**hybrid-aware speculative decoding**. Measured today, that kernel is the missing piece:
MLX verifies 8 tokens at 2.34× the cost of 1, so speculation nets ~1.0×.

Hardware: M4 Max (binned, 32-core GPU), 36 GB, macOS 15.6.1, MLX 0.32.1, llama.cpp b10809.
Model: Qwen3.5-4B (24 Gated-DeltaNet layers + 8 full-attention layers; tied 248k-vocab head).
Scripts: `bench/` (all numbers below reproduce with them).

---

## 1. Hardware ceilings (`bench/hw_roofline.py`)

| metric | measured | spec |
|---|---:|---:|
| streaming read bandwidth (GPU) | **~370 GB/s** | 410 GB/s |
| fp16 GEMM (4k/8k) | **11.9 TFLOPS** | — |
| tiny-op dispatch, chained | ~2.6 µs/op | — |
| single sync `eval` round-trip | ~160 µs | — |

Decode reads every weight once per step → **time/step ≥ bytes / 370 GB/s**.
Compute crosses bandwidth at M ≈ (11.9 TFLOPS × 2.4 GB / 370 GB/s) / 7 GFLOP/token ≈ **8–10 tokens/step**.
Below that, extra tokens per step are *free* in theory.

## 2. Baselines vs roofline (single stream, 4-bit)

| engine | bytes/token | roofline | measured decode | % of roof |
|---|---:|---:|---:|---:|
| llama.cpp Q4_K_M (`llama-bench tg128`) | 2.73 GB | 136 tok/s | **74.6 tok/s** | 55% |
| llama.cpp Q8_0 | 4.47 GB | 83 tok/s | 57.1 tok/s | 69% |
| MLX 4-bit (`mlx_lm generate`) | ~2.4 GB | ~155 tok/s | **115–120 tok/s** | ~77% |
| MLX via `mlx_lm.server` (doc 15) | — | — | 83 tok/s | server overhead ≈ 30% |

Prefill (llama.cpp pp512): 1171 tok/s ≈ 8.4 TFLOPS ≈ **70% of GEMM peak** → prefill
has ≤1.4× kernel headroom on M4. (M5's per-core matmul units change this: 3–6× prefill
is already published — BaseRT, Apple MLX-M5 blog.)

**Kernel-only ceiling:** ≈1.3× over MLX, ≈1.8× over llama.cpp. Not 4–8×.

## 3. The real bottleneck: the skinny-GEMM dead zone

4-bit matmul over a full-model-sized weight stack (`bench/mlx_dead_zone.py`, 1.63 GB):

| M (tokens/read) | ms | GB/s | cost vs M=1 | ideal cost |
|---:|---:|---:|---:|---:|
| 1 | 4.54 | 359 | 1.00× | 1.0× |
| 2 | 4.53 | 360 | 1.00× | 1.0× |
| 4 | 6.14 | 265 | 1.35× | 1.0× |
| 8 | 12.9 | 126 | **2.72×** | ~1.1× |
| 16 | 18.0 | 90 | **3.79×** | ~1.6× |
| 32 | 18.1 | 90 | 3.80× | ~3× |

MLX's GEMV (`qmv`) hits **97% of measured bandwidth** at M=1 — that kernel is done.
The tiled GEMM (`qmm`) taking over at M≥4–8 is built for large M and wastes the
bandwidth-bound regime: 8 tokens should cost ~1 weight read, they cost 2.7.

Full-model, pipelined steady state (`bench/step_cost_curve.py`, 512-token context):

| M | ms/step | cost vs M=1 | tok/s if all accepted |
|---:|---:|---:|---:|
| 1 | 8.67 | 1.00× | 115 |
| 2 | 9.11 | 1.05× | 220 |
| 4 | 11.65 | 1.34× | 343 |
| 8 | 20.28 | **2.34×** | 395 |
| 16 | 31.96 | **3.69×** | 501 |

Consequences, measured:
- **Speculative decoding nets nothing today.** Qwen3-4B + Qwen3-0.6B draft (mlx_lm):
  baseline 129 tok/s → K=2: 131, K=4: 126, K=6: **92**. Matches literature: EAGLE-3 in
  MLX 1.05× (mlx-lm #890), draft-model spec ≤1.61× (arXiv 2607.17283).
- **Hybrid models can't speculate at all:** `mlx_lm` → `ValueError: Speculative decoding
  requires a trimmable prompt cache (got ArraysCache)`. DeltaNet/SSM state can't be trimmed.
- **Batching underdelivers:** llama.cpp aggregate decode B=1/4/8/16/32 =
  76 / 110 / 112 / 187 / 235 tok/s vs a ~1000 tok/s compute ceiling.
- **Host overhead is real:** an unpipelined single step costs 18.4 ms vs 8.67 ms pipelined
  → ~10 ms/step of graph build + launch must stay hidden (any drafter that syncs every step
  re-exposes it).

## 4. The design that gets to 4–8×

Speedup = (tokens accepted per step τ) ÷ (step cost relative to one plain decode step).

| # | component | what it does | expected factor | evidence |
|---|---|---|---|---|
| K1 | **Skinny quantized GEMM** (M=2–16) | stream each weight block once, apply to all M activations in registers; extend `qmv`'s structure to M rows instead of using a tiled `qmm` | M=8 step 2.34× → ~1.2× | §3: GEMV is 97% of BW; M=2 already free |
| K2 | **Hybrid-aware verify** | DeltaNet chunk kernel emits per-position recurrent state (or keep the last-accepted state + replay ≤K tokens); KV trim for the 8 attention layers | enables spec on Qwen3.5/3.6, Mamba-style hybrids | §3 ValueError; no Mac engine does it today |
| K3 | **Block/parallel drafter** (DFlash/EAGLE-3-style head on target hidden states) | proposes 8–16 tokens in ~1 cheap pass | τ ≈ 3–6 | DFlash-mlx claims 4.37× on M5 Max (Qwen3.5-9B; bf16 baseline — **claimed**) |
| K4 | **Fused step / fewer dispatches** | fuse norm+QKV+RoPE, gate+up+SiLU, DeltaNet update; reuse command buffers; ~1 dispatch per sublayer | M=1 8.67 → ~7 ms (1.2×) | MLX at 77% of roof; ~30 dispatches/layer |
| K5 | **3-bit weights, BW-bound unpack** | 25% fewer bytes if dequant stays under memory time | ~1.25× | IQ2/IQ3 reported compute-bound on Metal → must design the unpack, not reuse it |
| K6 | Activation sparsity (TEAL-style 50%) | skip weight columns for near-zero activations | ~1.3–1.5× | arXiv 2511.04477 (M2 Max: 1.47×) — conflicts with K1 batching (union of masks) |

**Projected single-stream decode, Qwen3.5-4B on this M4 Max** (τ = mean accepted tokens/step, M=8 verify, drafter ≈ 1.5 ms):

| configuration | ms/step | τ | tok/s | vs llama.cpp (75) | vs MLX (118) |
|---|---:|---:|---:|---:|---:|
| MLX today, no spec | 8.67 | 1 | 115 | 1.5× | 1.0× |
| spec on **today's** kernel | 20.3 + 1.5 | 4 | 183 | 2.4× | 1.6× |
| **K1+K2+K3** | ~8.5 + 1.5 | 4 | **400** | **5.3×** | 3.4× |
| K1–K4 | ~7.5 + 1.5 | 4 | 445 | 5.9× | 3.8× |
| K1–K5 | ~6.2 + 1.5 | 4 | 520 | **6.9×** | **4.4×** |
| K1–K5, code/structured output | ~6.2 + 1.5 | 6 | 780 | 10× | 6.6× |

These rows are **projections** built from measured step costs plus an assumed τ; τ is the
dominant uncertainty and is workload-dependent (chat < code < extraction). K1 alone is
worth ~2.2× on the spec path, which is why it's the first thing to build.

## 5. What will *not* work (save the time)

- **Megakernel with grid-wide barriers.** Metal gives no forward-progress / co-residency
  guarantee across threadgroups (Apple forums thread 831017, unanswered). Use producer→consumer
  handoff between neighbouring threadgroups only, cap the grid to resident capacity, keep a
  multi-dispatch fallback.
- **ANE for decode.** ~85 GB/s to DRAM and ~190 µs per call (arXiv 2606.22283); Anemll
  Llama-3.2-1B ≈ 50–60 tok/s vs ~200 on the M4 Max GPU. Fine for low-power encoders, not a speedup.
- **Faster prefill kernels on M4.** Already 70% of GEMM peak. The 4–6× prefill wins are M5-only
  (Metal 4 tensor ops / MetalPerformancePrimitives, macOS 26.2+).
- **"Just batch it."** Raises aggregate throughput (MLX 207 tok/s @16 in doc 15), not the
  single-user latency that local inference is about.

## 6. Build plan (each phase has a measurable gate)

| phase | deliverable | gate (measured with `bench/`) |
|---|---|---|
| P1 | K1 as an `mx.fast.metal_kernel` prototype, 4-bit, group 64 | `qmm_skinny.py`: M=8 ≤ 1.25× M=1, M=16 ≤ 1.8×; bit-exact vs `mx.quantized_matmul` |
| P2 | K1 wired into Qwen3.5 forward | `step_cost_curve.py`: M=8 ≤ 1.3× full-model step |
| P3 | K2 state checkpointing for DeltaNet + KV trim | greedy spec output == greedy plain output, token-for-token over the golden set |
| P4 | K3 drafter (start: prompt-lookup + 0.8B draft; then trained block-draft head) | measured τ on `evals/golden_qa.json` + code prompts; end-to-end ≥ 3× llama.cpp |
| P5 | K4 fusion / dispatch reduction | M=1 ≥ 135 tok/s (≥ 87% of roof) |
| P6 | K5 3-bit with BW-bound unpack | qmv 3-bit ≥ 330 GB/s effective; evals still pass the deploy gate |

Escape hatch: if P1 can't beat the gate in an MLX custom kernel, drop to a standalone
Metal-cpp engine (own command-buffer encoding) — that is also the path for K4.

## 7. Open questions

- UNCONFIRMED: real τ for a Qwen3.5-4B drafter on this workload (only DFlash's M5 claims exist).
- UNCONFIRMED: cost of DeltaNet per-position state emission at M=8 (state = heads × d_k × d_v per layer; must stay ≪ weight bytes).
- UNCONFIRMED: whether `qmm`'s slowdown at M=4–16 is tile-shape choice (fixable by dispatch heuristics in MLX upstream) or structural.
- The README's "hybrid GQA" description of Qwen3.5-4B is wrong: 24 of 32 layers are Gated DeltaNet (`ssm_*` tensors), 8 are full attention (verified from GGUF tensor names).

## Sources

Measured: this doc's tables (`bench/*.py`, `llama-bench`, `llama-batched-bench`, `mlx_lm generate`).
External: Apple ML "Exploring LLMs with MLX on M5"; llama.cpp discussion #4167; vllm-mlx arXiv 2601.19139;
mlx-lm discussion #890 (EAGLE-3); arXiv 2607.17283 (spec decoding on Metal); arXiv 2511.04477 (sparsity);
github.com/bstnxbt/dflash-mlx (claimed); BaseRT arXiv 2607.19438; Hazy Research ThunderMittens blog;
mlx issues #3313, #3789; arXiv 2606.22283 (ANE); WWDC26 session 330 (Metal 4 tensors).
