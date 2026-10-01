# Hill-climb log: local decode speed → 4× llama.cpp

Method (after claude.dev "Automating eval design and hillclimbing"):

- **Eval**: `evals/eval_set.py` — 16 hand-written prompts mirroring the lab's chat product
  (RAG, chat, code, tool JSON, summarization, reasoning), stratified 10 train / 6 test.
- **Grader** (programmatic, cheapest that works): per-prompt decode tok/s ÷ llama.cpp decode tok/s
  on the same model (Q4_K_M vs MLX 4-bit), geometric mean per split.
- **Gate**: candidate tokens must equal MLX greedy's, except a divergence at a verified near-tie
  (target top1–top2 logit gap ≤ 0.5). Gate failure rejects the round regardless of speed.
- **Noise first**: replicate a config; a change is acted on only if it beats the replicate spread.
- **Rule**: one change per round. Keep iff train AND test both improve beyond noise; revert if only
  train improves or either drops. Changes are proposed from train results only.
- **Surfaces**: drafter (MTP / DFlash), draft precision, block size, verify kernels (dflash
  `verify_qmm`, skinny-mlx), verify mode, and model choice (2026 models with released drafters).
- **Target**: geomean speedup ≥ 4.0× on train and test.

Measurement hygiene: other GPU jobs on the machine (LoRA training, Blender renders) invalidate runs;
`scratchpad/wait_gpu.sh`-style gating waits for an idle GPU before every batch.

| round | model | change | train × llama.cpp | test × llama.cpp | gate | decision |
|---|---|---|---:|---:|---|---|
