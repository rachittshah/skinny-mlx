"""Models and engine configs: the hill-climb's modifiable surface. One new config per round."""

MODELS = {
    "4b": dict(mlx="mlx-community/Qwen3.5-4B-MLX-4bit", hf="Qwen/Qwen3.5-4B",
               gguf="~/inference-lab/models/Qwen3.5-4B-Q4_K_M.gguf", dflash_draft="z-lab/Qwen3.5-4B-DFlash"),
}

CONFIGS = {
    # baselines
    "llamacpp": dict(engine="llamacpp"),
    "mlx_greedy": dict(engine="mlx_greedy"),
    # round 0 candidates
    "flatspec_mtp_k2": dict(engine="flatspec", k=2),
    "dflash": dict(engine="dflash"),
}
