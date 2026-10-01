"""Models and engine configs: the hill-climb's modifiable surface. One new config per round."""

MODELS = {
    "4b": dict(mlx="mlx-community/Qwen3.5-4B-MLX-4bit", hf="Qwen/Qwen3.5-4B",
               gguf="~/inference-lab/models/Qwen3.5-4B-Q4_K_M.gguf", dflash_draft="z-lab/Qwen3.5-4B-DFlash"),
    "9b": dict(mlx="mlx-community/Qwen3.5-9B-MLX-4bit", hf="Qwen/Qwen3.5-9B",
               gguf="~/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_K_M.gguf",
               dflash_draft="z-lab/Qwen3.5-9B-DFlash"),
}

CONFIGS = {
    # baselines
    "llamacpp": dict(engine="llamacpp"),
    "mlx_greedy": dict(engine="mlx_greedy"),
    # round 0 candidates
    "flatspec_mtp_k2": dict(engine="flatspec", k=2),
    "dflash": dict(engine="dflash"),
    # hill-climb candidates (each round stacks onto the incumbent)
    "dflash_dq4": dict(engine="dflash", draft_quant="w4:gs64"),
    "dflash_dq8": dict(engine="dflash", draft_quant="w8:gs64"),
    "dflash_vqmm": dict(engine="dflash", env={"DFLASH_VERIFY_LINEAR": "1", "DFLASH_VERIFY_QMM": "1"}),
    "dflash_b8": dict(engine="dflash", block_tokens=8),
    "dflash_b12": dict(engine="dflash", block_tokens=12),
    "dflash_skinny": dict(engine="dflash", skinny=True, env={"DFLASH_VERIFY_LINEAR": "0"}),
    "dflash_verify_dflash": dict(engine="dflash", verify_mode="dflash"),
}


def stacked(base: str, **changes) -> dict:
    """Incumbent config + one change (hill-climb rounds compose winners)."""
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in CONFIGS[base].items()}
    env = {**c.get("env", {}), **changes.pop("env", {})}
    c.update(changes)
    if env:
        c["env"] = env
    return c
