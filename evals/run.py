"""Decode-speed eval runner: one engine config x one model over the eval set.

Metric (programmatic grader, per blog guidance "cheapest grader that works"):
  speedup = candidate decode tok/s / llama.cpp decode tok/s, per prompt, same model & 4-bit class,
  aggregated as a geometric mean per split. Target: >= 4.0 on train AND test.
Gate (lossless): candidate token ids must equal MLX greedy's, except a divergence at a verified
  near-tie (target top1-top2 logit gap <= TIE_GAP at that position). A config failing the gate
  on any prompt is rejected regardless of speed.
Every prompt uses the same chat-templated token ids (thinking disabled, as in the lab's product).

Usage:
  uv run python -m evals.run --model 4b --config llamacpp       # baseline (starts llama-server)
  uv run python -m evals.run --model 4b --config mlx_greedy     # reference ids for the gate
  uv run python -m evals.run --model 4b --config dflash
  uv run python -m evals.run --model 4b --summary               # table of all configs
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import requests

from bench._util import gpu_warm
from evals.configs import CONFIGS, MODELS
from evals.eval_set import PROMPTS

RESULTS = Path(__file__).resolve().parent / "results"
MAX_TOKENS = 256
TIE_GAP = 0.5  # logits; 0.125-0.25 is 1-2 bf16 ulps at typical top-logit magnitudes


def chat_ids(tok, text: str) -> list[int]:
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                   enable_thinking=False)


# ---------------------------------------------------------------- engines

def run_llamacpp(model: dict, cfg: dict, prompts: list[dict], repeats: int) -> list[dict]:
    port = 18181
    url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(["llama-server", "-m", str(Path(model["gguf"]).expanduser()), "--port", str(port),
                             "-fa", "on", "-c", "8192", "--jinja", "-ngl", "99", "--no-warmup"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                if requests.get(f"{url}/health", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        out = []
        for p in prompts:
            best = None
            for _ in range(repeats):
                gpu_warm(0.3)
                r = requests.post(f"{url}/v1/chat/completions", timeout=600, json={
                    "messages": [{"role": "user", "content": p["text"]}], "max_tokens": MAX_TOKENS,
                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).json()
                t = r["timings"]
                rec = dict(tps=t["predicted_per_second"], tokens=t["predicted_n"])
                best = rec if best is None or rec["tps"] > best["tps"] else best
            out.append(best)
        return out
    finally:
        proc.terminate()
        proc.wait()


def run_mlx_greedy(model: dict, cfg: dict, prompts: list[dict], repeats: int) -> list[dict]:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    m, tok = load(model["mlx"])
    if cfg.get("skinny"):
        from skinny_mlx.patch import patch_model

        patch_model(m)
    out = []
    for p in prompts:
        ids = chat_ids(tok, p["text"])
        for _ in stream_generate(m, tok, ids, max_tokens=8, sampler=make_sampler(temp=0.0)):
            pass  # warm
        best = None
        for _ in range(repeats):
            gpu_warm(0.3)
            toks, last = [], None
            for resp in stream_generate(m, tok, ids, max_tokens=MAX_TOKENS, sampler=make_sampler(temp=0.0)):
                toks.append(resp.token)
                last = resp
            rec = dict(tps=last.generation_tps, tokens=len(toks), ids=toks)
            best = rec if best is None or rec["tps"] > best["tps"] else best
        out.append(best)
    return out


def run_flatspec(model: dict, cfg: dict, prompts: list[dict], repeats: int) -> list[dict]:
    from mlx_lm import load

    from skinny_mlx.mtp import load_mtp_head
    from skinny_mlx.spec import MTPDrafter, speculative

    m, tok = load(model["mlx"])
    if cfg.get("skinny"):
        from skinny_mlx.patch import patch_model

        patch_model(m)
    head = load_mtp_head(m, repo=model["hf"])
    eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}
    out = []
    for p in prompts:
        ids = chat_ids(tok, p["text"])
        speculative(m, ids, 8, MTPDrafter(m, head, vocab=cfg.get("vocab", 32768)), cfg["k"])  # warm
        best = None
        for _ in range(repeats):
            gpu_warm(0.3)
            st = {}
            t = time.perf_counter()
            toks = speculative(m, ids, MAX_TOKENS, MTPDrafter(m, head, vocab=cfg.get("vocab", 32768)), cfg["k"], st)
            dt = time.perf_counter() - t  # includes prefill; flatspec has no split timer yet
            cut = next((i + 1 for i, x in enumerate(toks) if x in eos), len(toks))
            rec = dict(tps=len(toks) / dt, tokens=cut, ids=toks[:cut], tokens_per_step=st["tokens_per_step"])
            best = rec if best is None or rec["tps"] > best["tps"] else best
        out.append(best)
    return out


def run_dflash(model: dict, cfg: dict, prompts: list[dict], repeats: int) -> list[dict]:
    from dflash_mlx.engine.events import SummaryEvent
    from dflash_mlx.generate import generation_tps_from_summary
    from dflash_mlx.metal_limits import apply_metal_limits
    from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
    from dflash_mlx.runtime.bundle import load_runtime_bundle
    from dflash_mlx.runtime.context import build_offline_runtime_context

    apply_metal_limits()
    ctx = build_offline_runtime_context(verify_mode=cfg.get("verify_mode"), copyspec_mode=cfg.get("copyspec_mode"))
    if "enable_qmm" in cfg:
        ctx.verify.enable_qmm = cfg["enable_qmm"]
    bundle = load_runtime_bundle(model_ref=model["mlx"], draft_ref=model["dflash_draft"],
                                 draft_quant=cfg.get("draft_quant"), verify_config=ctx.verify)
    tok = bundle.tokenizer
    stop = get_stop_token_ids(tok)
    if cfg.get("skinny"):
        from skinny_mlx.patch import patch_model

        patch_model(bundle.target_model)

    def once(ids, n):
        summary = None
        for ev in stream_dflash_generate(target_model=bundle.target_model, target_ops=bundle.target_ops, tokenizer=tok,
                                         draft_model=bundle.draft_model, draft_backend=bundle.draft_backend,
                                         prompt="", prompt_tokens_override=ids, max_new_tokens=n,
                                         use_chat_template=False, stop_token_ids=stop, runtime_context=ctx,
                                         block_tokens=cfg.get("block_tokens")):
            if isinstance(ev, SummaryEvent):
                summary = ev
        return summary

    out = []
    for p in prompts:
        ids = chat_ids(tok, p["text"])
        once(ids, 16)  # warm
        best = None
        for _ in range(repeats):
            gpu_warm(0.3)
            s = once(ids, MAX_TOKENS)
            rec = dict(tps=generation_tps_from_summary(s), tokens=s.generation_tokens, ids=list(s.generated_token_ids),
                       acceptance=s.acceptance_ratio, tokens_per_cycle=s.tokens_per_cycle)
            best = rec if best is None or rec["tps"] > best["tps"] else best
        out.append(best)
    return out


ENGINES = {"llamacpp": run_llamacpp, "mlx_greedy": run_mlx_greedy, "flatspec": run_flatspec, "dflash": run_dflash}


# ---------------------------------------------------------------- gate + summary

def gate(model: dict, records: list[dict], ref: list[dict]) -> list[dict]:
    """Annotate each record with first divergence from MLX greedy and whether it is a near-tie."""
    import mlx.core as mx
    from mlx_lm import load

    m, tok = None, None
    for p, rec, r in zip(PROMPTS, records, ref):
        a, b = rec.get("ids"), r["ids"]
        n = min(len(a), len(b))
        div = next((i for i in range(n) if a[i] != b[i]), None)
        rec["diverge_at"], rec["tie_gap"] = div, None
        if div is not None:
            if m is None:
                m, tok = load(model["mlx"])
            ids = chat_ids(tok, p["text"]) + b[:div]
            lg = m(mx.array([ids]))[0, -1].astype(mx.float32)
            top = mx.sort(lg)[-2:].tolist()
            rec["tie_gap"] = round(top[1] - top[0], 4)
        rec["gate_ok"] = div is None or rec["tie_gap"] <= TIE_GAP
    return records


def geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def load_results(model_key: str, config: str) -> list[dict] | None:
    f = RESULTS / model_key / f"{config}.json"
    return json.loads(f.read_text())["records"] if f.exists() else None


def summarize(model_key: str) -> None:
    base = load_results(model_key, "llamacpp")
    mlxr = load_results(model_key, "mlx_greedy")
    rows = []
    for f in sorted((RESULTS / model_key).glob("*.json")):
        name = f.stem
        recs = json.loads(f.read_text())["records"]
        row = {"config": name}
        for sp in ("train", "test"):
            idx = [i for i, p in enumerate(PROMPTS) if p["split"] == sp]
            if base:
                row[f"{sp} vs llama.cpp"] = geomean([recs[i]["tps"] / base[i]["tps"] for i in idx])
            if mlxr:
                row[f"{sp} vs mlx"] = geomean([recs[i]["tps"] / mlxr[i]["tps"] for i in idx])
            row[f"{sp} tok/s"] = geomean([recs[i]["tps"] for i in idx])
        row["gate"] = all(r.get("gate_ok", True) for r in recs)
        rows.append(row)
    keys = list(rows[0].keys()) if rows else []
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r.values()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="4b", choices=list(MODELS))
    ap.add_argument("--config")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", default="", help="suffix for result file (noise runs)")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        return summarize(args.model)
    model, cfg = MODELS[args.model], CONFIGS[args.config]
    t0 = time.time()
    records = ENGINES[cfg["engine"]](model, cfg, PROMPTS, args.repeats)
    for p, r in zip(PROMPTS, records):
        r["id"], r["split"] = p["id"], p["split"]
    ref = load_results(args.model, "mlx_greedy")
    if cfg["engine"] not in ("llamacpp", "mlx_greedy") and ref:
        records = gate(model, records, ref)
    out = RESULTS / args.model / f"{args.config}{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": cfg, "model": model, "repeats": args.repeats, "wall_s": round(time.time() - t0),
                               "records": records}, indent=1))
    summarize(args.model)


if __name__ == "__main__":
    main()
