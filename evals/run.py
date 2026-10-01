"""Decode-speed eval runner with paired measurement against llama.cpp.

Metric (programmatic grader, per blog guidance "cheapest grader that works"):
  ratio = candidate decode tok/s / llama.cpp decode tok/s on the same model (Q4_K_M vs MLX 4-bit),
  measured only in idle-GPU windows; a pair is discarded if llama.cpp ran below 85% of its clean speed,
  per prompt, geometric mean per split. Target: >= 4.0 on train AND test.
Paired design: llama-server stays up for the session; per prompt we alternate
  llama.cpp, candidate, llama.cpp, candidate, ... and score each candidate run against the mean of
  its neighbouring baseline runs, then take the median over rounds. Slowly varying background
  load (other GPU jobs on the machine) cancels in the ratio instead of biasing it.
Gate (lossless): candidate token ids must equal MLX greedy's, except a divergence at a verified
  near-tie (target top1-top2 logit gap <= TIE_GAP there). Gate failure rejects the config.
Every engine gets identical chat-templated token ids (thinking disabled, as in the lab product).

Usage:
  uv run python -m evals.run --model 4b --config mlx_greedy     # first: reference ids for the gate
  uv run python -m evals.run --model 4b --config dflash
  uv run python -m evals.run --model 4b --summary
"""

import argparse
import glob
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import requests

from bench._util import gpu_warm
from evals.configs import CONFIGS, MODELS
from evals.eval_set import PROMPTS

RESULTS = Path(__file__).resolve().parent / "results"
CLEAN_BASE_MIN = 0.85  # a pair is valid only if llama.cpp ran at >= 85% of its clean (idle-GPU) speed
MAX_RETRIES = 20


def gpu_util() -> int:
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
    i = out.find('"Device Utilization %"=')
    return int(out[i + 23 :].split(",")[0].split("}")[0]) if i >= 0 else 0


def wait_quiet(max_util: int = 5, samples: int = 3, timeout_s: float = 3600) -> None:
    """Block until the GPU is idle (no foreign jobs) for `samples` consecutive 0.5 s readings."""
    t0, ok = time.time(), 0
    while ok < samples and time.time() - t0 < timeout_s:
        ok = ok + 1 if gpu_util() <= max_util else 0
        time.sleep(0.5)
MAX_TOKENS = 256
TIE_GAP = 0.5  # logits; 0.125-0.25 is 1-2 bf16 ulps at typical top-logit magnitudes


def chat_ids(tok, text: str) -> list[int]:
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                   enable_thinking=False)


# ---------------------------------------------------------------- baseline: llama.cpp server

class LlamaServer:
    def __init__(self, model: dict, port: int = 18181):
        self.url = f"http://127.0.0.1:{port}"
        gguf = sorted(glob.glob(str(Path(model["gguf"]).expanduser())))[-1]
        self.proc = subprocess.Popen(["llama-server", "-m", gguf, "--port", str(port), "-fa", "on", "-c", "8192",
                                      "--jinja", "-ngl", "99", "--no-warmup"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(180):
            try:
                if requests.get(f"{self.url}/health", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        self.generate(PROMPTS[0], 16)  # warm

    def generate(self, p: dict, n: int = MAX_TOKENS) -> dict:
        r = requests.post(f"{self.url}/v1/chat/completions", timeout=600, json={
            "messages": [{"role": "user", "content": p["text"]}], "max_tokens": n,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).json()
        t = r["timings"]
        return dict(tps=t["predicted_per_second"], tokens=t["predicted_n"])

    def close(self):
        self.proc.terminate()
        self.proc.wait()


# ---------------------------------------------------------------- candidate engines (factories)

def make_mlx_greedy(model: dict, cfg: dict):
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    m, tok = load(model["mlx"])
    if cfg.get("skinny"):
        from skinny_mlx.patch import patch_model

        patch_model(m)

    def gen(p, n=MAX_TOKENS):
        toks, last = [], None
        for resp in stream_generate(m, tok, chat_ids(tok, p["text"]), max_tokens=n, sampler=make_sampler(temp=0.0)):
            toks.append(resp.token)
            last = resp
        return dict(tps=last.generation_tps, tokens=len(toks), ids=toks)

    return gen


def make_flatspec(model: dict, cfg: dict):
    from mlx_lm import load

    from skinny_mlx.mtp import load_mtp_head
    from skinny_mlx.spec import MTPDrafter, speculative

    m, tok = load(model["mlx"])
    if cfg.get("skinny"):
        from skinny_mlx.patch import patch_model

        patch_model(m)
    head = load_mtp_head(m, repo=model["hf"])
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])

    def gen(p, n=MAX_TOKENS):
        st = {}
        t = time.perf_counter()
        toks = speculative(m, chat_ids(tok, p["text"]), n, MTPDrafter(m, head, vocab=cfg.get("vocab", 32768)), cfg["k"], st)
        dt = time.perf_counter() - t  # includes prefill (no split timer yet) -> conservative
        cut = next((i + 1 for i, x in enumerate(toks) if x in eos), len(toks))
        return dict(tps=len(toks) / dt, tokens=cut, ids=toks[:cut], tokens_per_step=st["tokens_per_step"])

    return gen


def make_dflash(model: dict, cfg: dict):
    from dflash_mlx.engine.events import SummaryEvent
    from dflash_mlx.generate import generation_tps_from_summary
    from dflash_mlx.metal_limits import apply_metal_limits
    from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
    from dflash_mlx.runtime.bundle import load_runtime_bundle
    from dflash_mlx.runtime.context import build_offline_runtime_context

    apply_metal_limits()
    ctx = build_offline_runtime_context(verify_mode=cfg.get("verify_mode"), copyspec_mode=cfg.get("copyspec_mode"))
    bundle = load_runtime_bundle(model_ref=model["mlx"], draft_ref=model["dflash_draft"],
                                 draft_quant=cfg.get("draft_quant"), verify_config=ctx.verify)
    tok = bundle.tokenizer
    stop = get_stop_token_ids(tok)
    cfg["_load_meta"] = {k: v for k, v in vars(bundle).items() if isinstance(v, (bool, int, str, float))}
    if cfg.get("skinny"):
        from skinny_mlx.patch import patch_model

        patch_model(bundle.target_model)

    def gen(p, n=MAX_TOKENS):
        s = None
        for ev in stream_dflash_generate(target_model=bundle.target_model, target_ops=bundle.target_ops, tokenizer=tok,
                                         draft_model=bundle.draft_model, draft_backend=bundle.draft_backend,
                                         prompt="", prompt_tokens_override=chat_ids(tok, p["text"]), max_new_tokens=n,
                                         use_chat_template=False, stop_token_ids=stop, runtime_context=ctx,
                                         block_tokens=cfg.get("block_tokens")):
            if isinstance(ev, SummaryEvent):
                s = ev
        return dict(tps=generation_tps_from_summary(s), tokens=s.generation_tokens, ids=list(s.generated_token_ids),
                    acceptance=s.acceptance_ratio, tokens_per_cycle=s.tokens_per_cycle)

    return gen


ENGINES = {"mlx_greedy": make_mlx_greedy, "flatspec": make_flatspec, "dflash": make_dflash}


def run_paired(model: dict, cfg: dict, rounds: int) -> list[dict]:
    server = LlamaServer(model)
    try:
        gen = ENGINES[cfg["engine"]](model, cfg)
        out = []
        for p in PROMPTS:
            gen(p, 16)  # warm this prompt's shapes
            cands, ratios, base, rejected = [], [], [], 0
            while len(ratios) < rounds and rejected < MAX_RETRIES:
                wait_quiet()  # other GPU jobs on the machine: measure only in idle windows
                gpu_warm(0.3)
                b0 = server.generate(p)
                c = gen(p)
                b1 = server.generate(p)
                if min(b0["tps"], b1["tps"]) < CLEAN_BASE_MIN * model["clean_base_tps"]:
                    rejected += 1  # contended pair: discard, don't average it in
                    continue
                base += [b0, b1]
                cands.append(c)
                ratios.append(c["tps"] / ((b0["tps"] + b1["tps"]) / 2))
            if not ratios:
                raise RuntimeError(f"{p['id']}: no clean pair in {MAX_RETRIES} tries (GPU never idle)")
            rec = dict(cands[max(range(rounds), key=lambda i: cands[i]["tps"])])
            rec.update(ratio=statistics.median(ratios), ratios=[round(r, 3) for r in ratios],
                       base_tps=statistics.median(b["tps"] for b in base), rejected_pairs=rejected)
            out.append(rec)
            print(f"  {p['id']:14s} {rec['tps']:6.1f} tok/s  vs llama.cpp {rec['base_tps']:5.1f}  ratio {rec['ratio']:.2f}"
                  f"  (rejected {rejected} contended pairs)",
                  flush=True)
        return out
    finally:
        server.close()


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


def geomean(xs) -> float:
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def load_results(model_key: str, config: str) -> list[dict] | None:
    f = RESULTS / model_key / f"{config}.json"
    return json.loads(f.read_text())["records"] if f.exists() else None


def split_ratio(recs: list[dict], split: str) -> float:
    return geomean(r["ratio"] for r, p in zip(recs, PROMPTS) if p["split"] == split)


def summarize(model_key: str) -> None:
    print(f"{'config':28s} {'train x llama':>13s} {'test x llama':>13s} {'train tok/s':>12s} {'gate':>5s}")
    for f in sorted((RESULTS / model_key).glob("*.json")):
        recs = json.loads(f.read_text())["records"]
        if "ratio" not in recs[0]:
            continue
        tps = geomean(r["tps"] for r, p in zip(recs, PROMPTS) if p["split"] == "train")
        ok = all(r.get("gate_ok", True) for r in recs)
        print(f"{f.stem:28s} {split_ratio(recs, 'train'):12.2f}x {split_ratio(recs, 'test'):12.2f}x {tps:12.1f} {str(ok):>5s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="4b", choices=list(MODELS))
    ap.add_argument("--config")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tag", default="", help="suffix for result file (noise replicates)")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        return summarize(args.model)
    model, cfg = MODELS[args.model], dict(CONFIGS[args.config])
    os.environ.update(cfg.get("env", {}))  # engine knobs read at import/load time
    t0 = time.time()
    records = run_paired(model, cfg, args.rounds)
    for p, r in zip(PROMPTS, records):
        r["id"], r["split"] = p["id"], p["split"]
    ref = load_results(args.model, "mlx_greedy")
    if cfg["engine"] != "mlx_greedy" and ref:
        records = gate(model, records, ref)
    out = RESULTS / args.model / f"{args.config}{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": cfg, "model": model, "rounds": args.rounds, "wall_s": round(time.time() - t0),
                               "records": records}, indent=1, default=str))
    summarize(args.model)


if __name__ == "__main__":
    main()
