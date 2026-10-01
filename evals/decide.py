"""Hill-climb decision rule: keep a candidate iff train AND test geomean speedups beat the incumbent
by more than the noise floor; revert otherwise. Noise floor = max(replicate spread, 3%).

Usage: uv run python -m evals.decide --model 4b --incumbent dflash --candidate dflash_dq4 [--replicate dflash_rep]
"""

import argparse
import math

from evals.run import load_results, split_ratio

MIN_NOISE = 0.03


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="4b")
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--replicate", help="second run of the incumbent, for the noise floor")
    a = ap.parse_args()
    inc, cand = load_results(a.model, a.incumbent), load_results(a.model, a.candidate)
    noise = MIN_NOISE
    if a.replicate:
        rep = load_results(a.model, a.replicate)
        noise = max(noise, *(abs(math.log(split_ratio(rep, s) / split_ratio(inc, s))) for s in ("train", "test")))
    gate = all(r.get("gate_ok", True) for r in cand)
    verdict = {}
    for s in ("train", "test"):
        i, c = split_ratio(inc, s), split_ratio(cand, s)
        verdict[s] = math.log(c / i) > noise
        print(f"{s:5s}: incumbent {i:.2f}x -> candidate {c:.2f}x ({(c / i - 1) * 100:+.1f}%, noise ±{noise * 100:.1f}%)")
    keep = gate and verdict["train"] and verdict["test"]
    print(f"gate {'ok' if gate else 'FAIL'} -> {'KEEP' if keep else 'REVERT'}")


if __name__ == "__main__":
    main()
