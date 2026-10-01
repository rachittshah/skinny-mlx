"""Shared timing helpers.

Apple GPUs scale clocks with load (DVFS): short bursty timings run at a lower clock and
swing ~2x run-to-run. Every measurement first holds the GPU busy, then times a window long
enough (>= ~50 ms) to stay at full clock.
"""

import time

import mlx.core as mx

_A = None


def gpu_warm(seconds: float = 0.3) -> None:
    """Hold the GPU busy with fp16 GEMMs so it is at full clock when timing starts."""
    global _A
    if _A is None:
        _A = mx.random.normal([2048, 2048]).astype(mx.float16)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        mx.eval(_A @ _A)


def best_time(run, reps: int = 5, warm: float = 0.3) -> float:
    """Min wall time of run() (which must eval its own work) after a warm-up."""
    run()
    best = float("inf")
    for _ in range(reps):
        gpu_warm(warm)
        t = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - t)
    return best
