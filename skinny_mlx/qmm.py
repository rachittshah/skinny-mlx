"""Skinny 4-bit quantized matmul for M = 2..16 tokens per weight read (the spec-verify regime).

MLX's qmv (M=1) streams weights at ~97% of bandwidth, but for M>=4 it switches to a tiled
qmm built for large M, and 8 tokens cost 2.7x one token. This kernel keeps qmv's shape —
one simdgroup streams ROWS weight rows once, contiguous 8-byte loads per lane — and applies
each dequantized weight to all M activation vectors held in registers.

Uses the affine identity  sum_i x_i (q_i s + b) = s * sum_i x_i q_i + b * sum_i x_i
so dequant is one multiply-add per group, not per weight.

Run:  uv run python -m skinny_mlx.qmm      (correctness + benchmark vs mx.quantized_matmul)
"""

import time

import mlx.core as mx
import numpy as np

GROUP = 64

_SOURCE = """
    // C^T[rows x tokens] = W[rows x K] . X^T[K x tokens] with 8x8 simdgroup MMA (ALU-bound on M1-M4).
    // Dequant cost is the bottleneck, so: (1) MMA on raw q and apply scale/bias once per 64-group:
    //   sum_k x(q s + b) = s * sum_k x q + b * xsum[group];
    // (2) weights repacked at load (repack_for_skinny) into MMA-fragment lane order: for each
    //   (8-row fragment, 64-group) each lane owns 8 contiguous bytes = exactly its 16 nibbles, so a
    //   simdgroup reads one coalesced 256 B block, and step s's pair sits at bits 4s and 16+4s:
    //   ((w >> 4s) & 0x000f000f) | 0x64006400 reinterpreted as half2 = 1024 + q, exact.
    // threadgroup = SK simdgroups splitting K over the same RF*8 rows; reduced via threadgroup memory.
    const uint lane = thread_index_in_simdgroup;
    const uint sk = simdgroup_index_in_threadgroup;
    const uint row0 = threadgroup_position_in_grid.y * (RF * 8);
    constexpr uint KG = K / 64, GS = KG / SK;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);   // fragment row owned by this lane
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;  // fragment cols fn, fn+1

    simdgroup_float8x8 C[RF][TF], G[RF][TF];
    for (uint r = 0; r < RF; r++) for (uint t = 0; t < TF; t++) C[r][t] = simdgroup_float8x8(0);
    simdgroup_half8x8 A, B[TF];

    for (uint g = sk * GS; g < (sk + 1) * GS; g++) {
        const uint k = g * 64;
        uint2 wv[RF];
        float sc[RF], bi[RF];
        _Pragma("clang loop unroll(full)") for (uint r = 0; r < RF; r++) {
            const uint row = row0 + r * 8 + fm;
            wv[r] = ((const device uint2*)w)[((row0 / 8 + r) * KG + g) * 32 + lane];
            sc[r] = float(scales[row * KG + g]);
            bi[r] = float(biases[row * KG + g]);
            for (uint t = 0; t < TF; t++) G[r][t] = simdgroup_float8x8(0);
        }
        _Pragma("clang loop unroll(full)") for (uint s = 0; s < 8; s++) {
            for (uint t = 0; t < TF; t++)
                simdgroup_load(B[t], x + t * 8 * K + k + s * 8, K, ulong2(0, 0), true);  // X^T [8k x 8tok]
            _Pragma("clang loop unroll(full)") for (uint r = 0; r < RF; r++) {
                const uint bits = ((wv[r][s / 4] >> (4 * (s % 4))) & 0x000f000fu) | 0x64006400u;
                const half2 q = as_type<half2>(bits) - half2(1024.0h);
                A.thread_elements()[0] = q.x;
                A.thread_elements()[1] = q.y;
                _Pragma("clang loop unroll(full)") for (uint t = 0; t < TF; t++) simdgroup_multiply_accumulate(G[r][t], A, B[t], G[r][t]);
            }
        }
        // C(fm, tok) += s[fm] * G(fm, tok) + b[fm] * xsum[tok][g]; lane holds tokens t*8+fn, t*8+fn+1
        for (uint t = 0; t < TF; t++) {
            const float xs0 = xsum[(t * 8 + fn) * KG + g], xs1 = xsum[(t * 8 + fn + 1) * KG + g];
            for (uint r = 0; r < RF; r++) {
                C[r][t].thread_elements()[0] += sc[r] * G[r][t].thread_elements()[0] + bi[r] * xs0;
                C[r][t].thread_elements()[1] += sc[r] * G[r][t].thread_elements()[1] + bi[r] * xs1;
            }
        }
    }

    threadgroup float red[SK][RF * 8][TF * 8];
    for (uint r = 0; r < RF; r++) for (uint t = 0; t < TF; t++)
        simdgroup_store(C[r][t], &red[sk][r * 8][t * 8], TF * 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = thread_index_in_threadgroup; i < RF * 8 * TF * 8; i += SK * 32) {
        const uint rr = i / (TF * 8), tt = i % (TF * 8);
        if (tt < M) {
            float v = 0.0f;
            for (uint s = 0; s < SK; s++) v += red[s][rr][tt];
            out[tt * N + row0 + rr] = v;
        }
    }
"""

_kernel = mx.fast.metal_kernel(
    name="skinny_qmm_4bit",
    input_names=["x", "xsum", "w", "scales", "biases"],
    output_names=["out"],
    source=_SOURCE,
)


def repack_for_skinny(w: mx.array) -> mx.array:
    """One-time load-time relayout of MLX 4-bit packs [N, K/8] into MMA-fragment lane order
    [N/8, K/64, 32 lanes, 2 words]. Lane (fm, fn) of 8-row fragment R, group g gets row R*8+fm,
    cols k = 64g + 8s + {fn, fn+1} for steps s=0..7; word h holds steps 4h..4h+3 with the first
    element of step s at bits 4s and the second at 16+4s. Same bytes, different order."""
    n, kp = w.shape
    k, kg = kp * 8, kp // 8
    a = np.array(w)
    q = np.stack([((a >> (4 * i)) & 0xF).astype(np.uint8) for i in range(8)], axis=-1).reshape(n // 8, 8, kg, 8, 8)  # [R, fm, g, s, c]
    lane = np.arange(32)
    qid = lane // 4
    fm = (qid & 4) + ((lane // 2) % 4)
    fn = (qid & 2) * 2 + (lane % 2) * 2
    e0 = q[:, fm, :, :, fn].astype(np.uint32)       # [32, R, g, s]
    e1 = q[:, fm, :, :, fn + 1].astype(np.uint32)
    words = np.zeros(e0.shape[:-1] + (2,), dtype=np.uint32)
    for h in range(2):
        for i in range(4):
            words[..., h] |= (e0[..., 4 * h + i] << (4 * i)) | (e1[..., 4 * h + i] << (16 + 4 * i))
    return mx.array(np.ascontiguousarray(words.transpose(1, 2, 0, 3)).reshape(n, kp))


def config_for(m: int, k: int) -> tuple[int, int]:
    """(RF row-fragments per simdgroup, SK K-splits per threadgroup), from `--sweep` on M4 Max:
    M<=8: RF=2,SK=8 (292 GB/s); M<=16: RF=1,SK=4. SK falls back to a divisor of the group count."""
    if CONFIG_OVERRIDE:
        return CONFIG_OVERRIDE
    rf, sk = (2, 8) if m <= 8 else (1, 4)
    while (k // 64) % sk:
        sk //= 2
    return rf, sk


CONFIG_OVERRIDE: tuple[int, int] | None = None


def skinny_qmm(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """x [M,K] fp16 @ dequant(w)[N,K]^T -> [M,N]. w must be repack_for_skinny()-ed MLX 4-bit, group 64."""
    m, k = x.shape
    n = w.shape[0]
    rf, sk = config_for(m, k)
    tf = (m + 7) // 8
    if m % 8:
        x = mx.concatenate([x, mx.zeros([tf * 8 - m, k], dtype=x.dtype)])
    xsum = x.astype(mx.float32).reshape(tf * 8, k // 64, 64).sum(-1)
    assert (k // 64) % sk == 0 and n % (8 * rf) == 0, (m, k, n)
    return _kernel(
        inputs=[x, xsum, w, scales, biases],
        template=[("M", m), ("TF", tf), ("RF", rf), ("SK", sk), ("K", k), ("N", n)],
        grid=(32 * sk, n // (8 * rf), 1),
        threadgroup=(32 * sk, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]


def _bench(fn, reps: int = 20) -> float:
    for _ in range(3):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    return (time.perf_counter() - t) / reps


D, F = 2560, 9216


def build_stack(layers: int = 32):
    """Random 4-bit weights shaped like the 4B model's gate/up/down/qkv; returns (mlx packs, repacked, bytes)."""
    shapes = [(F, D), (F, D), (D, F), (3 * D, D)] * layers
    Ws = [mx.quantize(mx.random.normal([o, i]).astype(mx.float16) * 0.02, group_size=GROUP, bits=4) for o, i in shapes]
    Rs = [(repack_for_skinny(q), s, b) for q, s, b in Ws]
    mx.eval(Ws, Rs)
    return Ws, Rs, sum(q.nbytes + s.nbytes + b.nbytes for q, s, b in Ws)


def main() -> None:
    Ws, Rs, nbytes = build_stack()

    # correctness on one layer, every M
    (q, s, b), (rq, _, _) = Ws[2], Rs[2]
    for M in [1, 2, 4, 8, 16]:
        x = mx.random.normal([M, F]).astype(mx.float16)
        ref = mx.quantized_matmul(x, q, s, b, transpose=True, group_size=GROUP, bits=4)
        got = skinny_qmm(x, rq, s, b)
        err = (mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max() / mx.abs(ref).max()).item()
        assert err < 2e-2, (M, err)
    print("correctness: max rel err < 2e-2 for M in 1..16")

    print(f"stack {nbytes / 1e9:.2f} GB, bandwidth floor {nbytes / 370e9 * 1e3:.2f} ms")
    print(f"{'M':>3} {'mlx ms':>8} {'skinny ms':>10} {'GB/s':>6} {'speedup':>8} {'cost vs M=1':>12}")
    base = None
    for M in [1, 2, 3, 4, 6, 8, 12, 16]:
        xs = {D: mx.random.normal([M, D]).astype(mx.float16), F: mx.random.normal([M, F]).astype(mx.float16)}
        mx.eval(xs)
        ref = _bench(lambda: [mx.quantized_matmul(xs[q.shape[1] * 8], q, s, b, transpose=True, group_size=GROUP, bits=4) for q, s, b in Ws])
        new = _bench(lambda: [skinny_qmm(xs[q.shape[1] * 8], q, s, b) for q, s, b in Rs])
        base = base or new
        print(f"{M:>3} {ref * 1e3:8.2f} {new * 1e3:10.2f} {nbytes / new / 1e9:6.0f} {ref / new:7.2f}x {new / base:11.2f}x")


def sweep() -> None:
    """Tile sweep (RF row-fragments x SK K-splits) on the full-model stack at M=8 and 16."""
    global CONFIG_OVERRIDE
    _, Rs, nbytes = build_stack()
    for M in [8, 16]:
        xs = {D: mx.random.normal([M, D]).astype(mx.float16), F: mx.random.normal([M, F]).astype(mx.float16)}
        mx.eval(xs)
        for rf in [1, 2, 4]:
            for sk in [1, 2, 4, 8]:
                CONFIG_OVERRIDE = (rf, sk)
                dt = _bench(lambda: [skinny_qmm(xs[q.shape[1] * 8], q, s, b) for q, s, b in Rs])
                print(f"M={M:2d} RF={rf} SK={sk}: {dt * 1e3:6.2f} ms {nbytes / dt / 1e9:4.0f} GB/s")
    CONFIG_OVERRIDE = None


if __name__ == "__main__":
    import sys

    sweep() if "--sweep" in sys.argv else main()
