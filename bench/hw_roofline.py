"""Hardware ceilings: streaming read GB/s, fp16 GEMM TFLOPS, per-op dispatch overhead. Run: uv run python bench/hw_roofline.py"""
import mlx.core as mx, time
# Streaming read bandwidth: reduce a large buffer (decode is a big streaming read of weights)
for gb in [1, 4, 8]:
    n = int(gb*2**30)//2
    a = mx.ones([n//1024,1024]).astype(mx.float16); mx.eval(a)
    for _ in range(3): mx.eval(a.sum())
    t=time.perf_counter(); R=10
    for _ in range(R): mx.eval(a.sum())
    dt=(time.perf_counter()-t)/R
    print(f"read {gb}GB: {gb*2**30/dt/1e9:.0f} GB/s")
# fp16 matmul peak
for s in [4096, 8192]:
    x=mx.random.normal([s,s]).astype(mx.float16); y=mx.random.normal([s,s]).astype(mx.float16); mx.eval(x,y)
    for _ in range(3): mx.eval(x@y)
    t=time.perf_counter(); R=10
    for _ in range(R): mx.eval(x@y)
    dt=(time.perf_counter()-t)/R
    print(f"fp16 gemm {s}: {2*s**3/dt/1e12:.1f} TFLOPS")
# dispatch overhead: tiny ops in a chain
x=mx.ones((16,)); mx.eval(x)
for n in [1, 100, 1000]:
    t=time.perf_counter()
    for _ in range(20):
        y=x
        for _ in range(n): y=y+1
        mx.eval(y)
    print(f"chain {n} tiny ops: {(time.perf_counter()-t)/20*1e3:.2f} ms  ({(time.perf_counter()-t)/20/n*1e6:.1f} us/op)")
