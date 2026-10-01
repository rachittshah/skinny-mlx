"""4-bit quantized_matmul cost vs tokens-per-weight-read M (the skinny-GEMM dead zone). Run: uv run python bench/mlx_dead_zone.py"""
# Does 4-bit matmul cost stay flat as tokens-per-weight-read (M) grows? Whole-model-sized stack of layers.
import mlx.core as mx, time
D, F, L = 2560, 9216, 32
Ws=[]
for _ in range(L):
    for (o,i) in [(F,D),(F,D),(D,F),(D*3,D)]:   # gate, up, down, ~qkv/out-ish
        w=mx.random.normal([o,i]).astype(mx.float16)
        Ws.append(mx.quantize(w, group_size=64, bits=4))
mx.eval(Ws)
nbytes=sum(q.nbytes+s.nbytes+b.nbytes for q,s,b in Ws)
f=mx.compile(lambda x, xf: [mx.quantized_matmul(x if q.shape[1]*8==D else xf, q,s,b,transpose=True,group_size=64,bits=4) for q,s,b in Ws])
print(f"weights {nbytes/1e9:.2f} GB, roofline at 370GB/s = {nbytes/370e9*1e3:.2f} ms")
base=None
for M in [1,2,4,8,16,32,64]:
    x=mx.ones([M,D],dtype=mx.float16); xf=mx.ones([M,F],dtype=mx.float16)
    for _ in range(3): mx.eval(f(x,xf))
    R=10; t=time.perf_counter()
    for _ in range(R): mx.eval(f(x,xf))
    dt=(time.perf_counter()-t)/R
    base=base or dt
    print(f"M={M:3d}: {dt*1e3:6.2f} ms  {nbytes/dt/1e9:4.0f} GB/s  cost vs M=1: {dt/base:4.2f}x  tokens/s-equiv: {M/dt:7.0f}")
