"""
bench_attention_nsight.py
=========================
测试 n 条 1000-token 序列做 prefill attention 的性能，配合 Nsight 使用。

用法：
------
# 1. Nsight Systems（时间线，推荐先用）
nsys profile \
    --trace=cuda,nvtx \
    --output=attn_n5 \
    python bench_attention_nsight.py --n 5

# 2. Nsight Compute（kernel 级指标，更慢）
ncu --set full \
    --nvtx \
    --nvtx-include "attention_forward" \
    --export attn_n5 \
    python bench_attention_nsight.py --n 5

# 3. 只看延迟，不用 Nsight
python bench_attention_nsight.py --n 1
python bench_attention_nsight.py --n 5
"""

import argparse
import torch
import torch.cuda.nvtx as nvtx
from flash_attn import flash_attn_varlen_func

# ─────────────────────────── 参数 ────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n",           type=int, default=1,    help="序列条数")
parser.add_argument("--seq_len",     type=int, default=1000, help="每条序列的 token 数")
parser.add_argument("--num_heads",   type=int, default=32,   help="Q heads（Qwen3-0.6B=16，7B=32）")
parser.add_argument("--num_kv_heads",type=int, default=8,    help="KV heads（GQA）")
parser.add_argument("--head_dim",    type=int, default=128,  help="head dim")
parser.add_argument("--warmup",      type=int, default=3,    help="warmup 次数")
parser.add_argument("--repeat",      type=int, default=10,   help="正式测量次数")
args = parser.parse_args()

N        = args.n
SEQ_LEN  = args.seq_len
Hq       = args.num_heads
Hkv      = args.num_kv_heads
D        = args.head_dim
DTYPE    = torch.bfloat16
DEVICE   = "cuda"
SCALE    = D ** -0.5

total_tokens = N * SEQ_LEN

print(f"\n{'='*55}")
print(f"  n={N}  seq_len={SEQ_LEN}  total_tokens={total_tokens}")
print(f"  Hq={Hq}  Hkv={Hkv}  head_dim={D}  dtype={DTYPE}")
print(f"{'='*55}\n")

# ─────────────────── 构造输入张量 ────────────────────────────
# q/k/v shape: (total_tokens, num_heads, head_dim)
q = torch.randn(total_tokens, Hq,  D, dtype=DTYPE, device=DEVICE)
k = torch.randn(total_tokens, Hkv, D, dtype=DTYPE, device=DEVICE)
v = torch.randn(total_tokens, Hkv, D, dtype=DTYPE, device=DEVICE)

# cu_seqlens: [0, 1000, 2000, ..., N*1000]
cu_seqlens = torch.arange(0, (N + 1) * SEQ_LEN, SEQ_LEN,
                           dtype=torch.int32, device=DEVICE)

# ─────────────────── 封装 kernel 调用 ────────────────────────
def run_attention():
    nvtx.range_push("attention_forward")
    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=SEQ_LEN,
        max_seqlen_k=SEQ_LEN,
        softmax_scale=SCALE,
        causal=True,
    )
    nvtx.range_pop()
    return out

# ─────────────────── Warmup ──────────────────────────────────
nvtx.range_push("warmup")
for _ in range(args.warmup):
    run_attention()
torch.cuda.synchronize()
nvtx.range_pop()
print(f"Warmup ({args.warmup} iters) done.")

# ─────────────────── 正式测量 ────────────────────────────────
# 用 CUDA Event 计时，精度 ~0.1µs，比 Python time 更准
start_event = torch.cuda.Event(enable_timing=True)
end_event   = torch.cuda.Event(enable_timing=True)

nvtx.range_push("benchmark")
latencies = []
for i in range(args.repeat):
    start_event.record()
    nvtx.range_push(f"iter_{i}")
    run_attention()
    nvtx.range_pop()
    end_event.record()
    torch.cuda.synchronize()
    latencies.append(start_event.elapsed_time(end_event))  # ms
nvtx.range_pop()

# ─────────────────── 结果汇总 ────────────────────────────────
latencies_t = torch.tensor(latencies)
mean_ms  = latencies_t.mean().item()
min_ms   = latencies_t.min().item()
max_ms   = latencies_t.max().item()
std_ms   = latencies_t.std().item()

throughput = total_tokens / (mean_ms / 1000)  # tokens/s

print(f"\n{'─'*45}")
print(f"  Repeat : {args.repeat} iters")
print(f"  Mean   : {mean_ms:.3f} ms")
print(f"  Min    : {min_ms:.3f} ms")
print(f"  Max    : {max_ms:.3f} ms")
print(f"  Std    : {std_ms:.3f} ms")
print(f"  Throughput : {throughput/1e6:.3f} M tokens/s")
print(f"{'─'*45}\n")

# ─────────────────── 多组对比（可选）────────────────────────
# 直接在命令行用不同 --n 跑，用 nsys 对比 timeline 即可。
# 如需在同一个脚本里跑多组（不用 Nsight），去掉下面的注释：
#
# for test_n in [1, 2, 4, 8]:
#     q2 = torch.randn(test_n * SEQ_LEN, Hq, D, dtype=DTYPE, device=DEVICE)
#     k2 = torch.randn(test_n * SEQ_LEN, Hkv, D, dtype=DTYPE, device=DEVICE)
#     v2 = torch.randn(test_n * SEQ_LEN, Hkv, D, dtype=DTYPE, device=DEVICE)
#     cu2 = torch.arange(0, (test_n+1)*SEQ_LEN, SEQ_LEN, dtype=torch.int32, device=DEVICE)
#     for _ in range(3): flash_attn_varlen_func(q2,k2,v2,cu2,cu2,SEQ_LEN,SEQ_LEN,SCALE,causal=True)
#     start_event.record()
#     for _ in range(10): flash_attn_varlen_func(q2,k2,v2,cu2,cu2,SEQ_LEN,SEQ_LEN,SCALE,causal=True)
#     end_event.record(); torch.cuda.synchronize()
#     t = start_event.elapsed_time(end_event) / 10
#     print(f"n={test_n:2d}  mean={t:.3f} ms  ratio={t/(latencies_t.mean().item()):.2f}x")
