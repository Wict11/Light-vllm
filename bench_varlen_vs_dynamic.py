"""
对比 always-varlen vs 动态选择 的性能。

测试方式：
  1. always-varlen: force_varlen=True，决策永远选 flash_attn_varlen_func
  2. dynamic:       按公式自动判断走 varlen 还是 kvcache

结论：paged cache 下 kvcache 优势大幅缩小，公式改为保守的 /32。
"""
import argparse
import csv
import math

import torch

from nanovllm.layers.attention import Attention
from nanovllm.utils.context import reset_context, set_context


def build_case(
    prefill_seqs: int,
    decode_seqs: int,
    prefill_q_len: int,
    decode_ctx_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
):
    num_seqs = prefill_seqs + decode_seqs
    if num_seqs <= 0:
        raise ValueError("total seqs must be > 0")

    query_lens = [prefill_q_len] * prefill_seqs + [1] * decode_seqs
    context_lens = [prefill_q_len] * prefill_seqs + [decode_ctx_len] * decode_seqs
    max_context = max(context_lens)
    block_size = 256
    max_blocks_per_seq = max(1, math.ceil(max_context / block_size))

    num_prefill_tokens = prefill_seqs * prefill_q_len
    num_decode_tokens = decode_seqs
    total_tokens = num_prefill_tokens + num_decode_tokens

    cu_q = [0]
    cu_k = [0]
    for ql, kl in zip(query_lens, context_lens):
        cu_q.append(cu_q[-1] + ql)
        cu_k.append(cu_k[-1] + kl)

    block_tables = []
    for seq_idx in range(num_seqs):
        start_block = seq_idx * max_blocks_per_seq
        block_tables.append(list(range(start_block, start_block + max_blocks_per_seq)))

    slot_mapping = []
    for seq_idx in range(num_seqs):
        tbl = block_tables[seq_idx]
        def pos_to_slot(pos):
            return tbl[pos // block_size] * block_size + pos % block_size
        if seq_idx < prefill_seqs:
            for p in range(prefill_q_len):
                slot_mapping.append(pos_to_slot(p))
        else:
            slot_mapping.append(pos_to_slot(decode_ctx_len - 1))

    device = "cuda"
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)

    cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=device)
    slot_t = torch.tensor(slot_mapping, dtype=torch.int32, device=device)
    context_lens_t = torch.tensor(context_lens, dtype=torch.int32, device=device)
    block_tables_t = torch.tensor(block_tables, dtype=torch.int32, device=device)

    total_blocks = num_seqs * max_blocks_per_seq
    k_cache = torch.randn(
        total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn(
        total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )

    attn = Attention(
        num_heads=num_heads,
        head_dim=head_dim,
        scale=head_dim ** -0.5,
        num_kv_heads=num_kv_heads,
    ).to(device)
    attn.k_cache = k_cache
    attn.v_cache = v_cache

    return {
        "attn": attn,
        "q": q,
        "k": k,
        "v": v,
        "cu_q": cu_q_t,
        "cu_k": cu_k_t,
        "max_q": max(query_lens),
        "max_k": max(context_lens),
        "slot_mapping": slot_t,
        "context_lens": context_lens_t,
        "block_tables": block_tables_t,
        "num_prefill_tokens": num_prefill_tokens,
        "num_decode_tokens": num_decode_tokens,
    }


def bench(attn, q, k, v, warmup: int, repeat: int):
    for _ in range(warmup):
        with torch.inference_mode():
            attn(q, k, v)
    torch.cuda.synchronize()

    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    lat = []
    for _ in range(repeat):
        st.record()
        with torch.inference_mode():
            attn(q, k, v)
        ed.record()
        torch.cuda.synchronize()
        lat.append(st.elapsed_time(ed))
    t = torch.tensor(lat)
    return float(t.mean().item()), float(torch.quantile(t, 0.95).item())


def run_case(case, mode: str, warmup: int, repeat: int):
    attn = case["attn"]
    if mode == "varlen":
        attn.force_varlen = True
    else:
        attn.force_varlen = False

    reset_context()
    set_context(
        True,
        case["cu_q"],
        case["cu_k"],
        case["max_q"],
        case["max_k"],
        case["slot_mapping"],
        case["context_lens"],
        case["block_tables"],
        num_prefill_tokens=case["num_prefill_tokens"],
        num_decode_tokens=case["num_decode_tokens"],
    )

    return bench(attn, case["q"], case["k"], case["v"], warmup, repeat)


def parse_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser("Compare always-varlen vs dynamic")
    parser.add_argument("--prefill_q_lens", type=str, default="8,16,32,64,128,256,512,1024,2048")
    parser.add_argument("--decode_seqs", type=str, default="8,32,64,128")
    parser.add_argument("--decode_ctx_lens", type=str, default="512,2048,4096")
    parser.add_argument("--prefill_seqs", type=str, default="1,2,4,8,16,32,64,128")
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--csv", type=str, default="varlen_vs_dynamic.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    prefill_q_lens = parse_list(args.prefill_q_lens)
    decode_seqs_list = parse_list(args.decode_seqs)
    decode_ctx_lens = parse_list(args.decode_ctx_lens)
    prefill_seqs_list = parse_list(args.prefill_seqs)

    rows = []
    total = len(prefill_seqs_list) * len(prefill_q_lens) * len(decode_seqs_list) * len(decode_ctx_lens)
    idx = 0

    for ps in prefill_seqs_list:
        for ql in prefill_q_lens:
            for d in decode_seqs_list:
                for ctx in decode_ctx_lens:
                    idx += 1
                    case = build_case(
                        prefill_seqs=ps,
                        decode_seqs=d,
                        prefill_q_len=ql,
                        decode_ctx_len=ctx,
                        num_heads=args.num_heads,
                        num_kv_heads=args.num_kv_heads,
                        head_dim=args.head_dim,
                        dtype=dtype,
                    )

                    varlen_mean, varlen_p95 = run_case(case, "varlen", args.warmup, args.repeat)
                    dyn_mean, dyn_p95 = run_case(case, "dynamic", args.warmup, args.repeat)

                    best = "varlen" if varlen_mean <= dyn_mean else "dynamic"
                    ratio = dyn_mean / varlen_mean
                    row = {
                        "prefill_seqs": ps,
                        "prefill_q_len": ql,
                        "decode_seqs": d,
                        "decode_ctx_len": ctx,
                        "varlen_mean_ms": varlen_mean,
                        "dynamic_mean_ms": dyn_mean,
                        "varlen_p95_ms": varlen_p95,
                        "dynamic_p95_ms": dyn_p95,
                        "dynamic_over_varlen": ratio,
                        "best": best,
                    }
                    rows.append(row)
                    print(
                        f"[{idx}/{total}] ps={ps} q={ql} d={d} ctx={ctx} "
                        f"varlen={varlen_mean:.3f}ms dynamic={dyn_mean:.3f}ms best={best}"
                    )

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved csv to: {args.csv}")


if __name__ == "__main__":
    main()
