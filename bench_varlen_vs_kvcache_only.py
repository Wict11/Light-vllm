import argparse
import csv
import math

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


def build_case(
    prefill_q_len: int,
    decode_seqs: int,
    decode_ctx_len: int,
    prefill_seqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
):
    if prefill_q_len < 1:
        raise ValueError("prefill_q_len must be >= 1")
    if decode_seqs < 0:
        raise ValueError("decode_seqs must be >= 0")
    if decode_ctx_len < 1:
        raise ValueError("decode_ctx_len must be >= 1")
    if prefill_seqs < 1:
        raise ValueError("prefill_seqs must be >= 1")

    device = "cuda"
    total_seqs = prefill_seqs + decode_seqs
    query_lens = [prefill_q_len] * prefill_seqs + [1] * decode_seqs
    k_lens = [prefill_q_len] * prefill_seqs + [decode_ctx_len] * decode_seqs

    # Dense cache for kvcache-only path.
    max_k = max(k_lens)
    k_cache = torch.randn(
        total_seqs,
        max_k,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    v_cache = torch.randn(
        total_seqs,
        max_k,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    # Flatten inputs for varlen path.
    cu_q = [0]
    cu_k = [0]
    q_list = []
    k_list = []
    v_list = []

    for i in range(total_seqs):
        ql = query_lens[i]
        kl = k_lens[i]
        q_i = torch.randn(ql, num_heads, head_dim, dtype=dtype, device=device)
        q_list.append(q_i)
        k_i = k_cache[i, :kl]
        v_i = v_cache[i, :kl]
        k_list.append(k_i)
        v_list.append(v_i)
        cu_q.append(cu_q[-1] + ql)
        cu_k.append(cu_k[-1] + kl)

    q_flat = torch.cat(q_list, dim=0)
    k_flat = torch.cat(k_list, dim=0)
    v_flat = torch.cat(v_list, dim=0)

    cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=device)

    decode_cache_lens = None
    if decode_seqs > 0:
        decode_cache_lens = torch.full(
            (decode_seqs,), decode_ctx_len, dtype=torch.int32, device=device
        )

    prefill_cache_lens = torch.full(
        (prefill_seqs,), prefill_q_len, dtype=torch.int32, device=device
    )

    return {
        "q_flat": q_flat,
        "k_flat": k_flat,
        "v_flat": v_flat,
        "cu_q": cu_q_t,
        "cu_k": cu_k_t,
        "max_q": max(query_lens),
        "max_k": max(k_lens),
        "q_prefill": torch.stack(q_list[:prefill_seqs], dim=0),
        "q_decode": torch.stack(q_list[prefill_seqs:], dim=0) if decode_seqs > 0 else None,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "prefill_cache_lens": prefill_cache_lens,
        "decode_cache_lens": decode_cache_lens,
    }


def run_varlen(case, scale: float):
    return flash_attn_varlen_func(
        case["q_flat"],
        case["k_flat"],
        case["v_flat"],
        cu_seqlens_q=case["cu_q"],
        cu_seqlens_k=case["cu_k"],
        max_seqlen_q=case["max_q"],
        max_seqlen_k=case["max_k"],
        softmax_scale=scale,
        causal=True,
    )


def run_kvcache_only(case, scale: float):
    outs = []

    # Prefill requests also use with_kvcache path.
    o_prefill = flash_attn_with_kvcache(
        case["q_prefill"],
        case["k_cache"][: case["q_prefill"].size(0)],
        case["v_cache"][: case["q_prefill"].size(0)],
        cache_seqlens=case["prefill_cache_lens"],
        softmax_scale=scale,
        causal=True,
    )
    outs.append(o_prefill.reshape(-1, o_prefill.size(-2), o_prefill.size(-1)))

    if case["q_decode"] is not None:
        p = case["q_prefill"].size(0)
        o_decode = flash_attn_with_kvcache(
            case["q_decode"],
            case["k_cache"][p:],
            case["v_cache"][p:],
            cache_seqlens=case["decode_cache_lens"],
            softmax_scale=scale,
            causal=True,
        )
        outs.append(o_decode.reshape(-1, o_decode.size(-2), o_decode.size(-1)))

    return torch.cat(outs, dim=0)


def bench(fn, warmup: int, repeat: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    lat = []
    for _ in range(repeat):
        st.record()
        fn()
        ed.record()
        torch.cuda.synchronize()
        lat.append(st.elapsed_time(ed))
    t = torch.tensor(lat)
    return float(t.mean().item()), float(torch.quantile(t, 0.95).item())


def parse_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser("Compare unified varlen vs kvcache-only")
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
    parser.add_argument("--csv", type=str, default="varlen_vs_kvcache_only.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    scale = args.head_dim ** -0.5

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
                        prefill_q_len=ql,
                        decode_seqs=d,
                        decode_ctx_len=ctx,
                        prefill_seqs=ps,
                        num_heads=args.num_heads,
                        num_kv_heads=args.num_kv_heads,
                        head_dim=args.head_dim,
                        dtype=dtype,
                    )

                    varlen_mean, varlen_p95 = bench(
                        lambda: run_varlen(case, scale), args.warmup, args.repeat
                    )
                    kvc_mean, kvc_p95 = bench(
                        lambda: run_kvcache_only(case, scale), args.warmup, args.repeat
                    )

                    best = "varlen" if varlen_mean <= kvc_mean else "kvcache_only"
                    ratio = kvc_mean / varlen_mean
                    row = {
                        "prefill_seqs": ps,
                        "prefill_q_len": ql,
                        "decode_seqs": d,
                        "decode_ctx_len": ctx,
                        "varlen_mean_ms": varlen_mean,
                        "kvcache_only_mean_ms": kvc_mean,
                        "varlen_p95_ms": varlen_p95,
                        "kvcache_only_p95_ms": kvc_p95,
                        "kvcache_over_varlen": ratio,
                        "best": best,
                    }
                    rows.append(row)
                    print(
                        f"[{idx}/{total}] ps={ps} q={ql} d={d} ctx={ctx} "
                        f"varlen={varlen_mean:.3f}ms kvcache_only={kvc_mean:.3f}ms best={best}"
                    )

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved csv to: {args.csv}")


if __name__ == "__main__":
    main()
