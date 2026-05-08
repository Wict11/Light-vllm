import argparse
import csv
import math
import os
from dataclasses import dataclass

import torch

from nanovllm.layers.attention import Attention
from nanovllm.utils.context import reset_context, set_context


@dataclass
class CaseConfig:
    prefill_seqs: int
    decode_seqs: int
    prefill_q_len: int
    prefill_prefix_len: int
    decode_ctx_len: int
    block_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype


def _build_case_tensors(cfg: CaseConfig):
    assert cfg.prefill_seqs >= 0
    assert cfg.decode_seqs >= 0
    assert cfg.prefill_q_len >= 1
    assert cfg.prefill_prefix_len >= 0
    assert cfg.decode_ctx_len >= 1

    num_seqs = cfg.prefill_seqs + cfg.decode_seqs
    if num_seqs <= 0:
        raise ValueError("prefill_seqs + decode_seqs must be > 0")

    query_lens = [cfg.prefill_q_len] * cfg.prefill_seqs + [1] * cfg.decode_seqs
    context_lens = [
        cfg.prefill_prefix_len + cfg.prefill_q_len
    ] * cfg.prefill_seqs + [cfg.decode_ctx_len] * cfg.decode_seqs

    max_context = max(context_lens)
    max_blocks_per_seq = max(1, math.ceil(max_context / cfg.block_size))

    num_prefill_tokens = cfg.prefill_seqs * cfg.prefill_q_len
    num_decode_tokens = cfg.decode_seqs
    total_tokens = num_prefill_tokens + num_decode_tokens

    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    for ql, kl in zip(query_lens, context_lens):
        cu_seqlens_q.append(cu_seqlens_q[-1] + ql)
        cu_seqlens_k.append(cu_seqlens_k[-1] + kl)

    block_tables = []
    for seq_idx in range(num_seqs):
        start_block = seq_idx * max_blocks_per_seq
        table = [start_block + i for i in range(max_blocks_per_seq)]
        block_tables.append(table)

    slot_mapping = []
    for seq_idx in range(num_seqs):
        table = block_tables[seq_idx]

        def pos_to_slot(pos: int) -> int:
            blk = pos // cfg.block_size
            off = pos % cfg.block_size
            return table[blk] * cfg.block_size + off

        if seq_idx < cfg.prefill_seqs:
            start = cfg.prefill_prefix_len
            for p in range(start, start + cfg.prefill_q_len):
                slot_mapping.append(pos_to_slot(p))
        else:
            slot_mapping.append(pos_to_slot(cfg.decode_ctx_len - 1))

    device = "cuda"
    q = torch.randn(
        total_tokens, cfg.num_heads, cfg.head_dim, dtype=cfg.dtype, device=device
    )
    k = torch.randn(
        total_tokens, cfg.num_kv_heads, cfg.head_dim, dtype=cfg.dtype, device=device
    )
    v = torch.randn(
        total_tokens, cfg.num_kv_heads, cfg.head_dim, dtype=cfg.dtype, device=device
    )

    cu_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
    cu_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
    slot_t = torch.tensor(slot_mapping, dtype=torch.int32, device=device)
    context_lens_t = torch.tensor(context_lens, dtype=torch.int32, device=device)
    block_tables_t = torch.tensor(block_tables, dtype=torch.int32, device=device)

    total_blocks = num_seqs * max_blocks_per_seq
    k_cache = torch.randn(
        total_blocks,
        cfg.block_size,
        cfg.num_kv_heads,
        cfg.head_dim,
        dtype=cfg.dtype,
        device=device,
    )
    v_cache = torch.randn(
        total_blocks,
        cfg.block_size,
        cfg.num_kv_heads,
        cfg.head_dim,
        dtype=cfg.dtype,
        device=device,
    )

    attn = Attention(
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        scale=cfg.head_dim ** -0.5,
        num_kv_heads=cfg.num_kv_heads,
    ).to(device)
    attn.k_cache = k_cache
    attn.v_cache = v_cache

    set_context(
        True,
        cu_q_t,
        cu_k_t,
        int(max(query_lens)),
        int(max(context_lens)),
        slot_t,
        context_lens_t,
        block_tables_t,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
    )

    return attn, q, k, v


def _run_once(attn, q, k, v):
    with torch.inference_mode():
        o = attn(q, k, v)
    return o


def bench_mode(cfg: CaseConfig, mode: str, warmup: int, repeat: int):
    os.environ["LIGHT_VLLM_MIXED_ATTN_MODE"] = mode
    attn, q, k, v = _build_case_tensors(cfg)

    for _ in range(warmup):
        _run_once(attn, q, k, v)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    lat = []
    for _ in range(repeat):
        start.record()
        _run_once(attn, q, k, v)
        end.record()
        torch.cuda.synchronize()
        lat.append(start.elapsed_time(end))

    out = _run_once(attn, q, k, v)
    mean_ms = float(torch.tensor(lat).mean().item())
    p95_ms = float(torch.quantile(torch.tensor(lat), 0.95).item())
    return mean_ms, p95_ms, out


def run_case(cfg: CaseConfig, warmup: int, repeat: int):
    unified_mean, unified_p95, out_unified = bench_mode(
        cfg, mode="unified", warmup=warmup, repeat=repeat
    )
    split_mean, split_p95, out_split = bench_mode(
        cfg, mode="split", warmup=warmup, repeat=repeat
    )
    auto_mean, auto_p95, out_auto = bench_mode(
        cfg, mode="auto", warmup=warmup, repeat=repeat
    )

    max_abs_diff = float((out_unified - out_split).abs().max().item())
    auto_vs_unified = float((out_auto - out_unified).abs().max().item())

    best_mode = "unified" if unified_mean <= split_mean else "split"
    return {
        "prefill_seqs": cfg.prefill_seqs,
        "decode_seqs": cfg.decode_seqs,
        "prefill_q_len": cfg.prefill_q_len,
        "prefill_prefix_len": cfg.prefill_prefix_len,
        "decode_ctx_len": cfg.decode_ctx_len,
        "unified_mean_ms": unified_mean,
        "split_mean_ms": split_mean,
        "auto_mean_ms": auto_mean,
        "unified_p95_ms": unified_p95,
        "split_p95_ms": split_p95,
        "auto_p95_ms": auto_p95,
        "best_mode": best_mode,
        "speedup_best_vs_unified": unified_mean / min(unified_mean, split_mean),
        "max_abs_diff_unified_split": max_abs_diff,
        "max_abs_diff_auto_unified": auto_vs_unified,
    }


def _parse_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Benchmark mixed-batch attention strategy")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    parser.add_argument("--prefill_seqs", type=int, default=2)
    parser.add_argument("--decode_seqs", type=int, default=16)
    parser.add_argument("--prefill_q_len", type=int, default=128)
    parser.add_argument("--prefill_prefix_len", type=int, default=128)
    parser.add_argument("--decode_ctx_len", type=int, default=1024)

    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep_prefill_seqs", type=str, default="1,2,4")
    parser.add_argument("--sweep_decode_seqs", type=str, default="1,4,8,16,32")
    parser.add_argument("--sweep_prefill_q_len", type=str, default="16,32,64,128,256")
    parser.add_argument("--sweep_prefill_prefix_len", type=str, default="0,64,256")
    parser.add_argument("--sweep_decode_ctx_len", type=str, default="256,512,1024,2048")
    parser.add_argument("--csv", type=str, default="")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if not args.sweep:
        cfg = CaseConfig(
            prefill_seqs=args.prefill_seqs,
            decode_seqs=args.decode_seqs,
            prefill_q_len=args.prefill_q_len,
            prefill_prefix_len=args.prefill_prefix_len,
            decode_ctx_len=args.decode_ctx_len,
            block_size=args.block_size,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
        )
        result = run_case(cfg, warmup=args.warmup, repeat=args.repeat)
        print(result)
        reset_context()
        return

    prefill_seqs_list = _parse_list(args.sweep_prefill_seqs)
    decode_seqs_list = _parse_list(args.sweep_decode_seqs)
    prefill_q_list = _parse_list(args.sweep_prefill_q_len)
    prefix_list = _parse_list(args.sweep_prefill_prefix_len)
    decode_ctx_list = _parse_list(args.sweep_decode_ctx_len)

    results = []
    total = (
        len(prefill_seqs_list)
        * len(decode_seqs_list)
        * len(prefill_q_list)
        * len(prefix_list)
        * len(decode_ctx_list)
    )
    idx = 0

    for p_seqs in prefill_seqs_list:
        for d_seqs in decode_seqs_list:
            for q_len in prefill_q_list:
                for pfx in prefix_list:
                    for d_ctx in decode_ctx_list:
                        idx += 1
                        cfg = CaseConfig(
                            prefill_seqs=p_seqs,
                            decode_seqs=d_seqs,
                            prefill_q_len=q_len,
                            prefill_prefix_len=pfx,
                            decode_ctx_len=d_ctx,
                            block_size=args.block_size,
                            num_heads=args.num_heads,
                            num_kv_heads=args.num_kv_heads,
                            head_dim=args.head_dim,
                            dtype=dtype,
                        )
                        try:
                            r = run_case(cfg, warmup=args.warmup, repeat=args.repeat)
                            results.append(r)
                            print(
                                f"[{idx}/{total}] p={p_seqs}, d={d_seqs}, q={q_len}, pfx={pfx}, ctx={d_ctx} "
                                f"unified={r['unified_mean_ms']:.3f}ms split={r['split_mean_ms']:.3f}ms "
                                f"best={r['best_mode']}"
                            )
                        except Exception as e:
                            print(
                                f"[{idx}/{total}] SKIP p={p_seqs}, d={d_seqs}, q={q_len}, pfx={pfx}, ctx={d_ctx} err={e}"
                            )
                        finally:
                            reset_context()

    if args.csv and results:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved csv to: {args.csv}")
    elif args.csv:
        print("No valid benchmark points produced; csv not written.")


if __name__ == "__main__":
    main()
