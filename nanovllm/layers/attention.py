import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

#     def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
#         context = get_context()
#         k_cache, v_cache = self.k_cache, self.v_cache
#         # k_cache.numel()返回张量中元素的总数
#         if k_cache.numel() and v_cache.numel():
#             store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
#         if context.is_prefill:
#             # 检查是否是混合批次（prefill + decode）
#             is_mixed = (
#                 hasattr(context, 'num_prefill_tokens') and
#                 hasattr(context, 'num_decode_tokens') and
#                 context.num_prefill_tokens is not None and
#                 context.num_decode_tokens is not None and
#                 context.num_prefill_tokens > 0 and
#                 context.num_decode_tokens > 0
#             )
            
#             if is_mixed:
#                 # 混合批次：分别处理 prefill 和 decode，然后合并
#                 num_prefill = context.num_prefill_tokens
#                 num_decode = context.num_decode_tokens

#                 # 计算有多少个 prefill 序列（cu_seqlens_q 前缀内的序列数）
#                 num_prefill_seqs = 0
#                 if context.cu_seqlens_q is not None:
#                     for i in range(len(context.cu_seqlens_q) - 1):
#                         if context.cu_seqlens_q[i + 1] <= num_prefill:
#                             num_prefill_seqs += 1
#                         else:
#                             break

#                 # 分离 prefill 和 decode 的 q/k/v
#                 q_prefill = q[:num_prefill]
#                 q_decode = q[num_prefill:]
#                 k_prefill = k[:num_prefill]
#                 v_prefill = v[:num_prefill]

#                 # 判断 prefill 序列是否真正有 prefix cache
#                 # （prefill 序列的 seqlen_k > seqlen_q 才代表有前缀缓存）
#                 prefill_has_prefix_cache = False
#                 if context.cu_seqlens_q is not None and context.cu_seqlens_k is not None:
#                     k_end = context.cu_seqlens_k[num_prefill_seqs].item()
#                     q_end = context.cu_seqlens_q[num_prefill_seqs].item()
#                     prefill_has_prefix_cache = k_end > q_end

#                 if prefill_has_prefix_cache:
#                     # 有 prefix cache：从 KV cache 读取 K/V
#                     k_prefill, v_prefill = k_cache, v_cache

#                 # 构建正确的 prefill cu_seqlens（支持多个 prefill 序列）
#                 cu_seqlens_q_prefill = context.cu_seqlens_q[:num_prefill_seqs + 1]
#                 cu_seqlens_k_prefill = (
#                     context.cu_seqlens_k[:num_prefill_seqs + 1]
#                     if prefill_has_prefix_cache
#                     else cu_seqlens_q_prefill
#                 )
#                 max_seqlen_q_p = int((cu_seqlens_q_prefill[1:] - cu_seqlens_q_prefill[:-1]).max().item())
#                 max_seqlen_k_p = int((cu_seqlens_k_prefill[1:] - cu_seqlens_k_prefill[:-1]).max().item())

#                 # 处理 prefill 部分
#                 o_prefill = flash_attn_varlen_func(
#                     q_prefill, k_prefill, v_prefill,
#                     max_seqlen_q=max_seqlen_q_p,
#                     cu_seqlens_q=cu_seqlens_q_prefill,
#                     max_seqlen_k=max_seqlen_k_p,
#                     cu_seqlens_k=cu_seqlens_k_prefill,
#                     softmax_scale=self.scale,
#                     # 只有 prefill 序列真正有 prefix cache 时才传 block_table
#                     block_table=(
#                         context.block_tables[:num_prefill_seqs]
#                         if prefill_has_prefix_cache and context.block_tables is not None
#                         else None
#                     ),
#                     causal=True
#                 )

#                 # 处理 decode 部分 - 使用 kvcache
#                 # 正确切分 decode 序列的 context_lens 和 block_table
#                 decode_context_lens = context.context_lens[num_prefill_seqs:] if context.context_lens is not None else None
#                 decode_block_tables = context.block_tables[num_prefill_seqs:] if context.block_tables is not None else None

#                 o_decode = flash_attn_with_kvcache(
#                     q_decode.unsqueeze(1),  # (num_decode, 1, num_heads, head_dim)
#                     k_cache,
#                     v_cache,
#                     cache_seqlens=decode_context_lens,
#                     block_table=decode_block_tables,
#                     softmax_scale=self.scale,
#                     causal=True
#                 ).squeeze(1)

#                 # 合并结果
#                 o = torch.cat([o_prefill, o_decode], dim=0)
#             else:
#                 # 纯 prefill 批次
#                 if context.block_tables is not None:    # prefix cache
#                     k, v = k_cache, v_cache
#                 # 注意：varlen 不接受 block_table 参数
#                 # print(f"q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
#                 o = flash_attn_varlen_func(
#                     q, k, v,
#                     max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
#                     max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
#                     softmax_scale=self.scale, causal=True
#                 )
#         else:
#             # 纯 decode 批次
#             o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
#                                         cache_seqlens=context.context_lens, block_table=context.block_tables, 
#                                         softmax_scale=self.scale, causal=True)
#         return o
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # k_cache.numel()返回张量中元素的总数
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            # 检查是否是混合批次（prefill + decode）
            is_mixed = (
                hasattr(context, 'num_prefill_tokens') and
                hasattr(context, 'num_decode_tokens') and
                context.num_prefill_tokens is not None and
                context.num_decode_tokens is not None and
                context.num_prefill_tokens > 0 and
                context.num_decode_tokens > 0
            )
            
            if is_mixed:
                use_unified_varlen = True  # 开关：True 表示把 decode 当作长度为 1 的 prefill 统一处理，False 保留原串行逻辑

                if use_unified_varlen:
                    # 统一使用 flash_attn_varlen_func 处理整个混合批次
                    o = flash_attn_varlen_func(
                        q, 
                        k_cache if context.block_tables is not None else k, 
                        v_cache if context.block_tables is not None else v,
                        max_seqlen_q=context.max_seqlen_q,
                        cu_seqlens_q=context.cu_seqlens_q,
                        max_seqlen_k=context.max_seqlen_k,
                        cu_seqlens_k=context.cu_seqlens_k,
                        softmax_scale=self.scale,
                        block_table=context.block_tables if context.block_tables is not None else None,
                        causal=True
                    )
                else:
                    # 混合批次：分别处理 prefill 和 decode，然后合并
                    num_prefill = context.num_prefill_tokens

                    # 计算有多少个 prefill 序列（cu_seqlens_q 前缀内的序列数）
                    num_prefill_seqs = 0
                    if context.cu_seqlens_q is not None:
                        for i in range(len(context.cu_seqlens_q) - 1):
                            if context.cu_seqlens_q[i + 1] <= num_prefill:
                                num_prefill_seqs += 1
                            else:
                                break

                    # 分离 prefill 和 decode 的 q/k/v
                    q_prefill = q[:num_prefill]
                    q_decode = q[num_prefill:]
                    k_prefill = k[:num_prefill]
                    v_prefill = v[:num_prefill]
                    
                    # 调试打印：检查拆分前后的 q, k, v 的形状（注意 Qwen 的 num_heads 和 num_kv_heads）
                    # print(f"\n[DEBUG Split Attention]")
                    # print(f"Total q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
                    # print(f"q_prefill={q_prefill.shape}, k_prefill={k_prefill.shape}")
                    # print(f"q_decode={q_decode.shape}")

                    # 判断 prefill 序列是否真正有 prefix cache
                    # （prefill 序列的 seqlen_k > seqlen_q 才代表有前缀缓存）
                    prefill_has_prefix_cache = False
                    if context.cu_seqlens_q is not None and context.cu_seqlens_k is not None:
                        k_end = context.cu_seqlens_k[num_prefill_seqs].item()
                        q_end = context.cu_seqlens_q[num_prefill_seqs].item()
                        prefill_has_prefix_cache = k_end > q_end

                    if prefill_has_prefix_cache:
                        # 有 prefix cache：从 KV cache 读取 K/V
                        k_prefill, v_prefill = k_cache, v_cache

                    # 构建正确的 prefill cu_seqlens（支持多个 prefill 序列）
                    cu_seqlens_q_prefill = context.cu_seqlens_q[:num_prefill_seqs + 1]
                    cu_seqlens_k_prefill = (
                        context.cu_seqlens_k[:num_prefill_seqs + 1]
                        if prefill_has_prefix_cache
                        else cu_seqlens_q_prefill
                    )
                    max_seqlen_q_p = int((cu_seqlens_q_prefill[1:] - cu_seqlens_q_prefill[:-1]).max().item())
                    max_seqlen_k_p = int((cu_seqlens_k_prefill[1:] - cu_seqlens_k_prefill[:-1]).max().item())

                    # 处理 prefill 部分
                    o_prefill = flash_attn_varlen_func(
                        q_prefill, k_prefill, v_prefill,
                        max_seqlen_q=max_seqlen_q_p,
                        cu_seqlens_q=cu_seqlens_q_prefill,
                        max_seqlen_k=max_seqlen_k_p,
                        cu_seqlens_k=cu_seqlens_k_prefill,
                        softmax_scale=self.scale,
                        # 只有 prefill 序列真正有 prefix cache 时才传 block_table
                        block_table=(
                            context.block_tables[:num_prefill_seqs]
                            if prefill_has_prefix_cache and context.block_tables is not None
                            else None
                        ),
                        causal=True
                    )

                    # 处理 decode 部分 - 使用 kvcache
                    # 按 q_decode 的真实 batch 大小对齐 decode 的辅助张量，避免 shape 不匹配
                    num_decode = int(q_decode.size(0))
                    if num_decode > 0:
                        total_seqs = int(context.cu_seqlens_q.numel() - 1) if context.cu_seqlens_q is not None else num_prefill_seqs + num_decode
                        decode_start = max(0, total_seqs - num_decode)

                        decode_context_lens = (
                            context.context_lens[decode_start:decode_start + num_decode]
                            if context.context_lens is not None else None
                        )
                        decode_block_tables = (
                            context.block_tables[decode_start:decode_start + num_decode]
                            if context.block_tables is not None else None
                        )

                        # flash-attn 要求 block_table 形状为 (batch_size, max_num_blocks_per_seq)
                        if decode_block_tables is not None and decode_block_tables.ndim == 1:
                            decode_block_tables = decode_block_tables.unsqueeze(0)

                        o_decode = flash_attn_with_kvcache(
                            q_decode.unsqueeze(1),  # (num_decode, 1, num_heads, head_dim)
                            k_cache,
                            v_cache,
                            cache_seqlens=decode_context_lens,
                            block_table=decode_block_tables,
                            softmax_scale=self.scale,
                            causal=True
                        ).squeeze(1)
                    else:
                        o_decode = q_decode

                    # 合并结果
                    o = torch.cat([o_prefill, o_decode], dim=0)
            else:
                # 纯 prefill 批次
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                # 注意：varlen 不接受 block_table 参数
                # print(f"q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
                # 这里也要传block_tables!!
                o = flash_attn_varlen_func(
                    q, k, v,
                    max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                    block_table=context.block_tables if context.block_tables is not None else None,
                    softmax_scale=self.scale, causal=True
                )
        else:
            # 纯 decode 批次
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o

