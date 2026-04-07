from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_num_prefills = config.max_num_prefills_per_batch  # 每批次最大prefill序列数
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.chunk_size = config.chunk_prefill_size  # 从配置读取 chunk 大小

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.mark_preempted()
        self.block_manager.deallocate(seq)
        if self.chunk_size < 999999:
            seq.prefilled_len = 0
        self.waiting.appendleft(seq)

    def _prefill_remaining(self, seq: Sequence) -> int:
        return seq.prefill_target_len - seq.num_cached_tokens - seq.prefilled_len

    def _clear_preempted_if_prefill_done(self, seq: Sequence):
        if seq.is_preempted and self._prefill_remaining(seq) <= 0:
            seq.clear_preempted()

    def _schedule_base(self) -> tuple[list[Sequence], bool, int, int]:
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            prefill_tokens = len(seq) - seq.num_cached_tokens
            if (
                num_batched_tokens + prefill_tokens > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += prefill_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True, num_batched_tokens, 0

        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False, 0, len(scheduled_seqs)
                
#     def schedule(self) -> tuple[list[Sequence], bool]:
#         # prefill
#         CHUNK_SIZE = self.chunk_size
#         scheduled_seqs = []
#         num_seqs = 0
#         num_batched_tokens = 0
#         # 标志位：当前 Batch 是否包含 Prefill 任务
#         has_prefill = False

#         # ------------------- 添加chunked prefill逻辑 ------------------
#         # 检查是否running队列中还有prefill未完成的序列
#         prefill_seq_found = None
#         for seq in list(self.running):
#             if num_seqs >= self.max_num_seqs:
#                 break
#             # 计算还需要prefill的长度（基于 prompt）
#             prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - seq.prefilled_len
#             if prompt_remaining <= 0:
#                 continue  # 已经prefill完成，跳过
#             this_chunk_size = min(prompt_remaining, CHUNK_SIZE)
#             # 计算token预算
#             if num_batched_tokens + this_chunk_size > self.max_num_batched_tokens:
#                 break
#             num_seqs += 1
#             num_batched_tokens += this_chunk_size
#             scheduled_seqs.append(seq)
#             has_prefill = True
#             prefill_seq_found = seq
#             break  # 每次只处理一个prefill序列的一个chunk
        
#         # 从 running 队列中临时移除已调度的 prefill 序列，避免在 decode 阶段重复处理
#         if prefill_seq_found is not None:
#             self.running.remove(prefill_seq_found)
#         # ------------------- 添加chunked prefill逻辑 ------------------

#         # 从 waiting 队列添加新的 prefill 请求（如果还没有 prefill）
#         if not has_prefill and self.waiting and num_seqs < self.max_num_seqs:
#             # 取出waiting队列的第一个序列
#             seq = self.waiting[0]
            
#             # ------------------ 添加chunked prefill逻辑 ------------------
#             # 计算本次可以处理的长度（基于 prompt）
#             prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - seq.prefilled_len
#             this_chunk_size = min(prompt_remaining, CHUNK_SIZE)
            
#             # 如果 prompt 已经完全 prefetched（prompt_remaining<=0），直接转入 running 等待 decode
#             if prompt_remaining <= 0:
#                 self.waiting.popleft()
#                 seq.status = SequenceStatus.RUNNING
#                 self.running.appendleft(seq)
#             else:
#                 # 使用实际的 chunk size 检查预算
#                 can_alloc = self.block_manager.can_allocate(seq)
#                 if this_chunk_size > 0 and num_batched_tokens + this_chunk_size <= self.max_num_batched_tokens and can_alloc:
#                     num_seqs += 1
#                     self.block_manager.allocate(seq)

#                     seq.status = SequenceStatus.RUNNING
#                     self.waiting.popleft()
#                     self.running.append(seq)
#                     scheduled_seqs.append(seq)
#                     num_batched_tokens += this_chunk_size  
#                     has_prefill = True
#                 elif not can_alloc:
#                     print(
#                         f"[SCHED DEBUG] cannot allocate KV blocks: free={len(self.block_manager.free_block_ids)}, "
#                         f"need={seq.num_blocks}, waiting={len(self.waiting)}, running={len(self.running)}"
#                     )
#                 elif this_chunk_size <= 0:
#                     print(f"[SCHED DEBUG] zero chunk size for seq {seq.seq_id}, prompt_remaining={prompt_remaining}, chunk_size_cfg={CHUNK_SIZE}")
#                 else:
#                     print(
#                         f"[SCHED DEBUG] token budget exceeded: num_batched_tokens={num_batched_tokens}, "
#                         f"chunk={this_chunk_size}, max={self.max_num_batched_tokens}"
#                     )

#         can_append_decode = False
#         if self.chunk_size < 999999 or not has_prefill:
#             can_append_decode = True

#         # decode
#         while self.running and num_seqs < self.max_num_seqs and can_append_decode:
#             seq = self.running.popleft()
#             while not self.block_manager.can_append(seq):
#                 # 如果添加不了新的block块了，启动抢占逻辑
#                 if self.running:
#                     # 如果有正在运行的序列，从running队列的右侧pop一个seq并抢占资源
#                     # 抢占逻辑就在下面的preempt函数里
#                     self.preempt(self.running.pop())
#                 else:
#                     # 如果没有正在运行的序列，直接将当前序列抢占
#                     self.preempt(seq)
#                     break
#             else:
#                 # 正常运行时
#                 # 检查这个 seq 是不是刚才已经作为 Prefill 块加进去了，如果是的话就放最后，继续下一个
#                 if seq in scheduled_seqs:
#                     # self.running.append(seq) # 已经在scheduler队列里了，最后会放进来
#                     continue
#                 # 检查是否是已经完成 Prefill 的 Decode 请求
#                 # remaining_prefill = len(seq) - seq.num_cached_tokens - seq.prefilled_len
#                 # 检查是否是已经完成 Prefill 的 Decode 请求（基于 prompt 长度判断）
#                 prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - seq.prefilled_len
#                 if prompt_remaining > 0:
#                     self.running.appendleft(seq) # 还没 Prefill 完的先不在这里处理
#                     continue

#                 if self.block_manager.can_append(seq) and num_batched_tokens + 1 <= self.max_num_batched_tokens:
#                     # self.block_manager.may_append(seq)
#                     # 只有在纯 decode 批次时才提前 append，混合批次不能提前 append
#                     if not has_prefill:
#                         self.block_manager.may_append(seq)
#                     num_batched_tokens += 1
#                     num_seqs += 1
#                     scheduled_seqs.append(seq)
#                 else:
#                     self.running.appendleft(seq) # 预算不够了，放回队头下次再说
#                     break

#         if not scheduled_seqs:
#             print(f"scheduled seqs EMPTY! running_seqs: {len(self.running)}, waiting_Seqs:{len(self.waiting)}")
#         assert scheduled_seqs
#         # running队列extendleft从队列左侧添加
#         # scheduled_seqs是按顺序添加的，例如[seq1,seq2,seq3]，running里还剩seq4,seq5没被选取到
#         # 如果直接添加，会变成[seq3,seq2,seq1]，但是调度的时候是从左边开始的，调度顺序就反了，
#         # 所以需要reversed(scheduled_seqs)，反转调度队列再添加到running队列的左边
#         self.running.extendleft(reversed(scheduled_seqs))

#         num_prefill_tokens = 0
#         num_decode_tokens = 0
#         for seq in scheduled_seqs:
#             prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - seq.prefilled_len
#             if prompt_remaining > 0:
#                 # Prefill 序列
#                 chunk_size = min(prompt_remaining, CHUNK_SIZE)
#                 num_prefill_tokens += chunk_size
#             else:
#                 # Decode 序列
#                 num_decode_tokens += 1
        
#         return scheduled_seqs, has_prefill, num_prefill_tokens, num_decode_tokens

#     def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill:bool=False) -> list[bool]:
#         # -------- 添加chunked prefill逻辑 ---------
#         '''
#         处理模型输出：
#         - 如果is_prefill为True，且序列长度超过1，则第一个是prefill的token_ids，后续是decode的token_ids
#         - 如果is_prefill为True且序列长度为1，则全部是prefill的token_ids
#         - 如果is_prefill为False，则全部是decode的token_ids
#         '''
#         # CHUNK_SIZE = 512
#         CHUNK_SIZE = self.chunk_size
#         if is_prefill:
#             if len(seqs) > 1:
#                 # 混合批次，第一个是prefill，后续是decode
#                 # prefill阶段不消耗token_ids
#                 prefill_seq = seqs[0]
#                 # remaining_len = len(prefill_seq) - prefill_seq.num_cached_tokens - prefill_seq.prefilled_len
#                 # prefill_chunk_size = min(remaining_len, CHUNK_SIZE)
#                 # prefill_seq.prefilled_len += prefill_chunk_size
#                 prompt_remaining = prefill_seq.num_prompt_tokens - prefill_seq.num_cached_tokens - prefill_seq.prefilled_len
#                 if prompt_remaining > 0:
#                     # 第一个序列是 prefill，更新 prefilled_len
#                     prefill_chunk_size = min(prompt_remaining, CHUNK_SIZE)
#                     prefill_seq.prefilled_len += prefill_chunk_size

#                 # decode阶段，处理后续序列（从seqs[1:]开始）
#                 # for seq, token_id in zip(seqs[1:], token_ids):
#                 for i, (seq, token_id) in enumerate(zip(seqs[1:], token_ids)):
#                     # print(f"  Processing decode seq {seq.seq_id}: token_id={token_id}, current_completion={seq.num_completion_tokens}")
#                     # 混合批次中的 decode 序列需要先 may_append 再 append_token
#                     self.block_manager.may_append(seq)
#                     seq.append_token(token_id)
#                     # print(f"  After append: seq {seq.seq_id} completion_tokens={seq.num_completion_tokens}")
#                     if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
#                         seq.status = SequenceStatus.FINISHED
#                         self.block_manager.deallocate(seq)
#                         self.running.remove(seq)
#             else:
#                 # 纯prefill，只更新prefilled_len
#                 prefill_seq = seqs[0]
#                 # remaining_len = len(prefill_seq) - prefill_seq.num_cached_tokens - prefill_seq.prefilled_len
#                 prompt_remaining = prefill_seq.num_prompt_tokens - prefill_seq.num_cached_tokens - prefill_seq.prefilled_len
#                 prefill_chunk_size = min(prompt_remaining, CHUNK_SIZE)
#                 prefill_seq.prefilled_len += prefill_chunk_size
#         else:
#             # decode阶段，处理所有序列
#             for seq, token_id in zip(seqs, token_ids):
#                 seq.append_token(token_id)
#                 if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
#                     seq.status = SequenceStatus.FINISHED
#                     self.block_manager.deallocate(seq)
#                     self.running.remove(seq)
#         # -------- 添加chunked prefill逻辑 ---------

    def _budget_can_add_prefill(self, num_seqs, num_batched_tokens, num_prefills, chunk_tokens):
        """检查是否还能加入一个 prefill 序列（三重约束）"""
        return (num_seqs < self.max_num_seqs
                and num_batched_tokens + chunk_tokens <= self.max_num_batched_tokens
                and num_prefills < self.max_num_prefills)

    def schedule(self) -> tuple[list[Sequence], bool, int, int]:
        if self.chunk_size >= 999999:
            return self._schedule_base()

        CHUNK_SIZE = self.chunk_size
        scheduled_seqs = []       # 本轮被调度的序列（prefill 在前，decode 在后）
        prefill_scheduled = []    # 被调度的 prefill 序列（用于从 running 中临时移除）
        num_seqs = 0
        num_batched_tokens = 0
        num_prefills = 0          # 当前批次已调度的 prefill 序列数
        has_prefill = False

        # ==================== 阶段1：从 running 队列中找 chunked prefill 续接 ====================
        # 遍历 running 中还未完成 prefill 的序列，按三重约束（token预算/序列数/prefill数）加入
        for seq in list(self.running):
            if not self._budget_can_add_prefill(num_seqs, num_batched_tokens, num_prefills, 1):
                break  # 预算或 prefill 数量已满，停止
            prompt_remaining = self._prefill_remaining(seq)
            if prompt_remaining <= 0:
                self._clear_preempted_if_prefill_done(seq)
                continue  # 已经 prefill 完成，跳过
            this_chunk_size = min(prompt_remaining, CHUNK_SIZE)
            if not self._budget_can_add_prefill(num_seqs, num_batched_tokens, num_prefills, this_chunk_size):
                break  # 这个 chunk 放不下了，停止（后面的可能更大，直接 break）
            num_seqs += 1
            num_batched_tokens += this_chunk_size
            num_prefills += 1
            scheduled_seqs.append(seq)
            prefill_scheduled.append(seq)
            has_prefill = True
            # break

        # 从 running 队列中临时移除已调度的 prefill 序列，避免在 decode 阶段重复处理
        for seq in prefill_scheduled:
            self.running.remove(seq)

        # ==================== 阶段2：从 waiting 队列取新 prefill ====================
        # 在 prefill 数量/token预算/序列数量还有余量时，持续从 waiting 取新序列
        while self.waiting and self._budget_can_add_prefill(num_seqs, num_batched_tokens, num_prefills, 1):
            seq = self.waiting[0]
            prompt_remaining = self._prefill_remaining(seq)
            this_chunk_size = min(prompt_remaining, CHUNK_SIZE)

            # 如果 prompt 已经完全 prefill 完成，直接转入 running 等待 decode
            if prompt_remaining <= 0:
                self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                self._clear_preempted_if_prefill_done(seq)
                self.running.appendleft(seq)
                continue

            # 检查约束：token 预算、KV block 分配
            can_alloc = self.block_manager.can_allocate(seq)
            if not self._budget_can_add_prefill(num_seqs, num_batched_tokens, num_prefills, this_chunk_size):
                break  # token 预算不够了
            if not can_alloc:
                break  # KV block 分配不了，后续的更分配不了，直接停止

            # 满足所有约束，加入调度
            num_seqs += 1
            num_batched_tokens += this_chunk_size
            num_prefills += 1
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            # self.running.append(seq)
            scheduled_seqs.append(seq)
            has_prefill = True
            # break

        # ==================== 阶段3：从 running 队列中加入 decode 序列 ====================
        can_append_decode = (self.chunk_size < 999999) or (not has_prefill)
        if can_append_decode:
            while self.running and num_seqs < self.max_num_seqs:
                seq = self.running.popleft()
                # 抢占逻辑：如果无法为该序列追加新 block
                while not self.block_manager.can_append(seq):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    # 跳过已经作为 prefill 加入的序列
                    if seq in scheduled_seqs:
                        continue
                    # 跳过还没完成 prefill 的序列（不应在 decode 阶段处理）
                    prompt_remaining = self._prefill_remaining(seq)
                    if prompt_remaining > 0:
                        self.running.appendleft(seq)
                        continue
                    self._clear_preempted_if_prefill_done(seq)
                    # 检查 token 预算（decode 每个序列消耗 1 个 token）
                    if num_batched_tokens + 1 <= self.max_num_batched_tokens:
                        # if not has_prefill:
                        #     self.block_manager.may_append(seq)
                        # 先在外部检查kv是否能放，在里头才能真正的放！！
                        self.block_manager.may_append(seq)
                        num_batched_tokens += 1
                        num_seqs += 1
                        scheduled_seqs.append(seq)
                    else:
                        self.running.appendleft(seq)  # 预算不够，放回队头
                        break

        # ==================== 断言与放回 ====================
        if not scheduled_seqs:
            print(f"scheduled seqs EMPTY! running_seqs: {len(self.running)}, waiting_seqs:{len(self.waiting)}")
        assert scheduled_seqs
        

        # 将调度的序列放回 running 队列左侧，保持顺序
        self.running.extendleft(reversed(scheduled_seqs))

        # ==================== 计算 prefill/decode token 数量 ====================
        num_prefill_tokens = 0
        num_decode_tokens = 0
        for seq in scheduled_seqs:
            prompt_remaining = self._prefill_remaining(seq)
            if prompt_remaining > 0:
                chunk_size = min(prompt_remaining, CHUNK_SIZE)
                num_prefill_tokens += chunk_size
            else:
                self._clear_preempted_if_prefill_done(seq)
                num_decode_tokens += 1
        # print(f"seqs_len: {len(scheduled_seqs)}, is_prefill: {has_prefill}, num_prefill_tokens: {num_prefill_tokens}, num_decode_tokens: {num_decode_tokens}")
        return scheduled_seqs, has_prefill, num_prefill_tokens, num_decode_tokens

#     def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill:bool=False) -> list[bool]:
#         '''
#         处理模型输出：
#         - prefill 序列：只更新 prefilled_len，不 append token（logits 已被丢弃）
#         - decode 序列：may_append + append_token，检查是否结束
#         - token_ids 只对应 decode 序列，按顺序一一对应
#         '''
#         CHUNK_SIZE = self.chunk_size
#         if is_prefill:
#             # 混合批次或纯 prefill：先处理所有 prefill 序列，再处理 decode 序列
#             # 分离 prefill 和 decode 序列
#             prefill_seqs = []
#             decode_seqs = []
#             for seq in seqs:
#                 prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - seq.prefilled_len
#                 if prompt_remaining > 0:
#                     prefill_seqs.append(seq)
#                 else:
#                     decode_seqs.append(seq)

#             # 更新所有 prefill 序列的 prefilled_len
#             for seq in prefill_seqs:
#                 prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - seq.prefilled_len
#                 prefill_chunk_size = min(prompt_remaining, CHUNK_SIZE)
#                 seq.prefilled_len += prefill_chunk_size

#             # 处理 decode 序列（token_ids 只对应 decode 序列）
#             for seq, token_id in zip(decode_seqs, token_ids):
#                 # 混合批次中的 decode 序列需要先 may_append 再 append_token
#                 self.block_manager.may_append(seq)
#                 seq.append_token(token_id)
#                 if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
#                     seq.status = SequenceStatus.FINISHED
#                     self.block_manager.deallocate(seq)
#                     self.running.remove(seq)
#         else:
#             # 纯 decode 批次：处理所有序列
#             for seq, token_id in zip(seqs, token_ids):
#                 seq.append_token(token_id)
#                 if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
#                     seq.status = SequenceStatus.FINISHED
#                     self.block_manager.deallocate(seq)
#                     self.running.remove(seq)
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill:bool=False) -> list[bool]:
        if self.chunk_size >= 999999:
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
            return

        '''
        处理模型输出：
        - 纯 prefill 序列：只更新 prefilled_len，不 append token
        - 刚好完成 prefill 的序列：更新 prefilled_len，并纳入 decode 接收首个 token
        - decode 序列：may_append + append_token，检查是否结束
        - token_ids 对应刚好完成 prefill 的序列以及 decode 序列，按顺序一一对应
        '''
        CHUNK_SIZE = self.chunk_size
        if is_prefill:
            decode_seqs = []
            
            # 首先遍历所有序列，处理 prefill 进度
            for seq in seqs:
                prompt_remaining = self._prefill_remaining(seq)
                if prompt_remaining > 0:
                    prefill_chunk_size = min(prompt_remaining, CHUNK_SIZE)
                    seq.prefilled_len += prefill_chunk_size
                    # 如果 prefill 刚好完成，它本轮就会输出一个 token，加入 decode_seqs
                    if prompt_remaining <= CHUNK_SIZE:
                        self._clear_preempted_if_prefill_done(seq)
                        decode_seqs.append(seq)
                else:
                    # 已经是纯 decode 状态的序列
                    self._clear_preempted_if_prefill_done(seq)
                    decode_seqs.append(seq)

            # 处理接收 token 的序列（token_ids 对应 decode_seqs 中的序列）
            for seq, token_id in zip(decode_seqs, token_ids):
                # 混合批次中需要先 may_append 再 append_token
                
                # ！！这里不先给token放到block里，等下一次调度再放！！
                # self.block_manager.may_append(seq)
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
        else:
            # 纯 decode 批次：处理所有序列
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)

