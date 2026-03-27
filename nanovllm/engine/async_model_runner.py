"""
异步模型执行器 - 支持 CUDA Stream 异步推理

核心特性：
1. 使用独立 CUDA Stream 异步执行
2. 非阻塞启动推理
3. 支持等待结果完成
"""

import torch
from typing import Optional, Tuple, Any

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.config import Config
from multiprocessing.synchronize import Event
import threading

class AsyncModelRunner:
    """
    异步模型执行器包装器
    
    包装标准 ModelRunner，添加异步执行支持：
    - run_async(): 非阻塞启动推理
    - wait_for_result(): 等待推理完成并获取结果
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化异步模型执行器
        
        Args:
            config: 配置对象
            rank: 当前进程的 rank
            event: 多进程同步事件
        """
        # 初始化标准的 ModelRunner
        self.model_runner = ModelRunner(config, rank, event)
        self._result = None
        self._thread: Optional[threading.Thread] = None # 用于异步启动一个线程
        self._exception: Optional[Exception] = None
        
        # 创建独立的推理 stream（只在主进程）
        if rank == 0:
            self.inference_stream = torch.cuda.Stream()
            self.pending_results = []  # [(result, event, args), ...]
            self.use_async = True
        else:
            # 其他 rank 不需要异步（主进程会协调）
            self.use_async = False
        
        self.rank = rank
        self.config = config
    
    def _background_call(self, seqs, is_prefill, num_prefill_tokens, num_decode_tokens):
        """后台线程：真正执行推理，阻塞直到 GPU 完成并返回"""
        try:
            self._result = self.model_runner.call(
                "run", seqs, is_prefill, num_prefill_tokens, num_decode_tokens
            )
        except Exception as e:
            self._exception = e

    def run_async(self, seqs, is_prefill: bool, num_prefill_tokens: int, num_decode_tokens: int) -> None:
        """
        异步启动推理，立即返回（不等待完成）
        非阻塞：把推理任务扔给后台线程，立即返回。
        主线程可以趁机执行 schedule(N+1)，实现 CPU 调度与 GPU 推理重叠。
        
        Args:
            seqs: 序列列表
            is_prefill: 是否是 prefill 阶段
            num_prefill_tokens: prefill token 数量
            num_decode_tokens: decode token 数量
            
        Returns:
            None (立即返回，不等待结果)
        """
        
        # ======================= CUDA Stream版本 ========================= #
        # call函数，会给子进程写入shm，通知其他进程同步启动run()函数，所以必须要，
        # 但call函数又会导致返回值阻塞，无法真正异步，所以额外开一个thread跑推理进程

#         if not self.use_async or self.rank != 0:
#             # 非主进程或未启用异步，直接同步执行
#             result = self.model_runner.call("run", seqs, is_prefill, num_prefill_tokens, num_decode_tokens)
#             self.pending_results = [(result, None, None)]
#             return
        
#         # 在独立 stream 中异步执行
#         with torch.cuda.stream(self.inference_stream):
#             # 执行推理
#             result = self.model_runner.run(seqs, is_prefill, num_prefill_tokens, num_decode_tokens)
            
#             # 创建同步事件
#             event = torch.cuda.Event()
#             event.record(self.inference_stream)
            
#             # 记录 pending 结果
#             self.pending_results.append((result, event, (seqs, is_prefill)))
        
#         # 立即返回，不等待
        
        
        # ================== Thread版本 =====================#
        self._result = None
        self._exception = None
        self._thread = threading.Thread(
            target=self._background_call,
            args=(seqs, is_prefill, num_prefill_tokens, num_decode_tokens),
            daemon=True,
        )
        self._thread.start() # 异步跑modelrunner
        # ⭐ 立即返回

    def wait_for_result(self) -> Optional[Any]:
        """
        阻塞等待后台线程完成（流水线唯一同步点），返回 token_ids。
        """
        # ======================= CUDA Stream版本 ========================= #
#         if not self.pending_results:
#             return None
        
#         result, event, args = self.pending_results.pop(0)
        
#         # 同步等待完成
#         if event is not None:
#             event.synchronize()
        
#         return result
        # ================== Thread版本 =====================#
        if self._thread is None:
            return None
        self._thread.join()   # ⭐ 唯一阻塞点，等待后台进程结束
        self._thread = None
        if self._exception is not None:
            raise self._exception
        return self._result

    def has_pending_results(self) -> bool:
        """检查是否有未完成的推理"""
        return len(self.pending_results) > 0

    def get_pending_count(self) -> int:
        """获取 pending 结果数量"""
        return len(self.pending_results)

    def call(self, method_name: str, *args, **kwargs):
        """
        调用 ModelRunner 的方法（兼容接口）
        
        注意：这是同步调用，主要用于非 run 的方法
        """
        return self.model_runner.call(method_name, *args, **kwargs)

    def exit(self):
        """清理资源"""
        # 等待所有 pending 完成
        while self.pending_results:
            self.wait_for_result()
        
        # 清理 stream
        if self.use_async:
            torch.cuda.synchronize()
            del self.inference_stream
        
        # 清理 model_runner
        self.model_runner.exit()

    def __getattr__(self, name):
        """
        代理其他属性到 model_runner
        保持接口兼容性
        """
        return getattr(self.model_runner, name)
