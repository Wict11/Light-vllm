# Light-vLLM

项目基于nano-vLLM，实现一些功能。
* **支持Qwen3、Qwen-MoE、LLama2的模型**
* **添加混合批次调度** - 参照vLLM v1逻辑，基于token预算、最大prefill数量约束调度P、D请求
* **添加chunked prefill功能** - 限制prefill单次最大插入序列长度
* **添加异步调度功能** - 基于占位符机制实现decode阶段的提前调度；基于有效长度实现chunked prefill阶段的提前调度
* **其他功能实现中。。**

## Chunked Prefill 设计逻辑

* 🚀 **Scheduler layer** - 为长序列切分chunk并添加至调度队列，优先级依次是：running队列中的prefill阶段序列、waiting队列中的序列、running队列中decode阶段的序列
* 📖 **LLM engine layer** - 额外传入num_prefill_tokens和num_decode_tokens数据，区分混合prefill和decode的批次
* 🧠 **Attention layer** - 针对混合批次，分别调用flash attn的函数接口来处理，最后合并数据并返回
* 💡 **Post progress** - 只有decode阶段序列要计算logits和更新产生的token

## 异步调度设计逻辑
**详情请看ASYNC_GUIDE.md文件**

## Installation

```bash
pip install git+https://github.com/Wict11/Light-vllm.git
```

## Manual Download

If you prefer to download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

**0.Test Configuration:**
- Hardware: A10 (24GB)
- Model: Qwen3-0.6B


**🌟1.chunked prefill性能结果：**

* **脚本：base_chunk_v4.py**

* **流量情况：**
五个背景短流（20tokens）
五个incast长流（1000tokens）
* **Baseline：Prefill First:**
<img width="1198" height="437" alt="image" src="https://github.com/user-attachments/assets/953d9f9f-c954-4bd3-8d6e-602a14f8e981" />

<img width="679" height="226" alt="image" src="https://github.com/user-attachments/assets/3d7a35a9-7d87-4cbc-a9aa-d0b250618f9e" />

* **Chunk_size = 512:**
  <img width="1188" height="438" alt="image" src="https://github.com/user-attachments/assets/93109a4d-580f-4f01-991c-36cf22909430" />

  <img width="736" height="201" alt="image" src="https://github.com/user-attachments/assets/4c8d25f2-900a-4152-a163-916c192f0281" />
* **结果分析：**
  对混合批次的decode序列性能影响：开chunked prefill后，最大TPOT减少～3.1x，
  对混合批次的prefill序列性能影响：开chunked prefill后，TTFT至多增加～1.2x
  
* **待做测试：最大可支持长度**

  理论来说，chunked prefill也能够优化Attention计算时产生的临时矩阵，内存占用是O(N^2)，
  但由于用的Flash Attn，所以没有很大的临时矩阵，临时内存开销是O(N)，所以chunked prefill其实在这方面没什么很好的优化
  
**🌟2.异步调度性能结果：**

* **脚本：bench_async.py**

* **多小流并发场景：**

* ***🎬场景一:128条800token流量并发***

  *  **串行结果：**
  
    <img width="1536" height="106" alt="image" src="https://github.com/user-attachments/assets/d4b19163-2cc5-410b-9b51-6e71f18ceed2" />

  * **并行结果：**

    <img width="1526" height="114" alt="image" src="https://github.com/user-attachments/assets/76c0e585-22c2-4546-a759-4827419b8cc0" />


  * **结果分析：**
      从13977tok/s优化至17193tok/s，**22.6%左右提升**

* ***🎬场景二:128条800token流量并发 + chunked prefill***
  *  **chunk size: 512tok**  

  *  **串行结果：**

    <img width="1532" height="106" alt="image" src="https://github.com/user-attachments/assets/a6e3471b-0e1e-4b4d-93b8-1ce42d0ad5e2" />


  * **并行结果：**

    <img width="1548" height="104" alt="image" src="https://github.com/user-attachments/assets/3a825b11-225f-411b-b6b8-25081b049338" />


  * **结果分析：**
      从8555tok/s优化至11002tok/s，**28.6%左右提升**

      开了chunked prefill吞吐会下降是因为目前只加一个chunk进入混合批次，没跑满最大单批次可添加的token数，后续再修改为根据总的token预算来
    





