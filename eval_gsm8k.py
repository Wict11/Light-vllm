import sys
import re
import argparse
import pandas as pd
from tqdm import tqdm
from modelscope.msdatasets import MsDataset
from transformers import AutoTokenizer

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams

def extract_answer(text: str) -> str:
    """
    更高级的答案提取，适配具有 <think> 漫长推理链的模型。
    """
    # 1. 尝试去掉 <think> 到 </think> 之间的所有内容（非贪婪匹配）
    text_without_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 如果没找到闭合标签（说明可能因为长度限制被截断了），那就从最后一个 </think> 往后找
    if text_without_think.strip() == "" or text_without_think == text:
        if "</think>" in text:
            text_without_think = text.split("</think>")[-1]
    
    # 把去掉思考部分后的文本转小写，尝试匹配我们的预设强标志或者最后出现的自然语言回答
    if "####" in text_without_think:
        text_without_think = text_without_think.split("####")[-1]
    elif "the answer is" in text_without_think.lower():
        text_without_think = re.split(r'the answer is', text_without_think, flags=re.IGNORECASE)[-1]
        
    # 过滤可能存在的千分位逗号和美元符号
    text_without_think = text_without_think.replace(',', '')
    text_without_think = text_without_think.replace('$', '')
    
    # 匹配独立的数字块（过滤掉诸如结尾的 "."）
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text_without_think)
    if numbers:
        return numbers[-1]
    
    # 实在找不到，回退去最原始文本里找最后一个数字
    fallback_numbers = re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', ''))
    if fallback_numbers:
        return fallback_numbers[-1]
    return ""

def main(args):
    print(f"Loading GSM8K test dataset from ModelScope...")
    # 使用 ModelScope 下载数据集，添加 trust_remote_code=True 消除报错
    dataset = MsDataset.load('modelscope/gsm8k', subset_name='main', split='test', trust_remote_code=True)
    
    # 因为 msdataset 返回的格式可能和 hf 稍有区别，可以用遍历转为 list 方便处理
    dataset = list(dataset)
    
    # 如果限制了数量
    if args.limit > 0:
        dataset = dataset[:args.limit]
        
    print(f"Loaded {len(dataset)} test samples.")

    # 1. 组装输入 Prompts 和 Tokenizer
    print("Loading tokenizer to apply chat template...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    prompts = []
    expected_answers = []
    
    for item in dataset:
        q = item['question']
        # 提取真实答案：GSM8K 原始答案格式是 "...解析... #### 123"
        ans_text = item['answer']
        ans = ans_text.split("####")[-1].strip()
        expected_answers.append(ans)
        
        # 使用模型的 Chat Template
        # 强化系统指令：明确要求在思考之后用 #### 给出答案
        system_instruction = "You are an expert math solver. Think step by step in <think> tags, and then output your final answer directly right after the '#### ' marker."
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": q}
        ]
        
        # 渲染为标准的带有模板的字符串格式
        prompt_with_template = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt_with_template)

    # 2. 初始化 Nano-vLLM 引擎
    print(f"Initializing nano-vllm engine with model: {args.model_path}")
    llm = LLM(
        args.model_path, 
        max_model_len=args.max_model_len, 
        # chunk_prefill_size=args.chunk_size
    )
    
    # 3. 设置采样参数
    sampling_params = SamplingParams(
        temperature=1e-9,  # 接近贪婪搜索，保证每次出同样的解
        max_tokens=args.max_tokens, # 放宽 token 数防止思考被打断
        ignore_eos=False
    )

    # 4. 运行推理 (并发处理)
    print("Starting generation...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. 解析输出并计算得分 (Accuracy)
    correct_count = 0
    total_count = len(prompts)
    
    print("\n--- Evaluation Results ---")
    for i in tqdm(range(total_count), desc="Evaluating"):
        out_obj = outputs[i]
        
        # --- 健壮的文本提取逻辑 ---
        if isinstance(out_obj, dict) and 'text' in out_obj:
            generated_text = out_obj['text']
        elif hasattr(out_obj, "outputs") and len(out_obj.outputs) > 0 and hasattr(out_obj.outputs[0], 'text'):
            generated_text = out_obj.outputs[0].text
        elif hasattr(out_obj, "text"):
            generated_text = out_obj.text
        else:
            generated_text = str(out_obj)
            
        # 剔除引擎多带上的 prompt 痕迹
        if "assistant\n" in generated_text:
            generated_text = generated_text.split("assistant\n")[-1]
        
        # 提取最终计算出的数字
        pred_ans = extract_answer(generated_text)
        gt_ans = expected_answers[i]
        
        # 简单比对字符串数值
        is_correct = (pred_ans.strip() == gt_ans.strip())
        if is_correct:
            correct_count += 1
            
        # 打印前 3 个看看提取逻辑是否生效、有没有因为 1536 token 还被截断的问题
        if i < 3:
            print(f"\n[Sample {i}]")
            print(f"Q: {dataset[i]['question']}")
            # 如果内容太多只截取前 100 字符和最后 150 字符
            if len(generated_text) > 300:
                short_text = generated_text[:100] + "\n...[omitted]...\n" + generated_text[-150:]
            else:
                short_text = generated_text
            print(f"Generated text: \n{short_text}")
            print(f"Extracted: {pred_ans} | Expected: {gt_ans} -> {'✅' if is_correct else '❌'}")
            
    # 计算最终精度
    accuracy = (correct_count / total_count) * 100
    print("\n" + "="*50)
    print(f"Total Evaluated: {total_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Final Accuracy:  {accuracy:.2f}%")
    print("="*50)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, 
                       default="/mnt/workspace/nano_vllm/nano-vllm/Qwen/Qwen3-0.6B",
                       help="Path to model")
    # 想快速看前 10 个用这行: python eval_gsm8k.py --limit 10
    parser.add_argument("--limit", type=int, default=10, help="测试样本数，-1为全部")
    # 对于带有长思考链的模型，需要较大的上下文支持（这里默认为2048以囊括提问+作答）
    parser.add_argument("--max_model_len", type=int, default=2048)
    # 为推理留出足够的空间
    parser.add_argument("--max_tokens", type=int, default=1536, help="给模型多少长度去想步骤")
    args = parser.parse_args()
    main(args)