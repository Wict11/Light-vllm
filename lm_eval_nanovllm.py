import argparse
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from lm_eval.api.model import LM
from lm_eval.evaluator import simple_evaluate
from nanovllm import LLM as NanoLLM
from nanovllm import SamplingParams


class NanoVLLMForLMEval(LM):
    """A minimal lm_eval adapter for nano-vllm focused on generation tasks."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str | None = None,
        max_model_len: int = 4096,
        chunk_prefill_size: int = 512,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        default_max_tokens: int = 512,
        default_temperature: float = 1e-9,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self._max_length = int(max_model_len)
        self.default_max_tokens = default_max_tokens
        self.default_temperature = max(default_temperature, 1e-9)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        self._model = NanoLLM(
            self.model_path,
            max_model_len=max_model_len,
            chunk_prefill_size=chunk_prefill_size,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer_path

    @property
    def eot_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def device(self):
        return "cuda:0"

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "NanoVLLMForLMEval currently supports generation-only tasks (generate_until)."
        )

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        raise NotImplementedError(
            "NanoVLLMForLMEval currently supports generation-only tasks (generate_until)."
        )

    @staticmethod
    def _normalize_until(until: Any) -> list[str]:
        if until is None:
            return []
        if isinstance(until, str):
            return [until]
        if isinstance(until, Iterable):
            return [x for x in until if isinstance(x, str) and x != ""]
        return []

    @staticmethod
    def _truncate_by_until(text: str, until_list: list[str]) -> str:
        cut = len(text)
        for stop in until_list:
            idx = text.find(stop)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut]

    @staticmethod
    def _extract_text(output_obj: Any) -> str:
        if isinstance(output_obj, dict) and "text" in output_obj:
            return str(output_obj["text"])
        if hasattr(output_obj, "outputs") and output_obj.outputs:
            first = output_obj.outputs[0]
            if hasattr(first, "text"):
                return str(first.text)
        if hasattr(output_obj, "text"):
            return str(output_obj.text)
        return str(output_obj)

    def _build_sampling_params(self, gen_kwargs: dict[str, Any]) -> SamplingParams:
        max_tokens = int(
            gen_kwargs.get(
                "max_gen_toks",
                gen_kwargs.get("max_tokens", self.default_max_tokens),
            )
        )

        temperature = gen_kwargs.get("temperature", self.default_temperature)
        try:
            temperature = float(temperature)
        except Exception:
            temperature = self.default_temperature
        if temperature <= 1e-10:
            temperature = 1e-9

        return SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=False,
        )

    def generate_until(
        self,
        requests: list[Any],
        disable_tqdm: bool = False,
    ) -> list[str]:
        prompts: list[str] = []
        sampling_params_list: list[SamplingParams] = []
        until_list_all: list[list[str]] = []

        for req in requests:
            args = req.args
            context = args[0]
            gen_kwargs = args[1] if len(args) > 1 else {}
            if gen_kwargs is None:
                gen_kwargs = {}

            prompts.append(context)
            sampling_params_list.append(self._build_sampling_params(gen_kwargs))
            until_list_all.append(self._normalize_until(gen_kwargs.get("until")))

        outputs = self._model.generate(
            prompts,
            sampling_params_list,
            use_tqdm=not disable_tqdm,
        )

        final_texts: list[str] = []
        for prompt, out_obj, until_list in zip(
            prompts, outputs, until_list_all, strict=True
        ):
            text = self._extract_text(out_obj)
            if text.startswith(prompt):
                text = text[len(prompt) :]
            text = self._truncate_by_until(text, until_list)
            final_texts.append(text)

        return final_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lm_eval with nano-vllm backend")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--tasks", type=str, default="gsm8k")
    parser.add_argument("--num_fewshot", type=int, default=8)
    parser.add_argument("--limit", type=float, default=20)
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--fewshot_as_multiturn", action="store_true")

    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--chunk_prefill_size", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enforce_eager", action="store_true")

    parser.add_argument("--default_max_tokens", type=int, default=512)
    parser.add_argument("--default_temperature", type=float, default=1e-9)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/lm_eval_runs/light_vllm_adapter",
    )
    parser.add_argument("--bootstrap_iters", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = NanoVLLMForLMEval(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        max_model_len=args.max_model_len,
        chunk_prefill_size=args.chunk_prefill_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        default_max_tokens=args.default_max_tokens,
        default_temperature=args.default_temperature,
    )

    task_list = [x.strip() for x in args.tasks.split(",") if x.strip()]
    if not task_list:
        raise ValueError("No task specified.")

    results = simple_evaluate(
        model=model,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        limit=None if args.limit < 0 else args.limit,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        bootstrap_iters=args.bootstrap_iters,
        log_samples=False,
    )

    if results is None:
        print("No results returned (non-main rank).")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_path = out_dir / f"results_{ts}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Saved results to:", out_path)
    print(json.dumps(results.get("results", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
