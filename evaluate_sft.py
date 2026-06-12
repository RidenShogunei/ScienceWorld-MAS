"""Evaluate Main/Sub adapters by held-out exact match and output validity."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from provenance import experiment_provenance
from sft_trainer import ensure_torch_set_submodule


MAIN_PATTERN = re.compile(r"\[subtask\](.*?)\[/subtask\]", re.DOTALL)
SUB_PATTERN = re.compile(
    r"\[action\](.*?)\[/action\]\s*\[subtask_done\](true|false)\[/subtask_done\]",
    re.DOTALL | re.IGNORECASE,
)


def load_samples(path: str | Path, category: str, limit: int) -> list[dict]:
    samples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            if sample.get("category") == category:
                samples.append(sample)
                if len(samples) >= limit:
                    break
    return samples


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def parse_output(category: str, text: str):
    match = (MAIN_PATTERN if category == "main" else SUB_PATTERN).search(text)
    if not match:
        return None
    if category == "main":
        return (normalize(match.group(1)),)
    return normalize(match.group(1)), match.group(2).lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/val.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--agent", choices=("main", "sub"), required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_torch_set_submodule()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if args.use_4bit:
        kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="auto",
        )
    elif torch.cuda.is_available():
        kwargs.update(dtype=torch.bfloat16, device_map={"": 0})
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    device = next(model.parameters()).device

    rows = []
    valid = exact = 0
    action_exact = done_exact = 0
    for sample in load_samples(args.data, args.agent, args.limit):
        prompt = tokenizer.apply_chat_template(
            sample["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        predicted = parse_output(args.agent, text)
        expected = parse_output(args.agent, sample["messages"][-1]["content"])
        is_valid = predicted is not None
        is_exact = is_valid and predicted == expected
        valid += int(is_valid)
        exact += int(is_exact)
        if args.agent == "sub" and is_valid:
            action_exact += int(predicted[0] == expected[0])
            done_exact += int(predicted[1] == expected[1])
        rows.append(
            {
                "prediction": text,
                "expected": sample["messages"][-1]["content"],
                "valid": is_valid,
                "exact": is_exact,
            }
        )

    metrics = {
        "agent": args.agent,
        "samples": len(rows),
        "valid_rate": valid / max(len(rows), 1),
        "exact_match": exact / max(len(rows), 1),
    }
    if args.agent == "sub":
        metrics["action_exact"] = action_exact / max(len(rows), 1)
        metrics["done_accuracy"] = done_exact / max(len(rows), 1)
    print(json.dumps(metrics, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "provenance": experiment_provenance(
                        {"base_model": args.base_model, "adapter": args.adapter, "data": args.data}
                    ),
                    "examples": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
