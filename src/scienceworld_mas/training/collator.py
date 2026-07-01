"""Causal-LM feature construction for supervised chat training."""

from __future__ import annotations

from dataclasses import dataclass

IGNORE_INDEX = -100


def example_to_features(tokenizer, messages: list[dict[str, str]], *, max_length: int) -> dict[str, list[int]]:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("training messages must end with an assistant response")

    prompt_messages = messages[:-1]
    completion = messages[-1]["content"]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    completion_ids = tokenizer(
        completion + (tokenizer.eos_token or ""),
        add_special_tokens=False,
    )["input_ids"]

    input_ids = list(prompt_ids) + list(completion_ids)
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class SupervisedDataCollator:
    tokenizer: object
    max_length: int

    def __call__(self, examples) -> dict:
        features = [
            example_to_features(self.tokenizer, list(example.messages), max_length=self.max_length)
            for example in examples
        ]
        max_len = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
            batch["labels"].append(item["labels"] + [IGNORE_INDEX] * pad)
        return batch
