"""LoRA supervised training for System1 SFT and System2 behavior cloning."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .collator import SupervisedDataCollator
from .examples import RoleName, load_training_examples


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    base_model: str
    data_dir: str
    role: RoleName
    output_dir: str
    train_split: str = "train"
    eval_split: str = "val"
    max_length: int = 2048
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_4bit: bool = False
    seed: int = 123
    log_every: int = 20
    max_steps: int | None = None


class ListDataset:
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _to_tensors(batch, torch):
    return {
        key: torch.tensor(value, dtype=torch.long)
        for key, value in batch.items()
    }


def train_supervised(config: SupervisedTrainingConfig) -> dict:
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if config.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if config.log_every < 1:
        raise ValueError("log_every must be >= 1")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_examples = load_training_examples(config.data_dir, role=config.role, split=config.train_split)
    if not train_examples:
        raise ValueError(f"no training examples for role={config.role} split={config.train_split}")
    eval_path = Path(config.data_dir) / config.role / f"{config.eval_split}.jsonl"
    eval_examples = (
        load_training_examples(config.data_dir, role=config.role, split=config.eval_split)
        if eval_path.exists()
        else []
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    model_kwargs = {"trust_remote_code": True}
    if config.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.train()

    collator = SupervisedDataCollator(tokenizer=tokenizer, max_length=config.max_length)

    def collate(examples):
        return _to_tensors(collator(examples), torch)

    train_loader = DataLoader(
        ListDataset(train_examples),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    eval_loader = (
        DataLoader(
            ListDataset(eval_examples),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        if eval_examples
        else None
    )
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    device = next(model.parameters()).device
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(name: str) -> None:
        checkpoint_dir = output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    def evaluate_loss() -> float | None:
        if eval_loader is None:
            return None
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in eval_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                losses.append(float(model(**batch).loss.detach().cpu()))
        model.train()
        return sum(losses) / len(losses) if losses else None

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    last_loss = math.nan
    best_eval_loss = math.inf
    pending_accumulation = 0

    def optimizer_step() -> None:
        nonlocal global_step, pending_accumulation
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        pending_accumulation = 0

    for epoch in range(config.epochs):
        for batch_index, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            last_loss = float(outputs.loss.detach().cpu())
            pending_accumulation += 1
            if pending_accumulation >= config.gradient_accumulation_steps:
                optimizer_step()
                if global_step % config.log_every == 0:
                    print(f"[train] epoch={epoch + 1} step={global_step} loss={last_loss:.4f}")
                if config.max_steps is not None and global_step >= config.max_steps:
                    break
        if pending_accumulation:
            optimizer_step()
            if global_step % config.log_every == 0:
                print(f"[train] epoch={epoch + 1} step={global_step} loss={last_loss:.4f}")

        eval_loss = evaluate_loss()
        if eval_loss is not None:
            print(f"[eval] epoch={epoch + 1} step={global_step} loss={eval_loss:.4f}")
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint("best")

        if config.max_steps is not None and global_step >= config.max_steps:
            break

    save_checkpoint("last")
    if math.isinf(best_eval_loss):
        save_checkpoint("best")
    return {
        "role": config.role,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "steps": global_step,
        "last_loss": last_loss,
        "best_eval_loss": None if math.isinf(best_eval_loss) else best_eval_loss,
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--role", choices=("system1", "system2"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_supervised(SupervisedTrainingConfig(**vars(args)))
    print(metrics)


if __name__ == "__main__":
    main()
