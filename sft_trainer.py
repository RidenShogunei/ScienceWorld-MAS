"""Train independent Main and Sub LoRA adapters on Multi-Square expert data."""

from __future__ import annotations

import os

import torch

# Force CUDA init before huggingface_hub import corrupts it
if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
    _ = torch.cuda.device_count()

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from agent_protocol import fit_sub_messages_for_inference, infer_protocol_from_schema, parse_env_sub_user
from provenance import experiment_provenance


def early_stop_state_after_epoch(
    *,
    val_loss: float,
    train_loss: float,
    best_val: float,
    train_loss_at_best_val: float,
    epochs_without_improvement: int,
    overfit_epochs: int,
    patience: int,
    overfit_patience: int,
    min_delta: float,
) -> dict[str, Any]:
    """Update early-stop counters after one epoch; stop on plateau or overfitting."""
    improved = val_loss < best_val - min_delta
    if improved:
        return {
            "best_val": val_loss,
            "train_loss_at_best_val": train_loss,
            "epochs_without_improvement": 0,
            "overfit_epochs": 0,
            "should_stop": False,
            "reason": None,
        }

    new_no_improve = epochs_without_improvement + 1
    is_overfit = (
        val_loss > best_val + min_delta
        and train_loss < train_loss_at_best_val - min_delta
    )
    new_overfit = overfit_epochs + 1 if is_overfit else 0

    should_stop = False
    reason = None
    if new_no_improve >= patience:
        should_stop = True
        reason = "patience"
    elif new_overfit >= overfit_patience:
        should_stop = True
        reason = "overfit"

    return {
        "best_val": best_val,
        "train_loss_at_best_val": train_loss_at_best_val,
        "epochs_without_improvement": new_no_improve,
        "overfit_epochs": new_overfit,
        "should_stop": should_stop,
        "reason": reason,
    }


def ensure_torch_set_submodule() -> None:
    """Backport torch.nn.Module.set_submodule for Transformers 5 on PyTorch 2.4."""
    if hasattr(torch.nn.Module, "set_submodule"):
        return

    def set_submodule(module, target: str, replacement: torch.nn.Module) -> None:
        if not target:
            raise ValueError("target must be a non-empty module path")
        atoms = target.split(".")
        parent = module
        for atom in atoms[:-1]:
            if not hasattr(parent, atom):
                raise AttributeError(f"{parent.__class__.__name__} has no submodule {atom}")
            parent = getattr(parent, atom)
            if not isinstance(parent, torch.nn.Module):
                raise AttributeError(f"{atom} is not a torch module")
        setattr(parent, atoms[-1], replacement)

    torch.nn.Module.set_submodule = set_submodule


class ChatDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        category: str,
        max_length: int,
        limit: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.category = category
        self.samples = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                if sample.get("category") != category:
                    continue
                self.samples.append(sample)
                if limit is not None and len(self.samples) >= limit:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        messages = sample["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("SFT samples must end with an assistant message")

        completion = messages[-1]["content"] + (self.tokenizer.eos_token or "")
        completion_ids = self.tokenizer(
            completion,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        completion_ids = completion_ids[: self.max_length]
        prompt_budget = max(self.max_length - len(completion_ids), 0)

        prompt_messages = messages[:-1]
        if self.category == "sub" and prompt_budget > 0:
            user = next(
                (message["content"] for message in prompt_messages if message.get("role") == "user"),
                "",
            )
            if "Valid actions:" in user:
                protocol = infer_protocol_from_schema(str(sample.get("schema", "")))
                contract, observation, valid_actions, history = parse_env_sub_user(user)
                prompt_messages = fit_sub_messages_for_inference(
                    self.tokenizer,
                    protocol,
                    task_context=contract,
                    observation=observation,
                    valid_actions=valid_actions,
                    recent_history=history,
                    max_input_length=prompt_budget,
                )

        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        input_ids = prompt_ids[-prompt_budget:] + completion_ids if prompt_budget else completion_ids
        labels = [-100] * min(len(prompt_ids), prompt_budget) + completion_ids
        if len(labels) != len(input_ids):
            raise RuntimeError("input/label alignment failed")
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(row["input_ids"].shape[0] for row in rows)
        batch: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            padding = max_length - row["input_ids"].shape[0]
            batch["input_ids"].append(
                torch.cat((row["input_ids"], torch.full((padding,), self.pad_token_id)))
            )
            batch["attention_mask"].append(
                torch.cat((row["attention_mask"], torch.zeros(padding, dtype=torch.long)))
            )
            batch["labels"].append(
                torch.cat((row["labels"], torch.full((padding,), -100, dtype=torch.long)))
            )
        return {key: torch.stack(values) for key, values in batch.items()}


def load_model(base_model: str, use_4bit: bool):
    ensure_torch_set_submodule()
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if use_4bit:
        kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
            device_map="auto",
        )
    else:
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.device_count() > 0 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    return model


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device: torch.device, max_batches: int | None) -> float:
    model.eval()
    losses = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        loss = model(**batch).loss
        if torch.isfinite(loss):
            losses.append(float(loss))
    model.train()
    return sum(losses) / max(len(losses), 1)


def save_adapter(model, tokenizer, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    unwrapped = model
    unwrapped.save_pretrained(destination)
    tokenizer.save_pretrained(destination)


def train_agent(args: argparse.Namespace, tokenizer, category: str) -> dict[str, float]:
    train_data = ChatDataset(args.train_data, tokenizer, category, args.max_length, args.train_limit)
    val_data = ChatDataset(args.val_data, tokenizer, category, args.max_length, args.val_limit)
    if not train_data or not val_data:
        raise ValueError(f"{category} requires non-empty train and validation data")

    collator = CausalCollator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model = load_model(args.base_model, args.use_4bit)
    init_adapter = None
    if category == "main":
        init_adapter = getattr(args, "init_main_adapter", None) or getattr(args, "init_adapter", None)
    elif category == "sub":
        init_adapter = getattr(args, "init_sub_adapter", None) or getattr(args, "init_adapter", None)
    if init_adapter:
        init_path = Path(init_adapter)
        if not init_path.exists():
            raise FileNotFoundError(f"init adapter not found: {init_path}")
        print(f"[{category}] loading adapter from {init_path}", flush=True)
        model = PeftModel.from_pretrained(model, str(init_path), is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    if torch.cuda.device_count() > 0 and not args.use_4bit:
        device = torch.device("cuda:0")
        print(f"[{category}] moving model to {device}...", flush=True)
        model = model.to(device)
    else:
        device = next(model.parameters()).device

    print(f"[{category}] device={device} creating optimizer...", flush=True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(f"[{category}] optimizer created, steps={len(train_loader)}", flush=True)
    total_updates = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    warmup_updates = int(total_updates * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / max(warmup_updates, 1), 1.0)
        if step < warmup_updates
        else max((total_updates - step) / max(total_updates - warmup_updates, 1), 0.0),
    )

    output_dir = Path(args.save_dir) / f"{category}_agent"
    best_dir = output_dir / "best"
    history = []
    best_val = float("inf")
    train_loss_at_best_val = float("inf")
    epochs_without_improvement = 0
    overfit_epochs = 0
    early_stopped = False
    early_stop_reason: str | None = None
    early_stop_enabled = args.early_stop_patience > 0
    update_step = 0
    started_at = time.time()
    optimizer.zero_grad(set_to_none=True)
    steps_per_epoch = len(train_loader)
    max_epochs = args.epochs

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        epoch_label = f"{epoch}/{max_epochs}" if not early_stop_enabled else f"{epoch}/{max_epochs}|es"
        pbar = tqdm(train_loader, desc=f"[{category}] epoch {epoch_label}", unit="step",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        for batch_index, batch in enumerate(pbar, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss) * args.gradient_accumulation_steps
            batches += 1

            if batch_index % args.gradient_accumulation_steps == 0 or batch_index == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                pbar.set_postfix(loss=f"{running_loss / max(batches, 1):.4f}",
                                 updates=f"{update_step}")
                if update_step % args.log_every_updates == 0:
                    elapsed = time.time() - started_at
                    updates_per_second = update_step / max(elapsed, 1e-6)
                    remaining = max(total_updates - update_step, 0)
                    eta_seconds = remaining / max(updates_per_second, 1e-6)
                    print(
                        f"[{category}] update={update_step}/{total_updates} "
                        f"loss={running_loss / max(batches, 1):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"elapsed_min={elapsed / 60:.1f} eta_min={eta_seconds / 60:.1f}",
                        flush=True,
                    )
                if args.save_every_updates and update_step % args.save_every_updates == 0:
                    save_adapter(model, tokenizer, output_dir / "latest")
                if args.max_updates is not None and update_step >= args.max_updates:
                    break

        train_loss = running_loss / max(batches, 1)
        val_loss = evaluate_loss(model, val_loader, device, args.max_eval_batches)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "updates": update_step,
        }
        history.append(epoch_record)
        print(f"[{category}] {json.dumps(epoch_record)}")
        save_adapter(model, tokenizer, output_dir / "last")

        if early_stop_enabled:
            stop_state = early_stop_state_after_epoch(
                val_loss=val_loss,
                train_loss=train_loss,
                best_val=best_val,
                train_loss_at_best_val=train_loss_at_best_val,
                epochs_without_improvement=epochs_without_improvement,
                overfit_epochs=overfit_epochs,
                patience=args.early_stop_patience,
                overfit_patience=args.early_stop_overfit_patience,
                min_delta=args.early_stop_min_delta,
            )
            epochs_without_improvement = stop_state["epochs_without_improvement"]
            overfit_epochs = stop_state["overfit_epochs"]
            epoch_record["epochs_without_improvement"] = epochs_without_improvement
            epoch_record["overfit_epochs"] = overfit_epochs

        if val_loss < best_val:
            best_val = val_loss
            train_loss_at_best_val = train_loss
            save_adapter(model, tokenizer, best_dir)

        if early_stop_enabled and stop_state["should_stop"]:
            early_stopped = True
            early_stop_reason = stop_state["reason"]
            print(
                f"[{category}] early stop after epoch {epoch}: reason={early_stop_reason} "
                f"(val_loss={val_loss:.6f} best_val={best_val:.6f} "
                f"no_improve={epochs_without_improvement} overfit_streak={overfit_epochs})",
                flush=True,
            )
            break

        if args.max_updates is not None and update_step >= args.max_updates:
            break

    metrics = {
        "agent": category,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "best_val_loss": best_val,
        "best_val_perplexity": math.exp(min(best_val, 20)),
        "updates": update_step,
        "elapsed_seconds": time.time() - started_at,
        "history": history,
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason,
        "epochs_completed": len(history),
        "run_config": {
            "epochs": args.epochs,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_overfit_patience": args.early_stop_overfit_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lr": args.lr,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "use_4bit": args.use_4bit,
            "init_adapter": getattr(args, "init_adapter", None),
            "seed": args.seed,
        },
        "provenance": experiment_provenance(
            {
                "base_model": args.base_model,
                "train_data": args.train_data,
                "val_data": args.val_data,
            }
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    del model
    if torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default="data/processed/train.jsonl")
    parser.add_argument("--val-data", default="data/processed/val.jsonl")
    parser.add_argument("--save-dir", default="artifacts/checkpoints/sft")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--init-adapter",
        default=None,
        help="Optional LoRA checkpoint for both agents (overridden by per-agent flags).",
    )
    parser.add_argument("--init-main-adapter", default=None)
    parser.add_argument("--init-sub-adapter", default=None)
    parser.add_argument("--agents", choices=("main", "sub", "both"), default="both")
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Maximum epochs; with --early-stop-patience > 0 training may stop earlier.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without val improvement (0 disables early stopping).",
    )
    parser.add_argument(
        "--early-stop-overfit-patience",
        type=int,
        default=2,
        help="Stop after this many consecutive overfit epochs (val up while train down).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum val_loss decrease to count as improvement.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--log-every-updates", type=int, default=100)
    parser.add_argument("--save-every-updates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    categories = ("main", "sub") if args.agents == "both" else (args.agents,)
    results = []
    for category in categories:
        print(f"[train] starting {category} agent")
        results.append(train_agent(args, tokenizer, category))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
