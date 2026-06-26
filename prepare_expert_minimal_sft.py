"""Prepare expert minimal SFT data: episode split, task rebalance, Sub prompt fit."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agent_protocol import (
    chat_prompt_token_count,
    fit_sub_messages_for_inference,
    infer_protocol_from_schema,
    main_system_prompt,
    parse_env_sub_user,
    sub_system_prompt,
)
from scienceworld_data import assign_split


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def episode_key(row: dict[str, Any]) -> str:
    return f"{row['task_name']}:{row['variation_id']}"


def rebalance_rows(rows: list[dict[str, Any]], *, max_per_task: int | None, seed: int) -> list[dict[str, Any]]:
    if max_per_task is None:
        return rows
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_name"]].append(row)
    rng = random.Random(seed)
    kept: list[dict[str, Any]] = []
    for task_name in sorted(by_task):
        items = by_task[task_name]
        if len(items) > max_per_task:
            items = rng.sample(items, max_per_task)
        kept.extend(items)
    kept.sort(key=lambda row: (row["rollout_id"], row.get("decision_index", 0), row.get("category", "")))
    return kept


def trim_sub_row(
    row: dict[str, Any],
    tokenizer,
    *,
    max_prompt_tokens: int,
) -> dict[str, Any]:
    protocol = infer_protocol_from_schema(str(row.get("schema", "minimal_contract_v1")))
    user = next(message["content"] for message in row["messages"] if message["role"] == "user")
    contract, observation, valid_actions, history = parse_env_sub_user(user)
    prompt_messages = fit_sub_messages_for_inference(
        tokenizer,
        protocol,
        task_context=contract,
        observation=observation,
        valid_actions=valid_actions,
        recent_history=history,
        max_input_length=max_prompt_tokens,
    )
    assistant = row["messages"][-1]["content"]
    out = dict(row)
    out["messages"] = prompt_messages + [{"role": "assistant", "content": assistant}]
    return out


def prepare_rows(
    rows: list[dict[str, Any]],
    tokenizer,
    *,
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        if row.get("category") == "sub":
            prepared.append(trim_sub_row(row, tokenizer, max_prompt_tokens=max_prompt_tokens))
        else:
            prepared.append(row)
    return prepared


def token_report(rows: list[dict[str, Any]], tokenizer, *, max_prompt_tokens: int) -> dict[str, Any]:
    over = 0
    lens: list[int] = []
    for row in rows:
        messages = row["messages"][:-1]
        count = chat_prompt_token_count(tokenizer, messages)
        lens.append(count)
        if count > max_prompt_tokens:
            over += 1
    lens.sort()
    if not lens:
        return {"n": 0, "over_rate": 0.0}
    return {
        "n": len(lens),
        "avg": round(sum(lens) / len(lens), 1),
        "p95": lens[int(0.95 * len(lens)) - 1],
        "max": lens[-1],
        "over_rate": round(over / len(lens), 4),
    }


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    source_rows = read_jsonl(input_dir / "train.jsonl")
    if not source_rows:
        raise SystemExit(f"no rows in {input_dir / 'train.jsonl'}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=args.local_tokenizer_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_prompt_tokens = args.max_input_length - args.max_completion_reserve
    balanced = rebalance_rows(source_rows, max_per_task=args.max_per_task, seed=args.seed)
    prepared = prepare_rows(balanced, tokenizer, max_prompt_tokens=max_prompt_tokens)

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    episodes = {episode_key(row) for row in prepared}
    for row in prepared:
        split = assign_split(
            episode_key(row),
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        row = dict(row)
        row["split"] = split
        by_split[split].append(row)

    for split, items in by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", items)

    manifest = {
        "input": str(input_dir),
        "source_rollouts": len(episodes),
        "samples": {split: len(items) for split, items in by_split.items()},
        "categories": {
            "main": sum(1 for row in prepared if row.get("category") == "main"),
            "sub": sum(1 for row in prepared if row.get("category") == "sub"),
        },
        "preprocessing": {
            "max_input_length": args.max_input_length,
            "max_completion_reserve": args.max_completion_reserve,
            "max_per_task": args.max_per_task,
            "split_seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
        },
        "schema": prepared[0].get("schema", "minimal_contract_v1"),
        "token_stats_after": token_report(prepared, tokenizer, max_prompt_tokens=max_prompt_tokens),
        "task_counts_after": dict(Counter(row["task_name"] for row in prepared).most_common()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/expert_minimal_sft_train50_success",
    )
    parser.add_argument(
        "--output-dir",
        default="data/expert_minimal_sft_train50_prepared",
    )
    parser.add_argument(
        "--tokenizer",
        default="artifacts/checkpoints/sft_minimax_native_train50/main_agent/best",
    )
    parser.add_argument("--local-tokenizer-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-input-length", type=int, default=6656)
    parser.add_argument("--max-completion-reserve", type=int, default=128)
    parser.add_argument("--max-per-task", type=int, default=250)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    manifest = prepare_dataset(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
