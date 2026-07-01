"""Model-backed hierarchical System1/System2 policy."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from scienceworld_mas.evaluation import ActionDecision, PolicyContext
from scienceworld_mas.training.examples import SYSTEM1_SYSTEM_PROMPT, SYSTEM2_SYSTEM_PROMPT


SYSTEM1_ADAPTER_NAME = "system1"
SYSTEM2_ADAPTER_NAME = "system2"


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True)
class ParsedExecutorOutput:
    action: str
    subgoal_done: bool


class ChatBackend(Protocol):
    """Small generation interface used by the hierarchical policy."""

    def generate(
        self,
        messages: tuple[dict[str, str], ...],
        *,
        adapter_name: str,
        settings: GenerationSettings,
    ) -> str:
        ...


def build_system1_messages(task_description: str, observation: str) -> tuple[dict[str, str], ...]:
    return (
        {"role": "system", "content": SYSTEM1_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Task:\n{task_description}\n\nObservation:\n{observation}",
        },
    )


def build_system2_messages(subgoal: str, observation: str) -> tuple[dict[str, str], ...]:
    return (
        {"role": "system", "content": SYSTEM2_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Subgoal:\n{subgoal}\n\nObservation:\n{observation}",
        },
    )


def parse_subgoal_output(raw: str) -> str | None:
    text = raw.strip()
    if not text:
        return None
    tagged = re.search(r"\[subgoal\](.*?)\[/subgoal\]", text, flags=re.IGNORECASE | re.DOTALL)
    if tagged:
        text = tagged.group(1).strip()
    if text.lower().startswith("subgoal:"):
        text = text.split(":", 1)[1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else None


def parse_executor_output(raw: str) -> ParsedExecutorOutput | None:
    action_match = re.search(r"\[action\](.*?)\[/action\]", raw, flags=re.IGNORECASE | re.DOTALL)
    done_match = re.search(
        r"\[subgoal_done\](true|false)\[/subgoal_done\]",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not action_match or not done_match:
        return None
    action = action_match.group(1).strip()
    if not action:
        return None
    return ParsedExecutorOutput(
        action=action,
        subgoal_done=done_match.group(1).strip().lower() == "true",
    )


class HierarchicalChatPolicy:
    """System1 plans subgoals and System2 executes them as ScienceWorld actions."""

    def __init__(
        self,
        backend: ChatBackend,
        *,
        system1_settings: GenerationSettings | None = None,
        system2_settings: GenerationSettings | None = None,
    ) -> None:
        self.backend = backend
        self.system1_settings = system1_settings or GenerationSettings()
        self.system2_settings = system2_settings or GenerationSettings()
        self._current_subgoal: str | None = None

    def reset_episode(self, task_description: str) -> None:
        self._current_subgoal = None

    def _plan(self, context: PolicyContext) -> tuple[str | None, str]:
        messages = build_system1_messages(context.task_description, context.observation)
        raw = self.backend.generate(
            messages,
            adapter_name=SYSTEM1_ADAPTER_NAME,
            settings=self.system1_settings,
        )
        return parse_subgoal_output(raw), raw

    def _execute(self, subgoal: str, context: PolicyContext) -> tuple[ParsedExecutorOutput | None, str]:
        messages = build_system2_messages(subgoal, context.observation)
        raw = self.backend.generate(
            messages,
            adapter_name=SYSTEM2_ADAPTER_NAME,
            settings=self.system2_settings,
        )
        return parse_executor_output(raw), raw

    def act(self, context: PolicyContext) -> ActionDecision:
        system1_raw = None
        if self._current_subgoal is None:
            subgoal, system1_raw = self._plan(context)
            if subgoal is None:
                return ActionDecision(
                    action=None,
                    raw_response=json.dumps(
                        {"system1_raw": system1_raw, "error": "failed_to_parse_subgoal"},
                        ensure_ascii=False,
                    ),
                    format_valid=False,
                )
            self._current_subgoal = subgoal

        parsed, system2_raw = self._execute(self._current_subgoal, context)
        raw_response = json.dumps(
            {
                "subgoal": self._current_subgoal,
                "system1_raw": system1_raw,
                "system2_raw": system2_raw,
            },
            ensure_ascii=False,
        )
        if parsed is None:
            return ActionDecision(
                action=None,
                raw_response=raw_response,
                format_valid=False,
            )
        if parsed.subgoal_done:
            self._current_subgoal = None
        return ActionDecision(
            action=parsed.action,
            raw_response=raw_response,
            format_valid=True,
        )


class HuggingFaceChatBackend:
    """Transformers/PEFT backend with two LoRA adapters on one base model."""

    def __init__(
        self,
        *,
        base_model: str,
        system1_adapter: str | Path,
        system2_adapter: str | Path,
        use_4bit: bool = False,
        torch_dtype: str = "auto",
        device_map: str = "auto",
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True}
        if device_map:
            model_kwargs["device_map"] = device_map
        if use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif torch_dtype != "auto":
            model_kwargs["torch_dtype"] = _torch_dtype(torch, torch_dtype)

        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        self.model = PeftModel.from_pretrained(
            base,
            str(system1_adapter),
            adapter_name=SYSTEM1_ADAPTER_NAME,
        )
        self.model.load_adapter(str(system2_adapter), adapter_name=SYSTEM2_ADAPTER_NAME)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _input_ids(self, messages: tuple[dict[str, str], ...]):
        encoded = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(encoded, "to"):
            return encoded.to(self.device)
        tensor = self.torch.tensor([encoded], dtype=self.torch.long)
        return tensor.to(self.device)

    def generate(
        self,
        messages: tuple[dict[str, str], ...],
        *,
        adapter_name: str,
        settings: GenerationSettings,
    ) -> str:
        self.model.set_adapter(adapter_name)
        input_ids = self._input_ids(messages)
        do_sample = settings.temperature > 0
        kwargs = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = settings.temperature
            kwargs["top_p"] = settings.top_p
        with self.torch.no_grad():
            output = self.model.generate(input_ids=input_ids, **kwargs)
        new_tokens = output[0, input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _torch_dtype(torch, name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported torch dtype: {name}")
    return mapping[name]


def load_hierarchical_hf_policy(
    *,
    base_model: str,
    system1_adapter: str | Path,
    system2_adapter: str | Path,
    use_4bit: bool = False,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    system1_max_new_tokens: int = 64,
    system2_max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> HierarchicalChatPolicy:
    backend = HuggingFaceChatBackend(
        base_model=base_model,
        system1_adapter=system1_adapter,
        system2_adapter=system2_adapter,
        use_4bit=use_4bit,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    return HierarchicalChatPolicy(
        backend,
        system1_settings=GenerationSettings(
            max_new_tokens=system1_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        ),
        system2_settings=GenerationSettings(
            max_new_tokens=system2_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        ),
    )
