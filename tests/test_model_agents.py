import json

from scienceworld_mas.agents import (
    GenerationSettings,
    HierarchicalChatPolicy,
    build_system1_messages,
    build_system2_messages,
    parse_executor_output,
    parse_subgoal_output,
)
from scienceworld_mas.evaluation import PolicyContext


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = {key: list(value) for key, value in outputs.items()}
        self.calls = []

    def generate(self, messages, *, adapter_name: str, settings: GenerationSettings) -> str:
        self.calls.append((adapter_name, messages, settings))
        return self.outputs[adapter_name].pop(0)


def context(step=0, observation="obs"):
    return PolicyContext(
        task_description="boil water",
        observation=observation,
        step_index=step,
        valid_actions=("look around",),
        history=(),
    )


def test_parse_subgoal_output_accepts_plain_or_tagged_text():
    assert parse_subgoal_output("find the beaker\nextra") == "find the beaker"
    assert parse_subgoal_output("[subgoal]heat water[/subgoal]") == "heat water"
    assert parse_subgoal_output("Subgoal: open the freezer") == "open the freezer"
    assert parse_subgoal_output("  ") is None


def test_parse_executor_output_requires_action_and_done_tags():
    parsed = parse_executor_output(
        "[action]look at beaker[/action][subgoal_done]false[/subgoal_done]"
    )

    assert parsed is not None
    assert parsed.action == "look at beaker"
    assert not parsed.subgoal_done
    assert parse_executor_output("look at beaker") is None


def test_hierarchical_policy_reuses_subgoal_until_done():
    backend = FakeBackend(
        {
            "system1": ["find beaker", "heat water"],
            "system2": [
                "[action]look at beaker[/action][subgoal_done]false[/subgoal_done]",
                "[action]pick up beaker[/action][subgoal_done]true[/subgoal_done]",
                "[action]activate stove[/action][subgoal_done]true[/subgoal_done]",
            ],
        }
    )
    policy = HierarchicalChatPolicy(backend)
    policy.reset_episode("boil water")

    first = policy.act(context(step=0))
    second = policy.act(context(step=1, observation="beaker seen"))
    third = policy.act(context(step=2, observation="holding beaker"))

    assert first.action == "look at beaker"
    assert second.action == "pick up beaker"
    assert third.action == "activate stove"
    assert [call[0] for call in backend.calls] == [
        "system1",
        "system2",
        "system2",
        "system1",
        "system2",
    ]


def test_hierarchical_policy_reports_format_error_with_raw_payload():
    backend = FakeBackend({"system1": ["find beaker"], "system2": ["look at beaker"]})
    policy = HierarchicalChatPolicy(backend)

    decision = policy.act(context())

    assert decision.action is None
    assert not decision.format_valid
    payload = json.loads(decision.raw_response)
    assert payload["subgoal"] == "find beaker"
    assert payload["system2_raw"] == "look at beaker"


def test_prompt_builders_match_training_prompt_shape():
    system1_messages = build_system1_messages("task", "obs")
    system2_messages = build_system2_messages("subgoal", "obs")

    assert system1_messages[-1]["content"] == "Task:\ntask\n\nObservation:\nobs"
    assert system2_messages[-1]["content"] == "Subgoal:\nsubgoal\n\nObservation:\nobs"
