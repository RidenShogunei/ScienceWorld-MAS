"""Tests for Plan A schema and generation helpers."""

from generate_contract_sft_data import ExpertStep

from plan_a.generate_sft_data import build_candidates, plan_for_step
from plan_a.schema import PlanContract, parse_plan_response


def test_plan_contract_roundtrip():
    plan = PlanContract(subgoal="Open the door to the hallway", focus_objects=["door", "hallway"])
    text = plan.to_tagged_json()
    parsed = parse_plan_response(text)
    assert parsed is not None
    assert parsed.subgoal == plan.subgoal
    assert parsed.focus_objects == plan.focus_objects


def test_plan_for_step():
    step = ExpertStep(
        source_index=1,
        step_index=0,
        task="find plant",
        observation="living room",
        subtask="Navigate to the greenhouse",
        expert_actions=["open door to hallway", "go to hallway"],
        low_observations=["living room", "hallway"],
        low_dones=[False, False],
        task_family="find-plant",
        split_key="high_source:1",
    )
    plan = plan_for_step(step, focus_limit=5)
    assert "greenhouse" in plan.subgoal.lower() or "Navigate" in plan.subgoal
    candidates = build_candidates(
        step.expert_actions,
        "open door to hallway",
        plan=plan,
        max_actions=8,
    )
    assert "open door to hallway" in candidates
