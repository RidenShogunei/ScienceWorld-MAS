"""Tests for L1 decision-state episode selection."""

from eval_episodes import EpisodeSpec

from l1.states import (
    build_episode_step_actions,
    replay_prefix_for_state,
    select_episode_specs,
)


def test_select_episode_specs_all_tasks_k2():
    specs = [
        EpisodeSpec("boil", 19, "dev"),
        EpisodeSpec("boil", 18, "dev"),
        EpisodeSpec("boil", 15, "dev"),
        EpisodeSpec("find-plant", 159, "dev"),
        EpisodeSpec("find-plant", 192, "dev"),
    ]
    selected = select_episode_specs(specs, tasks=None, variations_per_task=2)
    assert len(selected) == 4
    assert [s.variation_id for s in selected if s.task_name == "boil"] == [19, 18]
    assert [s.variation_id for s in selected if s.task_name == "find-plant"] == [159, 192]


def test_select_episode_specs_task_filter():
    specs = [
        EpisodeSpec("boil", 19, "dev"),
        EpisodeSpec("boil", 18, "dev"),
        EpisodeSpec("find-plant", 159, "dev"),
    ]
    selected = select_episode_specs(
        specs,
        tasks=["find-plant"],
        variations_per_task=1,
    )
    assert len(selected) == 1
    assert selected[0].task_name == "find-plant"


def test_replay_prefix_uses_stored_expert_actions():
    from l1.states import DecisionState

    states = [
        DecisionState(
            state_id="task:1:0",
            task_name="task",
            variation_id=1,
            split="dev",
            step_index=0,
            chunk_index=0,
            task="t",
            observation="o0",
            planner_observation="o0",
            inventory="",
            recent_history=[],
            candidate_actions=["look around"],
            expert_action="look around",
            expert_action_id=0,
            score_before=0.0,
            gold_contract={},
        ),
        DecisionState(
            state_id="task:1:1",
            task_name="task",
            variation_id=1,
            split="dev",
            step_index=1,
            chunk_index=0,
            task="t",
            observation="o1",
            planner_observation="o0",
            inventory="",
            recent_history=[{"action": "look around"}],
            candidate_actions=["open door"],
            expert_action="open door",
            expert_action_id=0,
            score_before=0.0,
            gold_contract={},
        ),
    ]
    index = build_episode_step_actions(states)
    assert replay_prefix_for_state(states[1], index) == ["look around"]
