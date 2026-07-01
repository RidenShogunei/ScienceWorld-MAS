from scienceworld_mas.agents import FirstValidActionPolicy, GoldActionPolicy
from scienceworld_mas.env import EpisodeSpec
from scienceworld_mas.evaluation import ActionDecision, PolicyContext


def context(valid_actions=()):
    return PolicyContext(
        task_description="task",
        observation="obs",
        step_index=0,
        valid_actions=tuple(valid_actions),
        history=(),
    )


def test_first_valid_action_policy_uses_context_valid_actions():
    policy = FirstValidActionPolicy()
    policy.reset_episode("task")
    decision = policy.act(context(["look", "open door"]))
    assert decision.action == "look"


def test_first_valid_action_policy_fails_without_valid_actions():
    policy = FirstValidActionPolicy()
    decision = policy.act(context([]))
    assert decision.action is None
    assert not decision.format_valid


def test_gold_action_policy_replays_gold_sequence():
    policy = GoldActionPolicy()
    policy.reset_episode("task")
    policy.prepare_episode(
        spec=EpisodeSpec("task", 0, "dev"),
        task_description="task",
        gold_actions=("a", "b"),
    )
    assert policy.act(context()).action == "a"
    assert policy.act(context()).action == "b"
    done = policy.act(context())
    assert isinstance(done, ActionDecision)
    assert done.action is None
    assert not done.format_valid
