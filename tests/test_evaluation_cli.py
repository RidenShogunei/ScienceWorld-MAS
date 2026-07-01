from scienceworld_mas.agents import FirstValidActionPolicy, GoldActionPolicy
from scienceworld_mas.evaluation.cli import build_policy


def test_build_policy():
    assert isinstance(build_policy("gold"), GoldActionPolicy)
    assert isinstance(build_policy("first-valid"), FirstValidActionPolicy)
