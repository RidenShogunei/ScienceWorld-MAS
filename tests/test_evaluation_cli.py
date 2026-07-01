from argparse import Namespace

import pytest

from scienceworld_mas.agents import FirstValidActionPolicy, GoldActionPolicy
from scienceworld_mas.evaluation.cli import build_policy


def test_build_policy():
    assert isinstance(build_policy("gold"), GoldActionPolicy)
    assert isinstance(build_policy("first-valid"), FirstValidActionPolicy)


def test_hf_hierarchical_policy_requires_model_paths():
    args = Namespace(
        policy="hf-hierarchical",
        base_model=None,
        system1_adapter=None,
        system2_adapter=None,
    )

    with pytest.raises(ValueError, match="--base-model"):
        build_policy(args)
