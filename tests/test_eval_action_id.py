"""Tests for action-id episodic eval helpers."""

from evaluate_environment import run_episode_action_id


def test_run_episode_action_id_dispatched(monkeypatch):
    """run_episode_action_id is importable and named for action-id path."""
    assert callable(run_episode_action_id)
