from dataclasses import dataclass

from scienceworld_mas.env import (
    EpisodeSpec,
    episode_list_metadata,
    generate_stratified_episodes,
    load_episode_list,
    save_episode_list,
)


@dataclass
class FakeRunner:
    task_names: list[str]
    variations_by_task: dict[str, list[int]]

    def variations(self, task_name: str, split: str) -> list[int]:
        assert split == "dev"
        return list(self.variations_by_task[task_name])


def test_generate_stratified_episodes_is_reproducible_and_balanced():
    runner = FakeRunner(
        task_names=["task-b", "task-a"],
        variations_by_task={
            "task-a": [1, 2, 3, 4],
            "task-b": [10, 11],
        },
    )
    first = generate_stratified_episodes(runner, "dev", k_per_task=3, seed=123)
    second = generate_stratified_episodes(runner, "dev", k_per_task=3, seed=123)
    assert first == second
    assert len(first) == 5
    assert sum(spec.task_name == "task-a" for spec in first) == 3
    assert sum(spec.task_name == "task-b" for spec in first) == 2


def test_episode_list_roundtrip(tmp_path):
    specs = [
        EpisodeSpec("boil", 1, "dev"),
        EpisodeSpec("freeze", 2, "dev"),
    ]
    metadata = episode_list_metadata(specs, split="dev", seed=123, k_per_task=5)
    path = tmp_path / "episodes.json"
    save_episode_list(path, specs, metadata)
    loaded_meta, loaded_specs = load_episode_list(path)
    assert loaded_meta == metadata
    assert loaded_specs == specs
