import json

from scienceworld_mas.bench import EpisodeScore, compute_benchmark_score
from scienceworld_mas.evaluation import EvaluationReport, write_evaluation_report


def test_write_evaluation_report_uses_metrics_contract(tmp_path):
    metrics = compute_benchmark_score(
        [
            EpisodeScore("task", 0, -100.0, steps=3),
            EpisodeScore("task", 1, 100.0, steps=5),
        ]
    )
    report = EvaluationReport(episode_list=None, metrics=metrics, episodes=())
    path = write_evaluation_report(report, tmp_path / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metrics"]["official_mean_score"] == 0.0
    assert payload["metrics"]["negative_score_rate"] == 0.5
