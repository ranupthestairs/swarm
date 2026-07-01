from __future__ import annotations

import json
from pathlib import Path

import pytest

from swarm.core.fly_batch import (
    FlyBatchJob,
    load_batch_jobs,
    print_batch_summary,
)


def test_load_batch_jobs_benchmark_format(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text(
        json.dumps(
            {
                "type2_open": [2001, 2002],
                "type1_city": [1001],
            }
        )
    )
    jobs = load_batch_jobs(path)
    assert jobs == [
        FlyBatchJob(seed=1001, challenge_type=1),
        FlyBatchJob(seed=2001, challenge_type=2),
        FlyBatchJob(seed=2002, challenge_type=2),
    ]


def test_load_batch_jobs_numeric_type_keys(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text(json.dumps({"1": [42], "6": [99]}))
    jobs = load_batch_jobs(path)
    assert jobs == [
        FlyBatchJob(seed=42, challenge_type=1),
        FlyBatchJob(seed=99, challenge_type=6),
    ]


def test_load_batch_jobs_list_format(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text(
        json.dumps(
            [
                {"type": 3, "seeds": [301, 302]},
                {"challenge_type": 5, "seed": 501},
            ]
        )
    )
    jobs = load_batch_jobs(path)
    assert jobs == [
        FlyBatchJob(seed=301, challenge_type=3),
        FlyBatchJob(seed=302, challenge_type=3),
        FlyBatchJob(seed=501, challenge_type=5),
    ]


def test_load_batch_jobs_runs_wrapper(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {"type": 4, "seeds": [401]},
                ]
            }
        )
    )
    jobs = load_batch_jobs(path)
    assert jobs == [FlyBatchJob(seed=401, challenge_type=4)]


def test_load_batch_jobs_fixture_file() -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "fly_batch_example.json"
    jobs = load_batch_jobs(fixture)
    assert len(jobs) == 3
    assert jobs[0] == FlyBatchJob(seed=1001, challenge_type=1)


def test_load_batch_jobs_rejects_empty(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text(json.dumps({"type1_city": []}))
    with pytest.raises(ValueError, match="No runs found"):
        load_batch_jobs(path)


def test_load_batch_jobs_skips_metadata_keys(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text(
        json.dumps(
            {
                "description": "quick smoke batch",
                "type1_city": [1001],
            }
        )
    )
    jobs = load_batch_jobs(path)
    assert jobs == [FlyBatchJob(seed=1001, challenge_type=1)]


def test_load_batch_jobs_combines_runs_and_mapping(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text(
        json.dumps(
            {
                "runs": [{"type": 2, "seeds": [2001]}],
                "type1_city": [1001],
            }
        )
    )
    jobs = load_batch_jobs(path)
    assert jobs == [
        FlyBatchJob(seed=2001, challenge_type=2),
        FlyBatchJob(seed=1001, challenge_type=1),
    ]


def test_load_batch_jobs_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Batch file not found"):
        load_batch_jobs(tmp_path / "missing.json")


def test_run_batch_continues_after_job_error(monkeypatch, tmp_path: Path) -> None:
    from swarm.core.fly_batch import FlyBatchJob, run_batch
    from swarm.core.fly_setup import FlyLaunchConfig

    calls = {"count": 0}

    def _fake_headless(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("sim exploded")
        return {
            "path": tmp_path / "ok.flytraj.json.gz",
            "result": {"score": 0.9},
            "seed": kwargs["launch"].seed,
            "challenge_type": kwargs["launch"].challenge_type,
            "type_label": "city",
            "success": True,
        }

    monkeypatch.setattr("swarm.core.fly_batch._prepare_agent_dir", lambda **kwargs: (tmp_path, None))
    monkeypatch.setattr("swarm.core.fly_batch._load_agent", lambda agent_dir: object())
    monkeypatch.setattr("swarm.core.fly_batch.make_env", lambda task, gui=False: type("E", (), {
        "action_space": type("S", (), {"low": __import__("numpy").zeros(5), "high": __import__("numpy").ones(5)})(),
        "close": lambda self: None,
    })())
    monkeypatch.setattr("swarm.core.fly_batch.run_headless_episode", _fake_headless)

    launch = FlyLaunchConfig(
        agent_path=tmp_path,
        agent_kind="source",
        seed=1,
        challenge_type=1,
    )
    results = run_batch(
        launch=launch,
        jobs=[
            FlyBatchJob(seed=1001, challenge_type=1),
            FlyBatchJob(seed=1002, challenge_type=1),
        ],
        repo_root=tmp_path,
    )
    assert len(results) == 2
    assert results[0]["error"] == "sim exploded"
    assert results[1]["success"] is True


def test_print_batch_summary(capsys) -> None:
    print_batch_summary(
        [
            {
                "success": True,
                "result": {"score": 0.8, "success_term": 1.0, "time_term": 0.7, "safety_term": 0.5},
            },
            {
                "success": False,
                "result": {"score": 0.01, "success_term": 0.0, "time_term": 0.0, "safety_term": 0.0},
            },
        ]
    )
    output = capsys.readouterr().out
    assert "BATCH SUMMARY" in output
    assert "Successes  : 1/2" in output
