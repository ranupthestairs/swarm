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
