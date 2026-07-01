from __future__ import annotations

from pathlib import Path

import numpy as np

from swarm.core.fly_trajectory import (
    FlyTrajectory,
    FlyTrajectoryFrame,
    agent_name_from_path,
    fly_runs_dir,
    format_score_summary,
    frame_obs_info,
    list_saved_runs,
    load_latest_trajectory,
    load_trajectory,
    run_video_filename,
    save_trajectory,
    trajectory_filename,
    video_path_for_trajectory,
    _peek_trajectory_result,
)


def test_trajectory_round_trip(tmp_path: Path) -> None:
    frame = FlyTrajectoryFrame(
        t_sim=0.02,
        frame=1,
        position=np.array([1.0, 2.0, 3.0]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        velocity=np.array([0.1, 0.2, 0.0]),
        action=np.array([0.0, 0.0, 1.0, 0.5, 0.0]),
        agent_info={"mode": "search", "goal_detected": False},
        search_area_vector=np.array([4.0, -1.0, 0.5]),
    )
    trajectory = FlyTrajectory(
        meta={
            "seed": 1001,
            "challenge_type": 1,
            "type_label": "city",
            "agent_path": "/tmp/agent",
            "agent_kind": "source",
            "result": {"success": False, "score": 0.01},
        },
        frames=[frame],
    )

    path = save_trajectory(trajectory, tmp_path / "run.flytraj.json.gz")
    loaded = load_trajectory(path)

    assert loaded.meta["seed"] == 1001
    assert len(loaded.frames) == 1
    assert loaded.frames[0].agent_info["mode"] == "search"
    obs_info = frame_obs_info(loaded.frames[0])
    np.testing.assert_allclose(obs_info["search_area_vector"], [4.0, -1.0, 0.5])


def test_saved_runs_live_in_repo_fly_runs(tmp_path: Path) -> None:
    frame = FlyTrajectoryFrame(
        t_sim=0.02,
        frame=1,
        position=np.array([1.0, 2.0, 3.0]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        velocity=np.array([0.1, 0.2, 0.0]),
        action=np.array([0.0, 0.0, 1.0, 0.5, 0.0]),
    )
    trajectory = FlyTrajectory(
        meta={
            "seed": 42,
            "challenge_type": 1,
            "type_label": "city",
            "agent_path": "/tmp/agent",
            "agent_kind": "source",
            "result": {"success": True, "score": 0.5},
        },
        frames=[frame],
    )
    runs_dir = fly_runs_dir(tmp_path)
    path = save_trajectory(trajectory, repo_root=tmp_path)
    assert path.parent == runs_dir
    runs = list_saved_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].seed == 42
    assert runs[0].score == 0.5


def test_saved_run_includes_score_breakdown(tmp_path: Path) -> None:
    frame = FlyTrajectoryFrame(
        t_sim=0.02,
        frame=1,
        position=np.array([1.0, 2.0, 3.0]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        velocity=np.array([0.1, 0.2, 0.0]),
        action=np.array([0.0, 0.0, 1.0, 0.5, 0.0]),
    )
    trajectory = FlyTrajectory(
        meta={
            "agent_name": "champion_UID_191",
            "seed": 1001,
            "challenge_type": 1,
            "type_label": "city",
            "agent_path": "/tmp/champion_UID_191",
            "agent_kind": "source",
            "result": {
                "success": True,
                "time_sec": 12.3,
                "score": 0.89,
                "success_term": 1.0,
                "time_term": 0.82,
                "safety_term": 0.55,
                "target_time_sec": 10.5,
                "min_clearance_m": 0.8,
                "collision": False,
            },
        },
        frames=[frame],
    )
    path = save_trajectory(trajectory, repo_root=tmp_path)
    runs = list_saved_runs(tmp_path)
    assert len(runs) == 1
    run = runs[0]
    assert run.success_term == 1.0
    assert run.time_term == 0.82
    assert run.safety_term == 0.55
    assert run.score_summary is not None
    assert "S:1.00" in run.score_summary
    assert "T:0.82" in run.score_summary
    assert "Saf:0.55" in run.score_summary
    peeked = _peek_trajectory_result(path)
    assert peeked is not None
    assert peeked["score"] == 0.89


def test_format_score_summary() -> None:
    summary = format_score_summary(
        {"score": 0.89, "success_term": 1.0, "time_term": 0.82, "safety_term": 0.55}
    )
    assert summary == "0.89  S:1.00  T:0.82  Saf:0.55"


def test_list_saved_runs_skips_corrupt_trajectory(tmp_path: Path) -> None:
    import gzip
    import os

    frame = FlyTrajectoryFrame(
        t_sim=0.02,
        frame=1,
        position=np.array([1.0, 2.0, 3.0]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        velocity=np.array([0.1, 0.2, 0.0]),
        action=np.array([0.0, 0.0, 1.0, 0.5, 0.0]),
    )
    good = FlyTrajectory(
        meta={
            "agent_name": "agent",
            "seed": 1,
            "challenge_type": 1,
            "type_label": "city",
            "agent_path": "/tmp/agent",
            "agent_kind": "source",
            "result": {"success": True, "score": 0.5},
        },
        frames=[frame],
    )
    good_path = save_trajectory(good, repo_root=tmp_path)

    bad = fly_runs_dir(tmp_path) / "agent_seed2_city_20260701_120100.flytraj.json.gz"
    with gzip.open(bad, "wb") as handle:
        handle.write(b"not-json")
    os.utime(bad, (bad.stat().st_mtime + 10, bad.stat().st_mtime + 10))

    runs = list_saved_runs(tmp_path)
    assert len(runs) == 2
    assert load_latest_trajectory(tmp_path) is not None
    assert load_latest_trajectory(tmp_path).meta["seed"] == 1
    assert _peek_trajectory_result(bad) is None
    assert good_path.is_file()


def test_browse_run_file_zenity_list_builds_args(tmp_path: Path) -> None:
    from swarm.core.fly_trajectory import SavedRunInfo, _browse_run_file_zenity_list

    runs = [
        SavedRunInfo(
            path=tmp_path / "a.flytraj.json.gz",
            display_name="seed1 city score=0.50",
        )
    ]
    # No zenity in CI/sandbox — just ensure the helper returns None without crashing.
    assert _browse_run_file_zenity_list([]) is None


def test_trajectory_filename_orders_agent_seed_type_timestamp() -> None:
    meta = {
        "agent_path": "/home/peter/miners/champion_UID_191",
        "agent_name": "champion_UID_191",
        "seed": 1001,
        "type_label": "city",
    }
    name = trajectory_filename(meta)
    assert name.startswith("champion_UID_191_seed1001_city_")
    assert name.endswith(".flytraj.json.gz")


def test_agent_name_from_zip_strips_extension() -> None:
    assert agent_name_from_path("/tmp/champion_UID_191.zip") == "champion_UID_191"


def test_video_path_matches_trajectory_basename(tmp_path: Path) -> None:
    meta = {
        "agent_name": "champion_UID_191",
        "seed": 1001,
        "type_label": "city",
    }
    traj_name = trajectory_filename(meta)
    traj_path = tmp_path / traj_name
    video_name = run_video_filename(meta)
    assert video_name.endswith(".mp4")
    assert video_name.startswith("champion_UID_191_seed1001_city_")
    assert video_path_for_trajectory(traj_path) == traj_path.with_name(video_name)
