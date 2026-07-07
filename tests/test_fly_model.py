from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import fly_model as fly


def test_prepare_agent_dir_accepts_source_directory(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    (source / "drone_agent.py").write_text("class DroneFlightController:\n    pass\n")

    agent_dir, temp_dir = fly._prepare_agent_dir(
        model_path=None,
        source_path=source,
        seed=42,
    )

    assert agent_dir == source.resolve()
    assert temp_dir is None


def test_parse_observation_extracts_search_vector() -> None:
    state = np.zeros(20, dtype=np.float32)
    state[0:3] = [1.0, 2.0, 3.0]
    state[6:9] = [0.3, 0.4, 0.0]
    state[-3:] = [4.0, -1.0, 2.0]

    info = fly._parse_observation({"state": state})

    assert info["speed_mps"] == pytest.approx(0.5)
    np.testing.assert_allclose(info["search_area_vector"], [4.0, -1.0, 2.0])
    np.testing.assert_allclose(info["search_area_center"], [5.0, 1.0, 5.0])


def test_snapshot_agent_debug_uses_get_debug_info() -> None:
    class _Agent:
        def get_debug_info(self):
            return {"mode": "landing", "goal_visibility_prob": 0.91}

    info = fly._snapshot_agent_debug(_Agent())

    assert info["mode"] == "landing"
    assert info["goal_visibility_prob"] == 0.91


def test_snapshot_agent_debug_includes_pad_lock_fields() -> None:
    class _Agent:
        platform_position = np.array([1.0, 0.0, 0.0], dtype=float)
        landing_platform = np.array([0.8, 0.0, 0.0], dtype=float)
        _last_dist_to_go = 0.2
        _last_goal_pos = np.array([1.0, 0.0, 0.0], dtype=float)
        move_in_auto_mode = False
        p_buffer = 0.15
        see_P = True
        is_find_P = False

    info = fly._snapshot_agent_debug(_Agent())

    np.testing.assert_allclose(info["raw_goal_position"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(info["landing_platform_position"], [0.8, 0.0, 0.0])
    assert info["pad_lock_dist_to_go"] == 0.2
    assert info["move_in_auto_mode"] is False


def test_snapshot_agent_debug_unwraps_wrapper_controller() -> None:
    class _Inner:
        _mode = "navigation"
        is_find_P = True
        see_P = True
        _goal_is_tracked = True
        _map_prediction_label = "mountain"
        _map_prediction_probability = 0.88
        _swarm_debug_eye_probability = 0.73
        landing_platform = np.array([10.0, 5.0, 2.0], dtype=float)
        tracking = False

    class _Wrapper:
        def __init__(self) -> None:
            self._main = _Inner()
            self._forest = _Inner()
            self._active = "main"
            self._tick = 42

    info = fly._snapshot_agent_debug(_Wrapper())

    assert info["mode"] == "navigation"
    assert info["goal_detected"] is True
    assert info["goal_visible"] is True
    assert info["goal_tracked"] is True
    assert info["goal_visibility_prob"] == pytest.approx(0.73)
    assert info["map_prediction"] == "mountain:0.880"
    assert info["active_controller"] == "main"
    assert info["controller_tick"] == 42


def test_build_hud_lines_includes_key_fields() -> None:
    from types import SimpleNamespace

    from swarm.core.fly_viewer import build_bottom_telemetry_lines

    task = SimpleNamespace(
        map_seed=1001,
        challenge_type=1,
        start=(0.0, 0.0, 1.0),
        goal=(10.0, 5.0, 2.0),
        search_radius=12.5,
        moving_platform=False,
    )
    lines = build_bottom_telemetry_lines(
        task=task,
        sim_state="running",
        t_sim=1.5,
        frame=75,
        obs_info={
            "position": np.array([1.0, 2.0, 3.0]),
            "speed_mps": 1.25,
            "search_area_vector": np.array([4.0, -1.0, 0.5]),
            "search_area_center": np.array([5.0, 1.0, 3.5]),
        },
        agent_info={
            "mode": "navigation",
            "goal_detected": True,
            "goal_visible": True,
            "goal_tracked": False,
            "goal_visibility_prob": 0.82,
            "predicted_goal_position": np.array([9.0, 4.0, 2.0]),
            "map_prediction": "city:0.91",
        },
        action=np.array([0.1, 0.2, 0.0, 0.8, 0.0], dtype=np.float32),
        camera_mode="chase",
    )

    joined = "\n".join(lines)
    assert "start" in joined
    assert "goal" in joined
    assert "Search" in joined
    assert "navigation" in joined
    assert "city:0.91" in joined
    assert "Goal detect" in joined
    assert "Predict pad" in joined
    assert "Speed" in joined


def test_build_hud_lines_includes_controller_fields() -> None:
    from types import SimpleNamespace

    from swarm.core.fly_viewer import build_bottom_telemetry_lines

    task = SimpleNamespace(
        map_seed=1001,
        challenge_type=1,
        start=(0.0, 0.0, 1.0),
        goal=(10.0, 5.0, 2.0),
        search_radius=12.5,
        moving_platform=False,
    )
    lines = build_bottom_telemetry_lines(
        task=task,
        sim_state="running",
        t_sim=1.5,
        frame=75,
        obs_info={
            "position": np.array([1.0, 2.0, 3.0]),
            "speed_mps": 1.25,
            "search_area_vector": np.array([4.0, -1.0, 0.5]),
            "search_area_center": np.array([5.0, 1.0, 3.5]),
        },
        agent_info={
            "mode": "navigation",
            "active_controller": "main",
            "forward": True,
            "first_order": False,
            "yaw_error_deg": 8.0,
            "command_action": np.array([0.1, 0.2, 0.0, 0.8, 0.0], dtype=float),
            "final_action": np.array([0.1, 0.2, 0.0, 0.75, 0.0], dtype=float),
            "goal_detected": True,
            "goal_visible": True,
            "goal_tracked": False,
            "goal_visibility_prob": 0.82,
            "predicted_goal_position": np.array([9.0, 4.0, 2.0]),
            "map_prediction": "city:0.91",
        },
        action=np.array([0.1, 0.2, 0.0, 0.75, 0.0], dtype=np.float32),
        camera_mode="chase",
    )

    joined = "\n".join(lines)
    assert "Control" in joined
    assert "Yaw" in joined
    assert "Throttle" in joined
    assert "cmd_spd=0.80" in joined
    assert "out_spd=0.75" in joined
