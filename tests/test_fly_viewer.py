from __future__ import annotations

import math

import numpy as np

from swarm.core.fly_viewer import (
    CAMERA_MODES,
    FlyRenderCamera,
    LEFT_PANEL_MIN_HEIGHT,
    build_bottom_telemetry_lines,
    browse_agent_directory,
    parse_seed_text,
    _goal_detection_lines,
)


def test_camera_modes_include_panel_buttons() -> None:
    assert "chase" in CAMERA_MODES
    assert "fpv" in CAMERA_MODES
    assert "overview" in CAMERA_MODES


def test_build_panel_lines_include_mission_fields() -> None:
    from types import SimpleNamespace

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
        sim_state="config",
        t_sim=0.0,
        frame=0,
        obs_info=None,
        agent_info=None,
        action=None,
        camera_mode="chase",
    )
    joined = "\n".join(lines)
    assert "Status CONFIG" in joined
    assert "start" in joined
    assert "goal" in joined
    assert "Camera chase" in joined


def test_goal_detection_lines_show_pad_estimate_errors() -> None:
    from types import SimpleNamespace

    task = SimpleNamespace(goal=(10.0, 0.0, 2.0))
    obs_info = {
        "position": np.array([0.0, 0.0, 1.0]),
        "search_area_center": np.array([8.0, 1.0, 2.0]),
    }
    agent_info = {
        "goal_detected": True,
        "goal_visible": True,
        "goal_tracked": True,
        "goal_visibility_prob": 0.87,
        "predicted_goal_position": np.array([9.5, 0.5, 2.1]),
        "platform_lost_steps": 0,
    }
    lines = _goal_detection_lines(agent_info, obs_info=obs_info, task=task)
    joined = "\n".join(lines)
    assert "Goal detect" in joined
    assert "locked=yes" in joined
    assert "visible=yes" in joined
    assert "prob=0.87" in joined
    assert "Predict pad" in joined
    assert "err_true_pad=" in joined
    assert "err_GPS_hint=" in joined


def test_left_panel_min_height_fits_controls() -> None:
    from swarm.core.fly_viewer import compute_left_panel_min_height

    btn_h = 26
    gap = 6
    y_cam = 586
    zoom_row_bottom = y_cam + 2 * (btn_h + gap) + btn_h
    assert LEFT_PANEL_MIN_HEIGHT >= zoom_row_bottom + 12
    assert compute_left_panel_min_height() == LEFT_PANEL_MIN_HEIGHT


def test_replay_controls_sit_inside_viewport() -> None:
    from swarm.core.fly_viewer import PANEL_WIDTH, REPLAY_BAR_HEIGHT, FlySimulatorWindow

    class _FakeWindow(FlySimulatorWindow):
        def __init__(self) -> None:
            self.view_width = 960
            self.left_panel_height = 700
            self._replay_ui_active = True

    window = _FakeWindow()
    timeline = window._replay_timeline_rect()
    speed = window._replay_speed_rect(1.0)
    assert timeline[0] >= PANEL_WIDTH
    assert speed[0] >= PANEL_WIDTH
    assert timeline[1] + timeline[3] <= window.left_panel_height
    assert speed[1] + speed[3] <= window.left_panel_height
    assert window._replay_bar_top() == window.left_panel_height - REPLAY_BAR_HEIGHT


def test_browse_agent_directory_without_tkinter(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "tkinter":
            raise ImportError("no tkinter")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    assert browse_agent_directory() is None


def test_parse_seed_text_clamps_and_fallback() -> None:
    assert parse_seed_text("1001") == 1001
    assert parse_seed_text("0", fallback=42) == 1
    assert parse_seed_text("abc", fallback=42) == 42
    assert parse_seed_text("", fallback=7) == 7
