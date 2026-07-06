from __future__ import annotations

import math

import numpy as np

from swarm.core.fly_viewer import (
    CAMERA_MODES,
    DroneCameraPose,
    FlyRenderCamera,
    LEFT_PANEL_MIN_HEIGHT,
    TELEMETRY_MAX_LINES,
    annotate_depth_direction_overlay,
    build_bottom_telemetry_lines,
    browse_agent_directory,
    colourise_depth_normalized,
    extract_flight_vector,
    parse_seed_text,
    project_world_direction_to_uv,
    _controller_state_lines,
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
        "raw_goal_position": np.array([9.6, 0.4, 2.0]),
        "landing_platform_position": np.array([9.5, 0.5, 2.1]),
        "pad_lock_dist_to_go": 0.18,
        "pad_lock_detector_visible": False,
        "move_in_auto_mode": False,
        "platform_lost_steps": 0,
    }
    lines = _goal_detection_lines(agent_info, obs_info=obs_info, task=task)
    joined = "\n".join(lines)
    assert "Goal detect" in joined
    assert "locked=yes" in joined
    assert "visible=yes" in joined
    assert "prob=0.87" in joined
    assert "Pad lock" in joined
    assert "goal_pos" in joined
    assert "landing_plat" in joined
    assert "dist_to_go=0.180" in joined
    assert "det_frame=no" in joined
    assert "Predict pad" in joined
    assert "err_true_pad=" in joined
    assert "err_GPS_hint=" in joined


def test_controller_state_lines_show_drone_agent_fields() -> None:
    lines = _controller_state_lines(
        {
            "active_controller": "main",
            "mode": "search",
            "forward": False,
            "first_order": True,
            "first_order_cnt": 12,
            "search_pattern": "spiral",
            "search_stage": 2,
            "goal_return": False,
            "landing_committed": False,
            "yaw_deg": 45.0,
            "yaw_target_deg": 60.0,
            "yaw_error_deg": 15.0,
            "command_action": np.array([0.1, 0.2, 0.0, 0.72, 0.15], dtype=float),
            "final_action": np.array([0.1, 0.2, 0.0, 0.65, 0.18], dtype=float),
            "prev_action": np.array([0.0, 0.3, 0.0, 0.60, 0.10], dtype=float),
            "gov_max_v_err": 1.8,
            "tilt_deg": 12.0,
            "eye_prob": 0.77,
            "eye_lost_steps": 3,
            "eye_filter_initialized": True,
            "detector_mode": "eye_pos6",
            "search_progress_m": 4.2,
            "altitude_m": 6.5,
        }
    )
    joined = "\n".join(lines)
    assert "Control" in joined
    assert "ctrl=main" in joined
    assert "search=spiral:st2" in joined
    assert "1st_cnt=12" in joined
    assert "Yaw" in joined
    assert "err=15°" in joined
    assert "Throttle" in joined
    assert "cmd_spd=0.72" in joined
    assert "out_spd=0.65" in joined
    assert "gov=1.80" in joined
    assert "Eye" in joined
    assert "det=eye_pos6" in joined


def test_build_panel_lines_include_controller_rows() -> None:
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
        sim_state="running",
        t_sim=1.0,
        frame=50,
        obs_info={
            "position": np.array([1.0, 2.0, 3.0]),
            "speed_mps": 1.0,
            "search_area_vector": np.array([4.0, -1.0, 0.5]),
            "search_area_center": np.array([5.0, 1.0, 3.5]),
        },
        agent_info={
            "mode": "navigation",
            "active_controller": "main",
            "forward": True,
            "yaw_error_deg": 4.0,
            "command_action": np.array([0.0, 1.0, 0.0, 0.8, 0.0], dtype=float),
            "final_action": np.array([0.0, 1.0, 0.0, 0.75, 0.0], dtype=float),
        },
        action=np.array([0.0, 1.0, 0.0, 0.75, 0.0], dtype=np.float32),
        camera_mode="chase",
    )
    joined = "\n".join(lines)
    assert "Control" in joined
    assert "Yaw" in joined
    assert "Throttle" in joined
    assert len(lines) <= TELEMETRY_MAX_LINES


def test_pad_lock_dist_to_go_matches_agent_formula() -> None:
    from swarm.core.fly_viewer import _pad_lock_dist_to_go

    goal = np.array([1.0, 0.0, 0.0], dtype=float)
    landing = np.array([0.8, 0.0, 0.0], dtype=float)
    assert math.isclose(_pad_lock_dist_to_go(goal, landing), 0.2)
    reverse = np.array([0.1, 0.0, 0.0], dtype=float)
    assert math.isclose(
        _pad_lock_dist_to_go(goal, landing, move_in_auto_mode=True, reverse_d=reverse),
        0.1,
    )


def test_left_panel_min_height_fits_controls() -> None:
    from swarm.core.fly_viewer import Y_CAMERA, compute_left_panel_min_height

    btn_h = 26
    gap = 6
    zoom_row_bottom = Y_CAMERA + 2 * (btn_h + gap) + btn_h
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


def test_colourise_depth_normalized_returns_rgb_uint8() -> None:
    depth = np.zeros((128, 128, 1), dtype=np.float32)
    depth[64:, :] = 1.0
    rgb = colourise_depth_normalized(depth)
    assert rgb.shape == (128, 128, 3)
    assert rgb.dtype == np.uint8
    assert int(rgb[0, 0, 0]) != int(rgb[-1, 0, 0])


def test_parse_seed_text_clamps_and_fallback() -> None:
    assert parse_seed_text("1001") == 1001
    assert parse_seed_text("3140505285") == 3140505285
    assert parse_seed_text("0", fallback=42) == 1
    assert parse_seed_text("abc", fallback=42) == 42
    assert parse_seed_text("", fallback=7) == 7


def test_chase_camera_smoothing_reduces_heading_jitter() -> None:
    camera = FlyRenderCamera((10.0, 0.0, 2.0), mode="chase")
    pos = np.array([0.0, 0.0, 1.0], dtype=float)
    quat_a = (0.0, 0.0, 0.0, 1.0)
    quat_b = (0.0, 0.0, 0.258819, 0.9659258)  # ~30 deg yaw

    eye_a, _ = camera.eye_and_target(pos, quat_a, dt=0.02)
    eye_b, _ = camera.eye_and_target(pos, quat_b, dt=0.02)

    camera_raw = FlyRenderCamera((10.0, 0.0, 2.0), mode="chase")
    raw_a, _ = camera_raw.eye_and_target(pos, quat_a, dt=0.02)
    camera_raw.reset_smoothing()
    raw_b, _ = camera_raw.eye_and_target(pos, quat_b, dt=0.02)

    smooth_delta = float(np.linalg.norm(eye_b - eye_a))
    raw_delta = float(np.linalg.norm(raw_b - raw_a))
    assert smooth_delta < raw_delta


def test_camera_mode_switch_resets_smoothing() -> None:
    camera = FlyRenderCamera((10.0, 0.0, 2.0), mode="chase")
    pos = np.array([0.0, 0.0, 1.0], dtype=float)
    camera.eye_and_target(pos, (0.0, 0.0, 0.258819, 0.9659258), dt=0.02)
    assert camera._smooth_fwd is not None

    camera.mode = "fpv"
    camera.eye_and_target(pos, (0.0, 0.0, 0.0, 1.0), dt=0.02)
    assert camera._smoothing_mode == "fpv"


def test_filter_saved_runs_matches_agent_seed_and_map() -> None:
    from pathlib import Path

    from swarm.core.fly_trajectory import SavedRunInfo
    from swarm.core.fly_viewer import filter_saved_runs

    runs = [
        SavedRunInfo(
            path=Path("/tmp/champion_UID_191_seed1001_city.flytraj.json.gz"),
            display_name="champion_UID_191 seed1001 city",
            agent_name="champion_UID_191",
            seed=1001,
            type_label="city",
            score_summary="0.89  S:1.00  T:0.82  Saf:0.55",
        ),
        SavedRunInfo(
            path=Path("/tmp/249_seed177550_warehouse.flytraj.json.gz"),
            display_name="249 seed177550 warehouse",
            agent_name="249",
            seed=177550,
            type_label="warehouse",
        ),
    ]
    assert len(filter_saved_runs(runs, "")) == 2
    assert len(filter_saved_runs(runs, "warehouse")) == 1
    assert filter_saved_runs(runs, "177550")[0].seed == 177550
    assert len(filter_saved_runs(runs, "champion")) == 1


def test_saved_runs_scrollbar_helpers() -> None:
    from pathlib import Path

    from swarm.core.fly_trajectory import SavedRunInfo
    from swarm.core.fly_viewer import SAVED_RUN_VISIBLE_ROWS, FlySimulatorWindow

    class _FakeWindow(FlySimulatorWindow):
        def __init__(self) -> None:
            self.saved_runs = [
                SavedRunInfo(
                    path=Path(f"/tmp/run_{idx}.json"),
                    display_name=f"run_{idx}",
                )
                for idx in range(10)
            ]
            self.run_scroll = 0
            self.run_search_text = ""

    window = _FakeWindow()
    assert window._saved_runs_max_scroll() == 10 - SAVED_RUN_VISIBLE_ROWS
    window.run_scroll = window._saved_runs_max_scroll()
    thumb_top = window._saved_runs_scrollbar_thumb_rect()[1]
    track_top = window._saved_runs_scrollbar_track_rect()[1]
    assert thumb_top >= track_top
    assert window._scroll_from_scrollbar_y(track_top) == 0
    track_bottom = track_top + window._saved_runs_scrollbar_track_rect()[3]
    assert window._scroll_from_scrollbar_y(track_bottom) == window._saved_runs_max_scroll()


def test_replay_goal_helpers_pick_mission_and_estimate() -> None:
    from types import SimpleNamespace

    from swarm.core.fly_viewer import (
        replay_estimated_goal_from_agent_info,
        replay_mission_goal,
    )

    trajectory = SimpleNamespace(meta={"goal": [10.0, 5.0, 2.0]})
    mission = replay_mission_goal(trajectory, task=None)
    assert mission is not None
    assert mission.tolist() == [10.0, 5.0, 2.0]

    agent_info = {
        "raw_goal_position": np.array([9.0, 4.0, 2.0]),
        "landing_platform_position": np.array([9.5, 4.5, 2.1]),
    }
    estimate = replay_estimated_goal_from_agent_info(agent_info)
    assert estimate is not None
    assert estimate.tolist() == [9.5, 4.5, 2.1]


def test_saved_run_detail_line_includes_timestamp_and_score() -> None:
    from pathlib import Path

    from swarm.core.fly_trajectory import SavedRunInfo
    from swarm.core.fly_viewer import _saved_run_detail_line

    run = SavedRunInfo(
        path=Path("/tmp/grit14_seed42_city.flytraj.json.gz"),
        display_name="grit14 seed42 city",
        score_summary="0.89  S:1.00  T:0.82  Saf:0.55",
        created_at="2026-06-30T14:51:00+00:00",
    )
    detail = _saved_run_detail_line(run)
    assert detail is not None
    assert "2026-06-30 07:51" in detail
    assert "0.89" in detail


def test_extract_flight_vector_prefers_command_action() -> None:
    action = np.array([1.0, 0.0, 0.0, 2.5, 0.0], dtype=float)
    obs_info = {
        "velocity": np.array([0.0, 1.0, 0.0]),
        "speed_mps": 1.0,
    }
    direction, speed, source = extract_flight_vector(action, obs_info)
    assert source == "command"
    assert speed == 2.5
    assert direction.tolist() == [1.0, 0.0, 0.0]


def test_extract_flight_vector_falls_back_to_velocity() -> None:
    obs_info = {
        "velocity": np.array([0.0, 2.0, 0.0]),
        "speed_mps": 2.0,
    }
    direction, speed, source = extract_flight_vector(None, obs_info)
    assert source == "velocity"
    assert speed == 2.0
    assert direction.tolist() == [0.0, 2.0, 0.0]


def test_project_world_direction_to_uv_center_for_forward() -> None:
    camera_pos = np.array([0.0, 0.0, 0.0], dtype=float)
    camera_target = np.array([1.0, 0.0, 0.0], dtype=float)
    camera_up = np.array([0.0, 0.0, 1.0], dtype=float)
    u, v, visible = project_world_direction_to_uv(
        np.array([1.0, 0.0, 0.0]),
        camera_pos=camera_pos,
        camera_target=camera_target,
        camera_up=camera_up,
        fov_deg=90.0,
        aspect=1.0,
    )
    assert visible
    assert math.isclose(u, 0.5, abs_tol=0.02)
    assert math.isclose(v, 0.5, abs_tol=0.02)


def test_annotate_depth_direction_overlay_draws_speed_strip() -> None:
    depth = np.zeros((32, 32, 3), dtype=np.uint8)
    pose = DroneCameraPose(
        camera_pos=np.array([0.0, 0.0, 0.0]),
        camera_target=np.array([1.0, 0.0, 0.0]),
        camera_up=np.array([0.0, 0.0, 1.0]),
        fov_deg=90.0,
        aspect=1.0,
    )
    annotated = annotate_depth_direction_overlay(
        depth,
        direction_world=np.array([0.0, 1.0, 1.0]),
        speed_mps=1.5,
        camera_pose=pose,
    )
    assert annotated.shape == depth.shape
    bottom_strip = annotated[-6:, :, 1]
    assert int(bottom_strip.max()) > 100
