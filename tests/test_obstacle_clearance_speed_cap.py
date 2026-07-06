from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_MINER_DIR = Path(__file__).resolve().parents[1] / "miners" / "new"
if str(_MINER_DIR) not in sys.path:
    sys.path.insert(0, str(_MINER_DIR))

from drone_agent import (  # noqa: E402
    DRONE_HULL_RADIUS_M,
    OBSTACLE_CLEARANCE_MIN_TRAVEL_M,
    OBSTACLE_CLEARANCE_RAMP_M,
    OBSTACLE_CLEARANCE_REQUIRED_M,
    OBSTACLE_REPEL_MAX_M,
    OBSTACLE_REPEL_MIN_M,
    OBSTACLE_TIGHT_STEER_LATERAL_M,
    SIM_DT,
    VILLAGE_SPAWN_GUARD_END_S,
    VILLAGE_SPAWN_MAX_SPEED_MPS,
    WORLD_SPEED_LIMIT_MPS,
    _clearance_along_direction,
    _compute_clearance_distances,
    _depth_lateral_clearances,
    _effective_clearance_m,
    _needs_tight_obstacle_steering,
    _pick_ctx,
    _repel_weight_for_distance,
    _side_hull_fov_clearance_m,
    _steer_for_closest_obstacle,
    _village_clearance_speed_scale,
    _village_early_spawn_speed_cap,
    cap_speed_for_obstacle_clearance,
    clearance_max_world_speed_mps,
)


def _dir_action(speed: float) -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, speed, 0.0], dtype=np.float32)


def test_cap_leaves_speed_unchanged_when_clearance_is_large() -> None:
    action = _dir_action(0.9)
    capped, debug = cap_speed_for_obstacle_clearance(action, OBSTACLE_CLEARANCE_RAMP_M + 1.0)
    assert float(capped[3]) == pytest.approx(0.9)
    assert debug["obstacle_speed_cap_applied"] is False


def test_cap_stops_when_clearance_is_below_min_travel() -> None:
    action = _dir_action(0.8)
    capped, debug = cap_speed_for_obstacle_clearance(action, OBSTACLE_CLEARANCE_MIN_TRAVEL_M - 0.1)
    assert float(capped[3]) == pytest.approx(0.0)
    assert debug["obstacle_speed_cap_applied"] is True


def test_cap_scales_speed_between_min_travel_and_ramp() -> None:
    action = _dir_action(1.0)
    clearance = 0.5 * (OBSTACLE_CLEARANCE_MIN_TRAVEL_M + OBSTACLE_CLEARANCE_RAMP_M)
    capped, debug = cap_speed_for_obstacle_clearance(action, clearance)
    assert 0.0 < float(capped[3]) < 1.0
    assert debug["obstacle_speed_cap_applied"] is True


def test_clearance_max_world_speed_uses_required_margin() -> None:
    assert clearance_max_world_speed_mps(OBSTACLE_CLEARANCE_REQUIRED_M) == pytest.approx(0.0)
    assert clearance_max_world_speed_mps(2.0) == pytest.approx((2.0 - OBSTACLE_CLEARANCE_REQUIRED_M) / SIM_DT)


def test_cap_physics_limit_uses_required_margin() -> None:
    action = _dir_action(1.0)
    clearance = 1.4
    capped, _debug = cap_speed_for_obstacle_clearance(action, clearance)
    expected = clearance_max_world_speed_mps(clearance) / WORLD_SPEED_LIMIT_MPS
    assert float(capped[3]) == pytest.approx(expected, rel=1e-3)
    assert float(capped[3]) < 1.0


def test_clearance_along_direction_uses_forward_cone() -> None:
    ctx = {
        "candidate_world_dirs": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }
    clearances = np.array([2.0, 0.8, 5.0], dtype=np.float32)
    forward = _clearance_along_direction(ctx, clearances, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert forward == pytest.approx(2.0)


def test_depth_lateral_clearances_use_columns() -> None:
    depth_m = np.full((32, 48), 5.0, dtype=np.float32)
    depth_m[:, :16] = 2.0
    lateral = _depth_lateral_clearances(depth_m)
    assert lateral["left_clearance_m"] < lateral["right_clearance_m"]
    assert lateral["lateral_clearance_m"] == pytest.approx(lateral["left_clearance_m"])


def test_effective_clearance_is_min_of_forward_lateral_fov_and_side_hull() -> None:
    ctx = {
        "candidate_world_dirs": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    }
    clearances = np.array([4.0, 1.2], dtype=np.float32)
    depth_m = np.full((32, 48), 5.0, dtype=np.float32)
    depth_m[:, :16] = 2.0
    effective, details = _effective_clearance_m(
        ctx,
        clearances,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        depth_m,
    )
    assert effective == pytest.approx(
        min(
            details["forward_clearance_m"],
            details["lateral_clearance_m"],
            details["fov_clearance_m"],
            details["side_hull_clearance_m"],
        )
    )


def test_side_hull_clearance_is_lower_at_fov_edges() -> None:
    depth_m = np.full((32, 48), 5.0, dtype=np.float32)
    center = _side_hull_fov_clearance_m(depth_m)
    depth_m[:, 0] = 2.0
    edge = _side_hull_fov_clearance_m(depth_m)
    assert edge < center


def test_needs_tight_steering_when_lateral_below_threshold() -> None:
    assert _needs_tight_obstacle_steering(
        fov_clearance_m=5.0,
        lateral_clearance_m=OBSTACLE_TIGHT_STEER_LATERAL_M - 0.1,
        side_hull_clearance_m=5.0,
    )


def test_village_spawn_speed_cap_limits_early_search() -> None:
    lateral = {
        "left_clearance_m": 6.0,
        "right_clearance_m": 4.0,
        "center_clearance_m": 5.0,
        "lateral_clearance_m": 4.0,
        "lateral_divergence_m": 2.0,
    }
    capped, scale = _village_early_spawn_speed_cap(
        1.0,
        sim_time_s=VILLAGE_SPAWN_GUARD_END_S - 1.0,
        lateral=lateral,
        fov_clearance_m=OBSTACLE_CLEARANCE_RAMP_M + 1.0,
        drone_speed_mps=3.0,
    )
    assert capped <= VILLAGE_SPAWN_MAX_SPEED_MPS / WORLD_SPEED_LIMIT_MPS + 1e-6
    assert scale < 1.0


def test_steer_triggers_on_lateral_clearance() -> None:
    ctx = {
        "candidate_world_dirs": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }
    clearances = np.array([5.0, 5.0], dtype=np.float32)
    preferred = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    _steered, _clr, tight = _steer_for_closest_obstacle(
        ctx,
        clearances,
        preferred,
        lateral_clearance_m=OBSTACLE_TIGHT_STEER_LATERAL_M - 0.5,
        side_hull_clearance_m=5.0,
    )
    assert tight is True


def test_repel_weight_is_stronger_when_closer() -> None:
    assert _repel_weight_for_distance(OBSTACLE_REPEL_MAX_M) == pytest.approx(0.0)
    assert _repel_weight_for_distance(OBSTACLE_REPEL_MIN_M) == pytest.approx(1.0)
    mid = 0.5 * (OBSTACLE_REPEL_MIN_M + OBSTACLE_REPEL_MAX_M)
    assert 0.0 < _repel_weight_for_distance(mid) < 1.0


def test_steer_for_closest_obstacle_moves_away_from_threat() -> None:
    ctx = {
        "candidate_world_dirs": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "candidate_dirs": np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "camera_forward": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "camera_right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "camera_up_vector": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    clearances = np.array([0.8, 4.0, 3.0], dtype=np.float32)
    preferred = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    steered, min_clr, tight = _steer_for_closest_obstacle(ctx, clearances, preferred)
    assert tight is True
    assert min_clr == pytest.approx(0.8)
    assert float(steered[1]) > 0.15


def test_pick_ctx_prefers_clearance_over_goal_angle() -> None:
    ctx = {
        "candidate_dirs": np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "camera_forward": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "camera_right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "camera_up_vector": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "preference_angles": np.array([0.05, 0.8], dtype=np.float32),
    }

    def _fake_clearance(_ctx, **kwargs):
        return np.array([4.0, 1.2], dtype=np.float32)

    import drone_agent as da

    original = da._compute_clearance_distances
    da._compute_clearance_distances = _fake_clearance
    try:
        picked = _pick_ctx(ctx, preferred_clearance_m=1.0, max_lookahead_m=10.0)
    finally:
        da._compute_clearance_distances = original

    assert float(picked[2]) == pytest.approx(1.0)
    assert float(picked[0]) == pytest.approx(0.0)


def test_village_clearance_speed_scale() -> None:
    assert _village_clearance_speed_scale(3.0) == pytest.approx(1.0)
    assert _village_clearance_speed_scale(OBSTACLE_CLEARANCE_REQUIRED_M) == pytest.approx(0.0)


def test_compute_clearance_distances_returns_open_space() -> None:
    ctx = {
        "candidate_dirs": np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        "surface_ranges": np.array([[20.0]], dtype=np.float32),
        "obstacle_ranges_sq": np.array([400.0], dtype=np.float32),
        "projections": np.array([[20.0]], dtype=np.float32),
    }
    clearances = _compute_clearance_distances(ctx, max_lookahead_m=15.0)
    assert float(clearances[0]) == pytest.approx(15.0)
    assert DRONE_HULL_RADIUS_M == pytest.approx(0.12)
