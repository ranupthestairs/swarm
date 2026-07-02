from __future__ import annotations

from typing import Optional, Iterable, Tuple
from pathlib import Path
import json
import torch
import onnxruntime as ort
import numpy as np

DEFAULT_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)
DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 20.0
MAX_REAR_DIST = 60.0
MAX_YAW_RATE = 3.141
SIM_DT = 1 / 50
CAMERA_FOV_DEG = 90.0
CAMERA_OFFSET_M = 0.13
CAMERA_UP_OFFSET_M = 0.05
m_STATE_DIM = 141
STATIC_LOCK_WINDOW_TICKS = 50
STATIC_LOCK_REQUIRED_HITS = 35
STATIC_LOCK_MIN_PROBABILITY = 0.8
STATIC_LOCK_MAX_MOTION_M = 0.5
STATIC_LOCK_MAX_AVG_DISTANCE_M = 6.0
STATIC_LOCK_MAX_REVERSE_SPEED = 0.08
VILLAGE_GOAL_VISIBLE_THRESHOLD = 0.8
VILLAGE_LANDING_MIN_PROBABILITY = 0.9
VILLAGE_SEARCH_VECTOR_DAMP_RADIUS_M = 3.0
VILLAGE_SEARCH_MIN_DIRECTION_TRUST = 0.2
VILLAGE_SEARCH_MIN_VERTICAL_LIMIT = 0.35
VILLAGE_SEARCH_MAX_VERTICAL_LIMIT = 0.6
VILLAGE_SEARCH_CRUISE_ALTITUDE_M = 6.5
VILLAGE_SEARCH_MIN_ALTITUDE_M = 6.0
SPIRAL_SEARCH_START_RADIUS_M = 1.5
SPIRAL_SEARCH_MAX_RADIUS_M = 22.0
SPIRAL_SEARCH_RADIUS_STEP_M = 0.4
SPIRAL_SEARCH_ANGLE_STEP_RAD = np.pi / 18.0
VILLAGE_SPIRAL_MAX_SPEED = 0.45
VILLAGE_SPIRAL_MAX_VERTICAL = 0.20
VILLAGE_SAFE_CRUISE_ALTITUDE_M = 6.5
VILLAGE_SAFE_HOVER_ABOVE_PAD_M = 2.5
VILLAGE_SAFE_CENTER_DISTANCE_M = 0.50
VILLAGE_SAFE_DESCENT_MAX_SPEED = 0.20
VILLAGE_SAFE_HORIZONTAL_MAX_SPEED = 0.38
VILLAGE_NAV_MIN_ALTITUDE_M = 6.0
VILLAGE_NAV_APPROACH_MIN_ALTITUDE_M = 5.5
VILLAGE_NAV_FLOOR_RELEASE_DISTANCE_M = 5.0
VILLAGE_CLOSE_GOAL_DISTANCE_M = 6.0
VILLAGE_GOAL_TRACK_RELEASE_DISTANCE_M = 5.0
VILLAGE_GOAL_TRACK_MIN_PROBABILITY = 0.05
VILLAGE_NAV_SLOW_ALTITUDE_M = 5.0
VILLAGE_NAV_MAX_LOW_SPEED = 1.0
VILLAGE_NAV_Z_GOAL_GAIN = 1.0
VILLAGE_NAV_Z_GOAL_GAIN_CLOSE = 1.0
VILLAGE_AVOID_SAFETY_MARGIN_M = 0.85
VILLAGE_AVOID_PREFERRED_CLEARANCE_M = 5.0
VILLAGE_AVOID_CLOSE_SAFETY_MARGIN_M = 1.6
VILLAGE_AVOID_CLOSE_PREFERRED_CLEARANCE_M = 3.5
VILLAGE_CLOSE_NAV_HOLD_DISTANCE_M = 12.0
VILLAGE_FINAL_APPROACH_DISTANCE_M = 12.0
MAP_XGB_MODEL_NAME = "map_xgboost_state_depth_stats_model.json"
MAP_LABELS = ("city", "open", "mountain", "village", "warehouse", "forest")
MAP_STATIC_LABELS = {"warehouse", "forest"}
MAP_STATIC_CONFIDENCE_THRESHOLD = 0.80
MAP_STATIC_MIN_PREDICTIONS = 2
MAP_PREDICT_EVERY_TICKS = 5
MAP_PREDICT_MAX_TICK = 120
MAP_UID53_RULE_LABELS = {"open", "village"}
MAP_PT_DETECTOR_LABELS = {"mountain"}
MAP_HIGH_TAKEOFF_LABELS = {"mountain", "village"}
MAP_RULE_MIN_PREDICTIONS = 2
MAP_RULE_MIN_PROBABILITY = 0.80
CONSIST_N = 6
CONSIST_SPREAD = 1.0
CONSIST_NEAR = 25.0


def _default_H_m() -> np.ndarray:
    dt = SIM_DT
    x = np.zeros((6, 1), dtype=np.float32)
    p = np.eye(6, dtype=np.float32)
    f_m = np.array(
        [
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    h_m = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    q_m = np.eye(6, dtype=np.float32) * 0.01
    r_m = np.eye(3, dtype=np.float32) * 0.05
    i_m = np.eye(6, dtype=np.float32)
    h_t = h_m.T
    r2 = np.concatenate([r_m, r_m.T], axis=0)
    return np.concatenate([x, p, f_m, h_t, q_m, r2, i_m], axis=1)


class Gains:
    K: float = 0.4


class _XGBMapPredictor:

    def __init__(self, model_path: Path):
        self.enabled = False
        self.num_class = 0
        self.base_score = None
        self.tree_info = []
        self.trees = []
        if not model_path.exists():
            return
        with model_path.open("r", encoding="utf-8") as handle:
            model = json.load(handle)
        learner = model["learner"]
        params = learner["learner_model_param"]
        self.num_class = int(params["num_class"])
        self.base_score = np.asarray(json.loads(params["base_score"]), dtype=np.float32)
        booster = learner["gradient_booster"]["model"]
        self.tree_info = [int(v) for v in booster["tree_info"]]
        self.trees = []
        for tree in booster["trees"]:
            self.trees.append(
                {
                    "left": np.asarray(tree["left_children"], dtype=np.int32),
                    "right": np.asarray(tree["right_children"], dtype=np.int32),
                    "split_idx": np.asarray(tree["split_indices"], dtype=np.int32),
                    "split_cond": np.asarray(tree["split_conditions"], dtype=np.float32),
                    "default_left": np.asarray(tree["default_left"], dtype=np.int8),
                    "weights": np.asarray(tree["base_weights"], dtype=np.float32),
                }
            )
        self.enabled = self.num_class == len(MAP_LABELS) and len(self.trees) == len(self.tree_info)

    def predict_proba(self, features: np.ndarray) -> np.ndarray | None:
        if not self.enabled:
            return None
        x = np.asarray(features, dtype=np.float32).reshape(-1)
        margins = self.base_score.astype(np.float32).copy()
        for tree, class_idx in zip(self.trees, self.tree_info):
            node = 0
            left = tree["left"]
            right = tree["right"]
            split_idx = tree["split_idx"]
            split_cond = tree["split_cond"]
            default_left = tree["default_left"]
            weights = tree["weights"]
            while left[node] != -1:
                feature_idx = int(split_idx[node])
                value = x[feature_idx] if feature_idx < x.size else np.nan
                if np.isnan(value):
                    node = int(left[node] if default_left[node] else right[node])
                elif float(value) < float(split_cond[node]):
                    node = int(left[node])
                else:
                    node = int(right[node])
            margins[class_idx] += weights[node]
        margins -= np.max(margins)
        probs = np.exp(margins)
        denom = float(np.sum(probs))
        if denom <= 1e-12:
            return None
        return (probs / denom).astype(np.float32)


def _map_depth_2d(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        arr = np.reshape(arr, arr.shape[:2])
    return np.clip(arr, 0.0, 1.0)


def _map_depth_feature_values(depth: np.ndarray) -> list[float]:
    d = _map_depth_2d(depth)
    gx = np.abs(np.diff(d, axis=1))
    gy = np.abs(np.diff(d, axis=0))
    percentiles = np.percentile(d, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    values = [
        float(d.min()),
        float(d.mean()),
        float(d.std()),
        float(d.max()),
        float(percentiles[0]),
        float(percentiles[1]),
        float(percentiles[2]),
        float(percentiles[3]),
        float(percentiles[4]),
        float(percentiles[5]),
        float(percentiles[6]),
        float(percentiles[7]),
        float(percentiles[8]),
        float((d <= 0.10).mean()),
        float((d <= 0.25).mean()),
        float((d >= 0.95).mean()),
        float((d >= 0.999).mean()),
        float((d <= 0.001).mean()),
        float(gx.mean() + gy.mean()),
        float(((gx > 0.05).mean() + (gy > 0.05).mean()) / 2.0),
    ]
    h, w = d.shape
    tile_means = []
    tile_mins = []
    tile_maxs = []
    for y0 in np.linspace(0, h, 5, dtype=int)[:-1]:
        y1 = int(y0 + h // 4)
        for x0 in np.linspace(0, w, 5, dtype=int)[:-1]:
            x1 = int(x0 + w // 4)
            tile = d[y0:y1, x0:x1]
            tile_means.append(float(tile.mean()))
            tile_mins.append(float(tile.min()))
            tile_maxs.append(float(tile.max()))
    values.extend(tile_means)
    values.extend(tile_mins)
    values.extend(tile_maxs)
    return values


class _ABTracker:

    def __init__(self, alpha=0.45, beta=0.10, dt=SIM_DT):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.dt = float(dt)
        self.pos = None
        self.vel = np.zeros(3, dtype=np.float32)
        self.initialized = False

    def reset(self):
        self.pos = None
        self.vel = np.zeros(3, dtype=np.float32)
        self.initialized = False

    def update(self, m):
        m = np.asarray(m, dtype=np.float32).reshape(3)
        if not self.initialized:
            self.pos = m.copy()
            self.vel = np.zeros(3, dtype=np.float32)
            self.initialized = True
            return
        pos_pred = self.pos + self.vel * self.dt
        r = m - pos_pred
        self.pos = pos_pred + self.alpha * r
        self.vel = self.vel + (self.beta / self.dt) * r
        vn = float(np.linalg.norm(self.vel))
        if vn > 3.0:
            self.vel = self.vel * (3.0 / vn)


def _prep_depth(depth_map: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim != 2:
        raise ValueError(f"expected depth map with shape (H,W) or (H,W,1), got {depth.shape}")

    if depth.shape[0] < 2 or depth.shape[1] < 2:
        raise ValueError(f"depth map is too small: {depth.shape}")

    return np.clip(depth, 0.0, 1.0)


def _norm_depth_m(depth: np.ndarray) -> np.ndarray:
    return DEPTH_MIN_M + depth * (DEPTH_MAX_M - DEPTH_MIN_M)

def _cam_vec_world(
    camera_vector: np.ndarray,
    camera_forward: np.ndarray,
    camera_right: np.ndarray,
    camera_up: np.ndarray,
) -> np.ndarray:
    world = (
        camera_right * float(camera_vector[0])
        + camera_up * float(camera_vector[1])
        + camera_forward * float(camera_vector[2])
    ).astype(np.float32)
    return _norm_vec(world)

def _cam_basis(
    camera_target: np.ndarray,
    *,
    camera_position: np.ndarray,
    camera_target_is_point: bool,
    camera_up: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_position = np.asarray(camera_position, dtype=np.float32).reshape(-1)
    if camera_position.shape != (3,):
        raise ValueError(f"expected camera_position shape (3,), got {camera_position.shape}")

    target = np.asarray(camera_target, dtype=np.float32).reshape(-1)
    if target.shape != (3,):
        raise ValueError(f"expected camera_target shape (3,), got {target.shape}")

    if camera_target_is_point:
        target = target - camera_position

    forward = _norm_vec(target)
    up_guess = DEFAULT_WORLD_UP if camera_up is None else _norm_vec(camera_up)

    right = np.cross(forward, up_guess).astype(np.float32)
    if float(np.linalg.norm(right)) <= 1e-8:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, fallback_up).astype(np.float32)

    right = _norm_vec(right)
    up = _norm_vec(np.cross(right, forward))
    return forward, right, up

def _parse_fov(fov_deg: float | Iterable[float]) -> tuple[float, float]:
    if np.isscalar(fov_deg):
        fov_x = float(fov_deg)
        fov_y = float(fov_deg)
    else:
        values = tuple(float(v) for v in fov_deg)
        if len(values) != 2:
            raise ValueError(f"expected scalar FOV or (horizontal, vertical), got {values}")
        fov_x, fov_y = values

    if not (0.0 < fov_x < 180.0 and 0.0 < fov_y < 180.0):
        raise ValueError(f"invalid FOV values: {(fov_x, fov_y)}")
    return fov_x, fov_y


def _min_pool(depth_m: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    row_idx = np.linspace(0, depth_m.shape[0], out_h + 1, dtype=np.int32)
    col_idx = np.linspace(0, depth_m.shape[1], out_w + 1, dtype=np.int32)
    temp = np.minimum.reduceat(depth_m, row_idx[:-1], axis=0)
    return np.minimum.reduceat(temp, col_idx[:-1], axis=1).astype(np.float32)

def _work_shape(height: int, width: int, working_resolution: int) -> tuple[int, int]:
    if working_resolution < 3:
        raise ValueError(f"working_resolution must be >= 3, got {working_resolution}")

    scale = min(1.0, float(working_resolution) / float(max(height, width)))
    out_h = max(3, int(round(height * scale)))
    out_w = max(3, int(round(width * scale)))

    if out_h % 2 == 0:
        out_h = max(3, out_h - 1)
    if out_w % 2 == 0:
        out_w = max(3, out_w - 1)

    return min(out_h, height), min(out_w, width)

def _cand_rays(
    height: int,
    width: int,
    fov_x_deg: float,
    fov_y_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(1.0, -1.0, height, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)

    tan_half_x = float(np.tan(np.deg2rad(np.float32(fov_x_deg)) * 0.5))
    tan_half_y = float(np.tan(np.deg2rad(np.float32(fov_y_deg)) * 0.5))

    ray_points = np.stack(
        [
            x_grid * tan_half_x,
            y_grid * tan_half_y,
            np.ones_like(x_grid, dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    ray_norms = np.linalg.norm(ray_points, axis=-1, keepdims=True)
    ray_dirs = (ray_points / np.maximum(ray_norms, 1e-8)).astype(np.float32)
    return ray_points, ray_dirs


def _cam_vecs_world(
    camera_vectors: np.ndarray,
    camera_forward: np.ndarray,
    camera_right: np.ndarray,
    camera_up: np.ndarray,
) -> np.ndarray:
    vectors = np.asarray(camera_vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"expected camera_vectors shape (N, 3), got {vectors.shape}")

    world = (
        vectors[:, 0:1] * camera_right[None, :]
        + vectors[:, 1:2] * camera_up[None, :]
        + vectors[:, 2:3] * camera_forward[None, :]
    ).astype(np.float32)
    norms = np.linalg.norm(world, axis=1, keepdims=True)
    return (world / np.maximum(norms, 1e-8)).astype(np.float32)


def _safe_ctx(
    depth_map: np.ndarray,
    camera_position: np.ndarray,
    camera_target: np.ndarray,
    fov_deg: float | Iterable[float],
    *,
    current_direction: np.ndarray | None = None,
    camera_target_is_point: bool = False,
    camera_up: np.ndarray | None = None,
    working_resolution: int = 49,
) -> dict:
    depth = _prep_depth(depth_map)
    depth_m = _norm_depth_m(depth)
    fov_x_deg, fov_y_deg = _parse_fov(fov_deg)
    camera_forward, camera_right, camera_up_vector = _cam_basis(
        camera_target,
        camera_position=np.asarray(camera_position, dtype=np.float32),
        camera_target_is_point=camera_target_is_point,
        camera_up=camera_up,
    )

    pooled_h, pooled_w = _work_shape(depth_m.shape[0], depth_m.shape[1], working_resolution)
    pooled_depth_m = _min_pool(depth_m, pooled_h, pooled_w)
    ray_points_cam, ray_dirs_cam = _cand_rays(pooled_h, pooled_w, fov_x_deg, fov_y_deg)

    surface_points_cam = (ray_points_cam * pooled_depth_m[..., None]).astype(np.float32)
    surface_ranges = np.linalg.norm(surface_points_cam, axis=-1)

    candidate_dirs = ray_dirs_cam.reshape(-1, 3)
    candidate_world_dirs = _cam_vecs_world(
        candidate_dirs,
        camera_forward=camera_forward,
        camera_right=camera_right,
        camera_up=camera_up_vector,
    )

    if current_direction is None:
        preferred_direction = camera_forward
    else:
        preferred_candidate = np.asarray(current_direction, dtype=np.float32).reshape(-1)
        if preferred_candidate.shape != (3,):
            raise ValueError(f"expected current_direction shape (3,), got {preferred_candidate.shape}")
        if float(np.linalg.norm(preferred_candidate)) <= 1e-8:
            preferred_direction = camera_forward
        else:
            preferred_direction = _norm_vec(preferred_candidate)

    preference_angles = np.arccos(
        np.clip(candidate_world_dirs @ preferred_direction, -1.0, 1.0)
    ).astype(np.float32)

    obstacle_ranges_sq = np.sum(
        surface_points_cam.reshape(-1, 3) ** 2, axis=1, dtype=np.float32
    )
    projections = candidate_dirs @ surface_points_cam.reshape(-1, 3).T

    return {
        "candidate_dirs": candidate_dirs,
        "camera_forward": camera_forward,
        "camera_right": camera_right,
        "camera_up_vector": camera_up_vector,
        "surface_ranges": surface_ranges,
        "obstacle_ranges_sq": obstacle_ranges_sq,
        "projections": projections,
        "preference_angles": preference_angles,
    }


def _pick_ctx(
    ctx: dict,
    *,
    drone_radius_m: float = 0.06,
    safety_margin_m: float = 0.03,
    preferred_clearance_m: float = 3.0,
    max_lookahead_m: float = 8.0,
) -> np.ndarray:
    candidate_dirs = ctx["candidate_dirs"]
    surface_ranges = ctx["surface_ranges"]
    obstacle_ranges_sq = ctx["obstacle_ranges_sq"]
    projections = ctx["projections"]
    preference_angles = ctx["preference_angles"]

    effective_radius = float(drone_radius_m + safety_margin_m)
    relevant_points = surface_ranges.ravel() <= float(max_lookahead_m + effective_radius)

    if np.any(relevant_points):
        rel_idx = np.flatnonzero(relevant_points)
        proj_rel = projections[:, rel_idx]
        orsq_rel = obstacle_ranges_sq[rel_idx]

        lateral_sq = np.clip(orsq_rel[None, :] - proj_rel * proj_rel, 0.0, None)

        collision_mask = (proj_rel > 0.0) & (lateral_sq < (effective_radius * effective_radius))
        collision_distances = np.full_like(proj_rel, np.inf, dtype=np.float32)

        if np.any(collision_mask):
            penetration = np.sqrt(
                np.maximum((effective_radius * effective_radius) - lateral_sq[collision_mask], 0.0)
            ).astype(np.float32)
            collision_distances[collision_mask] = np.maximum(
                proj_rel[collision_mask] - penetration,
                0.0,
            )

        min_collision_distance = np.min(collision_distances, axis=1)
        min_collision_distance[~np.isfinite(min_collision_distance)] = float(max_lookahead_m)
        clearance_distance = np.minimum(min_collision_distance, float(max_lookahead_m)).astype(np.float32)
    else:
        clearance_distance = np.full(candidate_dirs.shape[0], float(max_lookahead_m), dtype=np.float32)

    safe_candidates = np.flatnonzero(clearance_distance >= float(preferred_clearance_m))
    if safe_candidates.size > 0:
        best_order = np.lexsort(
            (
                -clearance_distance[safe_candidates],
                preference_angles[safe_candidates],
            )
        )
        best_idx = int(safe_candidates[best_order[0]])
    else:
        best_order = np.lexsort((preference_angles, -clearance_distance))
        best_idx = int(best_order[0])

    best_camera_direction = candidate_dirs[best_idx]
    return _cam_vec_world(
        best_camera_direction,
        camera_forward=ctx["camera_forward"],
        camera_right=ctx["camera_right"],
        camera_up=ctx["camera_up_vector"],
    )


def _norm_vec(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"expected a 3D vector, got shape {arr.shape}")

    norm = float(np.linalg.norm(arr))
    if norm <= eps:
        raise ValueError("cannot normalize a near-zero vector")
    return (arr / norm).astype(np.float32)

def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rotation_x = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ], dtype=np.float32)

    rotation_y = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ], dtype=np.float32)

    rotation_z = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ], dtype=np.float32)

    return (rotation_z @ rotation_y @ rotation_x).astype(np.float32)

def _cam_geom(
    drone_position: np.ndarray,
    drone_rpy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation_matrix = rpy_to_rot(drone_rpy[0], drone_rpy[1], drone_rpy[2])
    forward_direction = _norm_vec(rotation_matrix @ np.array([1.0, 0.0, 0.0], dtype=np.float32))
    up_guess = _norm_vec(rotation_matrix @ np.array([0.0, 0.0, 1.0], dtype=np.float32))
    right_direction = np.cross(forward_direction, up_guess).astype(np.float32)

    if float(np.linalg.norm(right_direction)) <= 1e-8:
        right_direction = np.cross(forward_direction, np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)

    right_direction = _norm_vec(right_direction)
    up_direction = _norm_vec(np.cross(right_direction, forward_direction))

    camera_position = (
        np.asarray(drone_position, dtype=np.float32)
        + forward_direction * CAMERA_OFFSET_M
        + up_guess * CAMERA_UP_OFFSET_M
    ).astype(np.float32)
    camera_target = (camera_position + forward_direction * 20.0).astype(np.float32)

    return camera_position, camera_target, right_direction, up_direction, forward_direction


def slerp_dir(a, b, t, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return b
    a /= na
    b /= nb

    if t <= 0.0: return a
    if t >= 1.0: return b
 
    cross = np.cross(a, b)
    cross_norm = np.linalg.norm(cross)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    angle = np.arctan2(cross_norm, dot)  # in [0, π]

    if angle < 1e-8: 
        v = b
    else:
        if np.pi - angle < 1e-8:  
            x = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(a, x)) > 0.9:
                x = np.array([0.0, 1.0, 0.0])
            k = np.cross(a, x)
            k /= np.linalg.norm(k)
        else:
            k = cross / cross_norm

        c, s = np.cos(t * angle), np.sin(t * angle)
        v = a * c + np.cross(k, a) * s + k * (np.dot(k, a)) * (1 - c)

    return v / np.linalg.norm(v)


class DroneFlightController:
    def __init__(
        self,
        *,
        m_model_path: Optional[Path] = None,
    ):
        script_dir = Path(__file__).resolve().parent

        if m_model_path is None:
            m_model_path = script_dir / "model.onnx"
        nav_model_path = script_dir / "navigation_model.onnx"
        self._map_predictor = _XGBMapPredictor(script_dir / MAP_XGB_MODEL_NAME)
        self._pt_model_path = script_dir / "model.pt"
        self._pt_model = None
        self._pt_device = None
        self._pt_M = None
        self._load_model(m_model_path)
        self._ab_tracker = _ABTracker()
        self.reset()
        
        self._load_navigation_model(nav_model_path)
        self.controller = LandingController(gains=Gains())
    
    def _getSecondOrderAction(self, new_action, state, attitude_action=0.0, is_smooth=True, eps=3e-5):
        if not is_smooth:
            self.prev_action = new_action.copy()
            return new_action.reshape(1, 5)

        def _to_vel(action):
            d = np.asarray(action[:3], dtype=np.float64)
            s = float(action[3])
            n = float(np.linalg.norm(d))
            return d * s if (n >= eps and s >= eps) else np.zeros(3)
        buffer = state[12:137].reshape(25, 5)
        v_k   = _to_vel(buffer[-1])
        v_km1 = _to_vel(buffer[-2])
        a_k   = v_k - v_km1   
        v_sp  = _to_vel(new_action)  

        alpha    = 0.3
        beta     = 0.2
        clip_a   = 0.05

        a_next = alpha * a_k + beta * (v_sp - v_k)
        a_next = np.clip(a_next, -clip_a, clip_a)
        v_next = v_k + a_next

        speed_next = float(np.linalg.norm(v_next))
        if speed_next < eps:
            self.prev_action[:3] = 0.0
            self.prev_action[3]  = 0.0
        else:
            self.prev_action[:3] = (v_next / speed_next).astype(np.float32)
            self.prev_action[3]  = float(speed_next)
        self.prev_action[4] = new_action[4]

        return_action = self.prev_action.copy()
        return_action[2] += attitude_action
        dir_norm = float(np.linalg.norm(return_action[:3]))
        if dir_norm < eps:
            return_action[:3] = 0.0
        else:
            return_action[:3] /= dir_norm
        return return_action.reshape(1, 5)


    def act(self, observation):
        state = np.asarray(observation.get("state", None), dtype=np.float32).squeeze()

        if self.start_state is None:
            self.start_state = state.copy()

        drone_position = np.array([state[0], state[1], state[2]], dtype=float)
        drone_rpy = np.array([state[3], state[4], state[5]], dtype=float)
        drone_velocity = np.array([state[6], state[7], state[8]], dtype=float)
        drone_speed = float(np.linalg.norm(drone_velocity))
        drone_altitude = state[-4] * 20.0

        search_area_vector = np.array([state[-3], state[-2], state[-1]], dtype=float)
        dist_raw_3d = float(np.linalg.norm(search_area_vector))
        depth = np.asarray(observation["depth"], dtype=np.float32)
        self._update_map_prediction(state, depth)
        self._mountain_flight = self._xgb_map_is(MAP_PT_DETECTOR_LABELS)
        self._high_takeoff_map = self._xgb_map_is(MAP_HIGH_TAKEOFF_LABELS)

        if (self._mode == "search" and self._goal_return_mode
                and self._last_known_goal_pos is not None):
            self._goal_return_search_steps += 1
            if self._goal_return_search_steps > 400:
                self._goal_return_mode = False
                self._last_known_goal_pos = None
                self.extra_search_vectors = None
                self.search_stage = 0
                self.search_state_0_rot_count = 0
            else:
                goal_vec = self._last_known_goal_pos - drone_position
                if float(np.linalg.norm(goal_vec)) > 0.1:
                    search_area_vector = goal_vec

        search_area_position = self._get_search_area_position(search_area_vector, drone_position)

        search_area_vector = search_area_position - drone_position
        distance_to_search_area = float(np.linalg.norm(search_area_vector))
        village_search_lift = self._xgb_map_is({"village"})
        search_progress_distance = (
            float(np.linalg.norm(search_area_vector[:2]))
            if village_search_lift
            else distance_to_search_area
        )
        if village_search_lift:
            pad_pos = self._village_pad_position()
            if pad_pos is not None:
                pad_hdist = float(
                    np.linalg.norm(drone_position[0:2] - pad_pos[0:2])
                )
            else:
                pad_hdist = None
            if pad_hdist is not None and pad_hdist < VILLAGE_FINAL_APPROACH_DISTANCE_M:
                hover_z = max(
                    VILLAGE_SEARCH_CRUISE_ALTITUDE_M,
                    float(pad_pos[2]) + VILLAGE_SAFE_HOVER_ABOVE_PAD_M,
                )
                to_pad = pad_pos - drone_position
                to_pad[2] = hover_z - float(drone_position[2])
                search_area_vector = to_pad.astype(np.float32)
                distance_to_search_area = float(np.linalg.norm(search_area_vector))
                search_progress_distance = pad_hdist
            else:
                near_goal = (
                    pad_hdist is not None
                    and pad_hdist < 2.5
                    and float(drone_position[2]) < VILLAGE_SEARCH_MIN_ALTITUDE_M
                )
                if not near_goal:
                    search_area_vector[2] = VILLAGE_SEARCH_CRUISE_ALTITUDE_M - float(drone_position[2])
        elif distance_to_search_area > 10.0:
            search_area_vector[2] += 1.5
        else:
            offset = 0.5 + distance_to_search_area / 10.0
            search_area_vector[2] += offset

        self._apply_static_map_override()

        is_goal_visible = False
        visible_goal_position = None
        visible_goal_position_cov = None
        visible_goal_velocity = None
        lock_goal_probability = None
        lock_goal_position = None

        if distance_to_search_area <= MAX_REAR_DIST or self._mode in ("navigation", "landing"):
            goal_visibility_prob, predicted_goal_position, predicted_goal_position_cov, predicted_platform_velocity = self._find_platform(depth, state)
            self._last_goal_visibility_prob = float(goal_visibility_prob)
            lock_goal_probability = goal_visibility_prob
            if goal_visibility_prob >= STATIC_LOCK_MIN_PROBABILITY:
                lock_goal_position = predicted_goal_position.copy()
            if self._xgb_map_is({"village"}) and goal_visibility_prob > 0.95 and predicted_goal_position is not None:
                pred_hdist = float(
                    np.linalg.norm(drone_position[0:2] - predicted_goal_position[0:2])
                )
                if pred_hdist < 35.0:
                    self._village_locked_goal_xy = np.asarray(
                        predicted_goal_position[0:2], dtype=np.float32
                    ).copy()
                    self._village_locked_goal_z = float(predicted_goal_position[2])
            if self._xgb_map_is({"village"}):
                if goal_visibility_prob > VILLAGE_GOAL_VISIBLE_THRESHOLD:
                    is_goal_visible = True
                    self._goal_is_tracked = True
                elif self._goal_is_tracked and self.landing_platform is not None:
                    hdist = float(
                        np.linalg.norm(drone_position[0:2] - self.landing_platform[0:2])
                    )
                    if (
                        hdist < VILLAGE_GOAL_TRACK_RELEASE_DISTANCE_M
                        and goal_visibility_prob > VILLAGE_GOAL_TRACK_MIN_PROBABILITY
                    ):
                        is_goal_visible = True
                    else:
                        is_goal_visible = False
                        self._goal_is_tracked = False
                else:
                    is_goal_visible = False
                    self._goal_is_tracked = False
            elif self._goal_is_tracked:
                is_goal_visible = bool(goal_visibility_prob >= self._goal_visible_exit_threshold)
            else:
                is_goal_visible = bool(goal_visibility_prob >= self._goal_visible_enter_threshold)

            if not self._xgb_map_is({"village"}):
                self._goal_is_tracked = is_goal_visible
            if is_goal_visible:
                visible_goal_position = predicted_goal_position.copy()
                visible_goal_position_cov = predicted_goal_position_cov.copy()
                visible_goal_velocity = predicted_platform_velocity.copy()
                visible_goal_velocity[2] = 0.0

        if self._static_map_rules_active():
            self._update_static_landing_lock(lock_goal_probability, lock_goal_position, drone_position)
        else:
            self._clear_static_landing_lock()

        slerp_steps = 0.06 / max(1.0, drone_speed / 2.5) if drone_speed >= 4.5 else 0.06

        if self._mode == "takeoff":
            yaw_to_search_area = np.arctan2(search_area_vector[1], search_area_vector[0])
            yaw_to_search_area_diff = (yaw_to_search_area - drone_rpy[2] + np.pi) % (2.0 * np.pi) - np.pi
            yaw_command = self._get_yaw_command(yaw_to_search_area / np.pi, drone_rpy)
            z_command = 0.0
            speed_command = 0.35
            min_altitude = VILLAGE_SAFE_CRUISE_ALTITUDE_M if self._xgb_map_is({"village"}) else (4.5 if self._high_takeoff_map and self.is_hatt else 1.5)
            if not hasattr(self, "start_state"): self.start_state = None
            if self.start_state is None: self.start_state = state.copy()
            spawn_alt = float(self.start_state[2])
            altitude_gain = float(drone_position[2] - spawn_alt)
            if spawn_alt > 15.0:
                ready_alt = altitude_gain >= 1.5
                if altitude_gain < 1.5:
                    z_command = 1.0
            elif spawn_alt >= 3.0:
                ready_alt = altitude_gain >= 0.5
                if altitude_gain < 0.5:
                    z_command = 1.0
            else:
                ready_alt = drone_altitude >= min_altitude
                if drone_altitude < min_altitude:
                    z_command = 1.0
            if ready_alt and is_goal_visible and abs(yaw_to_search_area_diff) < (np.pi / 6):
                if not (self._xgb_map_is({"village"}) and drone_altitude < VILLAGE_SAFE_CRUISE_ALTITUDE_M):
                    self.first_order_cnt = 20
                    self._mode = "navigation"
            elif ready_alt and abs(yaw_to_search_area_diff) < (np.pi / 24):
                if not (self._xgb_map_is({"village"}) and drone_altitude < VILLAGE_SAFE_CRUISE_ALTITUDE_M):
                    self.first_order_cnt = 30
                    self._mode = "search"

            action = np.array([0.0, 0.0, z_command, speed_command, yaw_command], dtype=np.float32)
        elif self._mode == "search":
            acceleration_rate = 0.07
            brake_rate = 0.05
            drone_speed_normalized = min(drone_speed,6.0) / 3.0

            if self._landing_platform_position is None:
                if search_progress_distance > 1.0:
                    if self._forward:
                        acceleration_rate = 0.313
                    yaw_command = np.arctan2(search_area_vector[1], search_area_vector[0])
                    speed_command = min(drone_speed_normalized + acceleration_rate, 2.0)
                    if abs(yaw_command - drone_rpy[2]) > 0.1 and abs(yaw_command - drone_rpy[2]) < (np.pi * 2 - 0.2):
                        speed_command = self._last_action[3]
                        speed_command = max(speed_command, 0.2)
                    if abs(yaw_command - drone_rpy[2]) > 0.3 and abs(yaw_command - drone_rpy[2]) < (np.pi * 2 - 0.5):
                        speed_command = self._last_action[3] - brake_rate
                        speed_command = max(speed_command, 0.2)
                        yaw_command = self._get_yaw_command(yaw_command / np.pi, drone_rpy) * np.pi
                    yaw_command = yaw_command / np.pi
                    if search_progress_distance < 3.0:
                        max_speed_command = speed_command = search_progress_distance / 3.0
                        if speed_command < self._last_action[3] - brake_rate:
                            speed_command = self._last_action[3] - brake_rate
                else:
                    if search_progress_distance < 1.0 and self.search_stage > 0:
                        self.search_stage += 1
                        yaw_command = self._rot_left(drone_rpy[2])
                        if self.search_stage >= len(self.extra_search_vectors):
                            self.search_stage = 0

                    if self.search_stage == 0:
                        self.search_state_0_rot_count += 1
                        if self.search_state_0_rot_count >= 100:
                            self.search_state_0_rot_count = 0
                            self.search_stage = 1
                        yaw_command = self._rot_left(drone_rpy[2])
                    speed_command = search_progress_distance / 3.0

                    if speed_command < self._last_action[3] - brake_rate:
                        speed_command = self._last_action[3] - brake_rate

                    speed_command = max(speed_command, 0.0)

                direction = search_area_vector

                if float(np.linalg.norm(direction)) > 1e-6:
                    direction_norm = direction / np.linalg.norm(direction)
                else:
                    direction_norm = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                if village_search_lift:
                    direction_norm = self._dampen_close_village_search_direction(
                        direction_norm,
                        search_progress_distance,
                    )
                    if self.search_stage > 0:
                        direction_norm = self._limit_village_spiral_direction(direction_norm)
                        speed_command = min(speed_command, VILLAGE_SPIRAL_MAX_SPEED)

                action = np.concatenate([direction_norm, [speed_command, yaw_command]], dtype=np.float32)
            else:
                direction = self._last_action[0:3]
                speed_command = self._last_action[3] - brake_rate
                speed_command = max(speed_command, 0.0)
                yaw_command = self._rot_left(drone_rpy[2])

                action = np.concatenate([direction, [speed_command, yaw_command]], dtype=np.float32)

            if village_search_lift:
                action = self._apply_village_search_safety(action, drone_position)

            if is_goal_visible:
                self._mode = "navigation"
                self.first_order_cnt = 20
                self._goal_return_mode = False
                self._goal_return_search_steps = 0
            elif (
                self._xgb_map_is({"village"})
                and self.landing_platform is not None
                and float(np.linalg.norm(drone_position[0:2] - self.landing_platform[0:2]))
                < VILLAGE_CLOSE_NAV_HOLD_DISTANCE_M
            ):
                self._mode = "navigation"
                self.first_order_cnt = 20
                self._goal_return_mode = False
                self._goal_return_search_steps = 0
        elif self._mode == "navigation":
            acceleration_rate = 0.08 if self._forward else 0.05
            brake_rate = 0.05
            drone_speed_normalized = min(drone_speed, 6.0) / 3.0
            nav_goal_position = visible_goal_position
            if (
                nav_goal_position is None
                and self._xgb_map_is({"village"})
                and self._goal_is_tracked
                and self.landing_platform is not None
            ):
                hdist = float(np.linalg.norm(drone_position[0:2] - self.landing_platform[0:2]))
                if hdist < VILLAGE_CLOSE_GOAL_DISTANCE_M:
                    nav_goal_position = self.landing_platform.copy()
            if nav_goal_position is not None:
                goal_position = nav_goal_position.copy()
                goal_position[2] += 0.5

                direction = goal_position - drone_position
                distance_to_goal = float(np.linalg.norm(direction))
                yaw_command = np.arctan2(direction[1], direction[0])
                max_speed = min(1.02 + (float(np.linalg.norm(direction[:2]))-4.0)/10.0, 2.0)
                if self._xgb_map_is({"village"}):
                    max_speed = min(max_speed, VILLAGE_NAV_MAX_LOW_SPEED)
                if drone_speed_normalized > max_speed:
                    speed_command = min(drone_speed_normalized - brake_rate, 2.0)
                else:
                    speed_command = min(drone_speed_normalized + acceleration_rate, max_speed)
                if abs(yaw_command - drone_rpy[2]) > 0.3 and abs(yaw_command - drone_rpy[2]) < (np.pi * 2 - 0.2):
                    speed_command = self._last_action[3]
                    speed_command = max(speed_command, 0.2)
                if abs(yaw_command - drone_rpy[2]) > 0.5 and abs(yaw_command - drone_rpy[2]) < (np.pi * 2 - 0.5):
                    speed_command = self._last_action[3] - brake_rate/4
                    speed_command = max(speed_command, 0.2)
                yaw_command /= np.pi
                if distance_to_goal > 1e-6:
                    village_cruise = self._xgb_map_is({"village"}) and not self._village_in_final_approach(
                        drone_position, visible_goal_position
                    )
                    if village_cruise:
                        pad_xy = self._village_pad_xy(goal_position)
                        hover_z = self._village_hover_altitude(goal_position)
                        goal_position[2] = hover_z
                        direction = goal_position - drone_position
                        direction[2] = 0.0
                        direction_norm = direction / np.linalg.norm(direction) if float(np.linalg.norm(direction)) > 1e-6 else np.array([0.0, 0.0, 0.0], dtype=np.float32)
                        distance_to_goal = float(np.linalg.norm(direction))
                    elif self._xgb_map_is({"village"}):
                        direction[2] *= 5.0
                        direction_norm = direction / np.linalg.norm(direction)
                    else:
                        direction[2] *= 5.0
                        direction_norm = direction / np.linalg.norm(direction)
                else:
                    direction_norm = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                if drone_altitude < 0.5:
                    direction_norm = np.array([direction_norm[0] * 0.1, direction_norm[1] * 0.1, 1.0], dtype=np.float32)
                if self._xgb_map_is({"village"}) and not self._village_in_final_approach(
                    drone_position, visible_goal_position
                ):
                    direction_norm, speed_command = self._apply_village_navigation_safety(
                        direction_norm,
                        speed_command,
                        drone_position,
                        distance_to_goal=distance_to_goal,
                        goal_position=goal_position,
                    )
                if distance_to_goal < 10.0 and self.is_find_P and not self._force_static_platform:
                    if self._first_plat_pos is None:
                        self._first_plat_pos = self.landing_platform.copy()
                    elif np.linalg.norm(self._first_plat_pos[0:2] - self.landing_platform[0:2]) > 0.7:
                        if not self._xgb_map_is({"village"}):
                            self.tracking = True

                if self._can_enter_landing_mode(distance_to_goal, drone_position, goal_position):
                    self._mode = "landing"
                    if self._xgb_map_is({"village"}):
                        self.tracking = False
                    self.controller.reset()
                    if self._static_landing_ready and self._static_lock_anchor is not None and self._is_static_landing_candidate():
                        self._landing_committed = True
                        self._landing_commit_position = self._static_lock_anchor.copy()
                        self._landing_platform_position = self._landing_commit_position.copy() + np.array([0.0, 0.0, 0.29])
                    else:
                        self._landing_committed = False
                        self._landing_commit_position = None
                        self._landing_platform_position = None

                action = np.concatenate([direction_norm, [speed_command, self._get_yaw_command(yaw_command, drone_rpy)]], dtype=np.float32)
            else:
                action = self._last_action.copy()

            if not is_goal_visible or visible_goal_position is None:
                village_hold_nav = (
                    self._xgb_map_is({"village"})
                    and self.landing_platform is not None
                    and float(np.linalg.norm(drone_position[0:2] - self.landing_platform[0:2]))
                    < VILLAGE_CLOSE_NAV_HOLD_DISTANCE_M
                )
                if village_hold_nav:
                    self._goal_lost_steps = 0
                else:
                    _spawn_alt = float(self.start_state[2]) if self.start_state is not None else 0.0
                    _patience = 15 if _spawn_alt > 15.0 else 10
                    self._goal_lost_steps = getattr(self, "_goal_lost_steps", 0) + 1
                    if self._goal_lost_steps > _patience:
                        if self._can_enter_goal_return_mode():
                            self._last_known_goal_pos = self.platform_position.copy()
                            self._goal_return_mode = True
                            self._goal_return_search_steps = 0
                            self._goal_return_attempts += 1
                            self.extra_search_vectors = None
                            self.search_stage = 0
                            self.search_state_0_rot_count = 0
                        self._mode = "search"
                        self._goal_lost_steps = 0
            else:
                self._goal_lost_steps = 0
        elif self._mode == "landing" and not self.tracking:
            committed_landing = bool(self._landing_committed and self._landing_commit_position is not None)
            if visible_goal_position is not None or committed_landing:
                if committed_landing:
                    if self._landing_platform_position is None:
                        self._landing_platform_position = self._landing_commit_position.copy() + np.array([0.0, 0.0, 0.29])
                    goal_position = self._landing_platform_position.copy()
                else:
                    goal_position = visible_goal_position.copy()
                    goal_position[2] += 0.3

                if self._landing_platform_position is None:
                    if self.move_in_auto_mode:
                        self._landing_platform_position = self.reverse_landing_platform.copy() + np.array([0.0, 0.0, 0.29])
                    else:
                        self._landing_platform_position = self.landing_platform.copy() + np.array([0.0, 0.0, 0.29])
                elif visible_goal_position is not None and not committed_landing:
                    dist_drone_to_anchor = float(np.linalg.norm(drone_position - self._landing_platform_position))
                    dxy = float(np.linalg.norm(goal_position[0:2] - self._landing_platform_position[0:2]))
                    if dist_drone_to_anchor < 1.5 and dxy < 0.5:
                        self._landing_platform_position[0:2] = (
                            0.8 * self._landing_platform_position[0:2] + 0.2 * goal_position[0:2]
                        )

                direction_to_landing_point = self._landing_platform_position - drone_position
                direction_to_current_goal_position = goal_position - drone_position
                distance_to_landing_point = float(np.linalg.norm(direction_to_landing_point))
                if committed_landing:
                    horizontal_distance_to_current_goal_position = float(
                        np.linalg.norm(self._landing_platform_position[0:2] - drone_position[0:2]))
                else:
                    horizontal_distance_to_current_goal_position = float(
                        np.linalg.norm(goal_position[0:2] - drone_position[0:2]))

                brake_rate = 0.01
                speed_command = self._last_action[3] - brake_rate
                speed_bound = min(distance_to_landing_point/6.0, 0.5)
                speed_command = max(speed_command, max(0.25, speed_bound))
                speed_command = min(speed_command, self._last_action[3])

                yaw_command = np.arctan2(direction_to_current_goal_position[1],
                                         direction_to_current_goal_position[0]) / np.pi

                landing_z_scale = 1.2 if self._xgb_map_is({"village"}) else 2.0
                if distance_to_landing_point > 0.2:
                    direction_to_landing_point[2] *= landing_z_scale
                    direction_norm = direction_to_landing_point / np.linalg.norm(direction_to_landing_point)
                    if distance_to_landing_point > 0.5 and self.move_in_auto_mode:
                        direction_norm[0:2] += 20.0*self.reverse_d[0:2]
                else:
                    direction_norm = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    speed_command = 0.0
                if (not committed_landing
                        and np.linalg.norm(self.reverse_landing_platform[0:2] - self._landing_platform_position[0:2]) > 0.7):
                    if not self._xgb_map_is({"village"}):
                        self.tracking = True

                if horizontal_distance_to_current_goal_position < 0.6:
                    direction_to_landing_point[2] *= landing_z_scale
                    direction_norm = direction_to_landing_point / np.linalg.norm(direction_to_landing_point)/3.0
                    direction_norm[2] = -1.0
                    speed_command = max(0.3, self._last_action[3] - brake_rate)

                if horizontal_distance_to_current_goal_position < 0.2:
                    direction_to_landing_point[2] *= landing_z_scale
                    direction_norm = np.array([0.0, 0.0, 0.0])
                    direction_norm[2] = -1.0
                    speed_command = max(0.2, self._last_action[3] - brake_rate)

                self._last_landing_hdist = horizontal_distance_to_current_goal_position
                action = np.concatenate([direction_norm, [speed_command, self._get_yaw_command(yaw_command, drone_rpy)]], dtype=np.float32)
            else:
                action = self._last_action.copy()

            if committed_landing:
                self._landing_patience = 0
            elif not is_goal_visible or visible_goal_position is None:
                self._landing_patience += 1
                _close_range = (self._last_landing_hdist is not None and self._last_landing_hdist < 0.5)
                _land_patience = 30 if _close_range else 10
                if self._landing_patience > _land_patience:
                    if self._can_enter_goal_return_mode():
                        self._last_known_goal_pos = self.platform_position.copy()
                        self._goal_return_mode = True
                        self._goal_return_search_steps = 0
                        self._goal_return_attempts += 1
                        self.extra_search_vectors = None
                        self.search_stage = 0
                        self.search_state_0_rot_count = 0
                    self._landing_platform_position = None
                    self._landing_patience = 0
                    self._mode = "search"
                elif _close_range:
                    action = np.array([0.0, 0.0, -1.0, 0.3, self._last_action[4]], dtype=np.float32)
            else:
                self._landing_patience = 0
        elif self._mode == "landing" and self.tracking:
            if visible_goal_position_cov is not None:
                goal_position = self.reverse_landing_platform.copy() + self.reverse_d*13
                horizontal_distance_to_current_goal_position = float(np.linalg.norm(goal_position[0:2] - drone_position[0:2]))
                if self._landing_platform_position is None:
                    self._landing_platform_position = visible_goal_position_cov.copy() + np.array([0.0, 0.0, 0.29])
                goal_position[2] = self._landing_platform_position[2]
                if horizontal_distance_to_current_goal_position < 1.0:
                    goal_position[2] -= 0.5

                self.controller.set_setpoint(goal_position)
                action = self.controller.update(drone_position, setpoint_v=self.reverse_d*13)

                direction_to_landing_point = goal_position - drone_position
                distance_to_landing_point = float(np.linalg.norm(direction_to_landing_point))
                unit_vect_to_landing_point = direction_to_landing_point / np.linalg.norm(direction_to_landing_point)

                brake_rate = 0.01
                speed_command = 0.3
                yaw_command = np.arctan2(direction_to_landing_point[1],direction_to_landing_point[0]) / np.pi

                action[2] = action[2] * 2.0
                direction_norm = action / np.linalg.norm(action)
                speed_command = min(np.linalg.norm(action), 1.0)
                speed_command = self._last_action[3] + np.sign(speed_command - self._last_action[3])*brake_rate
                if horizontal_distance_to_current_goal_position < 0.6:
                    direction_norm = action / np.linalg.norm(action)/2.0
                    direction_norm[2] = -1.0
                    speed_command = max(0.3, self._last_action[3] - brake_rate)

                self._last_landing_hdist = horizontal_distance_to_current_goal_position
                action = np.concatenate([direction_norm, [speed_command, self._get_yaw_command(yaw_command, drone_rpy)]], dtype=np.float32)
            else:
                action = self._last_action.copy()

            if not is_goal_visible or visible_goal_position is None:
                self._landing_patience += 1
                _close_range = (self._last_landing_hdist is not None and self._last_landing_hdist < 0.5)
                _land_patience = 30 if _close_range else 10
                if self._landing_patience > _land_patience:
                    if self._can_enter_goal_return_mode():
                        self._last_known_goal_pos = self.platform_position.copy()
                        self._goal_return_mode = True
                        self._goal_return_search_steps = 0
                        self._goal_return_attempts += 1
                        self.extra_search_vectors = None
                        self.search_stage = 0
                        self.search_state_0_rot_count = 0
                    self._landing_platform_position = None
                    self._landing_patience = 0
                    self._mode = "search"
                elif _close_range:
                    action = np.array([0.0, 0.0, -1.0, 0.3, self._last_action[4]], dtype=np.float32)
            else:
                self._landing_patience = 0
        else:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        direct_goal_direction = action[0:3]
        village_safety = self._xgb_map_is({"village"})
        use_ctx_avoidance = (
            (self._mode == "search" and distance_to_search_area > 3.0)
            or (self._mode == "navigation" and not village_safety)
        )
        if use_ctx_avoidance:
            goal_distance = None
            if visible_goal_position is not None:
                goal_distance = float(
                    np.linalg.norm(np.asarray(visible_goal_position, dtype=np.float32).reshape(3) - drone_position)
                )
                camera_position, camera_target, _, up_direction, _ = _cam_geom(
                    drone_position,
                    drone_rpy,
                )

                ctx = _safe_ctx(
                    depth_map=depth,
                    camera_position=camera_position,
                    camera_target=camera_target,
                    fov_deg=CAMERA_FOV_DEG,
                    current_direction=action[0:3],
                    camera_target_is_point=True,
                    camera_up=up_direction,
                    working_resolution=32,
                )

                waypoint_direction = _pick_ctx(
                    ctx,
                    safety_margin_m=VILLAGE_AVOID_SAFETY_MARGIN_M if village_safety else 0.35,
                    preferred_clearance_m=VILLAGE_AVOID_PREFERRED_CLEARANCE_M if village_safety else 10.0,
                    max_lookahead_m=15.0,
                )

                waypoint_direction_close = _pick_ctx(
                    ctx,
                    safety_margin_m=VILLAGE_AVOID_CLOSE_SAFETY_MARGIN_M if village_safety else 1.1,
                    preferred_clearance_m=VILLAGE_AVOID_CLOSE_PREFERRED_CLEARANCE_M if village_safety else 2.0,
                    max_lookahead_m=3.3,
                )
                dir_similarity = 1.0
                forward_threshold = 0.995 if village_safety else 0.99
                conflict_threshold = forward_threshold
                if np.linalg.norm(action[0:3]) > 0 and np.linalg.norm(waypoint_direction_close) > 0:
                    dir_1 = action[0:3] / np.linalg.norm(action[0:3])
                    dir_2 = waypoint_direction_close / np.linalg.norm(waypoint_direction_close)
                    dir_similarity = np.dot(dir_1, dir_2)

                    if dir_similarity < conflict_threshold:
                        waypoint_direction = waypoint_direction_close

                self._forward = dir_similarity >= forward_threshold
                action = np.concatenate([waypoint_direction, [action[3], action[4]]]).astype(np.float32)
                action = self.collipe_avoidance(depth, state, action, direct_goal_direction)

        elif village_safety and self._mode == "navigation":
            if not self._village_in_final_approach(drone_position, visible_goal_position):
                action = self.collipe_avoidance(depth, state, action, direct_goal_direction)

        if not self._inside_goal_zone and dist_raw_3d < 10.0:
            self._inside_goal_zone = True
        if self._inside_goal_zone and dist_raw_3d > 12.0 and self._mode not in ("search", "navigation", "landing"):
            _back_vec = np.asarray([state[-3], state[-2], state[-1]], dtype=np.float32)
            _bn = float(np.linalg.norm(_back_vec))
            if _bn > 1e-6:
                _back_dir = _back_vec / _bn
                action = np.concatenate([
                    _back_dir,
                    [min(float(action[3]), 0.5), action[4]],
                ]).astype(np.float32)

        if self._last_action is not None:
            slerp_dir_vector = slerp_dir(
                a=self._last_action[0:3],
                b=action[0:3],
                t=slerp_steps
            )

            action = np.concatenate([slerp_dir_vector, [action[3], action[4]]]).astype(np.float32)

        action = np.clip(action, -3.0, 3.0)

        self._last_action = action
        is_first_order = False
        if (self._mode == "navigation" or self._mode == "search") and not self._forward:
            is_first_order = True
            self.first_order_cnt = 35
        elif self.first_order_cnt > 0:
            self.first_order_cnt -= 1
            is_first_order = True
        else:
            is_first_order = False
        
        if self._xgb_map_is({"forest"}) and self._mode in ("search", "navigation"):
            action = np.asarray(action, dtype=np.float32).copy()
            action[3] = np.float32(min(float(action[3]), 0.8))
        if (
            self._xgb_map_is({"village"})
            and self._mode == "navigation"
            and float(drone_position[2]) < VILLAGE_NAV_SLOW_ALTITUDE_M
        ):
            action = np.asarray(action, dtype=np.float32).copy()
            action[3] = np.float32(min(float(action[3]), VILLAGE_NAV_MAX_LOW_SPEED))
        final_action = self._getSecondOrderAction(action, state, is_smooth=is_first_order)
        _tilt = max(abs(float(drone_rpy[0])), abs(float(drone_rpy[1])))
        _max_v_err = max(0.2, 2.2 - (2.0 / 0.7) * max(0.0, _tilt - 0.3))
        final_action = self._reference_governor(final_action, drone_velocity, max_v_err=_max_v_err, speed_limit=3.0)
        self._map_tick += 1
        return final_action

    def _reference_governor(self, action_1x5, drone_velocity_world, max_v_err=2.2, speed_limit=3.0):
        a = np.asarray(action_1x5, dtype=np.float32).reshape(-1).copy()
        new_v = a[0:3] * a[3] * speed_limit
        v_err = new_v - np.asarray(drone_velocity_world, dtype=np.float32)
        v_err_mag = float(np.linalg.norm(v_err))
        if v_err_mag > max_v_err:
            scale = max_v_err / v_err_mag
            new_v_safe = drone_velocity_world + v_err * scale
            new_speed_mag = float(np.linalg.norm(new_v_safe))
            if new_speed_mag > 1e-6:
                a[0:3] = (new_v_safe / new_speed_mag).astype(np.float32)
                a[3] = float(new_speed_mag / speed_limit)
            else:
                a[3] = 0.0
        if np.asarray(action_1x5).ndim == 2:
            return a.reshape(1, 5).astype(np.float32)
        return a.astype(np.float32)


    def collipe_avoidance(self, depth, state, action, gaol_direction):
        depth_input = np.asarray(depth, dtype=np.float32)
        if depth_input.ndim == 2:
            depth_input = depth_input[None, None, :, :]
        elif depth_input.ndim == 3 and depth_input.shape[-1] == 1:
            depth_input = np.transpose(depth_input, (2, 0, 1))[None, ...]
        elif depth_input.ndim != 4:
            raise ValueError(f"expected depth shape (H,W), (H,W,1), or (N,C,H,W), got {depth_input.shape}")

        state_input = np.asarray(state, dtype=np.float32).reshape(-1)
        action_input = np.asarray(action, dtype=np.float32).reshape(-1)
        gaol_direction = np.asarray(gaol_direction, dtype=np.float32).reshape(-1)
        if action_input.shape != (5,):
            raise ValueError(f"expected navigation action shape (5,), got {action_input.shape}")

        model_inputs = {}
        if "depth" in self._navigation_model_inputs:
            model_inputs["depth"] = depth_input
        if "state" in self._navigation_model_inputs:
            model_inputs["state"] = state_input
        if "action" in self._navigation_model_inputs:
            model_inputs["action"] = action_input
        if "goal_direction" in self._navigation_model_inputs:
            model_inputs["goal_direction"] = gaol_direction
        navigation_outputs = self._navigation_model_session.run(None, model_inputs)
        avoid_direction = navigation_outputs[0]
        if len(navigation_outputs) > 1:
            self.is_hatt = bool(np.asarray(navigation_outputs[1]).reshape(-1)[0])
        
        return np.asarray(avoid_direction, dtype=np.float32).reshape(-1)

    def reset(self):
        self._mode = "takeoff"
        self._goal_visibility_threshold = 0.5
        self._goal_visible_enter_threshold = 0.55
        self._goal_visible_exit_threshold = 0.35
        self._last_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.M = None
        self._pt_M = None
        if not hasattr(self, "_m_model_session"):
            self._m_model_session = None
        if not hasattr(self, "_m_state_dim"):
            self._m_state_dim = None
        self._landing_platform_position = None
        self._mountain_flight = False
        self._high_takeoff_map = False
        self._landing_patience = 0
        self._goal_is_tracked = False
        self.prev_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.start_state = None
        self.see_P = True
        self.platform_position = None
        self.reverse_landing_platform = None
        self.landing_platform = None
        self.platform_lost_step = 0
        self.p_buffer = 10.0
        self.reverse_d = np.array([0.0, 0.0, 0.0])
        self.move_in_auto_mode = False
        self.reverse_buffer = np.array([[0.0, 0.0, 0.0]])
        self.last_reverse_buffer = np.array([[0.0, 0.0, 0.0]])
        self.is_find_P = False
        self.tracking = False
        self._village_locked_goal_xy = None
        self._village_locked_goal_z = None
        self._forward = True
        self._first_plat_pos = None
        if hasattr(self, "_ab_tracker"):
            self._ab_tracker.reset()
        self.extra_search_vectors = None
        self.search_stage = 0
        self.search_state_0_rot_count = 0
        self.d_t_speed = np.array([0.0, 0.0, 0.0])
        self.first_order_cnt = 0
        self._last_known_goal_pos = None
        self._goal_return_mode = False
        self._goal_return_attempts = 0
        self._goal_return_search_steps = 0
        self._last_landing_hdist = None
        self._track_score = 0
        self._track_reject_streak = 0
        self._inside_goal_zone = False
        self.vk = np.array([0.0, 0.0, 0.0])
        self.vkm1 = np.array([0.0, 0.0, 0.0])
        self.is_hatt = True
        self._static_lock_anchor = None
        self._static_lock_window = []
        self._static_lock_positions = []
        self._static_lock_distances = []
        self._static_lock_avg_distance = None
        self._static_landing_ready = False
        self._landing_committed = False
        self._landing_commit_position = None
        self._map_tick = 0
        self._map_prob_sum = np.zeros(len(MAP_LABELS), dtype=np.float32)
        self._map_prob_count = 0
        self._map_prediction_label = None
        self._map_prediction_probability = 0.0
        self._map_prediction_probs = np.zeros(len(MAP_LABELS), dtype=np.float32)
        self._force_static_platform = False
        self._pred_buf = []
        self._last_detector_mode = "onnx"

    def _confirmed_map_label(
        self,
        *,
        min_count=MAP_RULE_MIN_PREDICTIONS,
        min_probability=MAP_RULE_MIN_PROBABILITY,
    ):
        count = int(getattr(self, "_map_prob_count", 0) or 0)
        label = getattr(self, "_map_prediction_label", None)
        probability = float(getattr(self, "_map_prediction_probability", 0.0) or 0.0)
        if count < int(min_count) or probability < float(min_probability):
            return None, probability, count
        return label, probability, count

    def _xgb_map_is(
        self,
        labels,
        *,
        min_count=MAP_RULE_MIN_PREDICTIONS,
        min_probability=MAP_RULE_MIN_PROBABILITY,
    ):
        label, _, _ = self._confirmed_map_label(
            min_count=min_count,
            min_probability=min_probability,
        )
        return label in labels

    def _uid53_rules_active(self):
        return self._xgb_map_is(MAP_UID53_RULE_LABELS)

    def _pt_detector_active(self):
        return self._xgb_map_is(MAP_PT_DETECTOR_LABELS)

    def _static_map_rules_active(self):
        return (
            bool(getattr(self, "_force_static_platform", False))
            and self._xgb_map_is(
                MAP_STATIC_LABELS,
                min_count=MAP_STATIC_MIN_PREDICTIONS,
                min_probability=MAP_STATIC_CONFIDENCE_THRESHOLD,
            )
        )

    def _clear_static_landing_lock(self):
        self._static_lock_anchor = None
        self._static_lock_window = []
        self._static_lock_positions = []
        self._static_lock_distances = []
        self._static_lock_avg_distance = None
        self._static_landing_ready = False
        self._landing_committed = False
        self._landing_commit_position = None

    def _map_features(self, state, depth):
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_arr.size < m_STATE_DIM:
            state_fixed = np.zeros(m_STATE_DIM, dtype=np.float32)
            state_fixed[:state_arr.size] = state_arr
        else:
            state_fixed = state_arr[:m_STATE_DIM].astype(np.float32)
        tick = float(self._map_tick)
        values = [tick, tick * SIM_DT]
        values.extend(float(v) for v in state_fixed.tolist())
        values.extend(_map_depth_feature_values(depth))
        return np.asarray(values, dtype=np.float32)

    def _update_map_prediction(self, state, depth):
        predictor = getattr(self, "_map_predictor", None)
        if predictor is None or not predictor.enabled:
            return
        tick = int(getattr(self, "_map_tick", 0))
        if tick > MAP_PREDICT_MAX_TICK and not self._force_static_platform:
            return
        if tick % MAP_PREDICT_EVERY_TICKS != 0:
            return
        probs = predictor.predict_proba(self._map_features(state, depth))
        if probs is None:
            return
        self._map_prob_sum += probs
        self._map_prob_count += 1
        avg_probs = self._map_prob_sum / max(1, self._map_prob_count)
        class_idx = int(np.argmax(avg_probs))
        class_prob = float(avg_probs[class_idx])
        class_label = MAP_LABELS[class_idx]
        self._map_prediction_label = class_label
        self._map_prediction_probability = class_prob
        self._map_prediction_probs = avg_probs.astype(np.float32)
        if (
            self._map_prob_count >= MAP_STATIC_MIN_PREDICTIONS
            and class_label in MAP_STATIC_LABELS
            and class_prob >= MAP_STATIC_CONFIDENCE_THRESHOLD
        ):
            self._force_static_platform = True
        elif (
            self._map_prob_count >= MAP_RULE_MIN_PREDICTIONS
            and class_label in MAP_UID53_RULE_LABELS
            and class_prob >= MAP_RULE_MIN_PROBABILITY
        ):
            self._force_static_platform = False
            self._clear_static_landing_lock()
        self._apply_static_map_override()

    def _apply_static_map_override(self):
        if not bool(getattr(self, "_force_static_platform", False)):
            return
        self.move_in_auto_mode = False
        self.tracking = False
        self.reverse_d = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if self.landing_platform is not None:
            self.reverse_landing_platform = self.landing_platform.copy()

    def _get_search_area_position(self, search_area_vector, drone_position):
        if self.extra_search_vectors is None:
            direction_xy = np.asarray(search_area_vector[:2], dtype=np.float32)
            direction_xy_norm = float(np.linalg.norm(direction_xy))
            if direction_xy_norm <= 1e-6:
                forward_xy = np.array([1.0, 0.0], dtype=np.float32)
            else:
                forward_xy = direction_xy / direction_xy_norm
            right_xy = np.array([-forward_xy[1], forward_xy[0]], dtype=np.float32)

            self.extra_search_vectors = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
            radius = SPIRAL_SEARCH_START_RADIUS_M
            angle = 0.0
            while radius <= SPIRAL_SEARCH_MAX_RADIUS_M:
                offset_xy = radius * (
                    np.cos(angle) * forward_xy
                    + np.sin(angle) * right_xy
                )
                self.extra_search_vectors.append(
                    np.array([offset_xy[0], offset_xy[1], 0.0], dtype=np.float32)
                )
                radius += SPIRAL_SEARCH_RADIUS_STEP_M
                angle += SPIRAL_SEARCH_ANGLE_STEP_RAD
        search_area_position = search_area_vector + drone_position
        search_area_position[:2] += self.extra_search_vectors[self.search_stage][:2]
        return search_area_position

    def _is_static_landing_candidate(self):
        reverse_speed = float(np.linalg.norm(np.asarray(self.reverse_d, dtype=np.float32).reshape(-1)[:2]))
        return (
            not bool(self.tracking)
            and not bool(self.move_in_auto_mode)
            and reverse_speed <= STATIC_LOCK_MAX_REVERSE_SPEED
        )

    def _village_in_final_approach(
        self,
        drone_position: np.ndarray,
        visible_goal_position: np.ndarray | None = None,
    ) -> bool:
        if not self._xgb_map_is({"village"}):
            return False
        refs: list[np.ndarray] = []
        locked = getattr(self, "_village_locked_goal_xy", None)
        if locked is not None:
            refs.append(np.asarray(locked, dtype=np.float32))
        if self.landing_platform is not None:
            refs.append(np.asarray(self.landing_platform[0:2], dtype=np.float32))
        if visible_goal_position is not None:
            refs.append(np.asarray(visible_goal_position[0:2], dtype=np.float32))
        if not refs:
            return False
        min_hdist = min(
            self._village_horizontal_distance(drone_position, ref) for ref in refs
        )
        return min_hdist < VILLAGE_FINAL_APPROACH_DISTANCE_M

    def _village_pad_xy(self, goal_position):
        locked = getattr(self, "_village_locked_goal_xy", None)
        if locked is not None:
            return np.asarray(locked, dtype=np.float32)
        if self.landing_platform is not None:
            return np.asarray(self.landing_platform[0:2], dtype=np.float32)
        return np.asarray(goal_position[0:2], dtype=np.float32)

    def _village_pad_position(self):
        locked = getattr(self, "_village_locked_goal_xy", None)
        if locked is not None:
            pad_z = float(getattr(self, "_village_locked_goal_z", 0.0) or 0.0)
            return np.array([float(locked[0]), float(locked[1]), pad_z], dtype=np.float32)
        if self.landing_platform is not None:
            return np.asarray(self.landing_platform, dtype=np.float32).reshape(3)
        if self.platform_position is not None:
            return np.asarray(self.platform_position, dtype=np.float32).reshape(3)
        return None

    def _village_horizontal_distance(self, drone_position, pad_xy):
        return float(np.linalg.norm(np.asarray(drone_position[0:2], dtype=np.float32) - pad_xy))

    def _village_hover_altitude(self, goal_position):
        pad_z = float(getattr(self, "_village_locked_goal_z", 0.0) or 0.0)
        if pad_z <= 0.0:
            if self.landing_platform is not None:
                pad_z = float(self.landing_platform[2])
            elif goal_position is not None:
                pad_z = float(goal_position[2])
        return max(VILLAGE_SAFE_CRUISE_ALTITUDE_M, pad_z + VILLAGE_SAFE_HOVER_ABOVE_PAD_M)

    def _can_enter_landing_mode(
        self,
        distance_to_goal: float,
        drone_position: np.ndarray | None = None,
        goal_position: np.ndarray | None = None,
    ) -> bool:
        if float(distance_to_goal) >= 4.0:
            return False
        if self._xgb_map_is({"village"}):
            prob = float(getattr(self, "_last_goal_visibility_prob", 0.0) or 0.0)
            return bool(
                self._goal_is_tracked
                and prob > VILLAGE_LANDING_MIN_PROBABILITY
            )
        return bool(self.is_find_P)

    def _apply_village_safe_landing(
        self,
        drone_position: np.ndarray,
        direction_to_landing_point: np.ndarray,
        horizontal_distance: float,
        speed_command: float,
        brake_rate: float,
    ) -> tuple[np.ndarray, float]:
        altitude = float(drone_position[2])
        pad_z = float(self._landing_platform_position[2]) if self._landing_platform_position is not None else altitude
        hover_z = max(VILLAGE_SAFE_CRUISE_ALTITUDE_M, pad_z + VILLAGE_SAFE_HOVER_ABOVE_PAD_M)
        hdist = float(horizontal_distance)

        if hdist > VILLAGE_SAFE_CENTER_DISTANCE_M:
            to_pad = np.asarray(direction_to_landing_point, dtype=np.float32).reshape(3).copy()
            to_pad[2] = 0.0
            xy_norm = float(np.linalg.norm(to_pad[0:2]))
            if xy_norm > 1e-6:
                direction = (to_pad / xy_norm).astype(np.float32)
            else:
                direction = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            if altitude < hover_z - 0.25:
                climb = np.array([direction[0] * 0.35, direction[1] * 0.35, 0.85], dtype=np.float32)
                climb_norm = float(np.linalg.norm(climb))
                direction = (climb / climb_norm).astype(np.float32) if climb_norm > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
            speed = min(float(speed_command), VILLAGE_SAFE_HORIZONTAL_MAX_SPEED)
            return direction, max(speed, 0.20)

        direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        speed = min(max(float(speed_command), 0.15), VILLAGE_SAFE_DESCENT_MAX_SPEED)
        return direction, speed

    def _can_enter_goal_return_mode(self):
        if self._xgb_map_is({"village"}):
            return False
        return (
            self.platform_position is not None
            and self._goal_return_attempts < 3
            and self._track_score >= 3
        )

    def _apply_village_navigation_safety(
        self,
        direction_norm: np.ndarray,
        speed_command: float,
        drone_position: np.ndarray,
        *,
        distance_to_goal: float | None = None,
        goal_position: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        if not self._xgb_map_is({"village"}) or self._mode != "navigation":
            return direction_norm, float(speed_command)

        direction = np.asarray(direction_norm, dtype=np.float32).reshape(3).copy()
        altitude = float(drone_position[2])
        hover_z = self._village_hover_altitude(goal_position) if goal_position is not None else VILLAGE_SAFE_CRUISE_ALTITUDE_M
        pad_xy = self._village_pad_xy(goal_position) if goal_position is not None else None
        hdist = self._village_horizontal_distance(drone_position, pad_xy) if pad_xy is not None else float("inf")

        if hdist > VILLAGE_SAFE_CENTER_DISTANCE_M:
            min_altitude = hover_z
            if altitude < min_altitude:
                if direction[2] < 0.0:
                    direction[2] = 0.0
                deficit = min_altitude - altitude
                direction[2] = max(float(direction[2]), min(0.85, deficit * 0.35))
            direction[0:2] = direction[0:2]
            direction[2] = min(float(direction[2]), 0.35)
            xy_norm = float(np.linalg.norm(direction[0:2]))
            if xy_norm > 1e-6:
                direction[0:2] = direction[0:2] / xy_norm * max(xy_norm, 0.65)
            speed_command = min(float(speed_command), VILLAGE_SAFE_HORIZONTAL_MAX_SPEED)
        else:
            min_altitude = max(VILLAGE_NAV_APPROACH_MIN_ALTITUDE_M, hover_z - 1.0)
            if altitude < min_altitude and direction[2] < 0.0:
                direction[2] = 0.0

        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction = (direction / norm).astype(np.float32)
        else:
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if altitude < VILLAGE_NAV_SLOW_ALTITUDE_M:
            speed_command = min(float(speed_command), VILLAGE_NAV_MAX_LOW_SPEED)
        return direction, float(speed_command)

    def _apply_village_search_safety(
        self,
        action: np.ndarray,
        drone_position: np.ndarray,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        altitude = float(drone_position[2])
        if altitude >= VILLAGE_SEARCH_MIN_ALTITUDE_M:
            return action
        action[2] = max(float(action[2]), 0.40)
        action[3] = min(float(action[3]), 0.55)
        return action

    def _dampen_close_village_search_direction(self, direction, distance_to_search_area):
        if distance_to_search_area >= VILLAGE_SEARCH_VECTOR_DAMP_RADIUS_M:
            return np.asarray(direction, dtype=np.float32)

        direction = np.asarray(direction, dtype=np.float32).reshape(3).copy()
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            return direction
        direction /= direction_norm

        trust = float(np.clip(
            distance_to_search_area / VILLAGE_SEARCH_VECTOR_DAMP_RADIUS_M,
            VILLAGE_SEARCH_MIN_DIRECTION_TRUST,
            1.0,
        ))

        prev_direction = np.asarray(self._last_action[:3], dtype=np.float32).reshape(3)
        prev_norm = float(np.linalg.norm(prev_direction))
        if prev_norm > 1e-6:
            prev_direction = prev_direction / prev_norm
            blended = trust * direction + (1.0 - trust) * prev_direction
            blended_norm = float(np.linalg.norm(blended))
            if blended_norm > 1e-6:
                direction = (blended / blended_norm).astype(np.float32)

        max_abs_z = float(
            VILLAGE_SEARCH_MIN_VERTICAL_LIMIT
            + trust * (VILLAGE_SEARCH_MAX_VERTICAL_LIMIT - VILLAGE_SEARCH_MIN_VERTICAL_LIMIT)
        )
        if abs(float(direction[2])) <= max_abs_z:
            return direction.astype(np.float32)

        xy = direction[:2].copy()
        xy_norm = float(np.linalg.norm(xy))
        if xy_norm <= 1e-6 and prev_norm > 1e-6:
            xy = prev_direction[:2].copy()
            xy_norm = float(np.linalg.norm(xy))
        if xy_norm <= 1e-6:
            return np.array([0.0, 0.0, np.sign(direction[2]) * max_abs_z], dtype=np.float32)

        xy_unit = xy / xy_norm
        xy_target_norm = float(np.sqrt(max(0.0, 1.0 - max_abs_z * max_abs_z)))
        limited = np.array(
            [
                xy_unit[0] * xy_target_norm,
                xy_unit[1] * xy_target_norm,
                np.sign(direction[2]) * max_abs_z,
            ],
            dtype=np.float32,
        )
        return limited

    def _limit_village_spiral_direction(self, direction):
        direction = np.asarray(direction, dtype=np.float32).reshape(3).copy()
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            return direction
        direction /= direction_norm

        if abs(float(direction[2])) <= VILLAGE_SPIRAL_MAX_VERTICAL:
            return direction.astype(np.float32)

        xy = direction[:2].copy()
        xy_norm = float(np.linalg.norm(xy))
        if xy_norm <= 1e-6:
            return np.array(
                [0.0, 0.0, np.sign(direction[2]) * VILLAGE_SPIRAL_MAX_VERTICAL],
                dtype=np.float32,
            )

        xy_unit = xy / xy_norm
        xy_target_norm = float(np.sqrt(max(0.0, 1.0 - VILLAGE_SPIRAL_MAX_VERTICAL ** 2)))
        return np.array(
            [
                xy_unit[0] * xy_target_norm,
                xy_unit[1] * xy_target_norm,
                np.sign(direction[2]) * VILLAGE_SPIRAL_MAX_VERTICAL,
            ],
            dtype=np.float32,
        )

    def _append_static_lock_hit(self, is_detection, estimate=None, distance=None):
        was_ready = bool(self._static_landing_ready)
        is_detection = bool(is_detection)
        self._static_lock_positions.append(
            np.asarray(estimate, dtype=np.float32).reshape(3).copy()
            if is_detection and estimate is not None
            else None
        )
        self._static_lock_distances.append(
            float(distance)
            if is_detection and distance is not None
            else None
        )
        if len(self._static_lock_positions) > STATIC_LOCK_WINDOW_TICKS:
            self._static_lock_positions = self._static_lock_positions[-STATIC_LOCK_WINDOW_TICKS:]
            self._static_lock_distances = self._static_lock_distances[-STATIC_LOCK_WINDOW_TICKS:]

        if was_ready:
            if self._static_lock_anchor is None:
                self._static_lock_window = [False for _ in self._static_lock_positions]
            else:
                anchor = np.asarray(self._static_lock_anchor, dtype=np.float32).reshape(3)
                self._static_lock_window = [
                    bool(pos is not None and np.linalg.norm(pos - anchor) <= STATIC_LOCK_MAX_MOTION_M)
                    for pos in self._static_lock_positions
                ]
            self._static_landing_ready = (
                self._static_lock_anchor is not None
                and self._is_static_landing_candidate()
            )
            return

        self._static_lock_window = [False for _ in self._static_lock_positions]
        valid = [
            (idx, pos)
            for idx, pos in enumerate(self._static_lock_positions)
            if pos is not None
        ]
        best_anchor = None
        best_mask = self._static_lock_window
        best_count = 0
        best_avg_distance = None

        for _, center in valid:
            rough_cluster = [
                pos
                for _, pos in valid
                if float(np.linalg.norm(pos - center)) <= STATIC_LOCK_MAX_MOTION_M
            ]
            if not rough_cluster:
                continue
            anchor = np.median(np.stack(rough_cluster, axis=0), axis=0).astype(np.float32)
            mask = [
                bool(pos is not None and np.linalg.norm(pos - anchor) <= STATIC_LOCK_MAX_MOTION_M)
                for pos in self._static_lock_positions
            ]
            count = sum(1 for item in mask if item)
            distances = [
                self._static_lock_distances[idx]
                for idx, item in enumerate(mask)
                if item and self._static_lock_distances[idx] is not None
            ]
            avg_distance = float(np.mean(distances)) if distances else None
            if count > best_count or (
                count == best_count
                and avg_distance is not None
                and (best_avg_distance is None or avg_distance < best_avg_distance)
            ):
                best_count = count
                best_mask = mask
                best_anchor = anchor
                best_avg_distance = avg_distance

        self._static_lock_window = best_mask
        if best_anchor is not None:
            self._static_lock_anchor = best_anchor
            self._static_lock_avg_distance = best_avg_distance
        elif not valid:
            self._static_lock_anchor = None
            self._static_lock_avg_distance = None

        window_ready = (
            len(self._static_lock_positions) == STATIC_LOCK_WINDOW_TICKS
            and best_count >= STATIC_LOCK_REQUIRED_HITS
            and self._static_lock_anchor is not None
            and self._static_lock_avg_distance is not None
            and self._static_lock_avg_distance <= STATIC_LOCK_MAX_AVG_DISTANCE_M
            and self._is_static_landing_candidate()
        )
        self._static_landing_ready = window_ready

    def _update_static_landing_lock(self, detection_probability, predicted_goal_position, drone_position=None):
        if self._landing_committed:
            return
        if not self._is_static_landing_candidate():
            self._append_static_lock_hit(False)
            return

        if detection_probability is None or float(detection_probability) < STATIC_LOCK_MIN_PROBABILITY:
            self._append_static_lock_hit(False)
            return

        if predicted_goal_position is None:
            self._append_static_lock_hit(False)
            return

        estimate = np.asarray(predicted_goal_position, dtype=np.float32).reshape(3)
        distance = None
        if drone_position is not None:
            distance = float(np.linalg.norm(estimate - np.asarray(drone_position, dtype=np.float32).reshape(3)))
        self._append_static_lock_hit(True, estimate, distance)

    def _find_platform(self, depth, state):
        pr, tr, pv, pq = self._predict_goal(depth, state, state[:3], state[3:6])
        using_pt = self._last_detector_mode == "pt"
        strict_village_detection = self._xgb_map_is({"village"})
        genuine = pr > 0.8
        is_visible = genuine

        if using_pt and not strict_village_detection:
            self._pred_buf.append(np.asarray(tr, dtype=np.float32).copy())
            if len(self._pred_buf) > CONSIST_N:
                self._pred_buf.pop(0)
            if (not is_visible) and len(self._pred_buf) >= CONSIST_N:
                arr = np.asarray(self._pred_buf, dtype=np.float32)
                spread = float(np.max(np.linalg.norm(arr - arr.mean(axis=0), axis=1)))
                search_center = (
                    np.asarray(state[0:3], dtype=np.float32)
                    + np.asarray(state[-3:], dtype=np.float32)
                )
                near = float(np.linalg.norm(np.asarray(tr, dtype=np.float32) - search_center)) < CONSIST_NEAR
                if spread < CONSIST_SPREAD and near:
                    is_visible = True
                    pr = max(pr, 0.85)
        else:
            self._pred_buf = []

        if is_visible and self._ab_tracker.initialized:
            elapsed_steps = max(1, self.platform_lost_step + 1)
            predicted = self._ab_tracker.pos + self._ab_tracker.vel * elapsed_steps * SIM_DT
            innovation = float(np.linalg.norm(tr - predicted))
            loose_innovation = using_pt or self._uid53_rules_active() or self._xgb_map_is({"forest"})
            max_innovation = 3.0 * elapsed_steps * SIM_DT + (2.0 if loose_innovation else 0.8)
            if innovation > max_innovation:
                is_visible = False
                pr = 0.0
                self._track_score = max(0, self._track_score - 2)
                self._track_reject_streak += 1
                if self._track_reject_streak >= 5 and self._track_score == 0:
                    self._ab_tracker.reset()
                    self._track_reject_streak = 0
            else:
                self._track_score = min(10, self._track_score + 1)
                self._track_reject_streak = 0
        elif is_visible:
            self._track_score = min(10, self._track_score + 1)
            self._track_reject_streak = 0

        if self.platform_position is None:
            self.platform_position = state[0:3] + state[-3:]
            self.reverse_landing_platform = self.platform_position.copy()
            self.landing_platform = self.platform_position.copy()
        if is_visible:
            self.see_P = True
            self.platform_position = tr
            self.platform_lost_step = 0
            if (not using_pt) or genuine:
                self._ab_tracker.update(tr)
        elif self.see_P:
            self.platform_lost_step += 1
            if self.platform_lost_step > 20:
                self.see_P = False
                self.is_find_P = False
        else:
            self.platform_position = state[0:3] + state[-3:]
        goal_pos = self.platform_position.copy()
        self.reverse_landing_platform = 0.7 * (self.landing_platform + self.reverse_d) + 0.3 * goal_pos
        self.landing_platform = 0.7 * self.landing_platform + 0.3 * goal_pos
        if self._force_static_platform:
            self._apply_static_map_override()
        else:
            self.detect_reverse_direction(goal_pos, self.reverse_landing_platform)
        dist_to_go = np.linalg.norm(goal_pos - self.landing_platform)
        if self.move_in_auto_mode:
            dist_to_go = np.linalg.norm(goal_pos - self.reverse_d - self.landing_platform)
        self.p_buffer = 0.7 * self.p_buffer + 0.3 * dist_to_go
        if dist_to_go < 0.1 and  self.p_buffer < 0.1 and is_visible:
            self.is_find_P = True
        else:
            self.is_find_P = False
        return pr, tr, pv, pq[:3]
    
    def detect_reverse_direction(self, a, b):
        if self._force_static_platform:
            self._apply_static_map_override()
            return
        self.reverse_buffer = np.concatenate([self.reverse_buffer, a.reshape(1,3).copy()])
        self.last_reverse_buffer = np.concatenate([self.last_reverse_buffer, b.reshape(1,3).copy()])
        if len(self.reverse_buffer) > 50:
            self.reverse_buffer = self.reverse_buffer[1:]
        if len(self.last_reverse_buffer) > 10:
            self.last_reverse_buffer = self.last_reverse_buffer[1:]
        if len(self.reverse_buffer) < 15:
            return
        d_vect_reverse = self.reverse_buffer[10:] - self.reverse_buffer[:-10]
        error = 0.0
        for x, v in enumerate(d_vect_reverse[:-1]):
            norm = np.linalg.norm(v) * np.linalg.norm(d_vect_reverse[x+1])
            pro = v @ d_vect_reverse[x+1]
            if norm < 1e-12 or pro < 0:
                continue
            error += pro / norm
            
        d_vect_reverse = self.last_reverse_buffer[1:] - self.last_reverse_buffer[:-1]
        mean_d_reverse = np.mean(d_vect_reverse, axis=0)
        platform_speed = np.linalg.norm(mean_d_reverse)
        if error > 30.0 and platform_speed > 0.012 and platform_speed < 0.05:
            self.move_in_auto_mode = True
            self.reverse_d = mean_d_reverse

    def _get_yaw_command(self, yaw_command_1, drone_rpy):
        drone_yaw = drone_rpy[2]
        drone_yaw_rescaled = drone_yaw/np.pi
        yaw_command_2 = yaw_command_1 + 2 if yaw_command_1 < 0 else yaw_command_1 - 2
        yaw_command = yaw_command_1 if abs(yaw_command_1 - drone_yaw_rescaled) < abs(yaw_command_2 - drone_yaw_rescaled) else yaw_command_2
        diff = yaw_command - drone_yaw_rescaled
        is_safe = abs(drone_rpy[0]) < 0.7 and abs(drone_rpy[1]) < 0.7
        if not is_safe:
            return drone_yaw_rescaled
        if abs(diff) < 0.02:
            return yaw_command_1
        
        if diff > 0.0:
            return self._rot_left(drone_yaw)
        else:
            return self._rot_right(drone_yaw)
        
    def _rot_left(self, drone_yaw):
        max_yaw_change = MAX_YAW_RATE * SIM_DT
        new_drone_yaw_angle = drone_yaw + max_yaw_change - 1e-4
        new_drone_yaw_angle_normalized = new_drone_yaw_angle / np.pi

        if new_drone_yaw_angle_normalized > 1.0:
            new_drone_yaw_angle_normalized = (new_drone_yaw_angle_normalized - 1.0) - 1.0

        if new_drone_yaw_angle_normalized < -1.0:
            new_drone_yaw_angle_normalized = (new_drone_yaw_angle_normalized + 1.0) + 1.0

        return np.clip(new_drone_yaw_angle_normalized, -1.0, 1.0)

    def _rot_right(self, drone_yaw):
        max_yaw_change = MAX_YAW_RATE * SIM_DT
        new_drone_yaw_angle = drone_yaw - max_yaw_change - 1e-4
        new_drone_yaw_angle_normalized = new_drone_yaw_angle / np.pi

        if new_drone_yaw_angle_normalized > 1.0:
            new_drone_yaw_angle_normalized = (new_drone_yaw_angle_normalized - 1.0) - 1.0

        if new_drone_yaw_angle_normalized < -1.0:
            new_drone_yaw_angle_normalized = (new_drone_yaw_angle_normalized + 1.0) + 1.0

        return np.clip(new_drone_yaw_angle_normalized, -1.0, 1.0)

    def _load_model(self, m_model_path: Path):
        m_model_path = Path(m_model_path)
        if not m_model_path.exists():
            raise FileNotFoundError(f"Model not found: {m_model_path}")

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.intra_op_num_threads = 2
            session = ort.InferenceSession(
                str(m_model_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load ONNX goal detector. If the model was exported with external data, "
                f"keep the referenced .onnx.data file next to {m_model_path}."
            ) from exc

        inputs = {input_info.name: input_info for input_info in session.get_inputs()}
        required_inputs = {"depth", "state", "M"}
        missing_inputs = sorted(required_inputs - set(inputs))
        if missing_inputs:
            raise KeyError(f"ONNX goal detector missing required inputs: {missing_inputs}")

        outputs = {output_info.name: output_info for output_info in session.get_outputs()}
        if "prediction" not in outputs:
            raise KeyError("ONNX goal detector missing required output: prediction")
        m_out_name = "M_out" if "M_out" in outputs else "M"
        if m_out_name not in outputs:
            raise KeyError("ONNX goal detector missing required output: M_out")

        state_shape = inputs["state"].shape
        state_dim = m_STATE_DIM
        if len(state_shape) == 2 and isinstance(state_shape[1], int):
            state_dim = int(state_shape[1])

        self._m_model_session = session
        self._m_model_inputs = inputs
        self._m_model_prediction_output = outputs["prediction"].name
        self._m_model_m_output = outputs[m_out_name].name
        self._m_state_dim = state_dim

    def _load_pt_model(self):
        if self._pt_model is not None:
            return
        pt_model_path = Path(self._pt_model_path)
        if not pt_model_path.exists():
            raise FileNotFoundError(f"Model not found: {pt_model_path}")

        device = torch.device("cpu")
        try:
            model = torch.jit.load(str(pt_model_path), map_location=device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load TorchScript goal detector from {pt_model_path}.") from exc

        model.eval()
        self._pt_model = model
        self._pt_device = device


    def _load_navigation_model(self, nav_model_path: Path):
        nav_model_path = Path(nav_model_path)
        if not nav_model_path.exists():
            raise FileNotFoundError(f"Model not found: {nav_model_path}")

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.intra_op_num_threads = 2
            session = ort.InferenceSession(
                str(nav_model_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load ONNX navigation model. If the model was exported with external data, "
                f"keep the referenced .onnx.data file next to {nav_model_path}."
            ) from exc

        inputs = {input_info.name: input_info for input_info in session.get_inputs()}
        missing_inputs = sorted({"depth", "action"} - set(inputs))
        if missing_inputs:
            raise KeyError(f"ONNX navigation model missing required inputs: {missing_inputs}")

        outputs = session.get_outputs()
        if len(outputs) < 1:
            raise KeyError("ONNX navigation model expected at least one output")

        self._navigation_model_session = session
        self._navigation_model_inputs = inputs
        self._navigation_model_output_name = outputs[0].name


    def _predict_goal_pt(
        self,
        depth: np.ndarray,
        state: np.ndarray,
        drone_position: np.ndarray,
        drone_rpy: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        self._load_pt_model()
        depth_input = np.asarray(depth, dtype=np.float32)
        if depth_input.ndim != 3 or depth_input.shape[-1] != 1:
            raise ValueError(f"expected depth shape (H,W,1), got {depth_input.shape}")

        camera_position, camera_target, _, _, _ = _cam_geom(
            drone_position,
            drone_rpy,
        )

        state_input = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_input.shape[0] != m_STATE_DIM:
            raise ValueError(
                "Goal detector state shape mismatch: "
                f"expected {m_STATE_DIM}, got {state_input.shape[0]}"
            )

        depth_b = depth_input[None, ...]
        cam_pos_b = camera_position[None, :].astype(np.float32, copy=False)
        cam_tgt_b = camera_target[None, :].astype(np.float32, copy=False)
        fov_b = np.asarray([CAMERA_FOV_DEG], dtype=np.float32)
        state_b = state_input[None, :]
        dev = self._pt_device
        with torch.inference_mode():
            prediction, self._pt_M = self._pt_model(
                torch.from_numpy(depth_b).to(dev),
                torch.from_numpy(state_b).to(dev),
                torch.from_numpy(cam_pos_b).to(dev),
                torch.from_numpy(cam_tgt_b).to(dev),
                torch.from_numpy(fov_b).to(dev),
                self._pt_M,
            )
            prediction = prediction.detach().cpu().numpy()
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != (1, 11):
            raise ValueError(f"expected output shape (1,11), got {prediction.shape}")

        visibility_prob = float(prediction[0, 0])
        pred_world = prediction[0, 1:4].astype(np.float32, copy=True)
        pred_world_cov = prediction[0, 4:7].astype(np.float32, copy=True)
        pred_q = prediction[0, 7:11].astype(np.float32, copy=True)
        dist_from_start = np.linalg.norm(pred_world - self.start_state[:3])
        if dist_from_start < 3.5:
            visibility_prob = 0.0
        return visibility_prob, pred_world, pred_world_cov, pred_q


    def _predict_goal(
        self,
        depth: np.ndarray,
        state: np.ndarray,
        drone_position: np.ndarray,
        drone_rpy: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        if self._pt_detector_active():
            self._last_detector_mode = "pt"
            return self._predict_goal_pt(depth, state, drone_position, drone_rpy)
        self._last_detector_mode = "onnx"
        depth_input = np.asarray(depth, dtype=np.float32)
        if depth_input.ndim != 3 or depth_input.shape[-1] != 1:
            raise ValueError(f"expected depth shape (H,W,1), got {depth_input.shape}")

        state_input = np.asarray(state, dtype=np.float32).reshape(-1)
        if self._m_state_dim is not None and state_input.shape[0] != self._m_state_dim:
            raise ValueError(
                "Goal detector state shape mismatch: "
                f"expected {self._m_state_dim}, got {state_input.shape[0]}"
            )

        depth_b = depth_input[None, ...]
        state_b = state_input[None, :]
        m_input = self.M if self.M is not None else _default_H_m()

        model_inputs = {
            "depth": depth_b,
            "state": state_b,
            "M": np.asarray(m_input, dtype=np.float32),
        }
        if "cam_pos" in self._m_model_inputs or "cam_tgt" in self._m_model_inputs:
            camera_position, camera_target, _, _, _ = _cam_geom(
                drone_position,
                drone_rpy,
            )
            if "cam_pos" in self._m_model_inputs:
                model_inputs["cam_pos"] = camera_position[None, :].astype(np.float32, copy=False)
            if "cam_tgt" in self._m_model_inputs:
                model_inputs["cam_tgt"] = camera_target[None, :].astype(np.float32, copy=False)
        if "fov" in self._m_model_inputs:
            model_inputs["fov"] = np.asarray([CAMERA_FOV_DEG], dtype=np.float32)
        prediction, self.M = self._m_model_session.run(
            [self._m_model_prediction_output, self._m_model_m_output],
            model_inputs,
        )
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != (1, 11):
            raise ValueError(f"expected output shape (1,11), got {prediction.shape}")

        visibility_prob = float(prediction[0, 0])
        pred_world = prediction[0, 1:4].astype(np.float32, copy=True)
        pred_world_cov = prediction[0, 4:7].astype(np.float32, copy=True)
        pred_q = prediction[0, 7:11].astype(np.float32, copy=True)
        dist_from_start = np.linalg.norm(pred_world - self.start_state[:3])
        if dist_from_start < 3.5:
            visibility_prob = 0.0
        return visibility_prob, pred_world, pred_world_cov, pred_q
 
class LandingController:    
    def __init__(
        self,
        gains: Gains = None,
        setpoint: Optional[np.ndarray] = None,
        error_deadband: float = 0.01,
    ):
        self.gains = gains or Gains()
        self.setpoint = setpoint if setpoint is not None else np.zeros(3)
        self.error_deadband = error_deadband
        
        
    def update(
        self,
        current_pos: np.ndarray,
        setpoint_v: Optional[np.ndarray] = None,
        feedforward: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        current_pos = np.array(current_pos)
        
        error = self.setpoint - current_pos
        error_magnitude = np.linalg.norm(error)
        if error_magnitude < self.error_deadband:
            error = np.zeros(3)
        
        p_term = self.gains.K * error
                
        ff_term = np.zeros(3)
        if setpoint_v is not None:
            ff_term = self.gains.K * 1.0 * np.array(setpoint_v)
            ff_term[2] = 0.0
        
        if feedforward is not None:
            ff_term += np.array(feedforward)
        
        output = p_term + ff_term
        return output
    
    
    def set_gains(self, K: float):
        """Update PID gains dynamically."""
        self.gains.K = K
    
    def set_setpoint(self, setpoint: np.ndarray):
        """Update target position."""
        self.setpoint = np.array(setpoint)
    
    def reset(self):
        """Reset controller state."""
