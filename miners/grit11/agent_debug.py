"""Debug helpers for Swarm drone agent — overlays, JSONL traces, snapshots."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

_ENGINE_KEYS = ("agent", "uid53", "uid94", "pt", "onnx", "single")
_ROUTE_ENGINE = {
    "village": "uid53",
    "open": "uid53",
    "warehouse": "uid94",
    "city": "pt",
    "mountain": "pt",
    "forest": "pt",
}


def _as_float_list(vec: object, n: int = 3) -> list[float]:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    return [float(arr[i]) for i in range(min(n, arr.size))]


def _dist3(a: object, b: object) -> Optional[float]:
    try:
        va = np.asarray(a, dtype=np.float64).reshape(-1)[:3]
        vb = np.asarray(b, dtype=np.float64).reshape(-1)[:3]
        if va.size < 3 or vb.size < 3:
            return None
        return float(np.linalg.norm(va - vb))
    except Exception:
        return None


def _active_engine_snapshot(router: Mapping[str, Any]) -> dict[str, Any]:
    if not router:
        return {}
    route = str(router.get("route") or "")
    key = _ROUTE_ENGINE.get(route)
    if key and isinstance(router.get(key), dict) and router[key]:
        return dict(router[key])
    for engine_key in _ENGINE_KEYS:
        snap = router.get(engine_key)
        if isinstance(snap, dict) and snap:
            return dict(snap)
    return {}


def _real_goal_distance_m(task: Mapping[str, Any], router: Mapping[str, Any]) -> Optional[float]:
    goal = task.get("goal")
    if goal is None:
        return None
    snap = _active_engine_snapshot(router)
    pos = snap.get("position")
    if pos is None:
        return None
    return _dist3(pos, goal)


def build_snapshot(
    agent_id: str,
    *,
    mode: str,
    position: object,
    velocity: object,
    speed_mps: float,
    dist_goal_vec: float,
    action: object,
    goal_detector: Optional[dict[str, Any]] = None,
    map_prediction: Optional[dict[str, Any]] = None,
    search_area_position: object | None = None,
    search_area_vector: object | None = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    act = np.asarray(action, dtype=np.float32).reshape(-1)
    snap: dict[str, Any] = {
        "agent_id": str(agent_id),
        "mode": str(mode),
        "position": _as_float_list(position, 3),
        "velocity": _as_float_list(velocity, 3),
        "speed_mps": float(speed_mps),
        "dist_goal_vec": float(dist_goal_vec),
        "action": [float(x) for x in act[:5]] if act.size >= 5 else [float(x) for x in act],
    }
    if goal_detector is not None:
        snap["goal_detector"] = dict(goal_detector)
    if map_prediction is not None:
        snap["map_prediction"] = dict(map_prediction)
    if search_area_position is not None:
        snap["search_area_position"] = _as_float_list(search_area_position, 3)
    if search_area_vector is not None:
        vec = np.asarray(search_area_vector, dtype=np.float64).reshape(-1)[:3]
        snap["search_area_vector"] = [float(x) for x in vec]
        snap["dist_search_area"] = float(np.linalg.norm(vec))
    if extra:
        snap["extra"] = dict(extra)
    return snap


def set_task_context(
    *,
    seed: int,
    challenge_type: int,
    map_label: str,
    start: Sequence[float],
    goal: Sequence[float],
    horizon: float,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "challenge_type": int(challenge_type),
        "map_label": str(map_label),
        "start": [float(x) for x in start[:3]],
        "goal": [float(x) for x in goal[:3]],
        "horizon_sec": float(horizon),
    }


MAP_LABELS_SHORT = ("city", "open", "mtn", "vil", "wh", "for")

# Match generate_video.py depth inset + agent onboard camera geometry.
DEPTH_INSET_SIZE = 128
DEPTH_INSET_MARGIN = 8
AGENT_CAMERA_OFFSET_M = 0.13
AGENT_CAMERA_UP_OFFSET_M = 0.05
AGENT_CAMERA_FOV_DEG = 90.0
FPV_OFFSET_FORWARD_M = 0.15
FPV_OFFSET_UP_M = 0.02
FPV_FOV_DEG = 90.0


def _rpy_to_rot_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rotation_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rotation_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rotation_z @ rotation_y @ rotation_x


def _norm_vec3(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def _look_at_view_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    forward = _norm_vec3(target - eye)
    right = _norm_vec3(np.cross(forward, up))
    cam_up = _norm_vec3(np.cross(right, forward))
    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = cam_up
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view.reshape(-1, order="F")


def _perspective_projection_matrix(
    fov_deg: float,
    aspect: float,
    *,
    near: float = 0.1,
    far: float = 500.0,
) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(float(fov_deg)) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / max(aspect, 1e-6)
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj.reshape(-1, order="F")


def _world_to_pixel(
    point: np.ndarray,
    view: np.ndarray,
    proj: np.ndarray,
    width: int,
    height: int,
    *,
    clamp_to_image: bool = True,
) -> Optional[tuple[float, float]]:
    view_m = np.asarray(view, dtype=np.float64).reshape((4, 4), order="F")
    proj_m = np.asarray(proj, dtype=np.float64).reshape((4, 4), order="F")
    p4 = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    clip = proj_m @ (view_m @ p4)
    w = float(clip[3])
    if w <= 1e-8:
        return None
    ndc = clip[:3] / w
    if ndc[2] < -1.0 or ndc[2] > 1.0:
        return None
    u = (ndc[0] + 1.0) * 0.5 * float(width)
    v = (1.0 - ndc[1]) * 0.5 * float(height)
    if u < 0.0 or u >= width or v < 0.0 or v >= height:
        if not clamp_to_image:
            return None
        u = min(max(u, 0.0), float(width - 1))
        v = min(max(v, 0.0), float(height - 1))
    return float(u), float(v)


def _camera_pose_from_drone(
    drone_position: Sequence[float],
    drone_rpy: Sequence[float],
    *,
    forward_offset_m: float,
    up_offset_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = _rpy_to_rot_matrix(drone_rpy)
    forward = _norm_vec3(rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64))
    up = _norm_vec3(rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
    eye = (
        np.asarray(drone_position, dtype=np.float64).reshape(3)
        + forward * float(forward_offset_m)
        + up * float(up_offset_m)
    )
    target = eye + forward * 20.0
    return eye, target, up


def _depth_sensor_pixel_to_frame(
    pixel_u: float,
    pixel_v: float,
    *,
    sensor_size: int,
    frame_w: int,
    frame_h: int,
) -> Optional[tuple[int, int]]:
    inset_size = DEPTH_INSET_SIZE
    inset_margin = DEPTH_INSET_MARGIN
    if inset_size + inset_margin * 2 > frame_w or inset_size + inset_margin * 2 > frame_h:
        return None
    ix0 = frame_w - inset_size - inset_margin
    iy0 = inset_margin
    scale_x = inset_size / max(1.0, float(sensor_size - 1))
    scale_y = inset_size / max(1.0, float(sensor_size - 1))
    fx = int(round(ix0 + pixel_u * scale_x))
    fy = int(round(iy0 + pixel_v * scale_y))
    fx = int(np.clip(fx, 0, frame_w - 1))
    fy = int(np.clip(fy, 0, frame_h - 1))
    return fx, fy


def _predicted_goal_pixel_on_frame(
    debug_snapshot: Mapping[str, Any],
    frame_w: int,
    frame_h: int,
) -> Optional[tuple[int, int, str]]:
    snap = _active_engine_snapshot(debug_snapshot)
    gd = snap.get("goal_detector") or {}
    pred = gd.get("pred")
    if pred is None:
        return None
    pred_xyz = np.asarray(pred, dtype=np.float64).reshape(-1)[:3]
    if pred_xyz.size < 3:
        return None

    source = str(gd.get("source") or "det")
    extra = snap.get("extra") or {}
    pos = snap.get("position")
    rpy = extra.get("rpy")

    if pos is not None and rpy is not None and len(rpy) >= 3:
        aspect = float(frame_w) / max(float(frame_h), 1.0)
        for label, fwd_off, up_off, fov in (
            ("fpv", FPV_OFFSET_FORWARD_M, FPV_OFFSET_UP_M, FPV_FOV_DEG),
            ("agent_cam", AGENT_CAMERA_OFFSET_M, AGENT_CAMERA_UP_OFFSET_M, AGENT_CAMERA_FOV_DEG),
        ):
            eye, target, up = _camera_pose_from_drone(
                pos,
                rpy,
                forward_offset_m=fwd_off,
                up_offset_m=up_off,
            )
            view = _look_at_view_matrix(eye, target, up)
            proj = _perspective_projection_matrix(fov, aspect)
            pix = _world_to_pixel(pred_xyz, view, proj, frame_w, frame_h, clamp_to_image=True)
            if pix is not None:
                return int(round(pix[0])), int(round(pix[1])), source

    pixel_center = gd.get("pixel_center")
    sensor_size = int(gd.get("depth_imgsz") or DEPTH_INSET_SIZE)
    if pixel_center is not None and len(pixel_center) >= 2:
        inset_pix = _depth_sensor_pixel_to_frame(
            float(pixel_center[0]),
            float(pixel_center[1]),
            sensor_size=sensor_size,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        if inset_pix is not None:
            return inset_pix[0], inset_pix[1], source

    return None


def _draw_predicted_goal_marker(
    frame: np.ndarray,
    debug_snapshot: Mapping[str, Any],
) -> None:
    try:
        import cv2
    except ImportError:
        return

    h, w = frame.shape[:2]
    marker = _predicted_goal_pixel_on_frame(debug_snapshot, w, h)
    if marker is None:
        return

    px, py, source = marker
    cv2.circle(frame, (px, py), 7, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (px, py), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    label = f"pred:{source}"
    cv2.putText(
        frame,
        label,
        (min(px + 10, max(0, w - 90)), max(14, py - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


def _fmt_xyz(vec: object) -> str:
    p = _as_float_list(vec, 3)
    return f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"


def _fmt_bool(value: object) -> str:
    return "T" if bool(value) else "F"


def format_overlay_lines(
    task: Mapping[str, Any],
    router: Mapping[str, Any],
    t_sim: float,
    step: int,
) -> list[str]:
    snap = _active_engine_snapshot(router)
    task_ctx = router.get("task") if isinstance(router.get("task"), dict) else task
    map_route = str(router.get("route") or "?")
    act_mode = str(snap.get("mode") or "?")
    pos = snap.get("position")
    vel = snap.get("velocity")
    speed = snap.get("speed_mps")
    action = snap.get("action") or []
    goal = (task_ctx or {}).get("goal")
    dist_real = _real_goal_distance_m(task_ctx or task, router)
    dist_vec = snap.get("dist_goal_vec")
    dist_search = snap.get("dist_search_area")
    search_pos = snap.get("search_area_position")
    map_pred = snap.get("map_prediction") or {}
    gd = snap.get("goal_detector") or {}
    extra = snap.get("extra") or {}

    lines: list[str] = []

    # --- run context ---
    seed = task_ctx.get("seed", "?")
    map_label = task_ctx.get("map_label", "?")
    lines.append(
        f"[run] seed={seed} {map_label} sim={t_sim:.2f}s step={step} map_cls={map_route}"
    )

    # --- act() FSM mode (primary debug signal) ---
    act_bits = [f"act_mode={act_mode}"]
    for key, label in (
        ("search_stage", "stage"),
        ("tracking", "land_track"),
        ("forward", "fwd"),
        ("first_order_cnt", "smooth"),
        ("inside_goal_zone", "in_zone"),
    ):
        if key not in extra:
            continue
        val = extra[key]
        if isinstance(val, bool):
            act_bits.append(f"{label}={_fmt_bool(val)}")
        else:
            act_bits.append(f"{label}={val}")
    lines.append("[act] " + "  ".join(act_bits))

    # --- goal / platform tracking ---
    track_bits: list[str] = []
    for key, label in (
        ("goal_tracked", "goal_trk"),
        ("yolo_tracked", "yolo_trk"),
        ("is_goal_visible", "goal_vis"),
        ("is_find_P", "find_P"),
        ("track_score", "score"),
        ("plat_lost_step", "lost"),
        ("detector_mode", "det"),
        ("is_hatt", "hatt"),
    ):
        if key not in extra:
            continue
        val = extra[key]
        if isinstance(val, bool):
            track_bits.append(f"{label}={_fmt_bool(val)}")
        else:
            track_bits.append(f"{label}={val}")
    if "last_vis_prob" in extra:
        track_bits.append(f"vis_prob={float(extra['last_vis_prob']):.2f}")
    yolo_hist = extra.get("yolo_history")
    if yolo_hist:
        track_bits.append("yolo_hist=" + "".join("1" if x else "0" for x in yolo_hist))
    if track_bits:
        lines.append("[track] " + "  ".join(track_bits))

    # --- landing sub-state ---
    land_bits: list[str] = []
    for key, label in (
        ("landing_committed", "commit"),
        ("static_landing_ready", "static_rdy"),
        ("landing_patience", "patience"),
        ("move_auto", "auto"),
    ):
        if key not in extra:
            continue
        val = extra[key]
        if isinstance(val, bool):
            land_bits.append(f"{label}={_fmt_bool(val)}")
        else:
            land_bits.append(f"{label}={val}")
    if extra.get("land_hdist") is not None:
        land_bits.append(f"hdist={float(extra['land_hdist']):.2f}m")
    if land_bits:
        lines.append("[land] " + "  ".join(land_bits))

    # --- attitude ---
    att_bits: list[str] = []
    if "altitude_agl" in extra:
        att_bits.append(f"agl={float(extra['altitude_agl']):.1f}m")
    rpy = extra.get("rpy")
    if rpy and len(rpy) >= 3:
        att_bits.append(f"rpy=({float(rpy[0]):.2f},{float(rpy[1]):.2f},{float(rpy[2]):.2f})")
    if att_bits:
        lines.append("[attitude] " + "  ".join(att_bits))

    # --- motion (scalars + vector) ---
    motion_bits: list[str] = []
    if speed is not None:
        motion_bits.append(f"speed={float(speed):.2f}m/s")
    if vel is not None:
        v = _as_float_list(vel, 3)
        motion_bits.append(f"vel=({v[0]:.2f},{v[1]:.2f},{v[2]:.2f})")
    if motion_bits:
        lines.append("[motion] " + "  ".join(motion_bits))

    # --- positions (all 3D points on one line) ---
    pos_bits: list[str] = []
    if pos is not None:
        pos_bits.append(f"drone={_fmt_xyz(pos)}")
    if goal is not None:
        pos_bits.append(f"goal={_fmt_xyz(goal)}")
    if search_pos is not None:
        pos_bits.append(f"search={_fmt_xyz(search_pos)}")
    visible_goal = extra.get("visible_goal")
    if visible_goal is not None:
        pos_bits.append(f"vis_goal={_fmt_xyz(visible_goal)}")
    platform_pos = extra.get("platform_pos")
    if platform_pos is not None:
        pos_bits.append(f"plat={_fmt_xyz(platform_pos)}")
    landing_plat = extra.get("landing_plat")
    if landing_plat is not None:
        pos_bits.append(f"land_plat={_fmt_xyz(landing_plat)}")
    landing_anchor = extra.get("landing_anchor")
    if landing_anchor is not None:
        pos_bits.append(f"anchor={_fmt_xyz(landing_anchor)}")
    if pos_bits:
        lines.append("[pos] " + "  ".join(pos_bits))

    # --- distances (all meters on one line) ---
    dist_bits: list[str] = []
    if dist_real is not None:
        dist_bits.append(f"to_goal={dist_real:.1f}m")
    if dist_search is not None:
        dist_bits.append(f"to_search={float(dist_search):.1f}m")
    if "search_progress" in extra:
        dist_bits.append(f"progress={float(extra['search_progress']):.1f}m")
    if extra.get("dist_vis_goal") is not None:
        dist_bits.append(f"to_vis={float(extra['dist_vis_goal']):.1f}m")
    if dist_vec is not None:
        dist_bits.append(f"vec={float(dist_vec):.1f}m")
    if extra.get("land_hdist") is not None:
        dist_bits.append(f"land_h={float(extra['land_hdist']):.2f}m")
    if dist_bits:
        lines.append("[dist] " + "  ".join(dist_bits))

    # --- map classifier ---
    if map_pred:
        label = str(map_pred.get("label", "?"))
        prob = map_pred.get("prob")
        probs = map_pred.get("probs") or []
        map_bits = [f"label={label}"]
        if prob is not None:
            map_bits.append(f"p={float(prob):.3f}")
        if "map_prob_count" in extra:
            map_bits.append(f"n={int(extra['map_prob_count'])}")
        if probs:
            short_probs = ",".join(
                f"{MAP_LABELS_SHORT[i]}:{float(probs[i]):.2f}"
                for i in range(min(len(probs), len(MAP_LABELS_SHORT)))
            )
            map_bits.append(short_probs)
        lines.append("[map] " + "  ".join(map_bits))

    # --- goal detector / YOLO (model output on one line) ---
    if gd and (gd.get("visibility") is not None or gd.get("pred") is not None):
        det_bits = [f"src={gd.get('source') or '?'}"]
        vis = gd.get("visibility")
        if vis is not None:
            det_bits.append(f"vis={float(vis):.2f}")
        if gd.get("confidence") is not None:
            det_bits.append(f"conf={float(gd['confidence']):.2f}")
        pred = gd.get("pred")
        if pred is not None:
            det_bits.append(f"pred={_fmt_xyz(pred)}")
        lines.append("[det] " + "  ".join(det_bits))

    # --- navigation / depth-ray ---
    nav_bits: list[str] = []
    if extra.get("dir_similarity") is not None:
        nav_bits.append(f"dir_sim={float(extra['dir_similarity']):.3f}")
    nav_avoid = extra.get("nav_avoid_dir")
    if nav_avoid:
        nav_bits.append(f"avoid={_fmt_xyz(nav_avoid)}")
    pre_nav = extra.get("pre_nav_action")
    if pre_nav and isinstance(pre_nav, (list, tuple)) and len(pre_nav) >= 3:
        nav_bits.append(f"pre_nav={_fmt_xyz(pre_nav[:3])}")
    if nav_bits:
        lines.append("[nav] " + "  ".join(nav_bits))

    # --- actions (direction vectors grouped) ---
    if action:
        a = [float(x) for x in action[:5]]
        if len(a) >= 5:
            lines.append(
                f"[action] dir=({a[0]:.2f},{a[1]:.2f},{a[2]:.2f})  sp={a[3]:.2f}  yaw={a[4]:.2f}"
            )
    pre_action = extra.get("pre_action")
    if pre_action and isinstance(pre_action, (list, tuple)) and len(pre_action) >= 5:
        p = [float(x) for x in pre_action[:5]]
        if len(action) < 5 or any(abs(p[i] - float(action[i])) > 1e-3 for i in range(5)):
            lines.append(
                f"[pre] dir=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})  sp={p[3]:.2f}  yaw={p[4]:.2f}"
            )

    return lines


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":"), default=_json_default))
        handle.write("\n")


def _json_default(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def apply_frame_overlay(
    frame: np.ndarray,
    task: object,
    debug_snapshot: Optional[dict[str, Any]],
    t_sim: float,
    step_idx: int,
    depth_map: object | None = None,
) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return frame

    task_ctx: dict[str, Any]
    if isinstance(debug_snapshot, dict) and isinstance(debug_snapshot.get("task"), dict):
        task_ctx = dict(debug_snapshot["task"])
    else:
        task_ctx = {
            "seed": int(getattr(task, "map_seed", 0)),
            "challenge_type": int(getattr(task, "challenge_type", 0)),
            "map_label": str(getattr(task, "challenge_type", "?")),
            "start": list(getattr(task, "start", (0.0, 0.0, 0.0))),
            "goal": list(getattr(task, "goal", (0.0, 0.0, 0.0))),
            "horizon_sec": float(getattr(task, "horizon", 0.0)),
        }

    lines = format_overlay_lines(task_ctx, debug_snapshot or {}, float(t_sim), int(step_idx))

    out = frame.copy()
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)

    h, w = out.shape[:2]
    line_h = 17
    max_lines = min(18, max(1, len(lines)))
    bar_h = min(h - 4, 6 + line_h * max_lines)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, out, 0.38, 0, out)
    y = 14
    for line in lines[:max_lines]:
        cv2.putText(
            out,
            str(line)[:140],
            (6, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        y += line_h

    _draw_predicted_goal_marker(out, debug_snapshot or {})

    snap = _active_engine_snapshot(debug_snapshot or {})
    gd = snap.get("goal_detector") or {}
    box = gd.get("pixel_box_5")
    imgsz = int(gd.get("depth_imgsz") or 128)
    if box and len(box) >= 4:
        inset_margin = 8
        inset_size = imgsz
        if inset_size + inset_margin * 2 <= w:
            ix0 = w - inset_size - inset_margin
            iy0 = inset_margin
            scale_x = inset_size / max(1.0, float(imgsz - 1))
            scale_y = inset_size / max(1.0, float(imgsz - 1))
            bx0 = int(ix0 + float(box[0]) * scale_x)
            by0 = int(iy0 + float(box[1]) * scale_y)
            bx1 = int(ix0 + float(box[2]) * scale_x)
            by1 = int(iy0 + float(box[3]) * scale_y)
            cv2.rectangle(out, (bx0, by0), (bx1, by1), (0, 220, 255), 2)
            conf = gd.get("confidence")
            if conf is None:
                conf = gd.get("visibility")
            if conf is not None:
                cv2.putText(
                    out,
                    f"YOLO {float(conf):.2f}",
                    (bx0, max(12, by0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

    _ = depth_map
    return out
