"""Per-step flight debug telemetry for video overlays and JSONL traces."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

MAP_LABELS = ("city", "open", "mountain", "village", "warehouse", "forest")
ENGINE_KEYS = ("pt", "onnx", "uid53", "uid94")
DEFAULT_DEPTH_IMGSZ = 128
MAX_OVERLAY_LINES = 24


def is_enabled() -> bool:
    return os.environ.get("SWARM_AGENT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _f3(vec) -> list[float]:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return [0.0, 0.0, 0.0]
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _task_context_from_task(task: Any) -> dict[str, Any]:
    if isinstance(task, dict):
        return {
            "seed": task.get("seed", "?"),
            "challenge_type": task.get("challenge_type", 0),
            "map_label": task.get("map_label", "?"),
            "start": task.get("start", [0, 0, 0]),
            "goal": task.get("goal", [0, 0, 0]),
            "horizon_sec": float(task.get("horizon_sec", task.get("horizon", 0.0))),
        }
    return set_task_context(
        seed=int(getattr(task, "map_seed", 0)),
        challenge_type=int(getattr(task, "challenge_type", 0)),
        map_label=str(getattr(task, "map_label", "?")),
        start=tuple(getattr(task, "start", (0.0, 0.0, 0.0))),
        goal=tuple(getattr(task, "goal", (0.0, 0.0, 0.0))),
        horizon=float(getattr(task, "horizon", 0.0)),
    )


def set_task_context(
    *,
    seed: int,
    challenge_type: int,
    map_label: str,
    start: tuple[float, float, float] | list[float],
    goal: tuple[float, float, float] | list[float],
    horizon: float = 0.0,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "challenge_type": int(challenge_type),
        "map_label": str(map_label),
        "start": _f3(start),
        "goal": _f3(goal),
        "horizon_sec": float(horizon),
    }


def build_snapshot(
    agent_id: str,
    *,
    mode: str,
    position: np.ndarray,
    velocity: np.ndarray,
    speed_mps: float,
    dist_goal_vec: float,
    action: np.ndarray,
    goal_detector: dict[str, Any] | None = None,
    map_prediction: dict[str, Any] | None = None,
    search_area_position: np.ndarray | list[float] | None = None,
    search_area_vector: np.ndarray | list[float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    act = np.asarray(action, dtype=np.float32).reshape(-1)
    snap: dict[str, Any] = {
        "agent": str(agent_id),
        "mode": str(mode),
        "position": _f3(position),
        "velocity": _f3(velocity),
        "speed_mps": float(speed_mps),
        "dist_goal_vec": float(dist_goal_vec),
        "action": [float(x) for x in act[:5]],
    }
    if search_area_position is not None:
        snap["search_area_position"] = _f3(search_area_position)
    if search_area_vector is not None:
        snap["search_area_vector"] = _f3(search_area_vector)
    if goal_detector:
        snap["goal_detector"] = goal_detector
    if map_prediction:
        snap["map_prediction"] = map_prediction
    if extra:
        snap.update(extra)
    return snap


def active_engine_key(route: str | None) -> str | None:
    label = str(route or "")
    if label in ("open", "village"):
        return "uid53"
    if label == "warehouse":
        return "uid94"
    if label == "mountain":
        return "pt"
    if label in ("city", "forest", "open"):
        return "onnx"
    return None


def _distance_3d(a, b) -> float | None:
    """Euclidean distance (metres) between two 3D points."""
    try:
        va = np.asarray(a, dtype=np.float64).reshape(-1)
        vb = np.asarray(b, dtype=np.float64).reshape(-1)
        if va.size < 3 or vb.size < 3:
            return None
        return float(np.linalg.norm(va[:3] - vb[:3]))
    except Exception:
        return None


def _drone_position_from_router(router: dict[str, Any]) -> list[float] | None:
    """Best-effort drone position from the active (or any populated) engine snapshot."""
    active_key = active_engine_key(str(router.get("route", "")))
    if active_key:
        pos = (router.get(active_key) or {}).get("position")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return _f3(pos)
    for key in ENGINE_KEYS:
        pos = (router.get(key) or {}).get("position")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return _f3(pos)
    return None


def _real_goal_distance_m(task: dict[str, Any], router: dict[str, Any]) -> float | None:
    """3D distance from drone to the task's true goal position (metres)."""
    goal = task.get("goal")
    if goal is None:
        task_ctx = router.get("task")
        if isinstance(task_ctx, dict):
            goal = task_ctx.get("goal")
    if goal is None:
        return None
    drone_pos = _drone_position_from_router(router)
    if drone_pos is None:
        return None
    return _distance_3d(drone_pos, goal)


def _format_search_lines(sub: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    pos = sub.get("search_area_position")
    vec = sub.get("search_area_vector")
    if pos:
        lines.append(
            f"search_area_pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})"
        )
    if vec:
        dist = float(np.linalg.norm(np.asarray(vec, dtype=np.float64)))
        lines.append(
            f"search_area_vec=({vec[0]:.1f},{vec[1]:.1f},{vec[2]:.1f})  "
            f"|vec|={dist:.1f}m"
        )
    return lines


def _format_engine_block(key: str, sub: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if not sub:
        return lines
    pos = sub.get("position", [0, 0, 0])
    lines.append(
        f"[{key}] mode={sub.get('mode', '?')}  "
        f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})  "
        f"spd={sub.get('speed_mps', 0):.2f}m/s"
    )
    lines.extend(_format_search_lines(sub))
    act = sub.get("action", [])
    if len(act) >= 5:
        lines.append(
            f"  act dir=({act[0]:.2f},{act[1]:.2f},{act[2]:.2f}) "
            f"spd={act[3]:.2f} yaw={act[4]:.2f}"
        )
    mp = sub.get("map_prediction")
    if mp:
        label_m = mp.get("label", "?")
        prob = mp.get("prob", 0.0)
        probs = mp.get("probs") or []
        top = " ".join(
            f"{MAP_LABELS[i][:3]}:{probs[i]:.2f}"
            for i in range(min(6, len(probs)))
        )
        lines.append(f"  map={label_m} ({prob:.2f})  [{top}]")
    gd = sub.get("goal_detector")
    if gd:
        vis = gd.get("visibility")
        pred = gd.get("pred")
        if vis is not None:
            line = f"  goal_det vis={vis:.3f}"
            if pred:
                line += f" pred=({pred[0]:.1f},{pred[1]:.1f},{pred[2]:.1f})"
            src = gd.get("source")
            if src:
                line += f" src={src}"
            px = gd.get("pixel_center")
            if px:
                line += f" px=({px[0]:.0f},{px[1]:.0f})"
            lines.append(line)
    return lines


def format_overlay_lines(
    task: dict[str, Any],
    router: dict[str, Any],
    t_sim: float,
    step: int,
) -> list[str]:
    lines: list[str] = []
    seed = task.get("seed", "?")
    label = task.get("map_label", "?")
    lines.append(f"seed={seed} {label}  sim={t_sim:.2f}s  step={step}")
    start = task.get("start", [0, 0, 0])
    goal = task.get("goal", [0, 0, 0])
    lines.append(
        f"start=({start[0]:.1f},{start[1]:.1f},{start[2]:.1f})  "
        f"goal=({goal[0]:.1f},{goal[1]:.1f},{goal[2]:.1f})"
    )
    dist_real_goal = _real_goal_distance_m(task, router)
    if dist_real_goal is not None:
        lines.append(f"dist_real_goal={dist_real_goal:.1f}m")
    route = router.get("route", "?")
    locked = router.get("route_locked")
    if locked is not None:
        lines.append(f"active_route={route}  locked={bool(locked)}")
    else:
        lines.append(f"active_route={route}")

    active_key = active_engine_key(str(route))
    if active_key:
        active_sub = router.get(active_key) or {}
        search_lines = _format_search_lines(active_sub)
        if search_lines:
            lines.append(f"--- active [{active_key}] search ---")
            lines.extend(search_lines)

    for key in ENGINE_KEYS:
        lines.extend(_format_engine_block(key, router.get(key) or {}))
    return lines


def _apply_text_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    out = frame.copy()
    h, w = out.shape[:2]
    line_h = 17
    max_lines = min(MAX_OVERLAY_LINES, max(1, len(lines)))
    bar_h = min(h - 4, 6 + line_h * max_lines)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, out, 0.38, 0, out)
    y = 14
    for line in lines[:max_lines]:
        cv2.putText(
            out,
            str(line)[:160],
            (6, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        y += line_h
    return out


def _draw_yolo_goal_box(frame: np.ndarray, goal_detector: dict[str, Any] | None) -> np.ndarray:
    import cv2

    if not goal_detector or goal_detector.get("source") != "yolo":
        return frame
    center = goal_detector.get("pixel_center")
    if not center or len(center) < 2:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    imgsz = float(goal_detector.get("depth_imgsz") or DEFAULT_DEPTH_IMGSZ)
    sx = w / max(1.0, imgsz)
    sy = h / max(1.0, imgsz)

    box = goal_detector.get("pixel_box_5")
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    else:
        cu, cv = float(center[0]), float(center[1])
        x1, y1, x2, y2 = cu - 2.0, cv - 2.0, cu + 2.0, cv + 2.0

    px1 = int(round(x1 * sx))
    py1 = int(round(y1 * sy))
    px2 = int(round(x2 * sx))
    py2 = int(round(y2 * sy))
    px1 = max(0, min(w - 1, px1))
    py1 = max(0, min(h - 1, py1))
    px2 = max(0, min(w - 1, px2))
    py2 = max(0, min(h - 1, py2))

    conf = goal_detector.get("confidence")
    if conf is None:
        conf = goal_detector.get("visibility")

    cv2.rectangle(out, (px1, py1), (px2, py2), (0, 255, 0), 2)
    if conf is not None:
        label = f"YOLO {float(conf):.2f}"
        ty = max(14, py1 - 6)
        cv2.putText(
            out,
            label,
            (px1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def _depth_to_inset_bgr(depth_map: np.ndarray, *, size: int = DEFAULT_DEPTH_IMGSZ) -> np.ndarray | None:
    """Normalized onboard depth (H,W) or (H,W,1) → 128×128 BGR inset matching agent input."""
    import cv2

    arr = np.asarray(depth_map, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    if arr.shape != (size, size):
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    gray = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_depth_inset(
    frame: np.ndarray,
    depth_map: np.ndarray | None,
    *,
    size: int = DEFAULT_DEPTH_IMGSZ,
    margin: int = 8,
) -> np.ndarray:
    """Paste agent depth view (128×128) at the top-right of the video frame."""
    import cv2

    if depth_map is None:
        return frame
    inset = _depth_to_inset_bgr(depth_map, size=size)
    if inset is None:
        return frame

    out = frame.copy()
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)

    h, w = out.shape[:2]
    if size + margin * 2 > w or size + margin * 2 > h:
        return out

    x0 = w - size - margin
    y0 = margin
    cv2.rectangle(out, (x0 - 2, y0 - 2), (x0 + size + 1, y0 + size + 1), (255, 255, 255), 1)
    out[y0 : y0 + size, x0 : x0 + size] = inset
    cv2.putText(
        out,
        "depth",
        (x0, y0 + size + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return out


def apply_frame_overlay(
    frame: np.ndarray,
    task: Any,
    router: dict[str, Any],
    t_sim: float,
    step: int,
    *,
    depth_map: np.ndarray | None = None,
) -> np.ndarray:
    """Draw text HUD + YOLO box + onboard depth inset (top-right, 128×128)."""
    task_ctx = _task_context_from_task(task)
    lines = format_overlay_lines(task_ctx, router, float(t_sim), int(step))
    out = _apply_text_overlay(frame, lines)

    active_key = active_engine_key(str(router.get("route", "")))
    if active_key:
        gd = (router.get(active_key) or {}).get("goal_detector")
        out = _draw_yolo_goal_box(out, gd)
    return apply_depth_inset(out, depth_map)


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
