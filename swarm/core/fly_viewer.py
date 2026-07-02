"""Pygame UI and off-screen cameras for ``swarm fly``."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from swarm.core.fly_setup import (
    MAP_TYPE_CHOICES,
    FlyLaunchConfig,
    load_last_agent_path,
    resolve_agent_path,
    save_last_agent_path,
)
from swarm.core.fly_trajectory import (
    SavedRunInfo,
    browse_run_file,
    list_saved_runs,
)

CAMERA_MODES: tuple[str, ...] = ("chase", "fpv", "top", "overview")
PANEL_WIDTH = 320
BOTTOM_PANEL_HEIGHT = 228
REPLAY_BAR_HEIGHT = 58
REPLAY_BAR_MARGIN = 12
LEFT_PANEL_BOTTOM_PADDING = 24
REPLAY_SPEED_CHOICES: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
# Default fly viewport (16:9). Lower than 960x540 for faster CPU rendering; override with --width/--height.
DEFAULT_VIEW_WIDTH = 640
DEFAULT_VIEW_HEIGHT = 360
VIDEO_FPS = 25
DEPTH_PREVIEW_SIZE = 140
DEPTH_PREVIEW_INSET = 10
DEPTH_PREVIEW_GAP = 6
DEPTH_COLORMAP_NAME = "inferno"
# Validator seeds are drawn from [0, 2**32 - 1] (up to 12 digits).
SEED_TEXT_MAX_LEN = 12

# Match scripts/generate_video.py chase/fpv tuning for stable live preview.
CHASE_DISTANCE_BACK_M = 2.5
CHASE_HEIGHT_ABOVE_M = 1.0
CHASE_SMOOTHING = 0.92
FPV_OFFSET_FORWARD_M = 0.15
FPV_OFFSET_UP_M = 0.02
FPV_SMOOTHING = 0.85
OVERVIEW_ORBIT_DEG_SEC = 5.0

SAVED_RUN_ROW_HEIGHT = 30
SAVED_RUN_VISIBLE_ROWS = 6
SAVED_RUN_SCROLLBAR_WIDTH = 10
MAP_TYPE_BOX_HEIGHT = 26
MAP_TYPE_OPTION_HEIGHT = 24

Y_MAP_LABEL = 202
Y_MAP_BOX = 222
Y_RUNS_LABEL = 258
Y_RUNS_TOP = 276
Y_SIMULATION_LABEL = 464
Y_BUILD = 474
Y_CTRL = 504
Y_REPLAY_ROW = 534
Y_EXPORT = 564
Y_CAMERA_LABEL = 590
Y_CAMERA = 614


def compute_left_panel_min_height() -> int:
    """Minimum left-panel height so every control row is clickable."""
    btn_h = 26
    gap = 6
    zoom_row_bottom = Y_CAMERA + 2 * (btn_h + gap) + btn_h
    return zoom_row_bottom + LEFT_PANEL_BOTTOM_PADDING


LEFT_PANEL_MIN_HEIGHT = compute_left_panel_min_height()


def _fmt_vec(values: Any, precision: int = 2) -> str:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 3:
        return "(?, ?, ?)"
    return (
        f"({arr[0]:.{precision}f}, {arr[1]:.{precision}f}, {arr[2]:.{precision}f})"
    )


def _fmt_bool(value: Any) -> str:
    if value is None:
        return "?"
    return "yes" if bool(value) else "no"


def _goal_detection_lines(
    agent_info: dict[str, Any],
    *,
    obs_info: dict[str, Any] | None = None,
    task: Any | None = None,
) -> list[str]:
    """Build goal-detection rows for the bottom telemetry panel."""
    lines: list[str] = []
    prob = agent_info.get("goal_visibility_prob")
    prob_txt = "-" if prob is None else f"{float(prob):.2f}"

    status_parts = [
        f"locked={_fmt_bool(agent_info.get('goal_detected'))}",
        f"visible={_fmt_bool(agent_info.get('goal_visible'))}",
        f"tracked={_fmt_bool(agent_info.get('goal_tracked'))}",
        f"prob={prob_txt}",
    ]
    lost = agent_info.get("platform_lost_steps")
    if lost is not None:
        status_parts.append(f"lost={int(lost)}")
    dist_buf = agent_info.get("goal_distance_buffer")
    if dist_buf is not None:
        status_parts.append(f"dist_buf={float(dist_buf):.2f}")
    lines.append("Goal detect  " + "  ".join(status_parts))

    predicted = agent_info.get("predicted_goal_position")
    if predicted is None:
        lines.append(
            "Predict pad  (none) — agent has not estimated the landing platform yet"
        )
        return lines

    pred = np.asarray(predicted, dtype=float).reshape(3)
    detail_parts = [f"Predict pad {_fmt_vec(pred)}"]
    if obs_info is not None:
        pos = np.asarray(obs_info["position"], dtype=float).reshape(3)
        detail_parts.append(f"dist={float(np.linalg.norm(pred - pos)):.1f}m")
        search_center = obs_info.get("search_area_center")
        if search_center is not None:
            center = np.asarray(search_center, dtype=float).reshape(3)
            detail_parts.append(f"err_GPS_hint={float(np.linalg.norm(pred - center)):.1f}m")
    if task is not None and getattr(task, "goal", None) is not None:
        true_goal = np.asarray(task.goal, dtype=float).reshape(3)
        detail_parts.append(f"err_true_pad={float(np.linalg.norm(pred - true_goal)):.1f}m")
    lines.append("  ".join(detail_parts))
    return lines


def build_bottom_telemetry_lines(
    *,
    task: Any | None,
    sim_state: str,
    t_sim: float,
    frame: int,
    obs_info: dict[str, Any] | None,
    agent_info: dict[str, Any] | None,
    action: np.ndarray | None,
    camera_mode: str,
    result: dict[str, Any] | None = None,
) -> list[str]:
    lines = [
        (
            f"Time {t_sim:6.2f}s   Frame {frame:5d}   "
            f"Status {sim_state.upper():9s}   Camera {camera_mode}"
        ),
    ]
    if task is not None:
        lines.append(
            f"Mission  start {_fmt_vec(task.start)}   goal {_fmt_vec(task.goal)}   "
            f"radius {float(task.search_radius):.1f}m"
        )
    if obs_info is not None:
        lines.append(
            f"Position {_fmt_vec(obs_info['position'])}   "
            f"Speed {float(obs_info['speed_mps']):.2f} m/s   "
            f"Search {_fmt_vec(obs_info['search_area_vector'])}   "
            f"Center {_fmt_vec(obs_info['search_area_center'])}"
        )
    if agent_info is not None:
        map_pred = agent_info.get("map_prediction")
        lines.append(
            f"Mode {agent_info.get('mode') or '-':12s}   "
            f"map={map_pred if map_pred is not None else '-'}"
        )
        lines.extend(
            _goal_detection_lines(agent_info, obs_info=obs_info, task=task)
        )
    if action is not None:
        act = np.asarray(action, dtype=float).reshape(-1)
        if act.size >= 5:
            lines.append(
                "Action "
                f"[dir_x={act[0]:+.2f}, dir_y={act[1]:+.2f}, dir_z={act[2]:+.2f}, "
                f"speed={act[3]:.2f}, yaw={act[4]:+.2f}]"
            )
    if result is not None:
        lines.append(
            f"Result success={_fmt_bool(result.get('success'))}   "
            f"time={float(result.get('time_sec', 0.0)):.2f}s   "
            f"collision={_fmt_bool(result.get('collision'))}"
        )
        from swarm.core.fly_trajectory import format_score_detail_lines

        lines.extend(format_score_detail_lines(result))
    return lines


def _resolve_browse_start_dir(initial_dir: str | Path | None) -> Path:
    if initial_dir:
        candidate = Path(initial_dir).expanduser()
        if candidate.is_file():
            candidate = candidate.parent
        if candidate.is_dir():
            return candidate.resolve()
    return Path.cwd().resolve()


def browse_agent_directory(
    *,
    initial_dir: str | Path | None = None,
    on_before_dialog: Any | None = None,
    on_after_dialog: Any | None = None,
) -> str | None:
    """Open a native folder picker for an agent source directory."""
    import sys

    start_dir = str(_resolve_browse_start_dir(initial_dir))
    if on_before_dialog is not None:
        on_before_dialog()
    try:
        if sys.platform.startswith("linux"):
            picked = _browse_agent_directory_zenity(start_dir)
            if picked:
                return picked
            picked = _browse_agent_directory_kdialog(start_dir)
            if picked:
                return picked
        picked = _browse_agent_directory_tk(start_dir)
        if picked:
            return picked
        if not sys.platform.startswith("linux"):
            picked = _browse_agent_directory_zenity(start_dir)
            if picked:
                return picked
            return _browse_agent_directory_kdialog(start_dir)
        return None
    finally:
        if on_after_dialog is not None:
            on_after_dialog()


def _browse_agent_directory_tk(start_dir: str) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    root.update()
    try:
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
    except tk.TclError:
        pass

    picked = filedialog.askdirectory(
        title="Select agent source folder",
        initialdir=start_dir,
        parent=root,
        mustexist=True,
    )
    root.update()
    root.destroy()
    return picked or None


def _browse_agent_directory_zenity(start_dir: str) -> str | None:
    import shutil
    import subprocess

    if not shutil.which("zenity"):
        return None
    start_path = Path(start_dir)
    if not start_path.is_dir():
        start_path = Path.cwd()
    try:
        result = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--directory",
                "--title=Select agent source folder",
                f"--filename={start_path.resolve()}/",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode == 0:
        picked = result.stdout.strip()
        return picked or None
    return None


def _browse_agent_directory_kdialog(start_dir: str) -> str | None:
    import shutil
    import subprocess

    if not shutil.which("kdialog"):
        return None
    try:
        result = subprocess.run(
            [
                "kdialog",
                "--getexistingdirectory",
                start_dir,
                "--title",
                "Select agent source folder",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode == 0:
        picked = result.stdout.strip()
        return picked or None
    return None


def _native_picker_available() -> bool:
    import shutil

    try:
        import tkinter  # noqa: F401
    except ImportError:
        pass
    else:
        return True
    return bool(shutil.which("zenity") or shutil.which("kdialog"))


def parse_seed_text(text: str, *, fallback: int = 42) -> int:
    cleaned = text.strip()
    if not cleaned:
        return max(1, int(fallback))
    try:
        return max(1, int(cleaned))
    except ValueError:
        return max(1, int(fallback))


def colourise_depth_normalized(depth: np.ndarray) -> np.ndarray:
    """Map the agent's normalized depth observation to an RGB preview frame."""
    array = np.asarray(depth, dtype=np.float32)
    if array.size == 0:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    plane = np.clip(array.reshape(array.shape[0], array.shape[1]), 0.0, 1.0)
    try:
        import matplotlib.cm as cm

        cmap = cm.colormaps.get_cmap(DEPTH_COLORMAP_NAME)
        rgb = cmap(1.0 - plane)[:, :, :3]
        return (rgb * 255).astype(np.uint8)
    except Exception:
        grey = (255 * (1.0 - plane)).astype(np.uint8)
        return np.stack([grey, grey, grey], axis=-1)


def _drone_basis(quat: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pybullet as p

    rot = np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
    forward = rot @ np.array([1.0, 0.0, 0.0], dtype=float)
    right = rot @ np.array([0.0, 1.0, 0.0], dtype=float)
    up = rot @ np.array([0.0, 0.0, 1.0], dtype=float)
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    up /= max(float(np.linalg.norm(up)), 1e-9)
    return forward, right, up


class FlyRenderCamera:
    """Off-screen camera used by the pygame viewport."""

    def __init__(self, goal: Sequence[float], *, mode: str = "chase") -> None:
        if mode not in CAMERA_MODES:
            raise ValueError(f"Unsupported camera mode: {mode}")
        self.goal = np.asarray(goal, dtype=float)
        self.mode = mode
        self.distance_scale = 1.0
        self._overview_yaw_deg = 0.0
        self._smooth_fwd: np.ndarray | None = None
        self._smooth_up: np.ndarray | None = None
        self._smoothing_mode = mode

    def reset_smoothing(self) -> None:
        self._smooth_fwd = None
        self._smooth_up = None

    def _sync_smoothing_mode(self) -> None:
        if self.mode != self._smoothing_mode:
            self.reset_smoothing()
            self._smoothing_mode = self.mode

    @staticmethod
    def _smooth_vector(
        current: np.ndarray,
        previous: np.ndarray | None,
        alpha: float,
    ) -> np.ndarray:
        if previous is None:
            smoothed = current
        else:
            smoothed = alpha * previous + (1.0 - alpha) * current
        norm = float(np.linalg.norm(smoothed))
        if norm < 1e-9:
            return current
        return smoothed / norm

    def eye_and_target(
        self,
        position: np.ndarray,
        quat: Sequence[float],
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._sync_smoothing_mode()
        forward, _, up = _drone_basis(quat)
        pos = np.asarray(position, dtype=float)

        if self.mode == "chase":
            fwd = self._smooth_vector(forward, self._smooth_fwd, CHASE_SMOOTHING)
            self._smooth_fwd = fwd
            back = CHASE_DISTANCE_BACK_M * self.distance_scale
            height = CHASE_HEIGHT_ABOVE_M * self.distance_scale
            eye = pos - fwd * back + np.array([0.0, 0.0, height])
            target = pos + np.array([0.0, 0.0, 0.15])
        elif self.mode == "fpv":
            fwd = self._smooth_vector(forward, self._smooth_fwd, FPV_SMOOTHING)
            body_up = self._smooth_vector(up, self._smooth_up, FPV_SMOOTHING)
            self._smooth_fwd = fwd
            self._smooth_up = body_up
            offset_fwd = FPV_OFFSET_FORWARD_M * self.distance_scale
            offset_up = FPV_OFFSET_UP_M * self.distance_scale
            eye = pos + fwd * offset_fwd + body_up * offset_up
            target = eye + fwd * 20.0
        elif self.mode == "top":
            eye = pos + np.array([0.0, 0.0, 20.0 * self.distance_scale])
            target = pos
        else:
            midpoint = (pos + self.goal) * 0.5
            span = float(np.linalg.norm(pos - self.goal))
            self._overview_yaw_deg = (
                self._overview_yaw_deg + OVERVIEW_ORBIT_DEG_SEC * dt
            ) % 360.0
            yaw_r = math.radians(self._overview_yaw_deg)
            pitch_r = math.radians(-35.0)
            dist = max(15.0, span * 1.3) * self.distance_scale
            eye = np.array(
                [
                    midpoint[0] + dist * math.cos(yaw_r) * math.cos(pitch_r),
                    midpoint[1] + dist * math.sin(yaw_r) * math.cos(pitch_r),
                    midpoint[2] - dist * math.sin(pitch_r),
                ],
                dtype=float,
            )
            target = midpoint
        return eye.astype(float), target.astype(float)


def render_rgb_frame(
    cli: int,
    *,
    eye: Sequence[float],
    target: Sequence[float],
    width: int,
    height: int,
    fov: float = 65.0,
    far: float = 250.0,
) -> np.ndarray:
    import pybullet as p

    view = p.computeViewMatrix(
        cameraEyePosition=list(eye),
        cameraTargetPosition=list(target),
        cameraUpVector=[0.0, 0.0, 1.0],
        physicsClientId=cli,
    )
    projection = p.computeProjectionMatrixFOV(
        fov=float(fov),
        aspect=width / max(height, 1),
        nearVal=0.05,
        farVal=float(far),
        physicsClientId=cli,
    )
    flags = int(getattr(p, "ER_NO_SEGMENTATION_MASK", 0))
    _, _, rgba, _, _ = p.getCameraImage(
        width=int(width),
        height=int(height),
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        shadow=0,
        lightDirection=[0.4, 0.4, 1.0],
        flags=flags,
        physicsClientId=cli,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


def export_video(frames: list[np.ndarray], output_path: Path, fps: int = VIDEO_FPS) -> Path:
    if not frames:
        raise ValueError("No frames recorded; run the simulation before exporting.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    from scripts.generate_video import _open_video_writer

    writer = _open_video_writer(output_path, fps=fps, width=width, height=height)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()
    return output_path


@dataclass
class FlyUiEvent:
    quit: bool = False
    build_map: bool = False
    start: bool = False
    pause: bool = False
    replay: bool = False
    open_run: bool = False
    load_run_path: str | None = None
    export: bool = False
    replay_seek: int | None = None
    replay_speed: float | None = None
    camera_mode: str | None = None
    zoom_in: bool = False
    zoom_out: bool = False


@dataclass
class ReplayUiState:
    active: bool
    frame: int
    total_frames: int
    t_sim: float
    duration: float
    speed: float


@dataclass(frozen=True)
class _Button:
    key: str
    label: str
    rect: tuple[int, int, int, int]


class FlySimulatorWindow:
    """Single-window fly UI: left config, 3D view, bottom telemetry."""

    def __init__(
        self,
        *,
        title: str,
        view_width: int = DEFAULT_VIEW_WIDTH,
        view_height: int = DEFAULT_VIEW_HEIGHT,
        camera_mode: str = "chase",
        seed: int = 42,
        challenge_type: int = 1,
        last_agent_path: str | Path | None = None,
        custom_path: str = "",
        use_last_agent: bool = True,
        repo_root: Path | None = None,
    ) -> None:
        import pygame

        self._pygame = pygame
        self.view_width = int(view_width)
        self.view_height = int(view_height)
        self.camera_mode = camera_mode
        self.seed = int(seed)
        self.challenge_type = int(challenge_type)
        self.custom_path = custom_path
        self.custom_active = False
        self.seed_active = False
        self.seed_text = str(int(seed))
        self.quit_requested = False
        self.map_loaded = False
        self._status_message = "Select agent, seed, and map type. Then Build Map."
        self._export_enabled = False
        self._replay_enabled = False
        self._sim_state = "config"
        self._camera_label_y = 562
        self._simulation_label_y = 424
        self.repo_root = repo_root
        self.saved_runs: list[SavedRunInfo] = []
        self.run_scroll = 0
        self.selected_run_path: str | None = None
        self.replay_speed = 1.0
        self._replay_ui_active = False
        self._timeline_dragging = False
        self._map_type_open = False
        self._run_scrollbar_dragging = False
        self._buttons: list[_Button] = []
        self.left_panel_height = max(self.view_height, LEFT_PANEL_MIN_HEIGHT)

        loaded_last = load_last_agent_path() if last_agent_path is None else None
        if last_agent_path is not None:
            candidate = Path(str(last_agent_path)).expanduser()
            self.last_agent_path = (
                str(candidate.resolve())
                if resolve_agent_path(candidate) is not None
                else None
            )
        elif loaded_last is not None:
            self.last_agent_path = str(loaded_last)
        else:
            self.last_agent_path = None

        if self.custom_path.strip():
            self.use_last_agent = False
        elif use_last_agent and self.last_agent_path:
            self.use_last_agent = True
        else:
            self.use_last_agent = False

        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode(
            (
                PANEL_WIDTH + self.view_width,
                self.left_panel_height + BOTTOM_PANEL_HEIGHT,
            ),
            pygame.RESIZABLE,
        )
        self._font = pygame.font.SysFont("dejavusans", 14)
        self._font_small = pygame.font.SysFont("dejavusans", 12)
        self._font_title = pygame.font.SysFont("dejavusans", 16, bold=True)
        self.refresh_saved_runs()
        self._layout_buttons()

    def refresh_saved_runs(self) -> None:
        self.saved_runs = list_saved_runs(self.repo_root)
        self.run_scroll = max(
            0,
            min(self.run_scroll, self._saved_runs_max_scroll()),
        )

    def apply_loaded_trajectory(self, trajectory: Any) -> None:
        """Sync launch controls from a loaded trajectory and close dropdowns."""
        meta = getattr(trajectory, "meta", None) or {}
        if "seed" in meta:
            self.seed = max(1, int(meta["seed"]))
            self.seed_text = str(self.seed)
        if "challenge_type" in meta:
            self.challenge_type = int(meta["challenge_type"])
        self._map_type_open = False
        self.seed_active = False
        self.custom_active = False

    def _saved_runs_max_scroll(self) -> int:
        return max(0, len(self.saved_runs) - SAVED_RUN_VISIBLE_ROWS)

    def _map_type_label(self, challenge_type: int) -> str:
        for value, label in MAP_TYPE_CHOICES:
            if value == challenge_type:
                return label
        return str(challenge_type)

    def _map_type_box_rect(self) -> tuple[int, int, int, int]:
        return (10, Y_MAP_BOX, PANEL_WIDTH - 20, MAP_TYPE_BOX_HEIGHT)

    def _map_type_option_rect(self, option_index: int) -> tuple[int, int, int, int]:
        box = self._map_type_box_rect()
        return (
            box[0],
            box[1] + box[3] + 2 + option_index * MAP_TYPE_OPTION_HEIGHT,
            box[2],
            MAP_TYPE_OPTION_HEIGHT,
        )

    def _saved_runs_list_rect(self) -> tuple[int, int, int, int]:
        height = SAVED_RUN_VISIBLE_ROWS * SAVED_RUN_ROW_HEIGHT
        width = PANEL_WIDTH - 20 - SAVED_RUN_SCROLLBAR_WIDTH - 4
        return (10, Y_RUNS_TOP, width, height)

    def _saved_runs_scrollbar_track_rect(self) -> tuple[int, int, int, int]:
        list_rect = self._saved_runs_list_rect()
        return (
            list_rect[0] + list_rect[2] + 4,
            list_rect[1],
            SAVED_RUN_SCROLLBAR_WIDTH,
            list_rect[3],
        )

    def _saved_runs_scrollbar_thumb_rect(self) -> tuple[int, int, int, int]:
        track = self._saved_runs_scrollbar_track_rect()
        total = len(self.saved_runs)
        if total <= SAVED_RUN_VISIBLE_ROWS:
            return track
        max_scroll = self._saved_runs_max_scroll()
        thumb_h = max(24, int(track[3] * SAVED_RUN_VISIBLE_ROWS / total))
        usable = max(1, track[3] - thumb_h)
        thumb_y = track[1] + int(usable * self.run_scroll / max_scroll)
        return (track[0], thumb_y, track[2], thumb_h)

    def _scroll_from_scrollbar_y(self, y: int) -> int:
        track = self._saved_runs_scrollbar_track_rect()
        thumb = self._saved_runs_scrollbar_thumb_rect()
        max_scroll = self._saved_runs_max_scroll()
        if max_scroll <= 0:
            return 0
        usable = max(1, track[3] - thumb[3])
        ratio = (y - track[1] - thumb[3] / 2) / usable
        ratio = max(0.0, min(1.0, ratio))
        return int(round(ratio * max_scroll))

    def set_replay_ui_active(self, active: bool) -> None:
        self._replay_ui_active = bool(active)

    def _layout_buttons(self) -> None:
        x0 = 10
        w_full = PANEL_WIDTH - 20
        w_half = (w_full - 6) // 2
        gap = 6
        btn_h = 26

        self._camera_label_y = Y_CAMERA_LABEL
        self._runs_label_y = Y_RUNS_LABEL
        self._run_rows_top = Y_RUNS_TOP
        self._simulation_label_y = Y_SIMULATION_LABEL

        self._buttons = [
            _Button("open_dir", "Open Dir...", (x0, 128, 92, 28)),
            _Button("build_map", "Build Map", (x0, Y_BUILD, w_full, btn_h)),
            _Button("start", "Start", (x0, Y_CTRL, w_half, btn_h)),
            _Button("pause", "Pause", (x0 + w_half + gap, Y_CTRL, w_half, btn_h)),
            _Button("replay", "Replay", (x0, Y_REPLAY_ROW, w_half, btn_h)),
            _Button("open_run", "Open Run", (x0 + w_half + gap, Y_REPLAY_ROW, w_half, btn_h)),
            _Button("export", "Export", (x0, Y_EXPORT, w_full, btn_h)),
            _Button("cam_chase", "Chase", (x0, Y_CAMERA, w_half, btn_h)),
            _Button("cam_fpv", "FPV", (x0 + w_half + gap, Y_CAMERA, w_half, btn_h)),
            _Button("cam_top", "Top", (x0, Y_CAMERA + btn_h + gap, w_half, btn_h)),
            _Button(
                "cam_overview",
                "Overview",
                (x0 + w_half + gap, Y_CAMERA + btn_h + gap, w_half, btn_h),
            ),
            _Button(
                "zoom_in",
                "Zoom +",
                (x0, Y_CAMERA + 2 * (btn_h + gap), w_half, btn_h),
            ),
            _Button(
                "zoom_out",
                "Zoom -",
                (x0 + w_half + gap, Y_CAMERA + 2 * (btn_h + gap), w_half, btn_h),
            ),
        ]

    def set_status(self, message: str) -> None:
        self._status_message = message

    def set_export_enabled(self, enabled: bool) -> None:
        self._export_enabled = bool(enabled)

    def set_replay_enabled(self, enabled: bool) -> None:
        self._replay_enabled = bool(enabled)

    def set_map_loaded(self, loaded: bool) -> None:
        self.map_loaded = bool(loaded)

    def set_sim_state(self, state: str) -> None:
        self._sim_state = str(state)

    def _last_agent_row_rect(self) -> tuple[int, int, int, int]:
        return (10, 76, PANEL_WIDTH - 20, 28)

    def _custom_field_rect(self) -> tuple[int, int, int, int]:
        return (108, 128, PANEL_WIDTH - 118, 28)

    def _seed_field_rect(self) -> tuple[int, int, int, int]:
        return (56, 168, PANEL_WIDTH - 66, 26)

    def _run_row_rect(self, row_index: int) -> tuple[int, int, int, int]:
        list_rect = self._saved_runs_list_rect()
        return (
            list_rect[0],
            list_rect[1] + row_index * SAVED_RUN_ROW_HEIGHT,
            list_rect[2],
            SAVED_RUN_ROW_HEIGHT - 2,
        )

    def _replay_bar_top(self) -> int:
        return self.left_panel_height - REPLAY_BAR_HEIGHT

    def _replay_timeline_rect(self) -> tuple[int, int, int, int]:
        top = self._replay_bar_top()
        x0 = PANEL_WIDTH + REPLAY_BAR_MARGIN
        width = max(120, self.view_width - 2 * REPLAY_BAR_MARGIN)
        return (x0, top + 20, width, 14)

    def _replay_speed_rect(self, speed: float) -> tuple[int, int, int, int]:
        top = self._replay_bar_top()
        idx = REPLAY_SPEED_CHOICES.index(speed) if speed in REPLAY_SPEED_CHOICES else 1
        x0 = PANEL_WIDTH + REPLAY_BAR_MARGIN + idx * 58
        return (x0, top + 38, 52, 18)

    def _replay_label_pos(self) -> tuple[int, int]:
        return (PANEL_WIDTH + REPLAY_BAR_MARGIN, self._replay_bar_top() + 4)

    def _pick_run_file(self) -> str | None:
        return browse_run_file(
            repo_root=self.repo_root,
            on_before_dialog=self._before_native_dialog,
            on_after_dialog=self._after_native_dialog,
        )

    def _timeline_frame_at_pos(self, pos: tuple[int, int], total_frames: int) -> int | None:
        if total_frames <= 1:
            return 0
        rect = self._replay_timeline_rect()
        bx, by, bw, bh = rect
        x, y = pos
        if not (bx <= x <= bx + bw and by <= y <= by + bh):
            return None
        ratio = (x - bx) / max(bw, 1)
        return int(round(ratio * (total_frames - 1)))

    def _hit_test_replay(self, pos: tuple[int, int]) -> str | None:
        if not self._replay_ui_active:
            return None
        x, y = pos
        if x < PANEL_WIDTH:
            return None
        rect = self._replay_timeline_rect()
        bx, by, bw, bh = rect
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return "replay_timeline"
        for speed in REPLAY_SPEED_CHOICES:
            sx, sy, sw, sh = self._replay_speed_rect(speed)
            if sx <= x <= sx + sw and sy <= y <= sy + sh:
                return f"replay_speed_{speed:g}"
        return None

    def _draw_replay_overlay(self, replay_ui: ReplayUiState) -> None:
        pygame = self._pygame
        top = self._replay_bar_top()
        overlay = pygame.Surface((self.view_width, REPLAY_BAR_HEIGHT), pygame.SRCALPHA)
        overlay.fill((12, 12, 16, 210))
        self.screen.blit(overlay, (PANEL_WIDTH, top))

        timeline_label = (
            f"Replay  {replay_ui.t_sim:5.1f}s / {replay_ui.duration:5.1f}s  "
            f"frame {replay_ui.frame}/{max(replay_ui.total_frames - 1, 0)}"
        )
        self.screen.blit(
            self._font_small.render(timeline_label, True, (220, 220, 225)),
            self._replay_label_pos(),
        )

        timeline = self._replay_timeline_rect()
        tx, ty, tw, th = timeline
        pygame.draw.rect(self.screen, (40, 40, 48), timeline, border_radius=4)
        if replay_ui.total_frames > 1:
            progress = replay_ui.frame / max(replay_ui.total_frames - 1, 1)
            fill_w = max(2, int(tw * progress))
            pygame.draw.rect(
                self.screen,
                (46, 125, 50),
                (tx, ty, fill_w, th),
                border_radius=4,
            )
        pygame.draw.rect(self.screen, (90, 90, 95), timeline, width=1, border_radius=4)

        for speed in REPLAY_SPEED_CHOICES:
            rect = self._replay_speed_rect(speed)
            active = abs(replay_ui.speed - speed) < 1e-6
            pygame.draw.rect(
                self.screen,
                (46, 125, 50) if active else (55, 55, 60),
                rect,
                border_radius=4,
            )
            pygame.draw.rect(self.screen, (90, 90, 95), rect, width=1, border_radius=4)
            label = self._font_small.render(f"{speed:g}x", True, (230, 230, 230))
            self.screen.blit(
                label,
                label.get_rect(center=(rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)),
            )

    def remember_agent(self, path: Path) -> None:
        resolved = save_last_agent_path(path)
        self.last_agent_path = str(resolved)
        self.use_last_agent = True
        self.custom_path = ""

    def _active_agent_display_path(self) -> str:
        if self.custom_path.strip():
            return self.custom_path.strip()
        if self.use_last_agent and self.last_agent_path:
            return self.last_agent_path
        return ""

    def _sync_seed_text(self) -> None:
        self.seed_text = str(int(self.seed))

    def _commit_seed_text(self) -> None:
        self.seed = parse_seed_text(self.seed_text, fallback=self.seed)
        self._sync_seed_text()

    def _before_native_dialog(self) -> None:
        pygame = self._pygame
        pygame.event.pump()
        try:
            pygame.display.iconify()
        except Exception:
            pass

    def _after_native_dialog(self) -> None:
        pygame = self._pygame
        try:
            self.screen = pygame.display.set_mode(
                (
                    PANEL_WIDTH + self.view_width,
                    self.left_panel_height + BOTTOM_PANEL_HEIGHT,
                ),
                pygame.RESIZABLE,
            )
        except Exception:
            pass

    def _pick_agent_directory(self) -> str | None:
        initial = self.custom_path.strip() or str(Path.cwd())
        return browse_agent_directory(
            initial_dir=initial,
            on_before_dialog=self._before_native_dialog,
            on_after_dialog=self._after_native_dialog,
        )

    def _deactivate_inputs(self) -> None:
        if self.seed_active:
            self._commit_seed_text()
        self.seed_active = False
        self.custom_active = False
        self._map_type_open = False

    def _resolve_agent_path(self) -> tuple[Path, str] | None:
        active = self._active_agent_display_path()
        if not active:
            return None
        return resolve_agent_path(Path(active).expanduser())

    def get_launch_config(self) -> FlyLaunchConfig | None:
        if self.seed_active:
            self._commit_seed_text()
            self.seed_active = False
        resolved = self._resolve_agent_path()
        if resolved is None:
            return None
        path, kind = resolved
        return FlyLaunchConfig(
            agent_path=path,
            agent_kind=kind,  # type: ignore[arg-type]
            seed=max(1, int(self.seed)),
            challenge_type=int(self.challenge_type),
        )

    def _hit_test(self, pos: tuple[int, int]) -> str | None:
        x, y = pos
        if x < PANEL_WIDTH:
            if self._map_type_open:
                for idx, (challenge_type, _label) in enumerate(MAP_TYPE_CHOICES):
                    rect = self._map_type_option_rect(idx)
                    bx, by, bw, bh = rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        return f"map_type_option_{challenge_type}"
            for button in self._buttons:
                bx, by, bw, bh = button.rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    return button.key
            map_box = self._map_type_box_rect()
            bx, by, bw, bh = map_box
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return "map_type_box"
            rect = self._last_agent_row_rect()
            bx, by, bw, bh = rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return "last_agent"
            field = self._custom_field_rect()
            bx, by, bw, bh = field
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return "custom_field"
            seed_field = self._seed_field_rect()
            bx, by, bw, bh = seed_field
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return "seed_field"
            track = self._saved_runs_scrollbar_track_rect()
            bx, by, bw, bh = track
            if (
                bx <= x <= bx + bw
                and by <= y <= by + bh
                and len(self.saved_runs) > SAVED_RUN_VISIBLE_ROWS
            ):
                return "run_scrollbar"
            for row in range(SAVED_RUN_VISIBLE_ROWS):
                rect = self._run_row_rect(row)
                bx, by, bw, bh = rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    return f"run_{self.run_scroll + row}"
        replay_key = self._hit_test_replay(pos)
        if replay_key is not None:
            return replay_key
        return None

    def pump(self) -> FlyUiEvent:
        pygame = self._pygame
        event = FlyUiEvent()
        for pg_event in pygame.event.get():
            if pg_event.type == pygame.QUIT:
                self.quit_requested = True
                event.quit = True
            elif pg_event.type == pygame.KEYDOWN:
                if pg_event.key == pygame.K_ESCAPE:
                    if self.seed_active or self.custom_active:
                        self._deactivate_inputs()
                    else:
                        self.quit_requested = True
                        event.quit = True
                elif self.seed_active:
                    if pg_event.key == pygame.K_BACKSPACE:
                        self.seed_text = self.seed_text[:-1]
                    elif pg_event.key == pygame.K_RETURN:
                        self._commit_seed_text()
                        self.seed_active = False
                    elif pg_event.unicode and pg_event.unicode.isdigit():
                        if len(self.seed_text) < SEED_TEXT_MAX_LEN:
                            self.seed_text += pg_event.unicode
                elif self.custom_active:
                    if pg_event.key == pygame.K_BACKSPACE:
                        self.custom_path = self.custom_path[:-1]
                    elif pg_event.key == pygame.K_RETURN:
                        self.custom_active = False
                        event.build_map = True
                    elif pg_event.unicode and pg_event.unicode.isprintable():
                        self.custom_path += pg_event.unicode
            elif pg_event.type == pygame.MOUSEBUTTONDOWN:
                if pg_event.button == 1:
                    key = self._hit_test(pg_event.pos)
                    if key == "custom_field":
                        if self.seed_active:
                            self._commit_seed_text()
                        self.seed_active = False
                        self._map_type_open = False
                        self.custom_active = True
                        self.use_last_agent = False
                    elif key == "seed_field":
                        if self.custom_active:
                            self.custom_active = False
                        self._map_type_open = False
                        self.seed_active = True
                    elif key == "map_type_box":
                        if self.seed_active:
                            self._commit_seed_text()
                        self.seed_active = False
                        self.custom_active = False
                        self._map_type_open = not self._map_type_open
                    elif key and key.startswith("map_type_option_"):
                        self.challenge_type = int(key.split("_", 3)[3])
                        self._map_type_open = False
                    elif key == "run_scrollbar":
                        self._run_scrollbar_dragging = True
                        self.run_scroll = self._scroll_from_scrollbar_y(pg_event.pos[1])
                    elif key == "last_agent" and self.last_agent_path:
                        self._deactivate_inputs()
                        self.use_last_agent = True
                        self.custom_path = ""
                        self.set_status("Using last run agent.")
                    else:
                        self._deactivate_inputs()
                    if key == "build_map":
                        event.build_map = True
                    elif key == "open_dir":
                        self._deactivate_inputs()
                        picked = self._pick_agent_directory()
                        if picked:
                            self.custom_path = picked
                            self.use_last_agent = False
                            self.set_status(f"Selected: {Path(picked).name}")
                        elif _native_picker_available():
                            self.set_status("Folder selection cancelled.")
                        else:
                            self.set_status(
                                "Folder picker unavailable. Install python3-tk or zenity."
                            )
                    elif key == "start":
                        event.start = True
                    elif key == "pause":
                        event.pause = True
                    elif key == "replay" and self._replay_enabled:
                        event.replay = True
                    elif key == "open_run":
                        self._deactivate_inputs()
                        self.refresh_saved_runs()
                        picked = self._pick_run_file()
                        if picked:
                            event.load_run_path = picked
                            self.selected_run_path = picked
                            self.set_status(f"Loaded run: {Path(picked).name}")
                        else:
                            self.set_status("Run selection cancelled.")
                    elif key and key.startswith("run_"):
                        suffix = key.split("_", 1)[1]
                        if suffix.isdigit():
                            run_index = int(suffix)
                            if 0 <= run_index < len(self.saved_runs):
                                picked = str(self.saved_runs[run_index].path)
                                event.load_run_path = picked
                                self.selected_run_path = picked
                                self.set_status(
                                    f"Loaded run: {self.saved_runs[run_index].display_name}"
                                )
                    elif key == "replay_timeline":
                        self._timeline_dragging = True
                        if self._replay_ui_active:
                            total = getattr(self, "_replay_total_frames", 1)
                            seek = self._timeline_frame_at_pos(pg_event.pos, total)
                            if seek is not None:
                                event.replay_seek = seek
                    elif key and key.startswith("replay_speed_"):
                        event.replay_speed = float(key.split("_", 2)[2])
                    elif key == "export" and self._export_enabled:
                        event.export = True
                    elif key == "cam_chase":
                        event.camera_mode = "chase"
                    elif key == "cam_fpv":
                        event.camera_mode = "fpv"
                    elif key == "cam_top":
                        event.camera_mode = "top"
                    elif key == "cam_overview":
                        event.camera_mode = "overview"
                    elif key == "zoom_in":
                        event.zoom_in = True
                    elif key == "zoom_out":
                        event.zoom_out = True
                elif pg_event.button == 4 and pg_event.pos[0] < PANEL_WIDTH:
                    list_rect = self._saved_runs_list_rect()
                    if (
                        list_rect[1] <= pg_event.pos[1] <= list_rect[1] + list_rect[3]
                        and self._saved_runs_max_scroll() > 0
                    ):
                        self.run_scroll = max(0, self.run_scroll - 1)
                elif pg_event.button == 5 and pg_event.pos[0] < PANEL_WIDTH:
                    list_rect = self._saved_runs_list_rect()
                    if (
                        list_rect[1] <= pg_event.pos[1] <= list_rect[1] + list_rect[3]
                        and self._saved_runs_max_scroll() > 0
                    ):
                        self.run_scroll = min(
                            self._saved_runs_max_scroll(),
                            self.run_scroll + 1,
                        )
            elif pg_event.type == pygame.MOUSEBUTTONUP:
                if pg_event.button == 1:
                    self._timeline_dragging = False
                    self._run_scrollbar_dragging = False
            elif pg_event.type == pygame.MOUSEMOTION:
                if self._run_scrollbar_dragging:
                    self.run_scroll = self._scroll_from_scrollbar_y(pg_event.pos[1])
                elif self._timeline_dragging and self._replay_ui_active:
                    total = getattr(self, "_replay_total_frames", 1)
                    seek = self._timeline_frame_at_pos(pg_event.pos, total)
                    if seek is not None:
                        event.replay_seek = seek
        return event

    def _draw_button(self, button: _Button, *, active: bool = False, enabled: bool = True) -> None:
        pygame = self._pygame
        x, y, w, h = button.rect
        if not enabled:
            color = (70, 70, 70)
            text_color = (160, 160, 160)
        elif active:
            color = (46, 125, 50)
            text_color = (255, 255, 255)
        else:
            color = (55, 55, 60)
            text_color = (230, 230, 230)
        pygame.draw.rect(self.screen, color, (x, y, w, h), border_radius=5)
        pygame.draw.rect(self.screen, (90, 90, 95), (x, y, w, h), width=1, border_radius=5)
        label = self._font_small.render(button.label, True, text_color)
        self.screen.blit(label, label.get_rect(center=(x + w // 2, y + h // 2)))

    def _draw_left_panel(self) -> None:
        pygame = self._pygame
        panel_h = self.left_panel_height
        pygame.draw.rect(self.screen, (18, 18, 22), (0, 0, PANEL_WIDTH, panel_h))
        self.screen.blit(self._font_title.render("Swarm Fly", True, (240, 240, 240)), (12, 10))
        status = self._status_message
        if len(status) > 42:
            status = status[:39] + "..."
        self.screen.blit(
            self._font_small.render(status, True, (170, 200, 255)),
            (12, 32),
        )
        self.screen.blit(self._font.render("Last run agent", True, (210, 210, 210)), (12, 56))
        last_rect = self._last_agent_row_rect()
        has_last = bool(self.last_agent_path)
        last_selected = self.use_last_agent and has_last and not self.custom_path.strip()
        pygame.draw.rect(
            self.screen,
            (40, 70, 45) if last_selected else (34, 34, 38),
            last_rect,
            border_radius=4,
        )
        if has_last:
            display = self.last_agent_path or ""
            if len(display) > 40:
                display = "..." + display[-37:]
            last_label = self._font_small.render(display, True, (230, 230, 230))
        else:
            last_label = self._font_small.render("No previous agent", True, (140, 140, 145))
        self.screen.blit(last_label, (last_rect[0] + 6, last_rect[1] + 6))

        self.screen.blit(
            self._font_small.render("Path:", True, (180, 180, 180)),
            (108, 112),
        )
        field = self._custom_field_rect()
        pygame.draw.rect(
            self.screen,
            (48, 48, 58) if self.custom_active else (34, 34, 38),
            field,
            border_radius=4,
        )
        display_path = self.custom_path
        if len(display_path) > 34:
            display_path = "..." + display_path[-31:]
        text = display_path + ("|" if self.custom_active else "")
        self.screen.blit(
            self._font_small.render(text or "type path...", True, (220, 220, 220)),
            (field[0] + 6, field[1] + 6),
        )

        self.screen.blit(self._font.render("Seed", True, (210, 210, 210)), (12, 172))
        seed_field = self._seed_field_rect()
        pygame.draw.rect(
            self.screen,
            (48, 48, 58) if self.seed_active else (34, 34, 38),
            seed_field,
            border_radius=4,
        )
        seed_display = self.seed_text + ("|" if self.seed_active else "")
        if not seed_display.strip("|"):
            seed_display = "|" if self.seed_active else str(self.seed)
        self.screen.blit(
            self._font_small.render(seed_display, True, (220, 220, 220)),
            (seed_field[0] + 6, seed_field[1] + 5),
        )
        self.screen.blit(self._font.render("Map type", True, (210, 210, 210)), (12, Y_MAP_LABEL))
        map_box = self._map_type_box_rect()
        pygame.draw.rect(
            self.screen,
            (48, 48, 58) if self._map_type_open else (34, 34, 38),
            map_box,
            border_radius=4,
        )
        pygame.draw.rect(self.screen, (90, 90, 95), map_box, width=1, border_radius=4)
        map_label = self._map_type_label(self.challenge_type)
        self.screen.blit(
            self._font_small.render(map_label, True, (230, 230, 230)),
            (map_box[0] + 8, map_box[1] + 6),
        )
        arrow = self._font_small.render("v" if self._map_type_open else ">", True, (180, 180, 185))
        self.screen.blit(arrow, (map_box[0] + map_box[2] - 16, map_box[1] + 6))

        self.screen.blit(self._font.render("Saved runs", True, (210, 210, 210)), (12, self._runs_label_y))
        list_rect = self._saved_runs_list_rect()
        pygame.draw.rect(
            self.screen,
            (28, 28, 32),
            list_rect,
            border_radius=4,
        )
        pygame.draw.rect(self.screen, (55, 55, 60), list_rect, width=1, border_radius=4)
        for row in range(SAVED_RUN_VISIBLE_ROWS):
            index = self.run_scroll + row
            if index >= len(self.saved_runs):
                break
            run = self.saved_runs[index]
            rect = self._run_row_rect(row)
            selected = self.selected_run_path == str(run.path)
            pygame.draw.rect(
                self.screen,
                (40, 70, 45) if selected else (34, 34, 38),
                rect,
                border_radius=4,
            )
            title = run.display_name.split("  |  ", 1)[0]
            if len(title) > 34:
                title = title[:31] + "..."
            self.screen.blit(
                self._font_small.render(title, True, (230, 230, 230)),
                (rect[0] + 6, rect[1] + 3),
            )
            score_line = run.score_summary
            if score_line is None and "  |  " in run.display_name:
                score_line = run.display_name.split("  |  ", 1)[1]
            if score_line:
                score_text = score_line
                if len(score_text) > 40:
                    score_text = score_text[:37] + "..."
                self.screen.blit(
                    self._font_small.render(score_text, True, (170, 210, 170)),
                    (rect[0] + 6, rect[1] + 15),
                )

        if len(self.saved_runs) > SAVED_RUN_VISIBLE_ROWS:
            track = self._saved_runs_scrollbar_track_rect()
            thumb = self._saved_runs_scrollbar_thumb_rect()
            pygame.draw.rect(self.screen, (40, 40, 48), track, border_radius=4)
            pygame.draw.rect(self.screen, (90, 90, 95), track, width=1, border_radius=4)
            pygame.draw.rect(self.screen, (110, 110, 118), thumb, border_radius=4)
            pygame.draw.rect(self.screen, (140, 140, 148), thumb, width=1, border_radius=4)

        if self._map_type_open:
            for idx, (challenge_type, label) in enumerate(MAP_TYPE_CHOICES):
                option_rect = self._map_type_option_rect(idx)
                selected = challenge_type == self.challenge_type
                pygame.draw.rect(
                    self.screen,
                    (46, 125, 50) if selected else (42, 42, 48),
                    option_rect,
                    border_radius=4,
                )
                pygame.draw.rect(
                    self.screen,
                    (90, 90, 95),
                    option_rect,
                    width=1,
                    border_radius=4,
                )
                self.screen.blit(
                    self._font_small.render(label, True, (230, 230, 230)),
                    (option_rect[0] + 8, option_rect[1] + 5),
                )

        self.screen.blit(self._font.render("Simulation", True, (210, 210, 210)), (12, self._simulation_label_y))
        self.screen.blit(self._font.render("Camera", True, (210, 210, 210)), (12, self._camera_label_y))

        for button in self._buttons:
            active = False
            enabled = True
            if button.key == "export":
                enabled = self._export_enabled
            elif button.key == "replay":
                enabled = self._replay_enabled
            elif button.key == "build_map":
                enabled = True
            elif button.key == "start":
                enabled = self.map_loaded and self._sim_state not in {"building", "config"}
                active = self._sim_state in {"running", "replay"}
            elif button.key == "pause":
                enabled = self.map_loaded and self._sim_state in {
                    "running",
                    "paused",
                    "replay",
                    "replay_paused",
                }
                active = self._sim_state in {"paused", "replay_paused"}
            elif button.key == f"cam_{self.camera_mode}":
                active = True
            self._draw_button(button, active=active, enabled=enabled)

    def _blit_camera_panel(
        self,
        frame_rgb: np.ndarray,
        *,
        x: int,
        y: int,
        size: int,
        label: str,
    ) -> None:
        pygame = self._pygame
        frame = np.asarray(frame_rgb, dtype=np.uint8)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        frame = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8).copy()
        surface = pygame.image.frombuffer(
            frame.tobytes(),
            (frame.shape[1], frame.shape[0]),
            "RGB",
        )
        if surface.get_width() != size or surface.get_height() != size:
            surface = pygame.transform.scale(surface, (size, size))

        backing = pygame.Surface((size + 8, size + 22), pygame.SRCALPHA)
        backing.fill((12, 12, 16, 210))
        self.screen.blit(backing, (x - 4, y - 18))
        self.screen.blit(
            self._font_small.render(label, True, (220, 220, 225)),
            (x, y - 16),
        )
        pygame.draw.rect(
            self.screen,
            (90, 90, 95),
            (x - 1, y - 1, size + 2, size + 2),
            width=1,
            border_radius=4,
        )
        self.screen.blit(surface, (x, y))

    def _draw_drone_camera_panels(
        self,
        *,
        drone_cam_rgb: np.ndarray | None,
        depth_rgb: np.ndarray | None,
    ) -> None:
        size = DEPTH_PREVIEW_SIZE
        inset = DEPTH_PREVIEW_INSET
        x = PANEL_WIDTH + self.view_width - size - inset
        y = inset
        if drone_cam_rgb is not None:
            self._blit_camera_panel(
                drone_cam_rgb,
                x=x,
                y=y,
                size=size,
                label="Drone cam",
            )
            y += size + DEPTH_PREVIEW_GAP
        if depth_rgb is not None:
            self._blit_camera_panel(
                depth_rgb,
                x=x,
                y=y,
                size=size,
                label="Depth",
            )

    def draw(
        self,
        frame_rgb: np.ndarray | None,
        bottom_lines: list[str],
        *,
        placeholder_text: str | None = None,
        replay_ui: ReplayUiState | None = None,
        drone_cam_rgb: np.ndarray | None = None,
        depth_rgb: np.ndarray | None = None,
    ) -> None:
        pygame = self._pygame
        self.screen.fill((24, 24, 28))
        self._draw_left_panel()

        view_rect = pygame.Rect(PANEL_WIDTH, 0, self.view_width, self.left_panel_height)
        pygame.draw.rect(self.screen, (30, 30, 35), view_rect)
        if frame_rgb is not None:
            frame_surface = pygame.image.frombuffer(
                np.asarray(frame_rgb, dtype=np.uint8).tobytes(),
                (frame_rgb.shape[1], frame_rgb.shape[0]),
                "RGB",
            )
            self.screen.blit(frame_surface, (PANEL_WIDTH, 0))
        elif placeholder_text:
            hint = self._font.render(placeholder_text, True, (180, 180, 190))
            self.screen.blit(
                hint,
                hint.get_rect(
                    center=(PANEL_WIDTH + self.view_width // 2, self.left_panel_height // 2),
                ),
            )

        if replay_ui is not None and replay_ui.active:
            self._replay_total_frames = max(1, int(replay_ui.total_frames))
            self._replay_ui_active = True
            self._draw_replay_overlay(replay_ui)
        else:
            self._replay_ui_active = False
            self._replay_total_frames = 1

        if frame_rgb is not None and (
            drone_cam_rgb is not None or depth_rgb is not None
        ):
            self._draw_drone_camera_panels(
                drone_cam_rgb=drone_cam_rgb,
                depth_rgb=depth_rgb,
            )

        pygame.draw.line(
            self.screen,
            (70, 70, 75),
            (PANEL_WIDTH, 0),
            (PANEL_WIDTH, self.left_panel_height),
            1,
        )
        pygame.draw.line(
            self.screen,
            (70, 70, 75),
            (0, self.left_panel_height),
            (PANEL_WIDTH + self.view_width, self.left_panel_height),
            1,
        )

        bottom_y = self.left_panel_height
        pygame.draw.rect(
            self.screen,
            (16, 16, 20),
            (0, bottom_y, PANEL_WIDTH + self.view_width, BOTTOM_PANEL_HEIGHT),
        )
        self.screen.blit(
            self._font_title.render("Drone Telemetry", True, (220, 220, 220)),
            (12, bottom_y + 8),
        )
        line_y = bottom_y + 32
        for line in bottom_lines[:11]:
            self.screen.blit(self._font_small.render(line, True, (205, 205, 205)), (12, line_y))
            line_y += 18

        pygame.display.flip()

    def close(self) -> None:
        try:
            self._pygame.quit()
        except Exception:
            pass


FlyCameraController = FlyRenderCamera
CAMERA_HELP = "Use the left panel controls."

def build_hud_lines(**kwargs: Any) -> list[str]:
    return build_bottom_telemetry_lines(**kwargs)
