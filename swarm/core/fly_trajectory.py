"""Save and load fly simulation trajectories for replay."""
from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

TRAJECTORY_VERSION = 1
RUN_FILE_SUFFIX = ".flytraj.json.gz"
_RUN_NAME_RE = re.compile(
    r"^(?P<agent>[\w\-]+)_seed(?P<seed>\d+)_(?P<label>[a-z_]+)_(?P<stamp>\d{8}_\d{6})"
    + re.escape(RUN_FILE_SUFFIX)
    + r"$",
    re.IGNORECASE,
)
_RUN_NAME_RE_LEGACY = re.compile(
    r"^(?P<stamp>\d{8}_\d{6})_seed(?P<seed>\d+)_(?P<label>[a-z_]+)_score(?P<score>[0-9.]+)"
    + re.escape(RUN_FILE_SUFFIX)
    + r"$",
    re.IGNORECASE,
)


def default_fly_repo_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "swarm").is_dir() and (candidate / "scripts" / "fly_model.py").is_file():
            return candidate.resolve()
    return Path.cwd().resolve()


def fly_runs_dir(repo_root: Path | None = None) -> Path:
    root = (repo_root or default_fly_repo_root()).resolve()
    return root / "fly_runs"


def trajectories_dir(repo_root: Path | None = None) -> Path:
    """Directory for saved fly runs (inside the swarm repo)."""
    return fly_runs_dir(repo_root)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _to_float_list(values: Any, size: int) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < size:
        arr = np.pad(arr, (0, size - arr.size))
    return [float(v) for v in arr[:size]]


def sanitize_agent_info(agent_info: dict[str, Any] | None) -> dict[str, Any]:
    if not agent_info:
        return {}
    clean: dict[str, Any] = {}
    for key, value in agent_info.items():
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            clean[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            clean[key] = value.item()
        elif isinstance(value, (bool, int, float, str)):
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = [_json_default(v) if isinstance(v, np.ndarray) else v for v in value]
    return clean


@dataclass(frozen=True)
class SavedRunInfo:
    path: Path
    display_name: str
    agent_name: str | None = None
    seed: int | None = None
    type_label: str | None = None
    score: float | None = None
    success: bool | None = None
    time_sec: float | None = None
    success_term: float | None = None
    time_term: float | None = None
    safety_term: float | None = None
    collision: bool | None = None
    score_summary: str | None = None
    created_at: str | None = None


@dataclass
class FlyTrajectoryFrame:
    t_sim: float
    frame: int
    position: np.ndarray
    quaternion: np.ndarray
    velocity: np.ndarray
    action: np.ndarray
    agent_info: dict[str, Any] = field(default_factory=dict)
    search_area_vector: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "t": float(self.t_sim),
            "frame": int(self.frame),
            "position": _to_float_list(self.position, 3),
            "quaternion": _to_float_list(self.quaternion, 4),
            "velocity": _to_float_list(self.velocity, 3),
            "action": _to_float_list(self.action, 5),
            "agent": sanitize_agent_info(self.agent_info),
        }
        if self.search_area_vector is not None:
            payload["search_area_vector"] = _to_float_list(self.search_area_vector, 3)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlyTrajectoryFrame:
        search = data.get("search_area_vector")
        return cls(
            t_sim=float(data.get("t", 0.0)),
            frame=int(data.get("frame", 0)),
            position=np.asarray(data.get("position", [0.0, 0.0, 0.0]), dtype=float),
            quaternion=np.asarray(data.get("quaternion", [0.0, 0.0, 0.0, 1.0]), dtype=float),
            velocity=np.asarray(data.get("velocity", [0.0, 0.0, 0.0]), dtype=float),
            action=np.asarray(data.get("action", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=float),
            agent_info=dict(data.get("agent") or {}),
            search_area_vector=(
                None if search is None else np.asarray(search, dtype=float)
            ),
        )


@dataclass
class FlyTrajectory:
    meta: dict[str, Any]
    frames: list[FlyTrajectoryFrame] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": TRAJECTORY_VERSION,
            "meta": self.meta,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlyTrajectory:
        version = int(data.get("version", 0))
        if version != TRAJECTORY_VERSION:
            raise ValueError(f"Unsupported trajectory version: {version}")
        frames = [FlyTrajectoryFrame.from_dict(item) for item in data.get("frames", [])]
        return cls(meta=dict(data.get("meta") or {}), frames=frames)

    @property
    def duration_sec(self) -> float:
        if not self.frames:
            return 0.0
        return float(self.frames[-1].t_sim)


def frame_obs_info(frame: FlyTrajectoryFrame) -> dict[str, Any]:
    position = np.asarray(frame.position, dtype=float)
    velocity = np.asarray(frame.velocity, dtype=float)
    if frame.search_area_vector is not None:
        search_area_vector = np.asarray(frame.search_area_vector, dtype=float)
    else:
        search_area_vector = np.zeros(3, dtype=float)
    return {
        "position": position,
        "velocity": velocity,
        "speed_mps": float(np.linalg.norm(velocity)),
        "search_area_vector": search_area_vector,
        "search_area_center": position + search_area_vector,
    }


def build_trajectory_meta(
    *,
    agent_path: Path,
    agent_kind: str,
    seed: int,
    challenge_type: int,
    type_label: str,
    task: Any,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    resolved_agent = agent_path.expanduser().resolve()
    agent_name = agent_name_from_path(resolved_agent)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "agent_path": str(resolved_agent),
        "agent_name": agent_name,
        "agent_kind": agent_kind,
        "seed": int(seed),
        "challenge_type": int(challenge_type),
        "type_label": type_label,
        "start": _to_float_list(getattr(task, "start", (0.0, 0.0, 0.0)), 3),
        "goal": _to_float_list(getattr(task, "goal", (0.0, 0.0, 0.0)), 3),
        "search_radius": float(getattr(task, "search_radius", 0.0)),
        "result": dict(result or {}),
    }


def _sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "agent"


def agent_name_from_path(agent_path: Path | str) -> str:
    path = Path(agent_path).expanduser()
    if path.suffix.lower() == ".zip":
        name = path.stem
    else:
        name = path.name
    return _sanitize_filename_part(name)


def trajectory_filename(meta: dict[str, Any]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    seed = int(meta.get("seed", 0))
    label = _sanitize_filename_part(str(meta.get("type_label", "run")))
    agent_path = meta.get("agent_path")
    agent_name = meta.get("agent_name")
    if not agent_name and agent_path:
        agent_name = agent_name_from_path(str(agent_path))
    agent_name = _sanitize_filename_part(str(agent_name or "agent"))
    return f"{agent_name}_seed{seed}_{label}_{stamp}{RUN_FILE_SUFFIX}"


def run_video_filename(meta: dict[str, Any]) -> str:
    """Video basename matching ``trajectory_filename`` for the same run meta."""
    traj_name = trajectory_filename(meta)
    return traj_name[: -len(RUN_FILE_SUFFIX)] + ".mp4"


def video_path_for_trajectory(trajectory_path: Path) -> Path:
    """Sibling ``.mp4`` path for a saved ``.flytraj.json.gz`` file."""
    path = Path(trajectory_path).expanduser().resolve()
    name = path.name
    if name.endswith(RUN_FILE_SUFFIX):
        return path.with_name(name[: -len(RUN_FILE_SUFFIX)] + ".mp4")
    return path.with_suffix(".mp4")


def default_run_video_path(
    meta: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> Path:
    """Default export path under ``fly_runs/`` using run naming metadata."""
    return fly_runs_dir(repo_root) / run_video_filename(meta)


def format_score_summary(result: dict[str, Any] | None) -> str | None:
    """Compact success/time/safety breakdown for run lists and status lines."""
    if not result:
        return None
    score = result.get("score")
    if score is None:
        return None
    success_term = result.get("success_term")
    time_term = result.get("time_term")
    safety_term = result.get("safety_term")
    if success_term is None and time_term is None and safety_term is None:
        return f"score {float(score):.2f}"
    return (
        f"{float(score):.2f}  "
        f"S:{float(success_term if success_term is not None else 0.0):.2f}  "
        f"T:{float(time_term if time_term is not None else 0.0):.2f}  "
        f"Saf:{float(safety_term if safety_term is not None else 0.0):.2f}"
    )


def format_score_detail_lines(result: dict[str, Any] | None) -> list[str]:
    """Human-readable score breakdown for telemetry and CLI output."""
    if not result:
        return []
    lines: list[str] = []
    score = result.get("score")
    if score is not None:
        lines.append(f"Score {float(score):.3f}  (validator flight_reward)")
    if result.get("success_term") is not None:
        from swarm.constants import REWARD_W_SAFETY, REWARD_W_SUCCESS, REWARD_W_TIME

        lines.append(
            "Terms "
            f"success={float(result['success_term']):.2f}*{REWARD_W_SUCCESS:.2f}  "
            f"time={float(result.get('time_term', 0.0)):.2f}*{REWARD_W_TIME:.2f}  "
            f"safety={float(result.get('safety_term', 0.0)):.2f}*{REWARD_W_SAFETY:.2f}"
        )
    target_time = result.get("target_time_sec")
    clearance = result.get("min_clearance_m")
    time_sec = result.get("time_sec")
    if target_time is not None or clearance is not None or time_sec is not None:
        time_txt = "-" if time_sec is None else f"{float(time_sec):.1f}s"
        target_txt = "-" if target_time is None else f"{float(target_time):.1f}s"
        clearance_txt = "-" if clearance is None else f"{float(clearance):.2f}m"
        lines.append(
            f"Flight time {time_txt}   Target {target_txt}   Min clearance {clearance_txt}"
        )
    return lines


def _peek_trajectory_result(path: Path, *, head_bytes: int = 65536) -> dict[str, Any] | None:
    """Read score fields from trajectory meta without loading frame data."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            chunk = handle.read(head_bytes)
    except OSError:
        return None
    key = '"result"'
    idx = chunk.find(key)
    if idx < 0:
        return None
    start = chunk.find("{", idx)
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(chunk[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(chunk[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return dict(payload) if isinstance(payload, dict) else None
    return None


def _saved_run_with_result(path: Path, run: SavedRunInfo, result: dict[str, Any] | None) -> SavedRunInfo:
    if not result:
        return run
    score = result.get("score")
    summary = format_score_summary(result)
    display = run.display_name
    if summary:
        display = f"{display}  |  {summary}"
    return SavedRunInfo(
        path=run.path,
        display_name=display,
        agent_name=run.agent_name,
        seed=run.seed,
        type_label=run.type_label,
        score=float(score) if score is not None else run.score,
        success=bool(result["success"]) if "success" in result else run.success,
        time_sec=float(result["time_sec"]) if result.get("time_sec") is not None else run.time_sec,
        success_term=(
            float(result["success_term"])
            if result.get("success_term") is not None
            else run.success_term
        ),
        time_term=(
            float(result["time_term"]) if result.get("time_term") is not None else run.time_term
        ),
        safety_term=(
            float(result["safety_term"])
            if result.get("safety_term") is not None
            else run.safety_term
        ),
        collision=bool(result["collision"]) if "collision" in result else run.collision,
        score_summary=summary or run.score_summary,
        created_at=run.created_at,
    )


def _saved_run_from_path(path: Path) -> SavedRunInfo:
    match = _RUN_NAME_RE.match(path.name)
    if match:
        agent = match.group("agent")
        seed = int(match.group("seed"))
        label = match.group("label")
        stamp = match.group("stamp")
        display = f"{agent} seed{seed} {label}"
        created = stamp.replace("_", " ", 1)
        return SavedRunInfo(
            path=path.resolve(),
            display_name=display,
            agent_name=agent,
            seed=seed,
            type_label=label,
            created_at=created,
        )
    legacy = _RUN_NAME_RE_LEGACY.match(path.name)
    if legacy:
        seed = int(legacy.group("seed"))
        label = legacy.group("label")
        score = float(legacy.group("score"))
        stamp = legacy.group("stamp")
        display = f"seed{seed} {label} score={score:.3f}"
        created = stamp.replace("_", " ", 1)
        return SavedRunInfo(
            path=path.resolve(),
            display_name=display,
            seed=seed,
            type_label=label,
            score=score,
            created_at=created,
        )
    return SavedRunInfo(path=path.resolve(), display_name=path.name)


def list_saved_runs(
    repo_root: Path | None = None,
    *,
    limit: int = 50,
) -> list[SavedRunInfo]:
    root = fly_runs_dir(repo_root)
    if not root.is_dir():
        return []
    candidates = sorted(
        root.glob(f"*{RUN_FILE_SUFFIX}"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    runs: list[SavedRunInfo] = []
    for path in candidates[:limit]:
        try:
            run = _saved_run_from_path(path)
            result = _peek_trajectory_result(path)
            runs.append(_saved_run_with_result(path, run, result))
        except OSError:
            continue
    return runs


def save_trajectory(
    trajectory: FlyTrajectory,
    path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> Path:
    if not trajectory.frames:
        raise ValueError("Cannot save an empty trajectory.")
    output = (
        Path(path)
        if path is not None
        else fly_runs_dir(repo_root) / trajectory_filename(trajectory.meta)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(trajectory.to_dict(), indent=2, default=_json_default)
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        handle.write(payload)
    trajectory.meta["path"] = str(output.resolve())
    return output


def load_trajectory(path: Path) -> FlyTrajectory:
    path = Path(path).expanduser()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    trajectory = FlyTrajectory.from_dict(data)
    trajectory.meta.setdefault("path", str(path.resolve()))
    return trajectory


def load_latest_trajectory(repo_root: Path | None = None) -> FlyTrajectory | None:
    for run in list_saved_runs(repo_root, limit=1):
        try:
            return load_trajectory(run.path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return None


def browse_run_file(
    *,
    repo_root: Path | None = None,
    on_before_dialog: Any | None = None,
    on_after_dialog: Any | None = None,
) -> str | None:
    """Open a native picker for a saved fly run."""
    import shutil
    import subprocess
    import sys

    start_dir = fly_runs_dir(repo_root)
    start_dir.mkdir(parents=True, exist_ok=True)
    runs = list_saved_runs(repo_root, limit=40)

    if on_before_dialog is not None:
        on_before_dialog()
    try:
        if sys.platform.startswith("linux"):
            picked = _browse_run_file_zenity_list(runs)
            if picked:
                return picked
            picked = _browse_run_file_zenity_file(start_dir)
            if picked:
                return picked
            picked = _browse_run_file_kdialog(start_dir)
            if picked:
                return picked
        picked = _browse_run_file_tk(start_dir)
        if picked:
            return picked
        if not sys.platform.startswith("linux"):
            picked = _browse_run_file_zenity_list(runs)
            if picked:
                return picked
            picked = _browse_run_file_zenity_file(start_dir)
            if picked:
                return picked
        return None
    finally:
        if on_after_dialog is not None:
            on_after_dialog()


def _browse_run_file_zenity_list(runs: list[SavedRunInfo]) -> str | None:
    import shutil
    import subprocess

    if not runs or not shutil.which("zenity"):
        return None
    args = [
        "zenity",
        "--list",
        "--title=Open saved fly run",
        "--text=Select a saved run:",
        "--column=Run",
        "--column=Path",
        "--print-column=2",
        "--width=760",
        "--height=420",
    ]
    for run in runs:
        args.extend([run.display_name, str(run.path)])
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if result.returncode == 0:
        picked = result.stdout.strip()
        return picked or None
    return None


def _browse_run_file_zenity_file(start_dir: Path) -> str | None:
    import shutil
    import subprocess

    if not shutil.which("zenity"):
        return None
    start_path = start_dir.resolve()
    try:
        result = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--title=Open saved fly run",
                f"--filename={start_path}/",
                f"--file-filter=Fly runs | *{RUN_FILE_SUFFIX}",
                "--file-filter=All files | *",
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


def _browse_run_file_kdialog(start_dir: Path) -> str | None:
    import shutil
    import subprocess

    if not shutil.which("kdialog"):
        return None
    try:
        result = subprocess.run(
            [
                "kdialog",
                "--getopenfilename",
                str(start_dir.resolve()),
                f"*{RUN_FILE_SUFFIX}",
                "Open saved fly run",
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


def _browse_run_file_tk(start_dir: Path) -> str | None:
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

    picked = filedialog.askopenfilename(
        title="Open saved fly run",
        initialdir=str(start_dir.resolve()),
        filetypes=[
            ("Fly runs", f"*{RUN_FILE_SUFFIX}"),
            ("All files", "*.*"),
        ],
        parent=root,
    )
    root.update()
    root.destroy()
    return picked or None
