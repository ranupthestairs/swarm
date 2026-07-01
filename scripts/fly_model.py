#!/usr/bin/env python3
"""
Real-time Swarm fly viewer with a pygame control panel.

Examples
--------
    python scripts/fly_model.py --source my_agent/ --seed 1001 --type 1
    python scripts/fly_model.py --model champion_UID_42.zip --seed 1001 --type 1
"""
from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gym_pybullet_drones.utils.enums import ActionType

from scripts.generate_video import _extract_zip, _load_agent
from swarm.constants import SIM_DT, SPEED_LIMIT
from swarm.core.fly_setup import FlyLaunchConfig, load_last_agent_path
from swarm.core.fly_trajectory import (
    FlyTrajectory,
    FlyTrajectoryFrame,
    build_trajectory_meta,
    default_run_video_path,
    format_score_detail_lines,
    frame_obs_info,
    load_latest_trajectory,
    load_trajectory,
    save_trajectory,
    video_path_for_trajectory,
)
from swarm.core.fly_viewer import (
    DEFAULT_VIEW_HEIGHT,
    DEFAULT_VIEW_WIDTH,
    FlyRenderCamera,
    FlySimulatorWindow,
    ReplayUiState,
    build_bottom_telemetry_lines,
    export_video,
)
from swarm.utils.env_factory import make_env
from swarm.validator.reward import flight_score_details
from swarm.validator.task_gen import task_for_seed_and_type

def _build_flight_result(
    *,
    success: bool,
    t_sim: float,
    task,
    info: dict[str, Any],
) -> dict[str, Any]:
    details = flight_score_details(
        success=success,
        t=t_sim,
        horizon=task.horizon,
        task=task,
        min_clearance=info.get("min_clearance"),
        collision=bool(info.get("collision", False)),
        legitimate_model=True,
    )
    return {
        "success": success,
        "time_sec": t_sim,
        "score": float(details["score"]),
        "collision": bool(info.get("collision", False)),
        "success_term": details["success_term"],
        "time_term": details["time_term"],
        "safety_term": details["safety_term"],
        "target_time_sec": details["target_time_sec"],
        "min_clearance_m": details["min_clearance_m"],
    }


TYPE_LABELS = {
    1: "city",
    2: "open",
    3: "mountain",
    4: "village",
    5: "warehouse",
    6: "forest",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fly a Swarm model with a pygame viewer and control panel.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional submission zip. If omitted, choose in the setup UI.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional agent source directory. If omitted, choose in the setup UI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional map seed. If omitted, choose in the setup UI.",
    )
    parser.add_argument(
        "--type",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=None,
        help="Optional challenge type. If omitted, choose in the setup UI.",
    )
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Skip the setup UI (requires --model or --source, and --type).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run simulation as fast as possible (no real-time pacing).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also print per-frame telemetry to the terminal.",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=25,
        help="Emit one terminal debug line every N simulation frames.",
    )
    parser.add_argument(
        "--camera",
        choices=["chase", "fpv", "top", "overview"],
        default="chase",
        help="Initial camera mode (default: chase).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_VIEW_WIDTH,
        help=f"3D viewport width (default: {DEFAULT_VIEW_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_VIEW_HEIGHT,
        help=f"3D viewport height (default: {DEFAULT_VIEW_HEIGHT}).",
    )
    parser.add_argument(
        "--video-out",
        type=Path,
        default=None,
        help="Default export path for the UI Export Video button.",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=None,
        help=(
            "JSON file listing map types and seeds. Runs headlessly (no viewer), "
            "saves each trajectory to fly_runs/."
        ),
    )
    return parser


def _prepare_agent_dir(
    *,
    model_path: Path | None,
    source_path: Path | None,
    seed: int,
) -> tuple[Path, Path | None]:
    if source_path is not None:
        agent_dir = source_path.resolve()
        if not agent_dir.is_dir():
            raise NotADirectoryError(f"Source is not a directory: {agent_dir}")
        if not (agent_dir / "drone_agent.py").is_file():
            raise FileNotFoundError(f"Missing drone_agent.py in source directory: {agent_dir}")
        return agent_dir, None

    if model_path is None:
        raise ValueError("Either --model or --source is required.")

    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if model_path.is_dir():
        if not (model_path / "drone_agent.py").is_file():
            raise FileNotFoundError(f"Missing drone_agent.py in directory: {model_path}")
        return model_path, None

    if model_path.suffix.lower() == ".zip" or zipfile.is_zipfile(model_path):
        temp_dir = Path(tempfile.mkdtemp(prefix=f"swarm_fly_seed{seed}_"))
        _extract_zip(model_path, temp_dir)
        return temp_dir, temp_dir

    raise FileNotFoundError(
        f"Unsupported model path (expected directory or zip): {model_path}"
    )


def _parse_observation(obs: dict[str, Any]) -> dict[str, Any]:
    state = np.asarray(obs["state"], dtype=np.float32)
    position = state[0:3]
    velocity = state[6:9]
    search_area_vector = state[-3:]
    return {
        "position": position,
        "velocity": velocity,
        "speed_mps": float(np.linalg.norm(velocity)),
        "search_area_vector": search_area_vector,
        "search_area_center": position + search_area_vector,
    }


def _snapshot_agent_debug(agent: Any) -> dict[str, Any]:
    if hasattr(agent, "get_debug_info") and callable(agent.get_debug_info):
        try:
            info = agent.get_debug_info()
            if isinstance(info, dict):
                return info
        except Exception:
            pass

    landing_platform = getattr(agent, "landing_platform", None)
    if landing_platform is None:
        landing_platform = getattr(agent, "platform_position", None)

    goal_visibility_prob = getattr(agent, "_last_goal_visibility_prob", None)
    if goal_visibility_prob is None:
        goal_visibility_prob = getattr(agent, "goal_visibility_prob", None)

    map_label = getattr(agent, "_map_prediction_label", None)
    map_prob = getattr(agent, "_map_prediction_probability", None)
    if map_label is not None and map_prob is not None:
        map_prediction = f"{map_label}:{float(map_prob):.3f}"
    else:
        map_prediction = None

    last_action = getattr(agent, "_last_action", None)
    if last_action is None:
        last_action = getattr(agent, "last_action", None)

    return {
        "mode": getattr(agent, "_mode", getattr(agent, "mode", None)),
        "goal_detected": getattr(agent, "is_find_P", getattr(agent, "goal_detected", None)),
        "goal_visible": getattr(agent, "see_P", getattr(agent, "goal_visible", None)),
        "goal_tracked": getattr(agent, "_goal_is_tracked", getattr(agent, "goal_tracked", None)),
        "goal_visibility_prob": goal_visibility_prob,
        "predicted_goal_position": landing_platform,
        "platform_lost_steps": getattr(agent, "platform_lost_step", None),
        "goal_distance_buffer": getattr(agent, "p_buffer", None),
        "map_prediction": map_prediction,
        "command_action": None if last_action is None else np.asarray(last_action, dtype=float),
        "tracking": getattr(agent, "tracking", None),
    }


def _fmt_vec3(values: Any, precision: int = 2) -> str:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return f"({arr[0]:.{precision}f},{arr[1]:.{precision}f},{arr[2]:.{precision}f})"


def _print_debug_frame(
    *,
    frame: int,
    t_sim: float,
    task,
    obs_info: dict[str, Any],
    agent_info: dict[str, Any],
    action: np.ndarray,
) -> None:
    act = np.asarray(action, dtype=float).reshape(-1)
    print(
        f"{frame:5d} {t_sim:5.2f} pos={_fmt_vec3(obs_info['position'])} "
        f"spd={obs_info['speed_mps']:.2f} mode={agent_info.get('mode')} "
        f"goal={agent_info.get('goal_detected')} act={act.round(2).tolist()}"
    )


def _default_video_path(
    *,
    seed: int,
    challenge_type: int,
    type_label: str,
    agent_path: Path | None,
    video_out: Path | None,
    repo_root: Path,
) -> Path:
    if video_out is not None:
        return Path(video_out).expanduser().resolve()
    meta = {
        "seed": int(seed),
        "type_label": type_label,
        "agent_path": str(agent_path) if agent_path is not None else None,
    }
    return default_run_video_path(meta, repo_root=repo_root)


def _get_drone_pose(env, drone_id: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    import pybullet as p

    cli = getattr(env, "CLIENT", getattr(env, "_cli", 0))
    pos, quat = p.getBasePositionAndOrientation(drone_id, physicsClientId=cli)
    return np.asarray(pos, dtype=float), quat


def _set_drone_pose(
    env,
    position: np.ndarray,
    quaternion: Sequence[float] | np.ndarray,
) -> None:
    import pybullet as p

    cli = getattr(env, "CLIENT", getattr(env, "_cli", 0))
    drone_id = env.DRONE_IDS[0]
    p.resetBasePositionAndOrientation(
        drone_id,
        np.asarray(position, dtype=float).tolist(),
        np.asarray(quaternion, dtype=float).tolist(),
        physicsClientId=cli,
    )


def _capture_trajectory_frame(
    *,
    t_sim: float,
    frame: int,
    env,
    action: np.ndarray,
    obs_info: dict[str, Any],
    agent_info: dict[str, Any],
) -> FlyTrajectoryFrame:
    position, quat = _get_drone_pose(env, env.DRONE_IDS[0])
    return FlyTrajectoryFrame(
        t_sim=float(t_sim),
        frame=int(frame),
        position=position,
        quaternion=np.asarray(quat, dtype=float),
        velocity=np.asarray(obs_info["velocity"], dtype=float),
        action=np.asarray(action, dtype=float).reshape(-1),
        agent_info=agent_info,
        search_area_vector=np.asarray(obs_info["search_area_vector"], dtype=float),
    )


def _trajectory_launch_config(trajectory: FlyTrajectory) -> FlyLaunchConfig:
    meta = trajectory.meta
    return FlyLaunchConfig(
        agent_path=Path(str(meta["agent_path"])).expanduser(),
        agent_kind=str(meta["agent_kind"]),  # type: ignore[arg-type]
        seed=int(meta["seed"]),
        challenge_type=int(meta["challenge_type"]),
    )


def _mission_matches_trajectory(task, trajectory: FlyTrajectory) -> bool:
    meta = trajectory.meta
    return (
        int(task.map_seed) == int(meta.get("seed", -1))
        and int(task.challenge_type) == int(meta.get("challenge_type", -1))
    )


def _apply_replay_frame(env, frame: FlyTrajectoryFrame) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    _set_drone_pose(env, frame.position, frame.quaternion)
    return frame_obs_info(frame), dict(frame.agent_info), np.asarray(frame.action, dtype=np.float32)


def _load_run_for_replay(path: str | Path) -> FlyTrajectory:
    trajectory = load_trajectory(Path(path))
    if not trajectory.frames:
        raise ValueError("Selected run has no trajectory frames.")
    return trajectory


def _apply_replay_at_index(
    env,
    trajectory: FlyTrajectory,
    index: int,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, float, int]:
    index = max(0, min(index, len(trajectory.frames) - 1))
    frame = trajectory.frames[index]
    obs_info, agent_info, action = _apply_replay_frame(env, frame)
    return obs_info, agent_info, action, float(frame.t_sim), int(frame.frame)


def _start_replay_session(
    *,
    trajectory: FlyTrajectory,
    env,
) -> tuple[int, dict[str, Any], dict[str, Any], np.ndarray, float, int]:
    obs_info, agent_info, action, t_sim, step_count = _apply_replay_at_index(env, trajectory, 0)
    return 1, obs_info, agent_info, action, t_sim, step_count


def _render_scene(
    env,
    camera: FlyRenderCamera,
    *,
    width: int,
    height: int,
    dt: float,
) -> np.ndarray:
    from swarm.core.fly_viewer import render_rgb_frame

    drone_id = env.DRONE_IDS[0]
    position, quat = _get_drone_pose(env, drone_id)
    eye, target = camera.eye_and_target(position, quat, dt)
    cli = getattr(env, "CLIENT", getattr(env, "_cli", 0))
    return render_rgb_frame(
        cli,
        eye=eye,
        target=target,
        width=width,
        height=height,
    )


def _reset_episode(env, agent, task):
    obs, _ = env.reset(seed=task.map_seed)
    reset_fn = getattr(agent, "reset", None)
    if callable(reset_fn):
        reset_fn()
    return obs


def _resolve_launch_config(args: argparse.Namespace) -> FlyLaunchConfig | None:
    if args.model is not None and args.source is not None:
        raise ValueError("Use either --model or --source, not both.")
    if not args.no_setup:
        return None
    if args.model is None and args.source is None:
        raise ValueError("--no-setup requires --model or --source.")
    if args.type is None:
        raise ValueError("--no-setup requires --type.")
    agent_path = args.source if args.source is not None else args.model
    assert agent_path is not None
    agent_path = agent_path.resolve()
    kind = "source" if args.source is not None else "zip"
    return FlyLaunchConfig(
        agent_path=agent_path,
        agent_kind=kind,
        seed=int(args.seed if args.seed is not None else 42),
        challenge_type=int(args.type),
    )


def _load_mission(
    launch: FlyLaunchConfig,
    *,
    camera_mode: str,
    view_width: int,
    view_height: int,
):
    model_path = launch.agent_path if launch.agent_kind == "zip" else None
    source_path = launch.agent_path if launch.agent_kind == "source" else None
    agent_dir, temp_dir = _prepare_agent_dir(
        model_path=model_path,
        source_path=source_path,
        seed=int(launch.seed),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        agent = _load_agent(agent_dir)
    reset_fn = getattr(agent, "reset", None)
    if callable(reset_fn):
        with contextlib.redirect_stdout(io.StringIO()):
            reset_fn()
    task = task_for_seed_and_type(
        SIM_DT,
        seed=int(launch.seed),
        challenge_type=int(launch.challenge_type),
    )
    env = make_env(task, gui=False)
    obs = _reset_episode(env, agent, task)
    camera = FlyRenderCamera(task.goal, mode=camera_mode)
    lo, hi = env.action_space.low.flatten(), env.action_space.high.flatten()
    return {
        "agent_dir": agent_dir,
        "temp_dir": temp_dir,
        "agent": agent,
        "task": task,
        "env": env,
        "obs": obs,
        "camera": camera,
        "lo": lo,
        "hi": hi,
    }


def _begin_live_flight(
    *,
    env,
    agent,
    task,
    reset_episode: bool,
) -> tuple[Any, dict[str, Any], dict[str, Any], float, int]:
    """Reset counters and optionally rewind the environment for a live flight."""
    if reset_episode:
        obs = _reset_episode(env, agent, task)
        obs_info = _parse_observation(obs)
        agent_info = _snapshot_agent_debug(agent)
        return obs, obs_info, agent_info, 0.0, 0
    obs_info = None
    agent_info = None
    return None, obs_info, agent_info, 0.0, 0


def _handle_start_pause(
    *,
    ui,
    window: FlySimulatorWindow,
    sim_state: str,
    env,
    agent,
    task,
) -> tuple[str, bool, Any | None, dict[str, Any] | None, dict[str, Any] | None, float, int]:
    """Apply Start/Pause controls. Returns updated sim state and optional reset data."""
    obs = None
    obs_info = None
    agent_info = None
    t_sim = 0.0
    step_count = 0
    reset_episode = False

    if ui.pause:
        if sim_state == "running":
            sim_state = "paused"
            window.set_status("Simulation paused.")
        elif sim_state == "replay":
            sim_state = "replay_paused"
            window.set_status("Replay paused.")
        elif sim_state == "paused":
            sim_state = "running"
            window.set_status("Simulation running.")
        elif sim_state == "replay_paused":
            sim_state = "replay"
            window.set_status("Replaying trajectory.")
        elif sim_state in {"finished", "replay_finished"}:
            window.set_status("Run finished. Press Start for a new flight.")
        elif sim_state == "ready":
            window.set_status("Press Start to begin the flight.")
        elif env is None:
            window.set_status("Build a map first.")

    if ui.start:
        if env is None:
            window.set_status("Build a map first.")
            return sim_state, False, obs, obs_info, agent_info, t_sim, step_count

        if sim_state == "replay_paused":
            sim_state = "replay"
            window.set_status("Replaying trajectory.")
        elif sim_state == "paused":
            sim_state = "running"
            window.set_status("Simulation running.")
        elif sim_state in {"ready", "finished", "replay_finished"}:
            reset_episode = sim_state in {"finished", "replay_finished"}
            if sim_state == "replay_finished":
                window.set_replay_ui_active(False)
            sim_state = "running"
            window.set_status("Simulation running.")
            window.set_export_enabled(False)
            window.set_replay_enabled(False)
        elif sim_state in {"replay", "replay_paused"}:
            window.set_replay_ui_active(False)
            reset_episode = True
            sim_state = "running"
            window.set_status("Simulation running.")
            window.set_export_enabled(False)
            window.set_replay_enabled(False)
        elif sim_state == "running":
            window.set_status("Simulation already running.")
        else:
            window.set_status("Build a map first.")

    if reset_episode and env is not None and agent is not None and task is not None:
        obs, obs_info, agent_info, t_sim, step_count = _begin_live_flight(
            env=env,
            agent=agent,
            task=task,
            reset_episode=True,
        )

    return sim_state, reset_episode, obs, obs_info, agent_info, t_sim, step_count


def fly_episode(
    *,
    args: argparse.Namespace,
    launch: FlyLaunchConfig | None = None,
    realtime: bool = True,
    debug: bool = False,
    debug_every: int = 25,
    camera_mode: str = "chase",
    view_width: int = DEFAULT_VIEW_WIDTH,
    view_height: int = DEFAULT_VIEW_HEIGHT,
    video_out: Path | None = None,
) -> dict:
    preset_agent = args.source if args.source is not None else args.model
    custom_path = ""
    use_last_agent = True
    last_agent_path = load_last_agent_path()
    if preset_agent is not None:
        custom_path = str(preset_agent.resolve())
        use_last_agent = False
        last_agent_path = preset_agent.resolve()

    window = FlySimulatorWindow(
        title="Swarm Fly",
        view_width=view_width,
        view_height=view_height,
        camera_mode=camera_mode,
        seed=int(args.seed if args.seed is not None else 42),
        challenge_type=int(args.type if args.type is not None else 1),
        last_agent_path=last_agent_path,
        custom_path=custom_path,
        use_last_agent=use_last_agent,
        repo_root=_REPO_ROOT,
    )

    last_trajectory = load_latest_trajectory(_REPO_ROOT)
    if last_trajectory is not None and last_trajectory.frames:
        window.set_replay_enabled(True)

    env = None
    agent = None
    task = None
    camera = FlyRenderCamera((0.0, 0.0, 0.0), mode=camera_mode)
    lo = hi = None
    temp_dir: Path | None = None
    obs_info = None
    agent_info = None
    obs = None

    sim_state = "config"
    t_sim = 0.0
    step_count = 0
    success = False
    info: dict[str, Any] = {}
    result: dict[str, Any] | None = None
    last_action = np.zeros(5, dtype=np.float32)
    recorded_frames: list[np.ndarray] = []
    trajectory_frames: list[FlyTrajectoryFrame] = []
    replay_index = 0
    replay_speed = 1.0
    current_launch_cfg: FlyLaunchConfig | None = None
    video_path: Path | None = None
    score = 0.0
    type_label = "unknown"
    challenge_type = int(window.challenge_type)
    seed = int(window.seed)

    last_step_at = time.perf_counter()
    last_preview_step_at = time.perf_counter()
    debug_every = max(1, int(debug_every))

    if launch is not None:
        window.seed = int(launch.seed)
        window.challenge_type = int(launch.challenge_type)
        window.custom_path = str(launch.agent_path)
        window.use_last_agent = False

    try:
        while not window.quit_requested:
            ui = window.pump()
            if ui.quit:
                break
            if ui.camera_mode is not None:
                camera.mode = ui.camera_mode
                window.camera_mode = ui.camera_mode
            if ui.zoom_in:
                camera.distance_scale = max(0.2, camera.distance_scale * 0.85)
            if ui.zoom_out:
                camera.distance_scale = min(8.0, camera.distance_scale * 1.15)

            if ui.load_run_path:
                try:
                    last_trajectory = _load_run_for_replay(ui.load_run_path)
                    window.set_replay_enabled(True)
                    window.refresh_saved_runs()
                except Exception as exc:
                    window.set_status(f"Failed to load run: {exc}")
                    print(f"Failed to load run: {exc}", file=sys.stderr)

            if ui.replay_speed is not None:
                replay_speed = max(0.1, float(ui.replay_speed))
                window.replay_speed = replay_speed

            if ui.replay_seek is not None and last_trajectory is not None and env is not None:
                seek = max(0, min(int(ui.replay_seek), len(last_trajectory.frames) - 1))
                obs_info, agent_info, last_action, t_sim, step_count = _apply_replay_at_index(
                    env,
                    last_trajectory,
                    seek,
                )
                replay_index = seek + 1
                if sim_state in {"replay", "replay_paused"}:
                    sim_state = "replay_paused"
                    window.set_status(f"Replay at {t_sim:.1f}s (paused).")

            if ui.build_map or (launch is not None and env is None):
                launch_cfg = window.get_launch_config()
                if launch_cfg is None:
                    window.set_status("Invalid agent selection.")
                else:
                    current_launch_cfg = launch_cfg
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                    if temp_dir is not None:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    window.set_status("Building map...")
                    window.draw(
                        None,
                        build_bottom_telemetry_lines(
                            task=None,
                            sim_state="building",
                            t_sim=0.0,
                            frame=0,
                            obs_info=None,
                            agent_info=None,
                            action=None,
                            camera_mode=camera.mode,
                        ),
                        placeholder_text="Building map...",
                    )
                    mission = _load_mission(
                        launch_cfg,
                        camera_mode=camera.mode,
                        view_width=view_width,
                        view_height=view_height,
                    )
                    env = mission["env"]
                    agent = mission["agent"]
                    task = mission["task"]
                    obs = mission["obs"]
                    camera = mission["camera"]
                    lo, hi = mission["lo"], mission["hi"]
                    temp_dir = mission["temp_dir"]
                    obs_info = _parse_observation(obs)
                    agent_info = _snapshot_agent_debug(agent)
                    seed = int(task.map_seed)
                    challenge_type = int(task.challenge_type)
                    type_label = TYPE_LABELS.get(challenge_type, f"type{challenge_type}")
                    video_path = _default_video_path(
                        seed=seed,
                        challenge_type=challenge_type,
                        type_label=type_label,
                        agent_path=launch_cfg.agent_path,
                        video_out=video_out,
                        repo_root=_REPO_ROOT,
                    )
                    sim_state = "ready"
                    t_sim = 0.0
                    step_count = 0
                    success = False
                    info = {}
                    result = None
                    recorded_frames.clear()
                    trajectory_frames.clear()
                    replay_index = 0
                    window.set_map_loaded(True)
                    window.set_status("Map ready. Press Start.")
                    window.set_export_enabled(False)
                    window.set_replay_enabled(False)
                    window.remember_agent(launch_cfg.agent_path)
                    print(
                        f"Loaded seed={seed} type={challenge_type} ({type_label}) "
                        f"start={tuple(round(v, 2) for v in task.start)} "
                        f"goal={tuple(round(v, 2) for v in task.goal)}"
                    )
                launch = None

            (
                sim_state,
                reset_episode,
                reset_obs,
                reset_obs_info,
                reset_agent_info,
                reset_t_sim,
                reset_step_count,
            ) = _handle_start_pause(
                ui=ui,
                window=window,
                sim_state=sim_state,
                env=env,
                agent=agent,
                task=task,
            )
            if ui.start or ui.pause:
                if sim_state in {"running", "replay"}:
                    last_step_at = time.perf_counter()
            if reset_episode:
                if reset_obs is not None:
                    obs = reset_obs
                if reset_obs_info is not None:
                    obs_info = reset_obs_info
                if reset_agent_info is not None:
                    agent_info = reset_agent_info
                t_sim = reset_t_sim
                step_count = reset_step_count
                success = False
                info = {}
                result = None
                recorded_frames.clear()
                trajectory_frames.clear()
                replay_index = 0

            if ui.replay and last_trajectory is not None and last_trajectory.frames:
                needs_reload = (
                    env is None
                    or task is None
                    or not _mission_matches_trajectory(task, last_trajectory)
                )
                if needs_reload:
                    replay_launch = _trajectory_launch_config(last_trajectory)
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                    if temp_dir is not None:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    window.set_status("Loading map for replay...")
                    window.draw(
                        None,
                        build_bottom_telemetry_lines(
                            task=None,
                            sim_state="building",
                            t_sim=0.0,
                            frame=0,
                            obs_info=None,
                            agent_info=None,
                            action=None,
                            camera_mode=camera.mode,
                        ),
                        placeholder_text="Loading replay map...",
                    )
                    mission = _load_mission(
                        replay_launch,
                        camera_mode=camera.mode,
                        view_width=view_width,
                        view_height=view_height,
                    )
                    env = mission["env"]
                    agent = mission["agent"]
                    task = mission["task"]
                    camera = mission["camera"]
                    lo, hi = mission["lo"], mission["hi"]
                    temp_dir = mission["temp_dir"]
                    seed = int(task.map_seed)
                    challenge_type = int(task.challenge_type)
                    type_label = TYPE_LABELS.get(challenge_type, f"type{challenge_type}")
                    window.set_map_loaded(True)
                replay_index, obs_info, agent_info, last_action, t_sim, step_count = (
                    _start_replay_session(trajectory=last_trajectory, env=env)
                )
                sim_state = "replay"
                replay_speed = window.replay_speed
                window.set_status(f"Replaying at {replay_speed:g}x.")
                window.set_export_enabled(False)
                window.set_replay_ui_active(True)
                last_step_at = time.perf_counter()

            if ui.export and sim_state == "finished" and recorded_frames:
                try:
                    if video_path is None and current_launch_cfg is not None:
                        video_path = _default_video_path(
                            seed=seed,
                            challenge_type=challenge_type,
                            type_label=type_label,
                            agent_path=current_launch_cfg.agent_path,
                            video_out=video_out,
                            repo_root=_REPO_ROOT,
                        )
                    assert video_path is not None
                    saved = export_video(recorded_frames, video_path)
                    window.set_status(f"Saved video: {saved}")
                    print(f"Saved video: {saved}")
                except Exception as exc:
                    window.set_status(f"Export failed: {exc}")
                    print(f"Export failed: {exc}", file=sys.stderr)

            now = time.perf_counter()
            frame_rgb = None
            if env is not None and task is not None:
                if sim_state == "running" and lo is not None and hi is not None:
                    if now - last_step_at >= SIM_DT:
                        try:
                            raw = agent.act(obs)
                            if raw is None:
                                raw = np.zeros(5, dtype=np.float32)
                        except Exception as exc:
                            print(f"Action error at t={t_sim:.2f}s: {exc}")
                            raw = np.zeros(5, dtype=np.float32)

                        act = np.clip(np.asarray(raw, dtype=np.float32).reshape(-1), lo, hi)
                        if getattr(env, "ACT_TYPE", None) == ActionType.VEL:
                            norm = max(float(np.linalg.norm(act[:3])), 1e-6)
                            act[:3] *= min(1.0, float(SPEED_LIMIT) / norm)
                            act = np.clip(act, lo, hi)
                        last_action = act

                        obs, _reward, terminated, truncated, info = env.step(act[None, :])
                        obs_info = _parse_observation(obs)
                        agent_info = _snapshot_agent_debug(agent)
                        t_sim += SIM_DT
                        step_count += 1
                        last_step_at = now
                        trajectory_frames.append(
                            _capture_trajectory_frame(
                                t_sim=t_sim,
                                frame=step_count,
                                env=env,
                                action=act,
                                obs_info=obs_info,
                                agent_info=agent_info,
                            )
                        )

                        if debug and (step_count % debug_every == 0):
                            _print_debug_frame(
                                frame=step_count,
                                t_sim=t_sim,
                                task=task,
                                obs_info=obs_info,
                                agent_info=agent_info,
                                action=act,
                            )

                        if terminated or truncated:
                            success = bool(info.get("success", False))
                            sim_state = "finished"
                            result = _build_flight_result(
                                success=success,
                                t_sim=t_sim,
                                task=task,
                                info=info,
                            )
                            score = float(result["score"])
                            print(
                                f"Flight score: {score:.3f} "
                                f"(success={result['success_term']:.2f} "
                                f"time={result['time_term']:.2f} "
                                f"safety={result['safety_term']:.2f})"
                            )
                            window.set_status(
                                f"Score {score:.3f} | "
                                f"{'success' if success else 'failed'} | "
                                f"{t_sim:.1f}s"
                            )
                            if trajectory_frames and current_launch_cfg is not None:
                                try:
                                    saved_meta = build_trajectory_meta(
                                        agent_path=current_launch_cfg.agent_path,
                                        agent_kind=current_launch_cfg.agent_kind,
                                        seed=seed,
                                        challenge_type=challenge_type,
                                        type_label=type_label,
                                        task=task,
                                        result=result,
                                    )
                                    last_trajectory = FlyTrajectory(
                                        meta=saved_meta,
                                        frames=list(trajectory_frames),
                                    )
                                    traj_path = save_trajectory(
                                        last_trajectory,
                                        repo_root=_REPO_ROOT,
                                    )
                                    video_path = video_path_for_trajectory(traj_path)
                                    window.set_replay_enabled(True)
                                    window.refresh_saved_runs()
                                    print(f"Saved run: {traj_path}")
                                except Exception as exc:
                                    print(f"Trajectory save failed: {exc}", file=sys.stderr)
                            window.set_export_enabled(True)
                elif (
                    sim_state == "replay"
                    and last_trajectory is not None
                    and now - last_step_at >= SIM_DT / max(replay_speed, 0.1)
                ):
                    if replay_index < len(last_trajectory.frames):
                        obs_info, agent_info, last_action, t_sim, step_count = (
                            _apply_replay_at_index(env, last_trajectory, replay_index)
                        )
                        replay_index += 1
                        last_step_at = now
                        if replay_index >= len(last_trajectory.frames):
                            sim_state = "replay_finished"
                            window.set_replay_ui_active(False)
                            window.set_status("Replay finished. Open another run or Replay again.")
                elif task.moving_platform and sim_state in {"ready", "paused"} and now - last_preview_step_at >= SIM_DT:
                    import pybullet as p

                    cli = getattr(env, "CLIENT", getattr(env, "_cli", 0))
                    p.stepSimulation(physicsClientId=cli)
                    last_preview_step_at = now

                frame_rgb = _render_scene(
                    env,
                    camera,
                    width=view_width,
                    height=view_height,
                    dt=SIM_DT,
                )
                if sim_state == "running":
                    recorded_frames.append(frame_rgb.copy())

            bottom_lines = build_bottom_telemetry_lines(
                task=task,
                sim_state=sim_state,
                t_sim=t_sim,
                frame=step_count,
                obs_info=obs_info,
                agent_info=agent_info,
                action=last_action,
                camera_mode=camera.mode,
                result=(
                    dict(last_trajectory.meta.get("result") or {})
                    if last_trajectory is not None
                    and sim_state in {"replay", "replay_paused", "replay_finished"}
                    and last_trajectory.meta.get("result")
                    else result
                ),
            )
            replay_ui = None
            if (
                last_trajectory is not None
                and sim_state in {"replay", "replay_paused", "replay_finished"}
            ):
                replay_ui = ReplayUiState(
                    active=sim_state in {"replay", "replay_paused"},
                    frame=max(0, replay_index - 1),
                    total_frames=len(last_trajectory.frames),
                    t_sim=t_sim,
                    duration=last_trajectory.duration_sec,
                    speed=replay_speed,
                )
            placeholder = None if frame_rgb is not None else "Select settings, then click Build Map"
            window.set_sim_state(sim_state)
            window.draw(
                frame_rgb,
                bottom_lines,
                placeholder_text=placeholder,
                replay_ui=replay_ui,
            )

            if realtime and sim_state in {"running", "replay"}:
                pace = SIM_DT if sim_state == "running" else SIM_DT / max(replay_speed, 0.1)
                time.sleep(max(0.0, pace - (time.perf_counter() - now)))
            else:
                time.sleep(0.02)
    finally:
        window.close()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    if task is None:
        return {
            "success": False,
            "time_sec": 0.0,
            "score": 0.0,
            "avg_speed": 0.0,
            "collision": False,
            "seed": seed,
            "challenge_type": challenge_type,
            "type_label": type_label,
            "video_path": None,
        }

    if not result:
        result = _build_flight_result(
            success=success,
            t_sim=t_sim,
            task=task,
            info=info,
        )

    return {
        "success": success,
        "time_sec": t_sim,
        "score": float(result["score"]),
        "avg_speed": float(obs_info["speed_mps"]) if obs_info and step_count else 0.0,
        "collision": bool(info.get("collision", False)),
        "seed": seed,
        "challenge_type": challenge_type,
        "type_label": type_label,
        "video_path": str(video_path) if recorded_frames else None,
    }


def _resolve_batch_launch_config(args: argparse.Namespace) -> FlyLaunchConfig:
    if args.model is not None and args.source is not None:
        raise ValueError("Use either --model or --source, not both.")
    if args.model is None and args.source is None:
        raise ValueError("Batch mode requires --model or --source.")
    if args.seed is not None or args.type is not None:
        raise ValueError("Do not pass --seed or --type with --batch; use the JSON file.")
    if getattr(args, "no_setup", False):
        raise ValueError("Do not use --no-setup with --batch.")
    agent_path = args.source if args.source is not None else args.model
    assert agent_path is not None
    kind = "source" if args.source is not None else "zip"
    return FlyLaunchConfig(
        agent_path=agent_path.resolve(),
        agent_kind=kind,
        seed=0,
        challenge_type=1,
    )


def run_batch_mode(args: argparse.Namespace) -> int:
    from swarm.core.fly_batch import load_batch_jobs, print_batch_summary, run_batch

    assert args.batch is not None
    batch_path = Path(args.batch).expanduser()
    if not batch_path.is_file():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    launch = _resolve_batch_launch_config(args)
    jobs = load_batch_jobs(batch_path)
    print(f"Batch file : {batch_path}")
    print(f"Agent      : {launch.agent_path}")
    print(f"Runs       : {len(jobs)}")
    print("Mode       : headless (trajectories saved to fly_runs/)")
    print("")

    results = run_batch(
        launch=launch,
        jobs=jobs,
        repo_root=_REPO_ROOT,
        debug=bool(args.debug),
        debug_every=int(args.debug_every),
    )
    print_batch_summary(results)
    failures = sum(1 for item in results if not item.get("success"))
    errors = sum(1 for item in results if item.get("error"))
    return 0 if failures == 0 and errors == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.batch is not None:
        try:
            return run_batch_mode(args)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 130
        except Exception as exc:
            print(f"Batch fly failed: {exc}", file=sys.stderr)
            return 1

    realtime = not args.fast

    try:
        launch = _resolve_launch_config(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        result = fly_episode(
            args=args,
            launch=launch,
            realtime=realtime,
            debug=bool(args.debug),
            debug_every=int(args.debug_every),
            camera_mode=str(args.camera),
            view_width=int(args.width),
            view_height=int(args.height),
            video_out=args.video_out,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except Exception as exc:
        print(f"Fly failed: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print("FLIGHT RESULT")
    print("=" * 60)
    print(f"Success    : {'yes' if result['success'] else 'no'}")
    print(f"Time       : {result['time_sec']:.2f}s")
    print(f"Score      : {result['score']:.3f}")
    print(f"Collision  : {'yes' if result['collision'] else 'no'}")
    for line in format_score_detail_lines(result)[1:]:
        print(line)
    if result.get("video_path"):
        print(f"Video path : {result['video_path']}")
    print("=" * 60)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
