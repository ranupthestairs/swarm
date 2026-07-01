#!/usr/bin/env python3
"""
Real-time 3D PyBullet viewer for flying a Swarm model.

Opens the full procedural world and runs your policy live so you can debug
flight behavior, collisions, and landing in real time.

Examples
--------
    python scripts/fly_model.py --model champion_UID_42.zip --seed 1001 --type 1
    python scripts/fly_model.py --model Submission/submission.zip --seed 42
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gym_pybullet_drones.utils.enums import ActionType

from scripts.generate_video import _extract_zip, _load_agent
from swarm.constants import SIM_DT, SPEED_LIMIT
from swarm.core.drone import track_drone
from swarm.utils.env_factory import make_env
from swarm.validator.reward import flight_reward
from swarm.validator.task_gen import random_task, task_for_seed_and_type

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
        description="Fly a Swarm model in the real-time 3D PyBullet viewer.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to submission zip (e.g. champion_UID_42.zip).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Map seed (default: 42).",
    )
    parser.add_argument(
        "--type",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=None,
        help="Challenge type. If omitted, inferred from the seed.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run as fast as possible without real-time pacing.",
    )
    return parser


def _resolve_challenge_type(seed: int, challenge_type: int | None) -> int:
    if challenge_type is not None:
        return challenge_type
    task = random_task(sim_dt=SIM_DT, seed=seed)
    return int(task.challenge_type)


def fly_episode(
    model_path: Path,
    *,
    seed: int,
    challenge_type: int,
    realtime: bool = True,
) -> dict:
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    work_dir = Path("/tmp") / f"swarm_fly_seed{seed}"
    extracted = _extract_zip(model_path, work_dir)

    with contextlib.redirect_stdout(io.StringIO()):
        agent = _load_agent(extracted)

    task = task_for_seed_and_type(SIM_DT, seed=seed, challenge_type=challenge_type)
    type_label = TYPE_LABELS.get(challenge_type, f"type{challenge_type}")

    print(f"Flying seed={seed}  type={challenge_type} ({type_label})")
    print(f"  Start : {tuple(round(v, 2) for v in task.start)}")
    print(f"  Goal  : {tuple(round(v, 2) for v in task.goal)}")
    print("  Close the PyBullet window or press Ctrl+C to stop.\n")

    env = make_env(task, gui=True)
    obs, _ = env.reset(seed=task.map_seed)

    cli_id = getattr(env, "CLIENT", getattr(env, "_cli", 0))
    lo, hi = env.action_space.low.flatten(), env.action_space.high.flatten()
    frames_per_cam = max(1, int(round(1.0 / (SIM_DT * 60.0))))

    t_sim = 0.0
    step_count = 0
    success = False
    speeds: list[float] = []
    last_pos = np.asarray(task.start, dtype=float)
    info: dict = {}

    try:
        while t_sim < task.horizon:
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

            prev = last_pos
            obs, _reward, terminated, truncated, info = env.step(act[None, :])
            last_pos = env._getDroneStateVector(0)[0:3]
            speeds.append(float(np.linalg.norm(last_pos - prev) / SIM_DT))
            t_sim += SIM_DT
            step_count += 1

            if step_count % frames_per_cam == 0:
                try:
                    track_drone(cli=cli_id, drone_id=env.DRONE_IDS[0])
                except Exception:
                    pass

            if realtime:
                time.sleep(SIM_DT)

            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        env.close()

    score = flight_reward(
        success=success,
        t=t_sim,
        horizon=task.horizon,
        task=task,
        min_clearance=info.get("min_clearance"),
        collision=bool(info.get("collision", False)),
    )

    return {
        "success": success,
        "time_sec": t_sim,
        "score": score,
        "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
        "collision": bool(info.get("collision", False)),
        "seed": seed,
        "challenge_type": challenge_type,
        "type_label": type_label,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    challenge_type = _resolve_challenge_type(args.seed, args.type)
    realtime = not args.fast

    try:
        result = fly_episode(
            args.model,
            seed=args.seed,
            challenge_type=challenge_type,
            realtime=realtime,
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
    print(f"Avg Speed  : {result['avg_speed']:.3f} m/s")
    print(f"Collision  : {'yes' if result['collision'] else 'no'}")
    print("=" * 60)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
