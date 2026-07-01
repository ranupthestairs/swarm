"""Headless batch fly runs from a JSON seed file."""
from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gym_pybullet_drones.utils.enums import ActionType

from scripts.generate_video import _load_agent
from swarm.constants import SIM_DT, SPEED_LIMIT
from swarm.core.fly_setup import FlyLaunchConfig
from swarm.core.fly_trajectory import (
    FlyTrajectory,
    build_trajectory_meta,
    format_score_summary,
    save_trajectory,
)
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import task_for_seed_and_type

BENCH_GROUP_ORDER = (
    "type1_city",
    "type2_open",
    "type3_mountain",
    "type4_village",
    "type5_warehouse",
    "type6_forest",
)
BENCH_GROUP_TO_TYPE = {
    "type1_city": 1,
    "type2_open": 2,
    "type3_mountain": 3,
    "type4_village": 4,
    "type5_warehouse": 5,
    "type6_forest": 6,
}
TYPE_LABELS = {
    1: "city",
    2: "open",
    3: "mountain",
    4: "village",
    5: "warehouse",
    6: "forest",
}
_LABEL_TO_TYPE = {label: challenge_type for challenge_type, label in TYPE_LABELS.items()}
_TYPE_PREFIX_RE = re.compile(r"^type(\d)(?:_|$)", re.IGNORECASE)


@dataclass(frozen=True)
class FlyBatchJob:
    seed: int
    challenge_type: int

    @property
    def type_label(self) -> str:
        return TYPE_LABELS.get(self.challenge_type, f"type{self.challenge_type}")


def _normalize_seed_value(seed: Any) -> int:
    if isinstance(seed, dict):
        if "seed" in seed:
            return int(seed["seed"])
        raise ValueError(f"Seed object must include 'seed': {seed!r}")
    return int(seed)


def _challenge_type_from_key(key: str) -> int | None:
    normalized = str(key).strip()
    if normalized in BENCH_GROUP_TO_TYPE:
        return BENCH_GROUP_TO_TYPE[normalized]
    if normalized.isdigit():
        value = int(normalized)
        if 1 <= value <= 6:
            return value
    match = _TYPE_PREFIX_RE.match(normalized)
    if match:
        return int(match.group(1))
    return _LABEL_TO_TYPE.get(normalized.lower())


def _jobs_from_mapping(raw: dict[str, Any]) -> list[FlyBatchJob]:
    jobs: list[FlyBatchJob] = []
    ordered_keys = [key for key in BENCH_GROUP_ORDER if key in raw]
    ordered_keys.extend(sorted(key for key in raw if key not in BENCH_GROUP_ORDER))
    for key in ordered_keys:
        seeds = raw.get(key)
        if seeds in (None, []):
            continue
        if not isinstance(seeds, list):
            raise ValueError(f"Seed group {key!r} must be a list when present.")
        challenge_type = _challenge_type_from_key(key)
        if challenge_type is None:
            raise ValueError(f"Unrecognized map type key: {key!r}")
        for seed in seeds:
            jobs.append(
                FlyBatchJob(
                    seed=_normalize_seed_value(seed),
                    challenge_type=challenge_type,
                )
            )
    return jobs


def _jobs_from_list(raw: list[Any]) -> list[FlyBatchJob]:
    jobs: list[FlyBatchJob] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("Each batch list entry must be an object with 'type' and 'seeds'.")
        challenge_type = entry.get("type", entry.get("challenge_type"))
        if challenge_type is None:
            raise ValueError(f"Batch entry missing type/challenge_type: {entry!r}")
        challenge_type = int(challenge_type)
        if challenge_type not in TYPE_LABELS:
            raise ValueError(f"Unsupported challenge type: {challenge_type}")
        seeds = entry.get("seeds")
        if seeds is None and "seed" in entry:
            seeds = [entry["seed"]]
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"Batch entry must include a non-empty seeds list: {entry!r}")
        for seed in seeds:
            jobs.append(FlyBatchJob(seed=_normalize_seed_value(seed), challenge_type=challenge_type))
    return jobs


def load_batch_jobs(seed_file: Path) -> list[FlyBatchJob]:
    """Load (challenge_type, seed) jobs from a JSON file."""
    path = Path(seed_file).expanduser()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "runs" in raw and isinstance(raw["runs"], list):
        jobs = _jobs_from_list(raw["runs"])
    elif isinstance(raw, list):
        jobs = _jobs_from_list(raw)
    elif isinstance(raw, dict):
        jobs = _jobs_from_mapping(raw)
    else:
        raise ValueError("Batch file must be a JSON object or list.")
    if not jobs:
        raise ValueError(f"No runs found in batch file: {path}")
    return jobs


def _prepare_agent_dir(*, model_path: Path | None, source_path: Path | None, seed: int):
    from scripts.fly_model import _prepare_agent_dir

    return _prepare_agent_dir(model_path=model_path, source_path=source_path, seed=seed)


def _reset_episode(env, agent, task):
    from scripts.fly_model import _reset_episode

    return _reset_episode(env, agent, task)


def _parse_observation(obs: dict[str, Any]) -> dict[str, Any]:
    from scripts.fly_model import _parse_observation

    return _parse_observation(obs)


def _snapshot_agent_debug(agent: Any) -> dict[str, Any]:
    from scripts.fly_model import _snapshot_agent_debug

    return _snapshot_agent_debug(agent)


def _capture_trajectory_frame(**kwargs):
    from scripts.fly_model import _capture_trajectory_frame

    return _capture_trajectory_frame(**kwargs)


def _build_flight_result(**kwargs):
    from scripts.fly_model import _build_flight_result

    return _build_flight_result(**kwargs)


def run_headless_episode(
    *,
    launch: FlyLaunchConfig,
    repo_root: Path,
    agent: Any,
    lo: np.ndarray,
    hi: np.ndarray,
    debug: bool = False,
    debug_every: int = 25,
) -> dict[str, Any]:
    """Run one episode without pygame rendering and save its trajectory."""
    task = task_for_seed_and_type(
        SIM_DT,
        seed=int(launch.seed),
        challenge_type=int(launch.challenge_type),
    )
    env = make_env(task, gui=False)
    obs = _reset_episode(env, agent, task)

    t_sim = 0.0
    step_count = 0
    success = False
    info: dict[str, Any] = {}
    trajectory_frames = []

    try:
        while True:
            try:
                raw = agent.act(obs)
                if raw is None:
                    raw = np.zeros(5, dtype=np.float32)
            except Exception as exc:
                print(f"Action error at t={t_sim:.2f}s: {exc}", file=sys.stderr)
                raw = np.zeros(5, dtype=np.float32)

            act = np.clip(np.asarray(raw, dtype=np.float32).reshape(-1), lo, hi)
            if getattr(env, "ACT_TYPE", None) == ActionType.VEL:
                norm = max(float(np.linalg.norm(act[:3])), 1e-6)
                act[:3] *= min(1.0, float(SPEED_LIMIT) / norm)
                act = np.clip(act, lo, hi)

            obs, _reward, terminated, truncated, info = env.step(act[None, :])
            obs_info = _parse_observation(obs)
            agent_info = _snapshot_agent_debug(agent)
            t_sim += SIM_DT
            step_count += 1
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

            if debug and (step_count % max(1, debug_every) == 0):
                print(
                    f"  frame={step_count:5d} t={t_sim:5.2f}s "
                    f"pos={np.round(obs_info['position'], 2).tolist()}",
                    file=sys.stderr,
                )

            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    result = _build_flight_result(success=success, t_sim=t_sim, task=task, info=info)
    type_label = TYPE_LABELS.get(int(launch.challenge_type), f"type{launch.challenge_type}")
    saved_meta = build_trajectory_meta(
        agent_path=launch.agent_path,
        agent_kind=launch.agent_kind,
        seed=int(launch.seed),
        challenge_type=int(launch.challenge_type),
        type_label=type_label,
        task=task,
        result=result,
    )
    trajectory = FlyTrajectory(meta=saved_meta, frames=trajectory_frames)
    traj_path = save_trajectory(trajectory, repo_root=repo_root)
    return {
        "path": traj_path,
        "result": result,
        "seed": int(launch.seed),
        "challenge_type": int(launch.challenge_type),
        "type_label": type_label,
        "success": success,
    }


def run_batch(
    *,
    launch: FlyLaunchConfig,
    jobs: Sequence[FlyBatchJob],
    repo_root: Path,
    debug: bool = False,
    debug_every: int = 25,
) -> list[dict[str, Any]]:
    """Run many episodes headlessly and save each trajectory under fly_runs/."""
    model_path = launch.agent_path if launch.agent_kind == "zip" else None
    source_path = launch.agent_path if launch.agent_kind == "source" else None
    first_seed = int(jobs[0].seed) if jobs else int(launch.seed)
    agent_dir, temp_dir = _prepare_agent_dir(
        model_path=model_path,
        source_path=source_path,
        seed=first_seed,
    )

    results: list[dict[str, Any]] = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            agent = _load_agent(agent_dir)
        lo = hi = None

        total = len(jobs)
        for index, job in enumerate(jobs, start=1):
            job_launch = FlyLaunchConfig(
                agent_path=launch.agent_path,
                agent_kind=launch.agent_kind,
                seed=int(job.seed),
                challenge_type=int(job.challenge_type),
            )
            if lo is None or hi is None:
                probe_task = task_for_seed_and_type(
                    SIM_DT,
                    seed=int(job.seed),
                    challenge_type=int(job.challenge_type),
                )
                probe_env = make_env(probe_task, gui=False)
                lo = probe_env.action_space.low.flatten()
                hi = probe_env.action_space.high.flatten()
                probe_env.close()

            reset_fn = getattr(agent, "reset", None)
            if callable(reset_fn):
                with contextlib.redirect_stdout(io.StringIO()):
                    reset_fn()

            print(
                f"[{index}/{total}] type{job.challenge_type} {job.type_label} "
                f"seed{job.seed} ...",
                flush=True,
            )
            outcome = run_headless_episode(
                launch=job_launch,
                repo_root=repo_root,
                agent=agent,
                lo=lo,
                hi=hi,
                debug=debug,
                debug_every=debug_every,
            )
            summary = format_score_summary(outcome["result"]) or ""
            status = "success" if outcome["success"] else "failed"
            print(
                f"    {status}  {summary}  -> {outcome['path'].name}",
                flush=True,
            )
            results.append(outcome)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def print_batch_summary(results: Sequence[dict[str, Any]]) -> None:
    if not results:
        print("No batch runs completed.")
        return
    successes = sum(1 for item in results if item.get("success"))
    scores = [float(item["result"]["score"]) for item in results if item.get("result")]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Runs       : {len(results)}")
    print(f"Successes  : {successes}/{len(results)}")
    print(f"Avg score  : {avg_score:.3f}")
    print("Saved under fly_runs/ — open them later with `swarm fly` → Open Run.")
    print("=" * 60)
