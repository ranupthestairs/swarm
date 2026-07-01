from __future__ import annotations

import json
import zipfile
from pathlib import Path

from swarm.core.fly_setup import (
    discover_agents,
    load_last_agent_path,
    save_last_agent_path,
)


def test_discover_agents_finds_source_and_zip(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    agent_dir = repo / "my_agent"
    agent_dir.mkdir()
    (agent_dir / "drone_agent.py").write_text("class DroneFlightController:\n    pass\n")

    zip_path = tmp_path / "champion_UID_9.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("drone_agent.py", "class DroneFlightController:\n    pass\n")

    options = discover_agents(repo_root=repo, cwd=tmp_path)
    labels = {option.label for option in options}
    assert "my_agent" in labels
    assert zip_path.name in labels


def test_save_and_load_last_agent_path(tmp_path: Path, monkeypatch) -> None:
    agent_dir = tmp_path / "my_agent"
    agent_dir.mkdir()
    (agent_dir / "drone_agent.py").write_text("class DroneFlightController:\n    pass\n")
    config_path = tmp_path / "last_agent.json"
    monkeypatch.setattr(
        "swarm.core.fly_setup.last_agent_config_path",
        lambda: config_path,
    )

    saved = save_last_agent_path(agent_dir)
    assert saved == agent_dir.resolve()
    assert load_last_agent_path() == agent_dir.resolve()
    assert json.loads(config_path.read_text(encoding="utf-8"))["path"] == str(agent_dir.resolve())
