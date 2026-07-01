"""Launch configuration UI for ``swarm fly``."""
from __future__ import annotations

import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

AgentKind = Literal["zip", "source"]

MAP_TYPE_CHOICES: tuple[tuple[int, str], ...] = (
    (1, "City"),
    (2, "Open"),
    (3, "Mountain"),
    (4, "Village"),
    (5, "Warehouse"),
    (6, "Forest"),
)


@dataclass(frozen=True)
class AgentOption:
    label: str
    path: Path
    kind: AgentKind


@dataclass(frozen=True)
class FlyLaunchConfig:
    agent_path: Path
    agent_kind: AgentKind
    seed: int
    challenge_type: int


def resolve_agent_path(path: Path) -> tuple[Path, AgentKind] | None:
    path = path.resolve()
    if _is_agent_source(path):
        return path, "source"
    if _is_agent_zip(path):
        return path, "zip"
    return None


def last_agent_config_path() -> Path:
    return Path.home() / ".config" / "swarm" / "last_agent.json"


def load_last_agent_path() -> Path | None:
    config_path = last_agent_config_path()
    if not config_path.is_file():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    raw = data.get("path") if isinstance(data, dict) else None
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if resolve_agent_path(path) is None:
        return None
    return path.resolve()


def save_last_agent_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolve_agent_path(resolved) is None:
        raise ValueError(f"Not a valid agent path: {resolved}")
    config_path = last_agent_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"path": str(resolved)}
    kind = resolve_agent_path(resolved)
    if kind is not None:
        payload["kind"] = kind[1]
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return resolved


def _is_agent_source(path: Path) -> bool:
    return path.is_dir() and (path / "drone_agent.py").is_file()


def _is_agent_zip(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".zip":
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return any(name.endswith("drone_agent.py") for name in zf.namelist())
    except Exception:
        return False


def _append_agent(
    options: list[AgentOption],
    seen: set[str],
    path: Path,
    *,
    label: str | None = None,
) -> None:
    path = path.resolve()
    key = str(path)
    if key in seen or not path.exists():
        return
    if _is_agent_source(path):
        options.append(AgentOption(label=label or path.name, path=path, kind="source"))
        seen.add(key)
    elif _is_agent_zip(path):
        options.append(AgentOption(label=label or path.name, path=path, kind="zip"))
        seen.add(key)


def discover_agents(*, repo_root: Path, cwd: Path | None = None) -> list[AgentOption]:
    """Find local agent zips and source folders for the setup screen."""
    cwd = (cwd or Path.cwd()).resolve()
    repo_root = repo_root.resolve()
    options: list[AgentOption] = []
    seen: set[str] = set()

    for pattern in ("champion_UID_*.zip", "submission.zip", "*.zip"):
        for path in sorted(cwd.glob(pattern)):
            _append_agent(options, seen, path)

    for path in (
        repo_root / "Submission" / "submission.zip",
        cwd / "Submission" / "submission.zip",
        repo_root / "swarm" / "submission_template",
    ):
        _append_agent(options, seen, path)

    search_roots = {cwd, repo_root}
    for root in search_roots:
        if _is_agent_source(root):
            _append_agent(options, seen, root, label=root.name)
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                _append_agent(options, seen, child)

    return options


def _truncate(text: str, max_len: int = 52) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


@dataclass(frozen=True)
class _Button:
    key: str
    label: str
    rect: tuple[int, int, int, int]


class FlySetupWindow:
    """Initial picker for agent, seed, and map type."""

    WIDTH = 920
    HEIGHT = 620

    def __init__(
        self,
        agents: list[AgentOption],
        *,
        seed: int = 42,
        challenge_type: int = 1,
        selected_index: int = 0,
    ) -> None:
        import pygame

        if not agents:
            raise ValueError("No local agents found. Add a folder with drone_agent.py or a submission zip.")
        self._pygame = pygame
        self.agents = agents
        self.seed = int(seed)
        self.challenge_type = int(challenge_type)
        self.selected_index = max(0, min(selected_index, len(agents) - 1))
        self.agent_scroll = 0
        self.custom_path = ""
        self.custom_active = False
        self.confirmed: FlyLaunchConfig | None = None
        self.cancelled = False
        self._message = "Select agent, seed, and map type. Then click Build & Open."

        pygame.init()
        pygame.display.set_caption("Swarm Fly — Setup")
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self._font = pygame.font.SysFont("dejavusans", 16)
        self._font_small = pygame.font.SysFont("dejavusans", 14)
        self._font_title = pygame.font.SysFont("dejavusans", 22, bold=True)
        self._buttons = self._layout_buttons()

    def _layout_buttons(self) -> list[_Button]:
        y = 500
        w, h, gap = 130, 36, 10
        x0 = 30
        buttons = [
            _Button("seed_m100", "Seed -100", (x0, y, w, h)),
            _Button("seed_m1", "Seed -1", (x0 + w + gap, y, w, h)),
            _Button("seed_p1", "Seed +1", (x0 + 2 * (w + gap), y, w, h)),
            _Button("seed_p100", "Seed +100", (x0 + 3 * (w + gap), y, w, h)),
            _Button("seed_rand", "Random Seed", (x0 + 4 * (w + gap), y, w, h)),
            _Button("build", "Build & Open", (self.WIDTH - 250, self.HEIGHT - 58, 150, 40)),
            _Button("quit", "Quit", (self.WIDTH - 90, self.HEIGHT - 58, 70, 40)),
        ]
        for idx, (challenge_type, label) in enumerate(MAP_TYPE_CHOICES):
            row, col = divmod(idx, 3)
            buttons.append(
                _Button(
                    f"type_{challenge_type}",
                    label,
                    (500 + col * 130, 120 + row * 46, 120, 36),
                )
            )
        return buttons

    def _agent_row_rect(self, row_index: int) -> tuple[int, int, int, int]:
        return (30, 118 + row_index * 30, 430, 26)

    def _custom_field_rect(self) -> tuple[int, int, int, int]:
        return (30, 430, 430, 34)

    def _button_at(self, pos: tuple[int, int]) -> str | None:
        x, y = pos
        for button in self._buttons:
            bx, by, bw, bh = button.rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return button.key
        visible_rows = 9
        for row in range(visible_rows):
            rect = self._agent_row_rect(row)
            bx, by, bw, bh = rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return f"agent_{self.agent_scroll + row}"
        field = self._custom_field_rect()
        bx, by, bw, bh = field
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return "custom_field"
        return None

    def _selected_agent(self) -> AgentOption | None:
        if self.custom_path.strip():
            path = Path(self.custom_path.strip()).expanduser()
            if _is_agent_source(path):
                return AgentOption(label=path.name, path=path.resolve(), kind="source")
            if _is_agent_zip(path):
                return AgentOption(label=path.name, path=path.resolve(), kind="zip")
            return None
        return self.agents[self.selected_index]

    def _confirm(self) -> None:
        agent = self._selected_agent()
        if agent is None:
            self._message = "Select a valid agent or enter a custom path to drone_agent.py / zip."
            return
        if self.challenge_type not in range(1, 7):
            self._message = "Choose a map type."
            return
        self.confirmed = FlyLaunchConfig(
            agent_path=agent.path,
            agent_kind=agent.kind,
            seed=max(1, int(self.seed)),
            challenge_type=int(self.challenge_type),
        )

    def run(self) -> FlyLaunchConfig | None:
        pygame = self._pygame
        clock = pygame.time.Clock()
        while not self.cancelled and self.confirmed is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.cancelled = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.cancelled = True
                    elif self.custom_active:
                        if event.key == pygame.K_BACKSPACE:
                            self.custom_path = self.custom_path[:-1]
                        elif event.key == pygame.K_RETURN:
                            self.custom_active = False
                            self._confirm()
                        elif event.unicode and event.unicode.isprintable():
                            self.custom_path += event.unicode
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        key = self._button_at(event.pos)
                        if key == "custom_field":
                            self.custom_active = True
                        else:
                            self.custom_active = False
                        if key == "build":
                            self._confirm()
                        elif key == "quit":
                            self.cancelled = True
                        elif key == "seed_m100":
                            self.seed = max(1, self.seed - 100)
                        elif key == "seed_m1":
                            self.seed = max(1, self.seed - 1)
                        elif key == "seed_p1":
                            self.seed += 1
                        elif key == "seed_p100":
                            self.seed += 100
                        elif key == "seed_rand":
                            self.seed = random.randint(1, 999_999)
                        elif key and key.startswith("type_"):
                            self.challenge_type = int(key.split("_", 1)[1])
                        elif key and key.startswith("agent_"):
                            self.selected_index = int(key.split("_", 1)[1])
                            self.custom_path = ""
                    elif event.button == 4:
                        self.agent_scroll = max(0, self.agent_scroll - 1)
                    elif event.button == 5:
                        max_scroll = max(0, len(self.agents) - 9)
                        self.agent_scroll = min(max_scroll, self.agent_scroll + 1)

            self._draw()
            clock.tick(60)
        self.close()
        return self.confirmed

    def _draw_button(self, button: _Button, *, active: bool = False) -> None:
        pygame = self._pygame
        x, y, w, h = button.rect
        color = (46, 125, 50) if active else (55, 55, 60)
        pygame.draw.rect(self.screen, color, (x, y, w, h), border_radius=6)
        pygame.draw.rect(self.screen, (90, 90, 95), (x, y, w, h), width=1, border_radius=6)
        label = self._font_small.render(button.label, True, (235, 235, 235))
        self.screen.blit(label, label.get_rect(center=(x + w // 2, y + h // 2)))

    def _draw(self) -> None:
        pygame = self._pygame
        self.screen.fill((24, 24, 28))
        title = self._font_title.render("Swarm Fly Setup", True, (240, 240, 240))
        self.screen.blit(title, (30, 20))
        subtitle = self._font.render(self._message, True, (170, 200, 255))
        self.screen.blit(subtitle, (30, 52))

        agent_title = self._font.render("Drone Agent", True, (220, 220, 220))
        self.screen.blit(agent_title, (30, 88))
        for row in range(9):
            index = self.agent_scroll + row
            if index >= len(self.agents):
                break
            option = self.agents[index]
            rect = self._agent_row_rect(row)
            selected = (not self.custom_path.strip()) and index == self.selected_index
            color = (40, 70, 45) if selected else (34, 34, 38)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            kind = "zip" if option.kind == "zip" else "src"
            text = self._font_small.render(
                f"[{kind}] {_truncate(option.label)}",
                True,
                (235, 235, 235),
            )
            self.screen.blit(text, (rect[0] + 8, rect[1] + 5))

        custom_label = self._font_small.render("Custom path (zip or folder):", True, (190, 190, 190))
        self.screen.blit(custom_label, (30, 408))
        field = self._custom_field_rect()
        field_color = (48, 48, 58) if self.custom_active else (34, 34, 38)
        pygame.draw.rect(self.screen, field_color, field, border_radius=4)
        pygame.draw.rect(self.screen, (90, 90, 95), field, width=1, border_radius=4)
        field_text = self.custom_path + ("|" if self.custom_active else "")
        self.screen.blit(self._font_small.render(field_text or "Click to type a path...", True, (220, 220, 220)), (field[0] + 8, field[1] + 8))

        map_title = self._font.render("Map Type", True, (220, 220, 220))
        self.screen.blit(map_title, (500, 88))
        seed_title = self._font.render(f"Seed: {self.seed}", True, (220, 220, 220))
        self.screen.blit(seed_title, (500, 250))

        for button in self._buttons:
            active = False
            if button.key == f"type_{self.challenge_type}":
                active = True
            if button.key == "build":
                active = False
            self._draw_button(button, active=active)

        pygame.display.flip()

    def close(self) -> None:
        try:
            self._pygame.quit()
        except Exception:
            pass


def run_fly_setup(
    *,
    repo_root: Path,
    cwd: Path | None = None,
    seed: int = 42,
    challenge_type: int = 1,
    agent_path: Path | None = None,
) -> FlyLaunchConfig | None:
    agents = discover_agents(repo_root=repo_root, cwd=cwd)
    selected_index = 0
    if agent_path is not None:
        resolved = agent_path.resolve()
        for idx, option in enumerate(agents):
            if option.path == resolved:
                selected_index = idx
                break
    window = FlySetupWindow(
        agents,
        seed=seed,
        challenge_type=challenge_type,
        selected_index=selected_index,
    )
    if agent_path is not None and not any(a.path.resolve() == agent_path.resolve() for a in agents):
        window.custom_path = str(agent_path)
    return window.run()
