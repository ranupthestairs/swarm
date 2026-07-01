from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from scripts.fly_model import _handle_start_pause


def _window() -> MagicMock:
    window = MagicMock()
    window.set_status = MagicMock()
    window.set_export_enabled = MagicMock()
    window.set_replay_enabled = MagicMock()
    window.set_replay_ui_active = MagicMock()
    return window


def _ui(*, start: bool = False, pause: bool = False) -> SimpleNamespace:
    return SimpleNamespace(start=start, pause=pause)


def test_pause_toggles_running_and_paused() -> None:
    window = _window()
    env = object()

    state, reset_episode, *_ = _handle_start_pause(
        ui=_ui(pause=True),
        window=window,
        sim_state="running",
        env=env,
        agent=None,
        task=None,
    )
    assert state == "paused"
    assert reset_episode is False

    state, reset_episode, *_ = _handle_start_pause(
        ui=_ui(pause=True),
        window=window,
        sim_state="paused",
        env=env,
        agent=None,
        task=None,
    )
    assert state == "running"
    assert reset_episode is False


def test_start_from_finished_requests_reset() -> None:
    window = _window()
    env = object()

    state, reset_episode, *_ = _handle_start_pause(
        ui=_ui(start=True),
        window=window,
        sim_state="finished",
        env=env,
        agent=None,
        task=None,
    )
    assert state == "running"
    assert reset_episode is True


def test_start_from_ready_does_not_reset() -> None:
    window = _window()

    state, reset_episode, *_ = _handle_start_pause(
        ui=_ui(start=True),
        window=window,
        sim_state="ready",
        env=object(),
        agent=None,
        task=None,
    )
    assert state == "running"
    assert reset_episode is False


def test_start_aborts_active_replay() -> None:
    window = _window()

    state, reset_episode, *_ = _handle_start_pause(
        ui=_ui(start=True),
        window=window,
        sim_state="replay",
        env=object(),
        agent=None,
        task=None,
    )
    assert state == "running"
    assert reset_episode is True
    window.set_replay_ui_active.assert_called_once_with(False)
