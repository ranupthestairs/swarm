import os
import sys
import importlib.util

_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


uid237 = _load("_route_uid237", "uid237.py")
best = _load("_route_best", "best.py")
forest = _load("_route_forest", "forest.py")

BEST_LABEL = ("village", "warehouse")
_FOREST_LABELS = ("forest",)
_LOCK_STABLE_TICKS = 16
_LOCK_MIN_TICK = 150

class DroneFlightController:
    def __init__(self, **kwargs):
        self._agent_uid237 = uid237.DroneFlightController()
        self._agent_best = best.DroneFlightController()
        self._agent_forest = forest.DroneFlightController()

    def reset(self):
        for a in (self._agent_uid237, self._agent_best, self._agent_forest):
            if hasattr(a, "reset"):
                a.reset()
        self._t = 0
        self._stable_label = None
        self._stable_count = 0
        self._route_locked = False
        self._forest_warmed = False

    def act(self, observation):
        if self._route_locked:
            action = self._agent_forest.act(observation)
            self._t += 1
            return action
        a_uid237 = self._agent_uid237.act(observation)
        a_best = self._agent_best.act(observation)
        a_forest = self._agent_forest.act(observation)
        label = getattr(self._agent_best, "_map_prediction_label", None)
                # Stability tracking on the label.
        if label is not None and label == self._stable_label:
            self._stable_count += 1
        else:
            self._stable_label = label
            self._stable_count = 1

        is_forest = label in _FOREST_LABELS

        if is_forest:
            # Warm the forest engine each tick once we've first seen "forest" so its
            # internal state (trackers, filters) is primed before we lock to it.
            self._forest_warmed = True
            try:
                self._agent_forest.act(observation)
            except Exception:
                pass

            # Lock to forest once the label has been stably "forest" long enough.
            if self._stable_count >= _LOCK_STABLE_TICKS and self._t >= _LOCK_MIN_TICK:
                self._route_locked = True
        self._t += 1
        if label is None or label in BEST_LABEL:
            return a_best
        else:
            return a_uid237
