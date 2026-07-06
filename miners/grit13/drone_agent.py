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


class DroneFlightController:
    def __init__(self, **kwargs):
        self._agent_uid237 = uid237.DroneFlightController()
        self._agent_best = best.DroneFlightController()
        self._agent_forest = forest.DroneFlightController()

    def reset(self):
        for a in (self._agent_uid237, self._agent_best, self._agent_forest):
            if hasattr(a, "reset"):
                a.reset()

    def act(self, observation):
        a_uid237 = self._agent_uid237.act(observation)
        a_best = self._agent_best.act(observation)
        a_forest = self._agent_forest.act(observation)
        label = getattr(self._agent_best, "_map_prediction_label", None)
        if label is None or label in BEST_LABEL:
            return a_best
        elif label == "forest":
            return a_forest
        else:
            return a_uid237
