#!/usr/bin/env python3
"""Minimal custom controller example for the AIO-Gym backend."""
from __future__ import annotations

import numpy as np

if __package__ is None or __package__ == "":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from aiogym.controllers import CONTROLLER_API_VERSION, ControllerContext, make_controller, register_controller
from aiogym.evaluation import BenchmarkProtocol, evaluate_controller


class ConstantTrimController:
    """A tiny actuator controller that holds a configurable normalized trim."""

    name = "Constant-trim"
    action_mode = "actuator"
    control_structure = "custom_constant_trim"
    controller_api_version = CONTROLLER_API_VERSION

    def __init__(self, model, trim=0.5):
        self.model = model
        self.trim = float(trim)
        self.nu = sum(model.actuator_counts())

    def reset(self, seed=None):
        return None

    def act(self, obs: np.ndarray, context: ControllerContext) -> np.ndarray:
        return np.full(self.nu, self.trim, dtype=np.float32)

    def metadata(self):
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "api": self.controller_api_version,
            "scenario": self.model.scenario,
            "action_mode": self.action_mode,
            "control_structure": self.control_structure,
            "trim": self.trim,
        }


def constant_trim_factory(model=None, scenario=None, config=None, policy=None):
    params = dict(config or {}).get("parameters", {})
    return ConstantTrimController(model, trim=params.get("trim", 0.5))


def main():
    register_controller("constant_trim", constant_trim_factory)
    protocol = BenchmarkProtocol.tracking(
        "cstr",
        episode_steps=8,
        dynamic=False,
        randomize=False,
        randomize_setpoints=False,
        randomize_plant=False,
        plant_drift=False,
    )
    controller = make_controller("constant_trim", scenario="cstr", config={"parameters": {"trim": 0.45}})
    result = evaluate_controller(controller, protocol.make_env(), seed_list=[0], protocol=protocol)
    print({
        "controller": result["name"],
        "objective": result["objective"],
        "metric": result["metric"],
        "value": round(result[result["metric"]], 3),
        "status": result["controller_status"],
    })


if __name__ == "__main__":
    main()
