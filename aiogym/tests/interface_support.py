#!/usr/bin/env python3
"""Shared fixtures and helpers for AIO-Gym interface tests."""
import json
import os
import tempfile

import numpy as np
import gymnasium as gym
import aiogym  # registers ids
from aiogym.controllers import as_controller, load_controller_config, make_controller, registered_controllers
from aiogym.env import AIOGymNativeEnv
from aiogym.evaluation import (
    BenchmarkCase,
    BenchmarkProtocol,
    build_evaluation_report,
    evaluate_controller,
    rollout_controller,
)
from aiogym.models import BUILTIN_MODELS, SCENARIOS, ProcessModelContract, builtin_gym_ids, define_model, make_model, register_model, unregister_model

OK = "OK"
try:
    import casadi as ca
except Exception:
    ca = None


def check(name, cond):
    print(f"  {OK if cond else 'FAIL'}  {name}")
    assert cond, name




class MiniTankModel(ProcessModelContract):
    scenario = "mini_tank"
    display_name = "Mini heated tank"
    summary = "Tiny custom model used to verify model registration and direct injection."
    n = 1
    dt_micro = 0.02
    state_names = ("h0", "T0")
    state_units = {"h0": "m", "T0": "degC"}
    state_bounds = {"h0": (0.0, 1.0), "T0": (0.0, 120.0)}
    action_names = ("feed_pump", "heater_0")
    action_kinds = {"feed_pump": "pump", "heater_0": "heater"}
    output_names = ("tank_temperature",)
    output_units = {"tank_temperature": "degC"}
    output_bounds = {"tank_temperature": (20.0, 80.0)}
    param_units = {"area": "m2", "pump_flow_max": "m3/s", "heater_gain": "degC/s", "t_cold": "degC", "t_amb": "degC"}
    param_bounds = {"area": (0.01, 2.0), "pump_flow_max": (0.0, 0.02), "heater_gain": (0.0, 20.0), "t_cold": (0.0, 40.0), "t_amb": (0.0, 45.0)}
    input_disturbances = ProcessModelContract.input_disturbances + (
        {"name": "feed_bias", "event": "feed_bias_step", "unit": "m3/s", "bounds": (-0.0001, 0.0001), "default": 0.0},
    )
    safety_constraints = ({"name": "level_bounds", "states": ("h0",), "bounds": (0.0, 1.0)},)

    def __init__(self):
        self.p = {"area": 0.2, "pump_flow_max": 0.001, "heater_gain": 4.0, "t_cold": 15.0, "t_amb": 20.0}

    def dynamics(self, x, u, env=None, backend="numeric", ca=None):
        values = self.action_vector(u)
        env = env or {}
        flow = values[0] * self.p["pump_flow_max"] + env.get("feed_bias", 0.0)
        dh = flow / self.p["area"]
        dT = 0.02 * (env["t_cold"] - x[1]) + values[1] * self.p["heater_gain"]
        return [dh, dT]

    def display_outputs(self, x, backend="numeric", ca=None):
        return {"levels": [max(0.0, x[0])], "temps": [x[1]]}

    def controlled_output(self, x, backend="numeric", ca=None):
        return [x[1]]

    def initial_state(self):
        return [0.25, 20.0]

    def clamp_state(self, x):
        return [min(max(x[0], 0.0), 1.0), x[1]]


class GenericVectorModel(ProcessModelContract):
    scenario = "generic_vector"
    display_name = "Generic vector model"
    summary = "Custom model that uses only generic x/u/y vectors."
    n = 1
    dt_micro = 0.02
    action_names = ("coolant_flow", "agitator_speed")
    action_units = {"coolant_flow": "kg/s", "agitator_speed": "rpm"}
    action_bounds = {"coolant_flow": (0.0, 1.0), "agitator_speed": (0.0, 1.0)}
    action_kinds = {"coolant_flow": "flow", "agitator_speed": "rotation"}
    output_names = ("product_quality",)
    output_bounds = {"product_quality": (0.0, 1.0)}
    default_y_sp = (0.6,)
    safety_constraints = ({"name": "quality_bounds", "states": ("x0",), "bounds": (0.0, 1.0)},)
    plant_regime = {"gain": (0.8, 1.2)}
    economic_config = {"temp_band": ((None, None),), "level_band": (), "value": "none", "w_value": 0.0, "w_energy": 0.0, "w_viol": 0.0}

    def __init__(self):
        self.p = {"gain": 0.4}

    def initial_state(self):
        return [0.25]

    def dynamics(self, x, u, env=None, backend="numeric", ca=None):
        if isinstance(u, dict):
            raise TypeError("generic vector model expects flat u")
        return [-0.2 * (x[0] - 0.5) + self.p["gain"] * float(u[0]) - 0.05 * float(u[1])]

    def controlled_output(self, x, backend="numeric", ca=None):
        return [x[0]]
