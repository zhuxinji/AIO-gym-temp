#!/usr/bin/env python3
"""Training-interface tests for AIO-Gym.

Exercises the Gymnasium/RL surface that
PC-Gym-style benchmarking and parallel training depend on. Run: python aiogym/tests/test_interface.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
import aiogym  # registers ids
from aiogym.controllers import as_controller, load_controller_config, make_controller, registered_controllers
from aiogym.env import AIOGymNativeEnv
from aiogym.evaluation import (
    BenchmarkConfig,
    BenchmarkProtocol,
    build_evaluation_report,
    evaluate_controller,
    rollout_controller,
)
from aiogym.models import BUILTIN_MODELS, SCENARIOS, ProcessModelContract, builtin_gym_ids, make_model, register_model, unregister_model

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
    param_units = {"area": "m2", "pump_flow_max": "m3/s", "heater_gain": "degC/s", "t_cold": "degC", "t_amb": "degC"}
    param_bounds = {"area": (0.01, 2.0), "pump_flow_max": (0.0, 0.02), "heater_gain": (0.0, 20.0), "t_cold": (0.0, 40.0), "t_amb": (0.0, 45.0)}
    input_disturbances = ProcessModelContract.input_disturbances + (
        {"name": "feed_bias", "event": "feed_bias_step", "unit": "m3/s", "bounds": (-0.0001, 0.0001), "default": 0.0},
    )
    safety_constraints = ({"name": "level_bounds", "states": ("h0",), "bounds": (0.0, 1.0)},)

    def __init__(self):
        self.p = {"area": 0.2, "pump_flow_max": 0.001, "heater_gain": 4.0, "t_cold": 15.0, "t_amb": 20.0}

    def actuator_counts(self):
        return (1, 0, 1)

    def derivatives(self, x, act, env):
        flow = act["pumps"][0] * self.p["pump_flow_max"] + env.get("feed_bias", 0.0)
        dh = flow / self.p["area"]
        dT = 0.02 * (env["t_cold"] - x[1]) + act["heaters"][0] * self.p["heater_gain"]
        return [dh, dT]

    def levels_temps(self, x):
        return [max(0.0, x[0])], [x[1]]

    def initial_state(self):
        return [0.25, 20.0]

    def clamp_state(self, x):
        return [min(max(x[0], 0.0), 1.0), x[1]]


def test_model_contract():
    """Every process model exposes a usable backend contract for tooling."""
    for scn in SCENARIOS:
        model = make_model(scn)
        card = model.model_card()
        n_act = sum(model.actuator_counts())
        params = card["parameters"]
        state_ok = len(card["states"]) == len(model.initial_state())
        action_ok = len(card["actions"]) == n_act
        vector_ok = card["state_vector"] == {"name": "x", "length": len(model.initial_state())}
        vector_ok = vector_ok and card["action_vector"] == {"name": "u", "length": n_act}
        vector_ok = vector_ok and card["dynamics_disturbances"] == list(model.dynamics_disturbance_names())
        param_ok = bool(params) and all("value" in row and "bounds" in row and "unit" in row for row in params.values())
        bounds_ok = all("bounds" in row and row["bounds"] is not None for row in card["states"] + card["actions"])
        meta_ok = card["scenario"] == scn and bool(card["name"]) and bool(card["constraints"])
        meta_ok = meta_ok and bool(card["plant_regime"]) and bool(card["economic_config"])
        meta_ok = meta_ok and isinstance(card["supervisory_layout"], list)
        u = [0.5] * n_act
        act = model.action_vector_to_dict(u)
        env = model.disturbance_defaults()
        dx_generic = model.dynamics(model.initial_state(), u, env)
        dx_dynamics_alias = model.dynamics(model.initial_state(), act, env)
        dx_casadi = dx_generic
        casadi_ok = len(model.disturbance_vector(env)) == len(model.dynamics_disturbance_names())
        if ca is not None:
            dx_casadi = np.asarray(model.dynamics(
                ca.DM(model.initial_state()),
                ca.DM(u),
                ca.DM(model.disturbance_vector(env)),
                backend="casadi",
                ca=ca,
            ), dtype=float).reshape(-1)
            casadi_ok = casadi_ok and np.allclose(dx_casadi, dx_generic)
        meas = model.measurement(model.initial_state(), env)
        generic_ok = np.allclose(model.action_vector(act), u)
        generic_ok = generic_ok and np.allclose(dx_generic, dx_dynamics_alias)
        generic_ok = generic_ok and casadi_ok
        generic_ok = generic_ok and meas["x"] == model.initial_state() and "y" in meas
        generic_ok = generic_ok and "levels" in meas and "temps" in meas
        check(f"{scn:10s} model contract states={len(card['states'])} actions={len(card['actions'])} params={len(params)}", state_ok and action_ok and vector_ok and param_ok and bounds_ok and meta_ok and generic_ok)


def test_custom_model_entrypoints():
    """Custom process models can be registered or passed directly to the env."""
    try:
        unregister_model(next(iter(BUILTIN_MODELS)))
        built_in_guard_ok = False
    except ValueError:
        built_in_guard_ok = True

    register_model("mini_tank", MiniTankModel)
    try:
        model = make_model("mini_tank")
        card = model.model_card()
        contract_ok = card["scenario"] == "mini_tank" and card["disturbance_defaults"]["feed_bias"] == 0.0
        env = AIOGymNativeEnv("mini_tank", dynamic=False, randomize=False, randomize_setpoints=False)
        obs, _ = env.reset(seed=0)
        obs2, reward, term, trunc, info = env.step(np.array([0.5, 0.5], np.float32))
        registered_ok = env.scenario == "mini_tank" and env._env()["feed_bias"] == 0.0
        registered_ok = registered_ok and env.observation_space.contains(obs) and np.isfinite(reward)

        direct_env = AIOGymNativeEnv(custom_model=MiniTankModel(), dynamic=False, randomize=False, randomize_setpoints=False)
        direct_env.reset(seed=0)
        direct_env._apply_disturbance("feed_bias_step")
        sampled = direct_env._env()["feed_bias"]
        direct_ok = direct_env.scenario == "mini_tank" and -0.0001 <= sampled <= 0.0001

        check("custom model registration + direct injection", built_in_guard_ok and contract_ok and registered_ok and direct_ok)
    finally:
        unregister_model("mini_tank")





def test_disturbance_schedule_config():
    """Dynamic disturbances are scheduled from the model disturbance schema."""
    for scn in SCENARIOS:
        env = AIOGymNativeEnv(scn, dynamic=True, randomize=False, randomize_setpoints=False)
        env.reset(seed=0)
        configured = {
            row["event"]
            for row in env.model.disturbance_schema()
            if row.get("dynamic", False) and row.get("event")
        }
        scheduled = [event for _, event in env._dist_events]
        names_ok = bool(scheduled) and all(isinstance(event, str) and event in configured for event in scheduled)
        check(f"{scn:10s} disturbance schedule uses model schema", names_ok)

def test_cstr_disturbance_semantics():
    """CSTR-specific disturbances should affect the backend model in the expected direction."""
    model = make_model("cstr")
    x = [0.5, 60.0]
    act = {"pumps": [0.5], "valves": [], "heaters": [0.5]}
    base = {"t_cold": 20.0, "t_amb": 20.0, "extra_outflow": 0.0}
    lean = {**base, "Caf": 0.6}
    rich = {**base, "Caf": 1.4}
    cold_coolant = {**base, "Tcool": 0.0}
    warm_coolant = {**base, "Tcool": 20.0}

    d_lean = model.dynamics(x, act, lean)
    d_base = model.dynamics(x, act, base)
    d_rich = model.dynamics(x, act, rich)
    d_cold = model.dynamics(x, act, cold_coolant)
    d_warm = model.dynamics(x, act, warm_coolant)

    check("CSTR Caf raises concentration derivative", d_rich[0] > d_base[0] > d_lean[0])
    check("CSTR colder coolant lowers temperature derivative", d_cold[1] < d_warm[1])
    check("CSTR production uses feed concentration", model.production(x, act, rich) > model.production(x, act, base) > model.production(x, act, lean))

    env = AIOGymNativeEnv("cstr", reward_mode="economic", action_mode="actuator", dynamic=False,
                          randomize=False, randomize_setpoints=False)
    env.reset(seed=0)
    env.caf = 1.25
    env.tcool = 3.0
    _, _, _, _, info = env.step(np.array([0.5, 0.5], np.float32))
    env_ok = info["cstr_feed_conc"] == 1.25 and info["cstr_coolant_temp"] == 3.0
    env_ok = env_ok and 0.0 <= info["cstr_conversion"] <= 1.0 and "cstr_ca_high" in info["cons_info"]
    check("CSTR env reports feed/coolant/conversion", env_ok)



def test_process_disturbance_semantics():
    """Cascade, quadruple, and HVAC process disturbances should have sane directions."""
    tank_env = {"t_cold": 15.0, "t_amb": 20.0, "extra_outflow": 0.0}

    cascade = make_model("cascade")
    cx = [0.30, 50.0, 0.30, 50.0, 0.30, 50.0]
    cact = {"pumps": [0.5], "valves": [0.5, 0.5, 0.5], "heaters": [0.5, 0.5, 0.5]}
    c_low_pump = cascade.dynamics(cx, cact, {**tank_env, "pump_flow_factor": 0.7})
    c_high_pump = cascade.dynamics(cx, cact, {**tank_env, "pump_flow_factor": 1.3})
    c_low_heat = cascade.dynamics(cx, cact, {**tank_env, "heater_efficiency": 0.6})
    c_high_heat = cascade.dynamics(cx, cact, {**tank_env, "heater_efficiency": 1.2})
    c_low_loss = cascade.dynamics(cx, cact, {**tank_env, "heat_loss_factor": 0.5})
    c_high_loss = cascade.dynamics(cx, cact, {**tank_env, "heat_loss_factor": 2.0})
    check("Cascade pump factor raises inlet-level derivative", c_high_pump[0] > c_low_pump[0])
    check("Cascade heater efficiency raises temp derivative", c_high_heat[1] > c_low_heat[1])
    check("Cascade heat-loss factor lowers temp derivative", c_high_loss[1] < c_low_loss[1])

    quad = make_model("quadruple")
    qx = [0.25, 50.0, 0.25, 50.0, 0.12, 45.0, 0.12, 45.0]
    qact = {"pumps": [0.5, 0.5], "valves": [], "heaters": [0.5, 0.5, 0.5, 0.5]}
    q_low_pump = quad.dynamics(qx, qact, {**tank_env, "pump_flow_factor": 0.7})
    q_high_pump = quad.dynamics(qx, qact, {**tank_env, "pump_flow_factor": 1.3})
    q_low_heat = quad.dynamics(qx, qact, {**tank_env, "heater_efficiency": 0.6})
    q_high_heat = quad.dynamics(qx, qact, {**tank_env, "heater_efficiency": 1.2})
    q_low_loss = quad.dynamics(qx, qact, {**tank_env, "heat_loss_factor": 0.5})
    q_high_loss = quad.dynamics(qx, qact, {**tank_env, "heat_loss_factor": 2.0})
    check("Quadruple pump factor raises lower-tank derivative", q_high_pump[0] > q_low_pump[0])
    check("Quadruple heater efficiency raises temp derivative", q_high_heat[1] > q_low_heat[1])
    check("Quadruple heat-loss factor lowers temp derivative", q_high_loss[1] < q_low_loss[1])

    hvac = make_model("hvac")
    hx = [20.0, 20.0]
    hact = {"pumps": [], "valves": [], "heaters": [0.75, 0.75]}
    h_base = {"t_cold": 5.0, "t_amb": 5.0, "extra_outflow": 0.0}
    h_low_eff = hvac.dynamics(hx, hact, {**h_base, "hvac_efficiency": 0.6})
    h_high_eff = hvac.dynamics(hx, hact, {**h_base, "hvac_efficiency": 1.2})
    h_no_load = hvac.dynamics(hx, hact, {**h_base, "heat_load": [0.0, 0.0]})
    h_load = hvac.dynamics(hx, hact, {**h_base, "heat_load": [800.0, 800.0]})
    check("HVAC efficiency raises heating derivative", h_high_eff[0] > h_low_eff[0])
    check("HVAC heat load raises room-temperature derivative", h_load[0] > h_no_load[0])

    env = AIOGymNativeEnv("cascade", dynamic=False, randomize=False, randomize_setpoints=False)
    env.reset(seed=0)
    env.pump_flow_factor = 1.2
    env.heater_efficiency = 0.8
    env.heat_loss_factor = 1.7
    _, _, _, _, info = env.step(np.full(env.action_space.shape[0], 0.5, np.float32))
    tank_info_ok = info["pump_flow_factor"] == 1.2 and info["heater_efficiency"] == 0.8 and info["heat_loss_factor"] == 1.7
    check("Tank env reports process factors", tank_info_ok)

    env = AIOGymNativeEnv("hvac", dynamic=False, randomize=False, randomize_setpoints=False)
    env.reset(seed=0)
    env.heat_load = [500.0, -100.0]
    env.hvac_efficiency = 0.9
    _, _, _, _, info = env.step(np.full(env.action_space.shape[0], 0.5, np.float32))
    hvac_info_ok = info["hvac_heat_load"] == [500.0, -100.0] and info["hvac_efficiency"] == 0.9
    hvac_info_ok = hvac_info_ok and "hvac_comfort_low" in info["cons_info"]
    check("HVAC env reports heat-load and efficiency", hvac_info_ok)

def test_plant_drift_semantics():
    """plant_drift should slowly move physical plant parameters within the regime bounds."""
    env = AIOGymNativeEnv("cascade", dynamic=False, randomize=False, randomize_setpoints=False,
                          randomize_plant=False, plant_drift=True, episode_steps=80)
    env.reset(seed=0)
    nominal_ua = env._p_nominal["ua_loss"]
    start_mult = dict(env._regime_mult)
    info = {}
    for _ in range(12):
        _, _, _, _, info = env.step(np.full(env.action_space.shape[0], 0.5, np.float32))
    drifted = env.model.p["ua_loss"] != nominal_ua and env._regime_mult != start_mult
    bounded = all(lo <= env._regime_mult[k] <= hi for k, (lo, hi) in env._regime.items())
    reported = "plant_mult" in info and info["plant_mult"] == env._regime_mult
    check("plant_drift moves bounded plant multipliers", drifted and bounded and reported)

def test_env_api():
    """Every scenario builds in both action modes with the right spaces + a valid step."""
    for scn in SCENARIOS:
        modes = ["actuator"]
        if make_model(scn).supervisory_layout:
            modes.append("setpoint")
        for mode in modes:
            e = AIOGymNativeEnv(scn, reward_mode="economic", action_mode=mode, dynamic=True, randomize_plant=True)
            obs, info = e.reset(seed=0)
            assert e.observation_space.contains(obs), f"{scn}/{mode} obs not in space"
            a = e.action_space.sample()
            obs2, r, term, trunc, info = e.step(a)
            exp_act = len(e.model.supervisory_layout) if mode == "setpoint" else e.nu
            check(f"{scn:10s}/{mode:8s} obs={obs.shape[0]} act={e.action_space.shape[0]}(exp {exp_act}) reward={r:.2f}", e.action_space.shape[0] == exp_act and np.isfinite(r) and "cons_info" in info)


def test_seeding():
    """Same seed -> identical rollout (reproducibility)."""
    def roll(seed):
        e = AIOGymNativeEnv("cstr", reward_mode="economic", dynamic=True, randomize_plant=True)
        o, _ = e.reset(seed=seed); xs = [o]
        for _ in range(30):
            o, *_ = e.step(np.full(e.action_space.shape[0], 0.5, np.float32)); xs.append(o)
        return np.concatenate(xs)
    check("deterministic on fixed seed", np.allclose(roll(42), roll(42)))
    check("different seeds differ", not np.allclose(roll(1), roll(2)))


def test_registered_ids():
    ids = builtin_gym_ids()
    ids_ok = set(ids) == set(BUILTIN_MODELS)
    for name in ids.values():
        e = gym.make(f"AIOGym/{name}-v0", reward_mode="economic", action_mode="actuator")
        e.reset(seed=0); e.step(e.action_space.sample())
    check("all built-in gym ids make + step", ids_ok)


def test_vectorized():
    """Parallel rollout via gymnasium SyncVectorEnv (the SB3 SubprocVecEnv contract)."""
    from gymnasium.vector import SyncVectorEnv
    n = 8
    venv = SyncVectorEnv([lambda: AIOGymNativeEnv("cstr", reward_mode="economic", action_mode="setpoint",
                                                  dynamic=True, randomize_plant=True) for _ in range(n)])
    obs, _ = venv.reset(seed=0)
    for _ in range(20):
        obs, r, term, trunc, info = venv.step(np.stack([venv.single_action_space.sample() for _ in range(n)]))
    check(f"vectorized {n} envs step, obs {obs.shape}", obs.shape[0] == n and np.all(np.isfinite(r)))
    venv.close()


def test_controller_evaluation_protocol():
    """PID/MPC/oracle-style controllers and learned policies share one evaluator."""
    fixed = BenchmarkProtocol.tracking("cstr", action_mode="actuator", episode_steps=8,
                                       dynamic=False, randomize=False,
                                       randomize_setpoints=False, randomize_plant=False,
                                       plant_drift=False)
    names = set(registered_controllers())
    registry_ok = {"pid", "mpc", "oracle", "cstr_grid_mpc", "policy", "sb3"}.issubset(names)
    expected_keys = {"name", "action_mode", "control_structure", "parameters", "scenarios"}
    expected_scenarios = {scn for scn in BUILTIN_MODELS if make_model(scn).supervisory_layout}
    config_shape_ok = True
    for ctl in ("pid", "mpc", "oracle", "cstr_grid_mpc", "sb3"):
        cfg = load_controller_config(ctl, "cstr")
        raw_scenarios = load_controller_config(ctl).get("scenarios", {})
        config_shape_ok = config_shape_ok and expected_keys.issubset(cfg)
        config_shape_ok = config_shape_ok and set(raw_scenarios) == expected_scenarios
    mpc_defaults = load_controller_config("mpc", "cstr")
    pid_defaults = load_controller_config("pid", "cstr")
    pid_controller = make_controller("pid", scenario="cstr")
    mpc_controller = make_controller("mpc", scenario="cstr", config={"P": 4})
    grid_controller = make_controller("cstr_grid_mpc", scenario="cstr", config={"grid": 11})
    oracle_controller = make_controller("oracle", scenario="cstr", config={"mode": "economic", "horizon": 4})
    registry_ok = registry_ok and config_shape_ok
    registry_ok = registry_ok and pid_defaults["parameters"]["gains"]["temp"] == [0.08, 0.02, 0.0]
    registry_ok = registry_ok and pid_controller.metadata()["api"] == "aiogym.controller.v1"
    registry_ok = registry_ok and pid_controller.metadata()["gains"]["temp"] == [0.08, 0.02, 0.0]
    registry_ok = registry_ok and mpc_defaults["parameters"]["P"] == 40
    registry_ok = registry_ok and mpc_controller.metadata()["control_structure"] == "fixed_sp_mpc"
    registry_ok = registry_ok and mpc_controller.metadata()["horizon"] == 4
    registry_ok = registry_ok and grid_controller.metadata()["control_structure"] == "cstr_grid_mpc"
    registry_ok = registry_ok and grid_controller.metadata()["grid"] == 11
    registry_ok = registry_ok and oracle_controller.metadata()["control_structure"] == "nmpc_oracle"
    registry_ok = registry_ok and oracle_controller.metadata()["horizon"] == 4
    check("controller registry builds built-ins", registry_ok)

    pid = evaluate_controller(pid_controller, fixed.make_env(), episodes=2,
                              seed=123, protocol=fixed, include_episodes=True)
    fixed_ok = (
        pid["name"] == "PID"
        and pid["episodes"] == 2
        and pid["seed_list"] == [123, 124]
        and pid["schema_version"] == "aiogym.evaluation.v2"
        and pid["protocol"]["objective"] == "tracking"
        and pid["controller"]["api"] == "aiogym.controller.v1"
        and pid["controller"]["control_structure"] == "fixed_sp_pid"
        and "tracking_iae" in pid
        and "energy_kwh" in pid
        and "constraint_violation_count" in pid
        and "metric_definitions" in pid
        and "result_schema" in pid
        and pid["reproducibility"]["seed_list"] == [123, 124]
        and pid["episode_metrics"][0]["seed"] == 123
        and "return_std" in pid
    )
    check("actuator controller evaluation includes reproducibility metadata", fixed_ok)

    sup = BenchmarkProtocol.economic("cstr", action_mode="setpoint", episode_steps=8,
                                     dynamic=False, randomize=False,
                                     randomize_setpoints=False, randomize_plant=False,
                                     plant_drift=False)
    env = sup.make_env()
    default_action = env.default_sp_action()

    class ConstantPolicy:
        name = "constant-setpoint-policy"

        def act(self, obs, deterministic=True):
            return default_action

    policy_controller = as_controller(ConstantPolicy(), action_mode="setpoint",
                                      control_structure="constant_setpoint_policy")
    pol = evaluate_controller(policy_controller, env, episodes=2, seed=321, protocol=sup)
    policy_ok = (
        pol["name"] == "constant-setpoint-policy"
        and pol["metric"] == "profit"
        and pol["protocol"]["action_mode"] == "setpoint"
        and pol["controller"]["api"] == "aiogym.controller.v1"
        and pol["controller"]["control_structure"] == "constant_setpoint_policy"
        and np.isfinite(pol["profit"])
    )
    check("policy evaluation uses the same rollout path", policy_ok)


def test_generic_controller_rollout():
    """The common rollout recorder should work across built-in process models."""
    rollout_ok = True
    for scn in SCENARIOS:
        if not make_model(scn).supervisory_layout:
            continue
        protocol = BenchmarkProtocol.tracking(
            scn,
            action_mode="actuator",
            episode_steps=4,
            dynamic=False,
            randomize=False,
            randomize_setpoints=False,
            randomize_plant=False,
            plant_drift=False,
        )
        env = protocol.make_env()
        artifact = rollout_controller(make_controller("pid", scenario=scn), env, seed=77, protocol=protocol)
        first = artifact["rollout"][0]
        rollout_ok = rollout_ok and artifact["protocol"]["scenario"] == scn
        rollout_ok = rollout_ok and artifact["steps"] == 4
        rollout_ok = rollout_ok and artifact["rollout_schema"]["disturbance"]
        rollout_ok = rollout_ok and len(first["action"]) == env.action_space.shape[0]
        rollout_ok = rollout_ok and {"obs", "state", "setpoint", "measurement", "disturbance", "profit", "constraint", "info"}.issubset(first)
    check("generic rollout recorder works across scenarios", rollout_ok)


def test_benchmark_config_and_report_schema():
    """Roadmap phase-3 evaluation protocols stay separated and reproducible."""
    protocols = {
        "tracking": BenchmarkProtocol.tracking("cstr", episode_steps=3, dynamic=False,
                                               randomize=False, randomize_setpoints=False,
                                               randomize_plant=False, plant_drift=False),
        "economic": BenchmarkProtocol.economic("cstr", episode_steps=3, dynamic=False,
                                               randomize=False, randomize_setpoints=False,
                                               randomize_plant=False, plant_drift=False),
        "robustness": BenchmarkProtocol.robustness("cstr", episode_steps=3),
        "safety": BenchmarkProtocol.safety("cstr", episode_steps=3),
    }
    protocol_ok = (
        protocols["tracking"].metadata()["metrics"][0] == "tracking_iae"
        and protocols["economic"].metadata()["metrics"][0] == "profit"
        and protocols["robustness"].metadata()["noise"] is True
        and protocols["safety"].metadata()["metrics"][0] == "constraint_violation_count"
    )
    config = BenchmarkConfig.from_protocol(protocols["tracking"], controller="pid", seeds=[11, 12])
    config_meta = config.metadata()
    config_ok = (
        config_meta["objective"] == "tracking"
        and config_meta["scenario"] == "cstr"
        and config_meta["controller"] == "pid"
        and config_meta["seed_list"] == [11, 12]
        and config_meta["episode_steps"] == 3
        and "tracking_iae" in config_meta["metric_definitions"]
    )

    result = evaluate_controller(
        make_controller("pid", scenario="cstr"),
        protocols["tracking"].make_env(),
        seed_list=[11, 12],
        protocol=protocols["tracking"],
        include_episodes=True,
    )
    report = build_evaluation_report([result])
    report_ok = (
        {"tracking", "economic", "safety", "robustness"}.issubset(report)
        and report["tracking"][0]["name"] == "PID"
        and "tracking_iae" in report["tracking"][0]
        and "profit" in report["economic"][0]
        and "constraint_violation_count" in report["safety"][0]
        and "return_std" in report["robustness"][0]
    )
    check("BenchmarkConfig + separated evaluation report schema", protocol_ok and config_ok and report_ok)


def test_oracle():
    """NMPC oracle solves and beats PID on CSTR economic (it's the upper bound)."""
    try:
        from aiogym.control.oracle import OracleAgent
    except RuntimeError as ex:
        print(f"  (skip oracle: {ex})"); return
    from aiogym.control.baselines import PIDAgent, evaluate
    from aiogym.models import make_model
    mk = lambda: AIOGymNativeEnv("cstr", reward_mode="economic", episode_steps=120, dynamic=True, randomize_plant=True)
    orc = evaluate(OracleAgent("cstr", horizon=12, mode="economic"), mk(), episodes=2)["profit"]
    pid = evaluate(PIDAgent(make_model("cstr")), mk(), episodes=2)["profit"]
    check(f"NMPC oracle {orc:.0f} > PID {pid:.0f}", orc > pid)


if __name__ == "__main__":
    print("model contracts:"); test_model_contract()
    print("custom model entrypoints:"); test_custom_model_entrypoints()
    print("disturbance schedule config:"); test_disturbance_schedule_config()
    print("CSTR disturbance semantics:"); test_cstr_disturbance_semantics()
    print("process disturbance semantics:"); test_process_disturbance_semantics()
    print("plant drift semantics:"); test_plant_drift_semantics()
    print("env API (built-in scenarios):"); test_env_api()
    print("reproducibility:"); test_seeding()
    print("registered gym ids:"); test_registered_ids()
    print("vectorized parallel rollout:"); test_vectorized()
    print("controller evaluation protocol:"); test_controller_evaluation_protocol()
    print("generic controller rollout:"); test_generic_controller_rollout()
    print("benchmark config/report schema:"); test_benchmark_config_and_report_schema()
    print("NMPC oracle baseline:"); test_oracle()
    print(f"\nALL INTERFACE TESTS PASS {OK}")
