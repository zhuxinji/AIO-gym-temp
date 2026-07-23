"""Environment, disturbance, seeding, registration, and vectorization tests."""
from aiogym.tests.interface_support import *  # noqa: F403

def test_disturbance_schedule_config():
    """Dynamic disturbances are scheduled from the model disturbance schema."""
    for scn in SCENARIO_IDS:
        env = AIOGymNativeEnv(scn, auto_events=True, randomize=False, randomize_setpoints=False)
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
    act = [0.5, 0.5]
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

    env = AIOGymNativeEnv("cstr", reward_mode="economic", action_mode="actuator", auto_events=False,
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
    cact = [0.5] * 7
    c_low_pump = cascade.dynamics(cx, cact, {**tank_env, "pump_flow_factor": 0.7})
    c_high_pump = cascade.dynamics(cx, cact, {**tank_env, "pump_flow_factor": 1.3})
    c_low_heat = cascade.dynamics(cx, cact, {**tank_env, "heater_efficiency": 0.6})
    c_high_heat = cascade.dynamics(cx, cact, {**tank_env, "heater_efficiency": 1.0})
    c_low_loss = cascade.dynamics(cx, cact, {**tank_env, "heat_loss_factor": 0.5})
    c_high_loss = cascade.dynamics(cx, cact, {**tank_env, "heat_loss_factor": 2.0})
    check("Cascade pump factor raises inlet-level derivative", c_high_pump[0] > c_low_pump[0])
    check("Cascade heater efficiency raises temp derivative", c_high_heat[1] > c_low_heat[1])
    check("Cascade heat-loss factor lowers temp derivative", c_high_loss[1] < c_low_loss[1])

    quad = make_model("quadruple")
    qx = [12.0, 13.0, 2.0, 1.5]
    qact = [0.5, 0.5]
    q_low_pump = quad.dynamics(qx, qact, {**tank_env, "pump_flow_factor": 0.7})
    q_high_pump = quad.dynamics(qx, qact, {**tank_env, "pump_flow_factor": 1.3})
    q_low_outlet = quad.dynamics(qx, qact, {**tank_env, "outlet_area_factor": 0.7})
    q_high_outlet = quad.dynamics(qx, qact, {**tank_env, "outlet_area_factor": 1.3})
    check("Quadruple pump factor raises lower-tank derivative", q_high_pump[0] > q_low_pump[0])
    check("Quadruple outlet-area factor lowers lower-tank derivative", q_high_outlet[0] < q_low_outlet[0])

    hvac = make_model("hvac")
    hx = [20.0, 20.0]
    hact = [0.75, 0.75]
    h_base = {"t_cold": 5.0, "t_amb": 5.0, "extra_outflow": 0.0}
    h_low_eff = hvac.dynamics(hx, hact, {**h_base, "hvac_efficiency": 0.6})
    h_high_eff = hvac.dynamics(hx, hact, {**h_base, "hvac_efficiency": 1.2})
    h_no_load = hvac.dynamics(hx, hact, {**h_base, "heat_load": [0.0, 0.0]})
    h_load = hvac.dynamics(hx, hact, {**h_base, "heat_load": [800.0, 800.0]})
    check("HVAC efficiency raises heating derivative", h_high_eff[0] > h_low_eff[0])
    check("HVAC heat load raises room-temperature derivative", h_load[0] > h_no_load[0])

    env = AIOGymNativeEnv("cascade", auto_events=False, randomize=False, randomize_setpoints=False)
    env.reset(seed=0)
    env.pump_flow_factor = 1.2
    env.heater_efficiency = 0.8
    env.heat_loss_factor = 1.7
    _, _, _, _, info = env.step(np.full(env.action_space.shape[0], 0.5, np.float32))
    tank_info_ok = info["pump_flow_factor"] == 1.2 and info["heater_efficiency"] == 0.8 and info["heat_loss_factor"] == 1.7
    check("Tank env reports process factors", tank_info_ok)

    env = AIOGymNativeEnv("hvac", auto_events=False, randomize=False, randomize_setpoints=False)
    env.reset(seed=0)
    env.heat_load = [500.0, -100.0]
    env.hvac_efficiency = 0.9
    _, _, _, _, info = env.step(np.full(env.action_space.shape[0], 0.5, np.float32))
    hvac_info_ok = info["hvac_heat_load"] == [500.0, -100.0] and info["hvac_efficiency"] == 0.9
    hvac_info_ok = hvac_info_ok and "hvac_comfort_low" in info["cons_info"]
    hvac_info_ok = hvac_info_ok and info["cons_violated"] and info["constraint"] > 0.0
    check("HVAC env reports heat-load and efficiency", hvac_info_ok)

def test_plant_drift_semantics():
    """plant_drift should slowly move physical plant parameters within the regime bounds."""
    env = AIOGymNativeEnv("cascade", auto_events=False, randomize=False, randomize_setpoints=False,
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
    for scn in SCENARIO_IDS:
        modes = ["actuator"]
        if make_model(scn).supervisory_layout:
            modes.append("setpoint")
        for mode in modes:
            e = AIOGymNativeEnv(scn, reward_mode="economic", action_mode=mode, auto_events=True, randomize_plant=True)
            obs, info = e.reset(seed=0)
            assert e.observation_space.contains(obs), f"{scn}/{mode} obs not in space"
            a = e.action_space.sample()
            obs2, r, term, trunc, info = e.step(a)
            exp_act = len(e.model.supervisory_layout) if mode == "setpoint" else e.nu
            check(f"{scn:10s}/{mode:8s} obs={obs.shape[0]} act={e.action_space.shape[0]}(exp {exp_act}) reward={r:.2f}", e.action_space.shape[0] == exp_act and np.isfinite(r) and "cons_info" in info)


def test_seeding():
    """Same seed -> identical rollout (reproducibility)."""
    def roll(seed):
        e = AIOGymNativeEnv("cstr", reward_mode="economic", auto_events=True, randomize_plant=True)
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
                                                  auto_events=True, randomize_plant=True) for _ in range(n)])
    obs, _ = venv.reset(seed=0)
    for _ in range(20):
        obs, r, term, trunc, info = venv.step(np.stack([venv.single_action_space.sample() for _ in range(n)]))
    check(f"vectorized {n} envs step, obs {obs.shape}", obs.shape[0] == n and np.all(np.isfinite(r)))
    venv.close()
