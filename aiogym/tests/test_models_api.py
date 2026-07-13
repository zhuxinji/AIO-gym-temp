"""Model contract, model-card, custom-model, and public API tests."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interface_support import *  # noqa: F403

def test_model_contract():
    """Every process model exposes a usable backend contract for tooling."""
    for scn in SCENARIOS:
        model = make_model(scn)
        card = model.model_card()
        n_act = sum(model.actuator_counts())
        params = card["parameters"]
        state_ok = len(card["states"]) == len(model.initial_state())
        action_ok = len(card["actions"]) == n_act
        output_ok = len(card["controlled_outputs"]) == len(model.controlled_output(model.initial_state()))
        output_ok = output_ok and card["controlled_output_vector"] == {"name": "y", "length": len(card["controlled_outputs"])}
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
        check(f"{scn:10s} model contract states={len(card['states'])} actions={len(card['actions'])} params={len(params)}", state_ok and action_ok and output_ok and vector_ok and param_ok and bounds_ok and meta_ok and generic_ok)


def test_model_card_export():
    """Model-card export covers the current registry, not a hand-written scenario list."""
    from aiogym.models import (
        MODEL_CARD_SCHEMA_VERSION,
        collect_model_cards,
        export_model_card_markdown,
        export_model_cards,
        render_model_card_markdown,
        validate_model_card,
    )

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cards = collect_model_cards()
    cards_ok = set(cards) == set(SCENARIOS)
    cards_ok = cards_ok and all(card["schema_version"] == MODEL_CARD_SCHEMA_VERSION for card in cards.values())
    for scenario, card in cards.items():
        validate_model_card(card, expected_scenario=scenario)

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = export_model_cards(tmpdir)
        exported = set(manifest["scenarios"])
        files_ok = exported == set(SCENARIOS)
        for scenario in SCENARIOS:
            path = os.path.join(tmpdir, f"{scenario}.json")
            files_ok = files_ok and os.path.exists(path)
            with open(path) as f:
                validate_model_card(json.load(f), expected_scenario=scenario)
        files_ok = files_ok and os.path.exists(os.path.join(tmpdir, "manifest.json"))

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = export_model_card_markdown(tmpdir)
        markdown_ok = set(manifest["scenarios"]) == set(SCENARIOS)
        markdown_ok = markdown_ok and os.path.exists(os.path.join(tmpdir, "README.md"))
        for scenario, card in cards.items():
            text = render_model_card_markdown(card)
            path = os.path.join(tmpdir, f"{scenario}.md")
            markdown_ok = markdown_ok and os.path.exists(path)
            markdown_ok = markdown_ok and f"`{scenario}`" in text
            markdown_ok = markdown_ok and "## State Vector" in text
            markdown_ok = markdown_ok and "## Dynamics And Assumptions" in text

    docs_path = os.path.join(root, "aiogym", "README.md")
    docs_ok = os.path.exists(docs_path)
    if docs_ok:
        with open(docs_path) as f:
            docs_text = f.read()
        for scenario in SCENARIOS:
            docs_ok = docs_ok and f"`{scenario}`" in docs_text
            card_path = os.path.join(root, "docs", "model_cards", f"{scenario}.md")
            docs_ok = docs_ok and os.path.exists(card_path)
            if os.path.exists(card_path):
                with open(card_path) as f:
                    card_text = f.read()
                docs_ok = docs_ok and f"`{scenario}`" in card_text
                docs_ok = docs_ok and "## Benchmark Semantics" in card_text

    check("model-card export covers registered built-ins", cards_ok and files_ok and markdown_ok and docs_ok)


def test_custom_model_entrypoints():
    """Custom process models can be registered or passed directly to the env."""
    try:
        unregister_model(next(iter(BUILTIN_MODELS)))
        built_in_guard_ok = False
    except ValueError:
        built_in_guard_ok = True

    register_model("mini_tank", MiniTankModel)
    register_model("generic_vector", GenericVectorModel)
    register_model("declarative_vector", define_model({
        "scenario": "declarative_vector",
        "display_name": "Declarative vector model",
        "states": {"x0": {"initial": 0.25, "bounds": (0.0, 1.0)}},
        "actions": {
            "coolant_flow": {"bounds": (0.0, 1.0), "kind": "flow"},
            "agitator_speed": {"bounds": (0.0, 1.0), "kind": "rotation"},
        },
        "params": {"gain": {"value": 0.4, "bounds": (0.0, 2.0)}},
        "dynamics": {"x0": "-0.2 * (x0 - 0.5) + gain * coolant_flow - 0.05 * agitator_speed"},
        "outputs": {"product_quality": {"expr": "x0", "bounds": (0.0, 1.0), "setpoint": 0.6}},
    }))
    try:
        model = make_model("mini_tank")
        card = model.model_card()
        contract_ok = card["scenario"] == "mini_tank" and card["disturbance_defaults"]["feed_bias"] == 0.0
        contract_ok = contract_ok and card["controlled_outputs"][0]["name"] == "tank_temperature"
        contract_ok = contract_ok and model.controlled_output_scales() == [60.0]
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

        generic = make_model("generic_vector")
        generic_schema = generic.action_schema()
        generic_ok = generic.action_dim() == 2 and not generic.uses_legacy_actions()
        generic_ok = generic_ok and [row["name"] for row in generic_schema] == ["coolant_flow", "agitator_speed"]
        generic_ok = generic_ok and [row["kind"] for row in generic_schema] == ["flow", "rotation"]
        generic_env = AIOGymNativeEnv("generic_vector", dynamic=False, randomize=False, randomize_setpoints=False, episode_steps=2)
        generic_obs, _ = generic_env.reset(seed=0)
        _, generic_reward, _, _, generic_info = generic_env.step(np.array([0.5, 0.1], np.float32))
        generic_ok = generic_ok and generic_env.action_space.shape == (2,)
        generic_ok = generic_ok and generic_env.observation_space.contains(generic_obs)
        generic_ok = generic_ok and np.isfinite(generic_reward) and "y" in generic_info and "y_sp" in generic_info

        declarative = make_model("declarative_vector")
        decl_dx = declarative.dynamics([0.25], [0.5, 0.1], {})
        declarative_ok = np.allclose(decl_dx, [0.245])
        declarative_ok = declarative_ok and declarative.default_setpoint_vector() == [0.6]
        declarative_env = AIOGymNativeEnv("declarative_vector", dynamic=False, randomize=False,
                                          randomize_setpoints=False, episode_steps=2)
        decl_obs, _ = declarative_env.reset(seed=0)
        _, decl_reward, _, _, decl_info = declarative_env.step(np.array([0.5, 0.1], np.float32))
        declarative_ok = declarative_ok and declarative_env.observation_space.contains(decl_obs)
        declarative_ok = declarative_ok and np.isfinite(decl_reward) and decl_info["y_sp"] == [0.6]
        if ca is not None:
            decl_casadi = np.asarray(declarative.dynamics(
                ca.DM([0.25]),
                ca.DM([0.5, 0.1]),
                ca.DM(declarative.disturbance_vector({})),
                backend="casadi",
                ca=ca,
            ), dtype=float).reshape(-1)
            declarative_ok = declarative_ok and np.allclose(decl_casadi, decl_dx)
            from aiogym.controllers.oracle import OracleAgent
            oracle_agent = OracleAgent("declarative_vector", horizon=2, mode="track", ipopt_max_iter=30)
            declarative_ok = declarative_ok and oracle_agent.metadata()["mode"] == "tracking"
            oracle_env = AIOGymNativeEnv("declarative_vector", dynamic=False, randomize=False,
                                         randomize_setpoints=False, episode_steps=1)
            oracle_env.reset(seed=0)
            oracle_act = oracle_agent.compute(
                oracle_env.model.measurement(oracle_env.integ.x, oracle_env._env()),
                {"y_sp": oracle_env.y_sp},
                oracle_env.control_dt,
            )
            declarative_ok = declarative_ok and len(oracle_act) == declarative.action_dim()
            declarative_ok = declarative_ok and oracle_agent.orc.last_error is None

        check("custom model registration + direct injection", built_in_guard_ok and contract_ok and registered_ok and direct_ok and generic_ok and declarative_ok)
    finally:
        unregister_model("mini_tank")
        unregister_model("generic_vector")
        unregister_model("declarative_vector")


def test_public_api_entrypoints():
    """Stable public API builds envs, runs benchmarks, and plots artifacts."""
    from aiogym import define_model as top_define_model
    from aiogym import make_env as top_make_env
    from aiogym import make_model as top_make_model
    from aiogym import plot_results as top_plot_results
    from aiogym import register_model as top_register_model
    from aiogym import run_benchmark as top_run_benchmark
    from aiogym import unregister_model as top_unregister_model
    from aiogym.env import make_env as env_make_env
    from aiogym.env_factory import make_env as factory_make_env
    from aiogym.evaluation import plot_results as eval_plot_results
    from aiogym.evaluation import run_benchmark as eval_run_benchmark
    from aiogym.models import define_model as models_define_model
    from aiogym.models import make_model as models_make_model
    from aiogym.models import register_model as models_register_model
    from aiogym.models import unregister_model as models_unregister_model

    public_surface_ok = not hasattr(aiogym, "api")
    public_surface_ok = public_surface_ok and not hasattr(aiogym, "write_benchmark_artifacts")
    public_surface_ok = public_surface_ok and top_define_model is models_define_model
    public_surface_ok = public_surface_ok and top_make_model is models_make_model
    public_surface_ok = public_surface_ok and top_register_model is models_register_model
    public_surface_ok = public_surface_ok and top_unregister_model is models_unregister_model
    public_surface_ok = public_surface_ok and top_make_env is factory_make_env
    public_surface_ok = public_surface_ok and env_make_env is not factory_make_env
    public_surface_ok = public_surface_ok and top_run_benchmark is eval_run_benchmark
    public_surface_ok = public_surface_ok and top_plot_results is eval_plot_results

    env = aiogym.make_env(model="cstr", objective="tracking", seed=7, episode_steps=3)
    override_env = aiogym.make_env(
        model="cstr",
        objective="tracking",
        seed=7,
        episode_steps=1,
        model_params={"Dmax": 0.015},
    )
    unknown_param_ok = False
    try:
        aiogym.make_env(model="cstr", model_params={"not_a_param": 1.0})
    except KeyError:
        unknown_param_ok = True
    env_ok = env.scenario == "cstr" and env.reward_mode == "tracking" and env.episode_steps == 3
    env_ok = env_ok and override_env.model.p["Dmax"] == 0.015 and override_env._p_nominal["Dmax"] == 0.015
    env_ok = env_ok and unknown_param_ok

    public_model = top_define_model({
        "scenario": "public_decl_model",
        "states": {"x0": {"initial": 0.1, "bounds": (0.0, 1.0)}},
        "actions": {"u0": {"bounds": (0.0, 1.0)}},
        "params": {"gain": 0.2},
        "dynamics": {"x0": "-x0 + gain * u0"},
        "outputs": {"y0": {"expr": "x0", "bounds": (0.0, 1.0), "setpoint": 0.4}},
    })
    top_register_model("public_decl_model", public_model, replace=True)
    try:
        public_model_ok = top_make_model("public_decl_model").scenario == "public_decl_model"
    finally:
        top_unregister_model("public_decl_model")

    with tempfile.TemporaryDirectory() as tmpdir:
        payload = aiogym.run_benchmark({
            "scenario": "cstr",
            "objective": "tracking",
            "controller": "pid",
            "seeds": [7],
            "episode_steps": 3,
            "model_params": {"Dmax": 0.015},
            "output_dir": tmpdir,
            "save_rollouts": True,
            "rollout_steps": 3,
        })
        benchmark_path = os.path.join(tmpdir, "benchmark.json")
        figures = aiogym.plot_results(tmpdir)
        api_ok = payload["schema_version"] == "aiogym.public_benchmark.v1"
        api_ok = api_ok and payload["scenario"] == "cstr" and payload["objective"] == "tracking"
        api_ok = api_ok and payload["results"][0]["model"]["parameters"]["Dmax"]["value"] == 0.015
        api_ok = api_ok and payload["results"][0]["protocol"]["model_params"]["Dmax"] == 0.015
        api_ok = api_ok and payload["rows"][0]["status"] in {"passed", "degraded"}
        api_ok = api_ok and os.path.exists(benchmark_path)
        for key in ("input_config", "benchmark_config", "model_card", "rows", "summary_csv",
                    "leaderboard", "results", "report", "rollouts"):
            api_ok = api_ok and os.path.exists(payload["artifacts"][key])
        for key in ("summary", "leaderboard", "rollout", "constraint_timeline"):
            api_ok = api_ok and os.path.exists(figures[key])

    check("public make_env/run_benchmark/plot_results API", public_surface_ok and env_ok and public_model_ok and api_ok)

def run_all():
    print("model contracts:"); test_model_contract()
    print("model-card export:"); test_model_card_export()
    print("custom model entrypoints:"); test_custom_model_entrypoints()
    print("public API entrypoints:"); test_public_api_entrypoints()


if __name__ == "__main__":
    run_all()
