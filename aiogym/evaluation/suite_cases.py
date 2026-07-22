"""Resolved benchmark-suite case expansion."""
from __future__ import annotations

from .._internal.config import parse_seed_list
from .cases import BenchmarkCase
from .metric_catalog import primary_metric_for_objective
from .protocols import resolve_protocol
from .suite_loading import expand_scenarios


def parse_csv(raw: str | None, default):
    if raw is None:
        return list(default)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("comma-separated options must contain at least one value")
    return values

def build_cases(args, *, load_suite):
    suite = load_suite(args.suite)
    explicit_cases = bool(suite.get("cases"))
    scenario_filter = (
        set(expand_scenarios(parse_csv(args.scenarios, suite["scenarios"])))
        if explicit_cases and args.scenarios is not None else None
    )
    explicit_objectives = (
        parse_csv(args.objectives, ()) if args.objectives is not None else None
    )
    controller_filter = (
        set(parse_csv(args.controllers, suite["controllers"]))
        if explicit_cases and args.controllers is not None else None
    )
    seeds = parse_seed_list(args.seed_list, args.seed, args.episodes)
    cases = []
    declarations = suite.get("cases") or [{
        "scenarios": parse_csv(args.scenarios, suite["scenarios"]),
        "controllers": parse_csv(args.controllers, suite["controllers"]),
    }]
    for declaration in declarations:
        scenarios = expand_scenarios(declaration.get("scenarios", declaration.get("scenario", suite["scenarios"])))
        if explicit_objectives is not None:
            objective_candidates = [("explicit", value) for value in explicit_objectives]
        elif "objectives" in declaration:
            objective_candidates = [
                ("case-config", value) for value in declaration["objectives"]
            ]
        elif "objective" in declaration:
            objective_candidates = [("case-config", declaration["objective"])]
        elif suite["objectives"]:
            objective_candidates = [
                ("suite-config", value) for value in suite["objectives"]
            ]
        else:
            objective_candidates = [("task-default", None)]
        controllers = list(declaration.get("controllers", [declaration["controller"]] if "controller" in declaration else suite["controllers"]))
        for scenario in scenarios:
            if scenario_filter is not None and scenario not in scenario_filter:
                continue
            for objective_source, objective_value in objective_candidates:
                action_mode = declaration.get("action_mode", suite["action_mode"])
                protocol_config = {"action_mode": action_mode}
                declaration_environment = dict(declaration.get("environment", {}))
                suite_environment = dict(suite.get("environment", {}))
                task = declaration.get("task", suite.get("task"))
                if task is not None:
                    protocol_config["task"] = task
                for key in (
                    "auto_events", "randomize", "randomize_setpoints", "randomize_plant",
                    "plant_drift", "integral_obs", "terminate_on_runaway", "noise",
                    "noise_pct", "tracking_q_y", "tracking_r_move", "tracking_r_steady", "model_params",
                ):
                    if key in declaration:
                        protocol_config[key] = declaration[key]
                    elif key in declaration_environment:
                        protocol_config[key] = declaration_environment[key]
                    elif key in suite:
                        protocol_config[key] = suite[key]
                    elif key in suite_environment:
                        protocol_config[key] = suite_environment[key]
                if args.episode_steps is not None:
                    protocol_config["episode_steps"] = int(args.episode_steps)
                elif "episode_steps" in declaration:
                    protocol_config["episode_steps"] = int(declaration["episode_steps"])
                elif "episode_steps" in suite:
                    protocol_config["episode_steps"] = int(suite["episode_steps"])
                if args.control_dt is not None:
                    protocol_config["control_dt"] = float(args.control_dt)
                elif "control_dt" in declaration:
                    protocol_config["control_dt"] = float(declaration["control_dt"])
                elif "control_dt" in suite:
                    protocol_config["control_dt"] = float(suite["control_dt"])
                protocol = resolve_protocol(
                    scenario,
                    objective_value if objective_source == "explicit" else None,
                    protocol_config,
                    case_objective=(
                        objective_value if objective_source == "case-config" else None
                    ),
                    suite_objective=(
                        objective_value if objective_source == "suite-config" else None
                    ),
                )
                task_meta = protocol.metadata()["task_identity"]
                objective = protocol.objective
                for controller in controllers:
                    if controller_filter is not None and controller not in controller_filter:
                        continue
                    controller_config = controller_config_for(args, controller, action_mode, objective)
                    controller_config = _merge_config(controller_config, suite.get("controller_configs", {}).get(controller, {}))
                    controller_config = _merge_config(controller_config, declaration.get("controller_configs", {}).get(controller, {}))
                    case_name = f"{objective}:{scenario}:{task_meta['name']}:{controller}"
                    case_spec = BenchmarkCase.from_protocol(
                        protocol,
                        controller=controller,
                        seeds=seeds,
                        controller_config=controller_config,
                        case_id=case_name,
                    )
                    cases.append({
                        "name": case_name,
                        "scenario": scenario,
                        "task": task_meta["name"],
                        "task_status": task_meta["status"],
                        "task_profile_hash": task_meta["profile_hash"],
                        "objective": objective,
                        "objective_source": protocol.objective_source,
                        "controller": controller,
                        "action_mode": action_mode,
                        "controller_config": controller_config,
                        "protocol": protocol,
                        "case_spec": case_spec,
                        "seeds": seeds,
                    })
    if not cases:
        raise ValueError("suite filters selected no benchmark cases")
    return suite, cases


def _merge_config(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in dict(override or {}).items():
        if key == "parameters" and isinstance(value, dict):
            out[key] = {**dict(out.get(key) or {}), **value}
        else:
            out[key] = value
    return out


def controller_config_for(args, controller: str, action_mode: str, objective: str | None = None):
    if controller == "sb3":
        if not args.sb3_path:
            raise SystemExit("controller 'sb3' requires --sb3-path")
        return {
            "path": args.sb3_path,
            "algo": args.sb3_algo,
            "action_mode": action_mode,
        }
    if controller == "onnx":
        if not args.onnx_path:
            raise SystemExit("controller 'onnx' requires --onnx-path")
        return {"path": args.onnx_path, "action_mode": action_mode}
    if controller == "oracle" and objective == "tracking":
        return {"profile": "tracking", "mode": "tracking"}
    return {}
