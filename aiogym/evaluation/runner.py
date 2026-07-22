"""One benchmark-case execution path shared by all public entrypoints."""
from __future__ import annotations

import traceback
from time import perf_counter
from typing import Any, Mapping, Sequence

from ..controllers import make_controller
from ..models import apply_model_params, make_model
from .evaluator import evaluate_controller
from .rollouts import rollout_controller
from .cases import BenchmarkCase
from .metric_catalog import primary_metric_for_objective
from .protocols import BenchmarkProtocol
from .rows import compact_result_row
from ..models.tasks import configure_model_for_task


def run_evaluation_case(
    *,
    scenario: str | None = None,
    controller: str | None = None,
    protocol: BenchmarkProtocol | None = None,
    seeds: Sequence[int] | None = None,
    controller_config: Mapping[str, Any] | None = None,
    include_episodes: bool = True,
    save_rollout: bool = False,
    rollout_steps: int | None = None,
    suite_case: str | None = None,
    case: BenchmarkCase | None = None,
    env_factory=None,
) -> dict[str, Any]:
    """Build and evaluate one controller with canonical model/config semantics."""

    if case is None:
        if protocol is None or scenario is None or controller is None or seeds is None:
            raise ValueError(
                "scenario, controller, protocol, and seeds are required when case is omitted"
            )
        case = BenchmarkCase.from_protocol(
            protocol,
            controller=controller,
            seeds=seeds,
            controller_config=controller_config,
            case_id=suite_case,
        )
    elif protocol is None:
        protocol = case.protocol

    scenario = case.environment.scenario
    controller = case.controller
    seed_list = tuple(int(seed) for seed in case.seeds)
    if not seed_list:
        raise ValueError("benchmark case must include at least one seed")
    config = dict(case.controller_config)
    if controller == "oracle":
        parameters = dict(config.get("parameters") or {})
        if "control_dt" not in config and "control_dt" not in parameters:
            config["control_dt"] = float(case.environment.control_dt)
        if case.objective.name == "tracking":
            # Tracking benchmarks rank normalized output-error cost. The
            # objective spec owns those output weights; controller profiles may keep
            # small input regularizers and finite-horizon terms for numerical
            # behavior, but they do not change the reported primary metric.
            config["q_y"] = case.objective.reward_options.get("tracking_q_y", 1.0)
    controller_model = apply_model_params(
        make_model(scenario), case.environment.model_params
    )
    configure_model_for_task(controller_model, case.environment.task)
    agent = make_controller(
        controller,
        model=controller_model,
        scenario=scenario,
        config=config,
    )
    make_environment = env_factory or (
        lambda environment, objective: environment.make_env(objective)
    )
    result = evaluate_controller(
        agent,
        make_environment(case.environment, case.objective),
        episodes=len(seed_list),
        seed=seed_list[0],
        seed_list=seed_list,
        protocol=protocol,
        objective_specification=case.objective,
        include_episodes=include_episodes,
    )
    from ..models.tasks import task_identity

    task_meta = task_identity(case.environment.task)
    row = compact_result_row(
        result,
        scenario=scenario,
        objective=case.objective.name,
        action_mode=case.environment.action_mode,
        task=task_meta["name"],
        task_status=task_meta["status"],
        task_profile_hash=task_meta["profile_hash"],
        suite_case=suite_case or case.case_id,
        controller=controller,
    )
    benchmark_config = case.metadata()
    rollout = None
    if save_rollout:
        rollout = rollout_controller(
            agent,
            make_environment(case.environment, case.objective),
            seed=seed_list[0],
            max_steps=rollout_steps,
            protocol=protocol,
            objective_specification=case.objective,
        )
    return {
        "controller": agent,
        "result": result,
        "row": row,
        "config": benchmark_config,
        "case": case.metadata(),
        "rollout": rollout,
    }


def execute_benchmark_case(
    case: BenchmarkCase,
    *,
    include_episodes: bool = True,
    save_rollout: bool = False,
    rollout_steps: int | None = None,
    suite_case: str | None = None,
    include_tracebacks: bool = False,
) -> dict[str, Any]:
    """Execute one resolved case and return a uniform success/failure artifact."""

    started = perf_counter()
    case_name = suite_case or case.case_id
    try:
        artifact = run_evaluation_case(
            case=case,
            include_episodes=include_episodes,
            save_rollout=save_rollout,
            rollout_steps=rollout_steps,
            suite_case=case_name,
        )
    except Exception as ex:
        from ..models.tasks import task_identity

        task_meta = task_identity(case.environment.task)
        error = {"type": ex.__class__.__name__, "message": str(ex)}
        if include_tracebacks:
            error["traceback"] = traceback.format_exc()
        row = {
            "suite_case": case_name,
            "scenario": case.environment.scenario,
            "task": task_meta["name"],
            "task_status": task_meta["status"],
            "task_profile_hash": task_meta["profile_hash"],
            "objective": case.objective.name,
            "objective_source": case.objective.source,
            "objective_status": "not-evaluated",
            "action_mode": case.environment.action_mode,
            "controller_name": case.controller,
            "execution_status": "failed",
            "metric": primary_metric_for_objective(case.objective.name),
            "message": str(ex),
            "episodes": 0,
            "seed": case.seeds[0] if case.seeds else None,
            "seed_list": list(case.seeds),
            "suite_runtime_seconds": float(perf_counter() - started),
        }
        return {"status": "failed", "row": row, "error": error, "case": case}

    row = artifact["row"]
    row["suite_runtime_seconds"] = float(perf_counter() - started)
    rollout = artifact.get("rollout")
    if rollout is not None and suite_case is not None:
        from ..models.tasks import task_identity

        rollout.update({
            "scenario": case.environment.scenario,
            "task": task_identity(case.environment.task)["name"],
            "objective": case.objective.name,
            "controller": case.controller,
            "suite_case": case_name,
        })
    artifact["status"] = row["execution_status"]
    artifact["rollout"] = rollout
    return artifact
