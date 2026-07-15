"""One benchmark-case execution path shared by all public entrypoints."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..controllers import make_controller
from ..models import apply_model_params, make_model
from .core import evaluate_controller, rollout_controller
from .protocols import BenchmarkConfig, BenchmarkProtocol
from .rows import compact_result_row


def run_evaluation_case(
    *,
    scenario: str,
    controller: str,
    protocol: BenchmarkProtocol,
    seeds: Sequence[int],
    controller_config: Mapping[str, Any] | None = None,
    include_episodes: bool = True,
    save_rollout: bool = False,
    rollout_steps: int | None = None,
    suite_case: str | None = None,
) -> dict[str, Any]:
    """Build and evaluate one controller with canonical model/config semantics."""

    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ValueError("benchmark case must include at least one seed")
    config = dict(controller_config or {})
    if controller == "oracle":
        parameters = dict(config.get("parameters") or {})
        if "control_dt" not in config and "control_dt" not in parameters:
            config["control_dt"] = float(protocol.control_dt)
    controller_model = apply_model_params(make_model(scenario), protocol.model_params)
    agent = make_controller(
        controller,
        model=controller_model,
        scenario=scenario,
        config=config,
    )
    result = evaluate_controller(
        agent,
        protocol.make_env(),
        episodes=len(seed_list),
        seed=seed_list[0],
        seed_list=seed_list,
        protocol=protocol,
        include_episodes=include_episodes,
    )
    task_meta = protocol.metadata()["task_identity"]
    row = compact_result_row(
        result,
        scenario=scenario,
        objective=protocol.objective,
        action_mode=protocol.action_mode,
        task=task_meta["name"],
        task_status=task_meta["status"],
        task_profile_hash=task_meta["profile_hash"],
        suite_case=suite_case,
        controller=controller,
    )
    benchmark_config = BenchmarkConfig.from_protocol(
        protocol,
        controller=controller,
        seeds=seed_list,
        controller_config=config,
    ).metadata()
    rollout = None
    if save_rollout:
        rollout = rollout_controller(
            agent,
            protocol.make_env(),
            seed=seed_list[0],
            max_steps=rollout_steps,
            protocol=protocol,
        )
    return {
        "controller": agent,
        "result": result,
        "row": row,
        "config": benchmark_config,
        "rollout": rollout,
    }
