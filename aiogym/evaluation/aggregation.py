"""Evaluation schema and report aggregation."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .metric_catalog import (
    EVALUATION_SCHEMA_VERSION,
    METRIC_DEFINITIONS,
    METRIC_DIRECTIONS,
    PROTOCOL_METRICS,
    ROLLOUT_SCHEMA,
)
from .metrics.robustness import robustness_extrema


def result_schema():
    return {
        "version": EVALUATION_SCHEMA_VERSION,
        "rollout": dict(ROLLOUT_SCHEMA),
        "episode_metrics": dict(METRIC_DEFINITIONS),
        "tables": {
            "tracking": list(PROTOCOL_METRICS["tracking"]),
            "economic": list(PROTOCOL_METRICS["economic"]),
            "safety": list(PROTOCOL_METRICS["safety"]),
            "robustness": list(PROTOCOL_METRICS["robustness"]),
        },
    }

def build_evaluation_report(results: Sequence[Mapping[str, Any]]):
    """Return roadmap-style tracking/economic/safety/robustness tables."""

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "tracking": [_table_row(result, PROTOCOL_METRICS["tracking"]) for result in results],
        "economic": [_table_row(result, PROTOCOL_METRICS["economic"]) for result in results],
        "safety": [_table_row(result, PROTOCOL_METRICS["safety"]) for result in results],
        "robustness": [_robustness_row(result) for result in results],
        "metric_definitions": dict(METRIC_DEFINITIONS),
    }

def _aggregate_metric_keys(per_episode):
    keys = set()
    for row in per_episode:
        for key, value in row.items():
            if key in ("episode", "seed", "steps"):
                continue
            if isinstance(value, (int, float, np.number)):
                keys.add(key)
    return sorted(keys)

def _table_row(result: Mapping[str, Any], keys: Sequence[str]):
    row = {
        "controller": result.get("controller_name"),
        "objective": result.get("objective"),
        "task": result.get("task", "default"),
        "task_status": result.get("task_status", "implicit-default"),
        "task_profile_hash": result.get("task_profile_hash"),
        "control_structure": dict(result.get("controller", {})).get("control_structure"),
        "controller_status": result.get("controller_status", "ok"),
        "controller_diagnostics": result.get("controller_diagnostics", {}),
        "episodes": result.get("episodes"),
        "seed_list": result.get("seed_list", []),
    }
    for key in keys:
        if key in result:
            row[key] = result[key]
        std_key = f"{key}_std"
        if std_key in result:
            row[std_key] = result[std_key]
    return row
def _robustness_row(result: Mapping[str, Any]):
    row = _table_row(result, PROTOCOL_METRICS["robustness"])
    row.update(robustness_extrema(
        result.get("episode_metrics", []),
        PROTOCOL_METRICS["robustness"],
        METRIC_DIRECTIONS,
    ))
    return row
