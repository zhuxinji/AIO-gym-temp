"""Robustness aggregations over evaluation episodes."""
from __future__ import annotations


def robustness_extrema(episode_metrics, metric_keys, metric_directions=None):
    directions = dict(metric_directions or {})
    summary = {}
    for key in metric_keys:
        vals = [float(ep[key]) for ep in episode_metrics if key in ep]
        if vals:
            best, worst = (min, max) if directions.get(key) == "minimize" else (max, min)
            summary[f"{key}_best"] = best(vals)
            summary[f"{key}_worst"] = worst(vals)
    return summary
