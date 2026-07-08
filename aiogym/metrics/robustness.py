"""Robustness aggregations over perturbed benchmark episodes."""
from __future__ import annotations


def robustness_extrema(episode_metrics, metric_keys):
    summary = {}
    for key in metric_keys:
        vals = [float(ep[key]) for ep in episode_metrics if key in ep]
        if vals:
            summary[f"{key}_best"] = max(vals)
            summary[f"{key}_worst"] = min(vals)
    return summary
