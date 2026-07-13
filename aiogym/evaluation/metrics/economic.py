"""Economic and resource-use metrics for evaluation rollouts."""
from __future__ import annotations


def economic_step_metrics(info, dt: float):
    return {
        "profit": float(info.get("profit", 0.0)),
        "prod": float(info.get("prod", 0.0)),
        "production": float(info.get("prod", 0.0)),
        "energy_kwh": float(info.get("energy_kw", 0.0)) * dt / 3600.0,
    }
