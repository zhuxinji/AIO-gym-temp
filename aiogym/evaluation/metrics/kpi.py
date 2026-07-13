"""Composite KPI scorer for evaluation.

The normalized score is:

    100 - normalized-output-tracking - excess-energy - safety

The scorer is used both as a KPI reward source and as a common comparison metric
for PID, MPC, oracle, and RL controllers.
"""
from __future__ import annotations

from dataclasses import dataclass

from .tracking import normalized_tracking_errors

W_TRACKING, W_ENERGY, W_SAFETY = 20.0, 0.5, 60.0


@dataclass(frozen=True)
class KPIStep:
    penalty: float
    output_error: float
    mean_output_error: float
    excess_kw: float
    safety_active: bool


def kpi_step(model, y, y_sp, action_energy_kw, ideal_energy_kw, safety_active,
             *, energy_scored=True):
    """Compute one KPI step without updating an accumulator."""

    errors = _normalized_output_errors(model, list(y), list(y_sp))
    output_error = sum(errors)
    mean_output_error = output_error / max(1, len(errors))
    excess_kw = max(0.0, action_energy_kw - ideal_energy_kw) if energy_scored else 0.0
    penalty = (
        W_TRACKING * mean_output_error
        + W_ENERGY * excess_kw
        + W_SAFETY * (1.0 if safety_active else 0.0)
    )
    return KPIStep(
        penalty=float(penalty),
        output_error=float(output_error),
        mean_output_error=float(mean_output_error),
        excess_kw=float(excess_kw),
        safety_active=bool(safety_active),
    )


class KPIScorer:
    def __init__(self, model):
        self.model = model
        self.energy_scored = bool(getattr(model, "energy_scored", True))
        self.reset()

    def reset(self):
        self.elapsed = 0.0
        self.output_error_integral = 0.0
        self.mean_output_error_integral = 0.0
        self.excess = 0.0
        self.safety_active_sec = 0.0

    def step_penalty_y(self, y, y_sp, action_energy_kw, ideal_energy_kw, safety_active, dt):
        """Accumulate one step from generic controlled outputs y and setpoints y_sp."""

        step = kpi_step(
            self.model,
            y,
            y_sp,
            action_energy_kw,
            ideal_energy_kw,
            safety_active,
            energy_scored=self.energy_scored,
        )
        self.accumulate(step, dt)
        return step.penalty

    def accumulate(self, step: KPIStep, dt):
        """Accumulate a previously evaluated KPI step."""

        self.elapsed += dt
        self.output_error_integral += step.output_error * dt
        self.mean_output_error_integral += step.mean_output_error * dt
        self.excess += step.excess_kw * dt / 3600.0
        if step.safety_active:
            self.safety_active_sec += dt

    def report(self):
        e = max(self.elapsed, 1e-6)
        avgY = self.mean_output_error_integral / e
        hours = e / 3600.0
        avgX = self.excess / hours if hours > 0 else 0.0
        sFrac = self.safety_active_sec / e
        pY = W_TRACKING * avgY
        pE = W_ENERGY * avgX if self.energy_scored else 0.0
        pS = W_SAFETY * sFrac
        score = max(0.0, min(100.0, 100.0 - pY - pE - pS))
        return {
            "score": score,
            "avg_output_error": avgY,
            "output_tracking_iae": self.output_error_integral,
            "excess_kwh": self.excess,
            "safety_active_frac": sFrac,
            "comp_tracking": pY,
            "comp_energy": pE,
            "comp_safety": pS,
        }


def _normalized_output_errors(model, y, y_sp):
    return [abs(err) for err in normalized_tracking_errors(model, y, y_sp)]
