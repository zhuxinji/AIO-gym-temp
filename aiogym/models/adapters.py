"""Boundary adapters for browser-era model observations and physical actions."""
from __future__ import annotations


class BrowserModelAdapter:
    """Keep legacy browser layout conversion out of generic environment logic."""

    def __init__(self, model):
        self.model = model

    @property
    def enabled(self) -> bool:
        return bool(self.model.uses_legacy_actions())

    @property
    def observation_dim(self) -> int:
        slots = self.model.legacy_observation_level_target_slots()
        return 3 * int(self.model.n) + len(slots) + 2

    def action(self, values):
        return self.model.action_vector_to_dict(values) if self.enabled else self.model.action_vector(values)

    def setpoints(self, y_sp):
        return self.model.legacy_observation_setpoints(y_sp)

    def observation(self, levels, temps, t_cold, t_amb, y_sp):
        level_targets, output_targets = self.setpoints(y_sp)
        return browser_observation_vector(
            self.model, levels, temps, t_cold, t_amb, level_targets, output_targets
        )


def browser_observation_vector(model, levels, temps, t_cold, t_amb,
                               level_targets, output_targets):
    """Build the browser-compatible observation vector for a legacy model."""

    n = model.n
    values = []
    for i in range(n):
        values.append(levels[i] if i < len(levels) else 0.0)
    for i in range(n):
        values.append(temps[i])
    for i in range(n):
        values.append(output_targets[i])
    for i in model.legacy_observation_level_target_slots():
        values.append(level_targets[i])
    values.append(t_cold)
    values.append(t_amb)
    return values
