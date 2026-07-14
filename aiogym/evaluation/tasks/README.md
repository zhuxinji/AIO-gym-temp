# Task profiles

Task profiles declare scenario-specific timing and reserve stable sections for
initial conditions, setpoints, disturbances, constraints, and acceptance rules.
They are opt-in, versioned inputs rather than hidden environment defaults.

The bundled `legacy-default` profiles reproduce the current direct-environment
timing only. New physically grounded tasks should use new names and record their
benchmark source instead of overwriting these compatibility profiles.

`quadruple/minimum-phase-tracking` is the first reference-derived task. It uses
the minimum-phase Johansson configuration, an exact nonlinear equilibrium, and
two deterministic lower-tank setpoint moves.
