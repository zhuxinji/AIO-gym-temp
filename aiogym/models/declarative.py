"""Declarative model definitions for custom process scenarios."""
from __future__ import annotations

import ast
from collections.abc import Mapping

from .core import ProcessModelContract


_BIN_OPS = {
    ast.Add: lambda a, b, ops: a + b,
    ast.Sub: lambda a, b, ops: a - b,
    ast.Mult: lambda a, b, ops: a * b,
    ast.Div: lambda a, b, ops: a / b,
    ast.Pow: lambda a, b, ops: a**b,
}
_UNARY_OPS = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}


def define_model(spec: Mapping) -> ProcessModelContract:
    """Create a process model from a formula-only declaration.

    The accepted formula language is intentionally small: names, numeric
    constants, arithmetic operators, and common math functions such as
    ``exp``, ``sqrt``, ``sin``, ``cos``, ``log``, ``abs``, ``min``, and ``max``.
    This keeps the same declaration usable for numeric simulation and CasADi
    oracle/NMPC construction.
    """

    return DeclarativeProcessModel(spec)


class DeclarativeProcessModel(ProcessModelContract):
    """Process model generated from a dictionary of variables and formulas."""

    def __init__(self, spec: Mapping):
        self._spec = dict(spec)
        self.scenario = _required_str(spec, "scenario")
        self.display_name = str(spec.get("display_name") or spec.get("name") or self.scenario)
        self.summary = str(spec.get("summary") or "Declarative process model.")
        self.dt_micro = float(spec.get("dt_micro", 0.02))

        states = _required_mapping(spec, "states")
        actions = _required_mapping(spec, "actions")
        outputs = _required_mapping(spec, "outputs")
        dynamics = _required_mapping(spec, "dynamics")
        params = dict(spec.get("params") or {})

        self.state_names = tuple(states.keys())
        self.action_names = tuple(actions.keys())
        self.output_names = tuple(outputs.keys())
        self.n = max(1, int(spec.get("n", len(self.output_names) or len(self.state_names))))

        self._x0 = []
        self.state_units = {}
        self.state_bounds = {}
        for name, row in states.items():
            meta = _as_meta(row, default_key="initial")
            if "initial" not in meta:
                raise ValueError(f"state {name!r} must define an initial value")
            self._x0.append(float(meta["initial"]))
            self.state_units[name] = str(meta.get("unit", ""))
            self.state_bounds[name] = _bounds(meta.get("bounds"))

        self.action_units = {}
        self.action_bounds = {}
        self.action_kinds = {}
        for name, row in actions.items():
            meta = _as_meta(row)
            self.action_units[name] = str(meta.get("unit", "fraction"))
            self.action_bounds[name] = _bounds(meta.get("bounds", (0.0, 1.0)))
            self.action_kinds[name] = str(meta.get("kind", "input"))

        self.p = {}
        self.param_units = {}
        self.param_bounds = {}
        for name, row in params.items():
            meta = _as_meta(row, default_key="value")
            self.p[name] = meta.get("value", row if not isinstance(row, Mapping) else 0.0)
            self.param_units[name] = str(meta.get("unit", ""))
            self.param_bounds[name] = _bounds(meta.get("bounds"))

        self.output_units = {}
        self.output_bounds = {}
        self.default_y_sp = []
        self._output_exprs = []
        for name, row in outputs.items():
            meta = _as_meta(row, default_key="expr")
            expr = meta.get("expr")
            if expr is None:
                raise ValueError(f"output {name!r} must define an expr")
            self._output_exprs.append(_Expression(str(expr), f"outputs.{name}"))
            self.output_units[name] = str(meta.get("unit", ""))
            bounds = _bounds(meta.get("bounds"))
            self.output_bounds[name] = bounds
            if "setpoint" in meta:
                self.default_y_sp.append(float(meta["setpoint"]))
            elif bounds[0] is not None and bounds[1] is not None:
                self.default_y_sp.append(0.5 * (float(bounds[0]) + float(bounds[1])))
            else:
                self.default_y_sp.append(0.0)
        self.default_y_sp = tuple(self.default_y_sp)

        self._dynamics_exprs = []
        for name in self.state_names:
            if name not in dynamics:
                raise ValueError(f"dynamics must define a derivative for state {name!r}")
            self._dynamics_exprs.append(_Expression(str(dynamics[name]), f"dynamics.{name}"))

        self.input_disturbances = tuple(_disturbance_rows(spec.get("disturbances") or {}))
        self.event_disturbances = tuple(spec.get("event_disturbances", ProcessModelContract.event_disturbances))
        self.safety_constraints = tuple(spec.get("constraints") or _state_bound_constraints(self.state_names, self.state_bounds))
        self.plant_regime = dict(spec.get("plant_regime") or {"nominal": (1.0, 1.0)})
        self.economic_config = dict(spec.get("economic_config") or ProcessModelContract.economic_config)
        self.supervisory_layout = tuple(spec.get("supervisory_layout") or ())
        self.energy_scored = bool(spec.get("energy_scored", False))

    def initial_state(self):
        return list(self._x0)

    def _dynamics(self, x, u, d, ops):
        context = self._context(x, u, d)
        return ops.vector([expr.eval(context, ops) for expr in self._dynamics_exprs])

    def controlled_output(self, x, backend="numeric", ca=None):
        ops = _casadi_ops_proxy(ca) if backend == "casadi" else _numeric_ops_proxy()
        context = self._context(x, [0.0] * self.action_dim(), {})
        return [expr.eval(context, ops) for expr in self._output_exprs]

    def runtime_env(self, disturbance_values):
        env = self.disturbance_defaults()
        env.update(dict(disturbance_values or {}))
        return env

    def _context(self, x, u, d):
        context = {}
        for i, name in enumerate(self.state_names):
            context[name] = x[i]
        for i, name in enumerate(self.action_names):
            context[name] = u[i]
        context.update(self.p)
        context.update(dict(d or {}))
        return context


class _Expression:
    def __init__(self, source: str, label: str):
        self.source = source
        self.label = label
        self.tree = ast.parse(source, mode="eval").body

    def eval(self, context: Mapping, ops):
        return _eval_node(self.tree, context, ops, self.label)


def _eval_node(node, context, ops, label):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise ValueError(f"{label} references unknown name {node.id!r}")
        return context[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, context, ops, label))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left, context, ops, label)
        right = _eval_node(node.right, context, ops, label)
        return _BIN_OPS[type(node.op)](left, right, ops)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(f"{label} only supports simple function calls")
        args = [_eval_node(arg, context, ops, label) for arg in node.args]
        return _call_func(node.func.id, args, ops, label)
    raise ValueError(f"{label} contains unsupported expression syntax")


def _call_func(name, args, ops, label):
    if name in {"sqrt", "exp", "sin", "cos", "tan", "log", "abs"}:
        if len(args) != 1:
            raise ValueError(f"{label}: {name}() expects one argument")
        return getattr(ops, name)(args[0])
    if name in {"min", "max"}:
        if len(args) < 2:
            raise ValueError(f"{label}: {name}() expects at least two arguments")
        func = getattr(ops, name)
        value = args[0]
        for arg in args[1:]:
            value = func(value, arg)
        return value
    raise ValueError(f"{label} uses unsupported function {name!r}")


def _numeric_ops_proxy():
    from .backends import _NUMERIC_OPS

    return _NUMERIC_OPS


def _casadi_ops_proxy(ca):
    from .backends import _casadi_ops

    if ca is None:
        raise ValueError("backend='casadi' requires ca=...")
    return _casadi_ops(ca)


def _required_mapping(spec, key):
    value = spec.get(key)
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"declarative model must define a non-empty {key!r} mapping")
    return value


def _required_str(spec, key):
    value = spec.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"declarative model must define a non-empty {key!r}")
    return value


def _as_meta(value, default_key=None):
    if isinstance(value, Mapping):
        return dict(value)
    if default_key is None:
        return {}
    return {default_key: value}


def _bounds(value):
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("bounds must be a two-item tuple/list or None")
    lo, hi = value
    return (None if lo is None else float(lo), None if hi is None else float(hi))


def _disturbance_rows(rows):
    for name, row in rows.items():
        meta = _as_meta(row, default_key="default")
        out = {
            "name": name,
            "event": meta.get("event", f"{name}_step"),
            "unit": meta.get("unit", ""),
            "bounds": _bounds(meta.get("bounds")),
            "default": meta.get("default", 0.0),
            "description": meta.get("description", name),
            "dynamic": bool(meta.get("dynamic", True)),
        }
        yield out


def _state_bound_constraints(names, bounds):
    constraints = []
    for name in names:
        bound = bounds.get(name)
        if bound is not None:
            constraints.append({"name": f"{name}_bounds", "states": (name,), "bounds": bound})
    return constraints or ({"name": "state_bounds", "states": tuple(names), "bounds": (None, None)},)
