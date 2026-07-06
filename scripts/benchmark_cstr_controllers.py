#!/usr/bin/env python3
"""CSTR controller smoke benchmark.

Runs PID, MPC, NMPC oracle, and optionally an SB3 checkpoint through the same
BenchmarkProtocol + evaluate_controller path. This is intentionally small: it
tests the controller/config/evaluator loop before the larger benchmark runner
exists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aiogym.controllers import make_controller
from aiogym.evaluation import (
    BenchmarkProtocol,
    evaluate_controller,
    metric_for_reward_mode,
    rollout_controller,
)


def protocol_factory(objective: str):
    return {
        "economic": BenchmarkProtocol.economic,
        "tracking": BenchmarkProtocol.tracking,
        "kpi": BenchmarkProtocol.kpi,
    }[objective]


def evaluate_named(name: str, protocol: BenchmarkProtocol, episodes: int, seed: int,
                   config: dict | None = None):
    controller = make_controller(name, scenario="cstr", config=config or {})
    return evaluate_controller(controller, protocol.make_env(), episodes=episodes,
                               seed=seed, protocol=protocol)


def compact_row(result: dict):
    controller = result["controller"]
    metric = result["metric"]
    return {
        "name": result["name"],
        "control_structure": controller.get("control_structure"),
        "metric": metric,
        metric: result[metric],
        f"{metric}_std": result[f"{metric}_std"],
        "kpi": result["kpi"],
        "profit": result["profit"],
        "return": result["return"],
        "track": result["track"],
        "constraint": result["constraint"],
        "episodes": result["episodes"],
        "seed": result["seed"],
    }


def controller_specs(args, baseline_protocol: BenchmarkProtocol):
    specs = [
        {"name": "pid", "protocol": baseline_protocol, "episodes": args.episodes,
         "seed": args.seed, "config": {}},
        {"name": "mpc", "protocol": baseline_protocol, "episodes": args.episodes,
         "seed": args.seed, "config": {}},
    ]

    if args.include_grid_mpc:
        specs.append({"name": "cstr_grid_mpc", "protocol": baseline_protocol,
                      "episodes": args.episodes, "seed": args.seed, "config": {}})

    if not args.skip_oracle:
        mode = "track" if args.objective == "tracking" else baseline_protocol.reward_mode
        specs.append({"name": "oracle", "protocol": baseline_protocol,
                      "episodes": args.oracle_episodes, "seed": args.seed,
                      "config": {"mode": mode}})

    if args.sb3_path:
        sb3_protocol = protocol_factory(args.objective)(
            "cstr",
            action_mode=args.sb3_action_mode,
            episode_steps=args.episode_steps,
            control_dt=args.control_dt,
        )
        specs.append({
            "name": "sb3",
            "protocol": sb3_protocol,
            "episodes": args.episodes,
            "seed": args.seed,
            "config": {
                "path": args.sb3_path,
                "algo": args.sb3_algo,
                "action_mode": args.sb3_action_mode,
            },
        })

    return specs


def run_spec(spec: dict):
    return evaluate_named(
        spec["name"],
        spec["protocol"],
        spec["episodes"],
        spec["seed"],
        config=spec.get("config"),
    )


def rollout_spec(spec: dict, max_steps: int | None = None):
    controller = make_controller(spec["name"], scenario="cstr", config=spec.get("config") or {})
    protocol = spec["protocol"]
    return rollout_controller(
        controller,
        protocol.make_env(),
        seed=spec["seed"],
        max_steps=max_steps,
        protocol=protocol,
    )


def plot_summary(rows: list[dict], path: str):
    labels = [row["name"] for row in rows]
    metrics = [
        ("profit", "Profit"),
        ("kpi", "KPI"),
        ("track", "Tracking error"),
        ("constraint", "Constraint"),
    ]
    colors = ["#3b82f6", "#14b8a6", "#f59e0b", "#ef4444", "#8b5cf6", "#64748b"]

    width, height = 1200, 760
    panels = [
        (70, 95, 500, 250),
        (650, 95, 500, 250),
        (70, 430, 500, 250),
        (650, 430, 500, 250),
    ]
    parts = [_svg_header(width, height), _svg_text(60, 45, "CSTR controller smoke benchmark", size=24, weight="700")]
    for panel, (key, title) in zip(panels, metrics):
        x, y, w, h = panel
        vals = [float(row.get(key, 0.0)) for row in rows]
        parts.append(_svg_text(x, y - 22, title, size=16, weight="700"))
        parts.extend(_svg_axes(x, y, w, h))
        if not vals:
            continue
        lo = min(min(vals), 0.0)
        hi = max(max(vals), 0.0)
        if hi == lo:
            hi = lo + 1.0
        zero = _map_y(0.0, lo, hi, y, h)
        parts.append(f'<line x1="{x}" y1="{zero:.2f}" x2="{x + w}" y2="{zero:.2f}" stroke="#94a3b8" stroke-dasharray="4 4"/>')
        gap = 18
        bar_w = max(10, (w - gap * (len(vals) + 1)) / max(len(vals), 1))
        for i, val in enumerate(vals):
            bx = x + gap + i * (bar_w + gap)
            by = _map_y(val, lo, hi, y, h)
            top, bottom = min(by, zero), max(by, zero)
            parts.append(
                f'<rect x="{bx:.2f}" y="{top:.2f}" width="{bar_w:.2f}" '
                f'height="{max(1.0, bottom - top):.2f}" fill="{colors[i % len(colors)]}"/>'
            )
            parts.append(_svg_text(bx + bar_w / 2, y + h + 24, labels[i], size=11, anchor="middle"))
            parts.append(_svg_text(bx + bar_w / 2, top - 6, _fmt(val), size=10, anchor="middle", fill="#334155"))
        parts.append(_svg_text(x - 12, y + 8, _fmt(hi), size=10, anchor="end", fill="#64748b"))
        parts.append(_svg_text(x - 12, y + h, _fmt(lo), size=10, anchor="end", fill="#64748b"))
    parts.append("</svg>")
    _write_text(path, "\n".join(parts))


def _series(rows: list[dict], key_path: tuple[str, ...], default: float = 0.0):
    vals = []
    for row in rows:
        value = row
        for key in key_path:
            if not isinstance(value, dict) or key not in value:
                value = default
                break
            value = value[key]
        vals.append(float(value))
    return vals


def _vector_series(rows: list[dict], key: str, idx: int, default: float = 0.0):
    vals = []
    for row in rows:
        value = row.get(key, [])
        if isinstance(value, list) and len(value) > idx:
            vals.append(float(value[idx]))
        else:
            vals.append(default)
    return vals


def _info_vector_series(rows: list[dict], key: str, idx: int, default: float = 0.0):
    vals = []
    for row in rows:
        value = row.get("info", {}).get(key, [])
        if isinstance(value, list) and len(value) > idx:
            vals.append(float(value[idx]))
        else:
            vals.append(default)
    return vals


def _setpoint_series(rows: list[dict], key: str, idx: int, default: float = 0.0):
    vals = []
    for row in rows:
        value = row.get("setpoint", {}).get(key, [])
        if isinstance(value, list) and len(value) > idx:
            vals.append(float(value[idx]))
        else:
            vals.append(default)
    return vals


def plot_cstr_rollouts(rollouts: list[dict], path: str):
    width, height = 1200, 1120
    left, right = 90, 40
    panel_h, gap = 150, 48
    panels = []
    top = 95
    for i in range(5):
        panels.append((left, top + i * (panel_h + gap), width - left - right, panel_h))

    colors = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#475569"]
    parts = [_svg_header(width, height), _svg_text(60, 45, "CSTR rollout comparison", size=24, weight="700")]
    series_by_panel = [[] for _ in panels]
    for artifact in rollouts:
        rows = artifact.get("rollout", [])
        if not rows:
            continue
        label = artifact.get("name", "controller")
        t = [float(row.get("time", i)) for i, row in enumerate(rows)]
        ca = _vector_series(rows, "next_state", 0)
        temp = _info_vector_series(rows, "temps", 0)
        t_sp = _setpoint_series(rows, "t_sp", 0)
        pump = _vector_series(rows, "action", 0)
        cooling = _vector_series(rows, "action", 1)
        profit = _series(rows, ("info", "profit"))
        constraint = _series(rows, ("info", "constraint"))

        series_by_panel[0].append({"label": label, "x": t, "y": ca, "style": "solid"})
        series_by_panel[1].append({"label": label, "x": t, "y": temp, "style": "solid"})
        series_by_panel[1].append({"label": f"{label} setpoint", "x": t, "y": t_sp, "style": "dash", "muted": True})
        series_by_panel[2].append({"label": f"{label} pump", "x": t, "y": pump, "style": "solid"})
        series_by_panel[2].append({"label": f"{label} cooling", "x": t, "y": cooling, "style": "dash"})
        series_by_panel[3].append({"label": label, "x": t, "y": profit, "style": "solid"})
        series_by_panel[4].append({"label": label, "x": t, "y": constraint, "style": "solid"})

    titles = ["Concentration Ca", "Temperature", "Actuator commands", "Stage profit", "Constraint penalty"]
    for idx, (panel, series_list, title) in enumerate(zip(panels, series_by_panel, titles)):
        x, y, w, h = panel
        parts.append(_svg_text(x, y - 20, title, size=15, weight="700"))
        parts.extend(_svg_axes(x, y, w, h))
        xs = [v for series in series_list for v in series["x"]]
        ys = [v for series in series_list for v in series["y"]]
        if not xs or not ys:
            continue
        xlo, xhi = min(xs), max(xs)
        ylo, yhi = min(ys), max(ys)
        if xhi == xlo:
            xhi = xlo + 1.0
        if yhi == ylo:
            yhi = ylo + 1.0
        pad = (yhi - ylo) * 0.08
        ylo -= pad
        yhi += pad
        for i, series in enumerate(series_list):
            color = colors[i % len(colors)]
            if series.get("muted"):
                color = "#94a3b8"
            dash = ' stroke-dasharray="6 5"' if series.get("style") == "dash" else ""
            points = _polyline_points(series["x"], series["y"], xlo, xhi, ylo, yhi, x, y, w, h)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        parts.append(_svg_text(x - 12, y + 8, _fmt(yhi), size=10, anchor="end", fill="#64748b"))
        parts.append(_svg_text(x - 12, y + h, _fmt(ylo), size=10, anchor="end", fill="#64748b"))
        if idx == len(panels) - 1:
            parts.append(_svg_text(x + w / 2, y + h + 36, "Time", size=13, anchor="middle", fill="#334155"))

    legend_x, legend_y = 800, 32
    names = [artifact.get("name", "controller") for artifact in rollouts if artifact.get("rollout")]
    for i, name in enumerate(names):
        lx = legend_x + (i % 3) * 125
        ly = legend_y + (i // 3) * 22
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="{colors[i % len(colors)]}" stroke-width="3"/>')
        parts.append(_svg_text(lx + 34, ly + 4, name, size=12, fill="#334155"))
    parts.append("</svg>")
    _write_text(path, "\n".join(parts))


def figure_paths(out_path: str):
    base, _ = os.path.splitext(out_path)
    return {
        "summary": f"{base}_summary.svg",
        "cstr_rollout": f"{base}_cstr_rollout.svg",
    }


def _svg_header(width: int, height: int):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}</style>'
    )


def _svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start",
              fill: str = "#0f172a", weight: str = "400"):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}">{_escape(text)}</text>'
    )


def _svg_axes(x: float, y: float, w: float, h: float):
    return [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#f8fafc" stroke="#cbd5e1"/>',
        f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#475569"/>',
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="#475569"/>',
    ]


def _polyline_points(xs, ys, xlo, xhi, ylo, yhi, x, y, w, h):
    points = []
    for xv, yv in zip(xs, ys):
        px = x + (float(xv) - xlo) / (xhi - xlo) * w
        py = _map_y(float(yv), ylo, yhi, y, h)
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def _map_y(value: float, lo: float, hi: float, y: float, h: float):
    return y + h - (float(value) - lo) / (hi - lo) * h


def _fmt(value: float):
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _escape(text: str):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="economic", choices=["economic", "tracking", "kpi"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--oracle-episodes", type=int, default=1)
    ap.add_argument("--episode-steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--control-dt", type=float, default=0.5)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--include-grid-mpc", action="store_true")
    ap.add_argument("--sb3-path", default=None)
    ap.add_argument("--sb3-algo", default="sac", choices=["sac", "ppo", "td3"])
    ap.add_argument("--sb3-action-mode", default="setpoint", choices=["actuator", "setpoint"])
    ap.add_argument("--out", default="aiogym/runs/bench_cstr_smoke.json")
    ap.add_argument("--save-rollouts", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--rollout-steps", type=int, default=None)
    args = ap.parse_args()

    make_protocol = protocol_factory(args.objective)
    baseline_protocol = make_protocol(
        "cstr",
        action_mode="actuator",
        episode_steps=args.episode_steps,
        control_dt=args.control_dt,
    )

    specs = controller_specs(args, baseline_protocol)
    results = [run_spec(spec) for spec in specs]
    rows = [compact_row(r) for r in results]

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "cstr_controller_smoke",
        "objective": args.objective,
        "metric": metric_for_reward_mode(baseline_protocol.reward_mode),
        "protocol": baseline_protocol.metadata(),
        "rows": rows,
        "results": results,
    }

    rollouts = []
    if args.save_rollouts or args.plot:
        rollouts = [rollout_spec(spec, max_steps=args.rollout_steps) for spec in specs]
        payload["rollouts"] = rollouts

    if args.plot:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, "tmp", "matplotlib"))
        figures = figure_paths(args.out)
        plot_summary(rows, figures["summary"])
        plot_cstr_rollouts(rollouts, figures["cstr_rollout"])
        payload["figures"] = figures

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"saved {args.out}")
    if args.plot:
        for kind, path in payload["figures"].items():
            print(f"saved {kind} figure {path}")
    for row in payload["rows"]:
        metric = row["metric"]
        print(
            f"{row['name']:12s} {row['control_structure']:18s} "
            f"{metric}={row[metric]:9.2f} +/- {row[f'{metric}_std']:.2f} "
            f"kpi={row['kpi']:8.2f} profit={row['profit']:8.2f} "
            f"track={row['track']:8.2f} constraint={row['constraint']:8.2f}"
        )


if __name__ == "__main__":
    main()
