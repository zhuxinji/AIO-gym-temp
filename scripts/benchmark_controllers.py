#!/usr/bin/env python3
"""Generic controller benchmark runner for all built-in AIO-Gym scenarios."""
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
    BenchmarkConfig,
    BenchmarkProtocol,
    build_evaluation_report,
    evaluate_controller,
    metric_for_reward_mode,
    rollout_controller,
)
from aiogym.models import SCENARIOS, make_model


def protocol_factory(objective: str):
    return {
        "economic": BenchmarkProtocol.economic,
        "tracking": BenchmarkProtocol.tracking,
        "robustness": BenchmarkProtocol.robustness,
        "safety": BenchmarkProtocol.safety,
        "kpi": BenchmarkProtocol.kpi,
    }[objective]


def parse_controllers(raw: str):
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("--controllers must include at least one controller")
    return names


def parse_seed_list(raw: str | None, seed: int, episodes: int):
    if not raw:
        return [seed + i for i in range(episodes)]
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seed-list must contain at least one integer seed")
    return seeds


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
        "normalized_score": result["normalized_score"],
        "profit": result["profit"],
        "return": result["return"],
        "track": result["track"],
        "constraint": result["constraint"],
        "tracking_iae": result.get("tracking_iae"),
        "energy_kwh": result.get("energy_kwh"),
        "constraint_violation_count": result.get("constraint_violation_count"),
        "constraint_violation_severity": result.get("constraint_violation_severity"),
        "safety_margin_min": result.get("safety_margin_min"),
        "episodes": result["episodes"],
        "seed": result["seed"],
        "seed_list": result["seed_list"],
    }


def controller_specs(args, baseline_protocol: BenchmarkProtocol):
    specs = []
    for name in parse_controllers(args.controllers):
        if name == "cstr_grid_mpc" and args.scenario != "cstr":
            raise ValueError("cstr_grid_mpc is only valid with --scenario cstr")
        if name == "sb3":
            if not args.sb3_path:
                raise ValueError("controller 'sb3' requires --sb3-path")
            sb3_protocol = protocol_factory(args.objective)(
                args.scenario,
                action_mode=args.sb3_action_mode,
                episode_steps=args.episode_steps,
                control_dt=args.control_dt,
            )
            specs.append({
                "name": "sb3",
                "protocol": sb3_protocol,
                "seed_list": parse_seed_list(args.seed_list, args.seed, args.episodes),
                "config": {
                    "path": args.sb3_path,
                    "algo": args.sb3_algo,
                    "action_mode": args.sb3_action_mode,
                },
            })
            continue
        if name in ("oracle", "nmpc"):
            mode = "track" if args.objective == "tracking" else baseline_protocol.reward_mode
            specs.append({
                "name": name,
                "protocol": baseline_protocol,
                "seed_list": parse_seed_list(args.seed_list, args.seed, args.oracle_episodes),
                "config": {"mode": mode},
            })
            continue
        specs.append({
            "name": name,
            "protocol": baseline_protocol,
            "seed_list": parse_seed_list(args.seed_list, args.seed, args.episodes),
            "config": {},
        })
    return specs


def run_spec(spec: dict, scenario: str):
    controller = make_controller(spec["name"], scenario=scenario, config=spec.get("config") or {})
    seeds = spec["seed_list"]
    return evaluate_controller(
        controller,
        spec["protocol"].make_env(),
        episodes=len(seeds),
        seed=seeds[0],
        seed_list=seeds,
        protocol=spec["protocol"],
        include_episodes=True,
    )


def rollout_spec(spec: dict, scenario: str, max_steps: int | None = None):
    controller = make_controller(spec["name"], scenario=scenario, config=spec.get("config") or {})
    protocol = spec["protocol"]
    return rollout_controller(
        controller,
        protocol.make_env(),
        seed=spec["seed_list"][0],
        max_steps=max_steps,
        protocol=protocol,
    )


def figure_paths(out_path: str, scenario: str):
    base, _ = os.path.splitext(out_path)
    return {
        "summary": f"{base}_summary.svg",
        "rollout": f"{base}_{scenario}_rollout.svg",
    }


def plot_summary(rows: list[dict], path: str, scenario: str):
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
    title = f"{scenario} controller benchmark"
    parts = [_svg_header(width, height), _svg_text(60, 45, title, size=24, weight="700")]
    for panel, (key, panel_title) in zip(panels, metrics):
        x, y, w, h = panel
        vals = [float(row.get(key, 0.0)) for row in rows]
        parts.append(_svg_text(x, y - 22, panel_title, size=16, weight="700"))
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


def plot_rollouts(rollouts: list[dict], path: str, scenario: str):
    model = make_model(scenario)
    if scenario == "cstr":
        panels = [
            {"title": "Concentration Ca", "series": [state_series(0)]},
            {"title": "Temperature", "series": [info_series("temps", 0), setpoint_series("t_sp", 0, dashed=True)]},
            {"title": "Actuator commands", "series": action_series(model.action_names[:2])},
            {"title": "Stage profit", "series": [metric_series("profit")]},
            {"title": "Constraint penalty", "series": [metric_series("constraint")]},
        ]
    elif scenario == "hvac":
        panels = [
            {"title": "Zone temperatures", "series": info_vector_family("temps")},
            {"title": "Temperature setpoints", "series": setpoint_vector_family("t_sp")},
            {"title": "Actuator commands", "series": action_series(model.action_names)},
            {"title": "Tracking error", "series": [metric_series("track")]},
            {"title": "Constraint penalty", "series": [metric_series("constraint")]},
        ]
    else:
        panels = [
            {"title": "Levels", "series": info_vector_family("levels")},
            {"title": "Temperatures", "series": info_vector_family("temps")},
            {"title": "Actuator commands", "series": action_series(model.action_names)},
            {"title": "Tracking error", "series": [metric_series("track")]},
            {"title": "Constraint penalty", "series": [metric_series("constraint")]},
        ]
    _plot_series_panels(rollouts, panels, path, f"{scenario} rollout comparison")


def state_series(index: int, label: str | None = None, dashed: bool = False):
    return {"kind": "state", "index": index, "label": label or f"x{index}", "dashed": dashed}


def info_series(key: str, index: int, label: str | None = None, dashed: bool = False):
    return {"kind": "info_vector", "key": key, "index": index, "label": label or f"{key}{index}", "dashed": dashed}


def setpoint_series(key: str, index: int, label: str | None = None, dashed: bool = False):
    return {"kind": "setpoint_vector", "key": key, "index": index, "label": label or f"{key}{index}", "dashed": dashed, "muted": dashed}


def metric_series(key: str, dashed: bool = False):
    return {"kind": "metric", "key": key, "label": key, "dashed": dashed}


def action_series(names):
    return [{"kind": "action", "index": i, "label": name} for i, name in enumerate(names)]


def info_vector_family(key: str):
    return [info_series(key, i) for i in range(_max_info_vector_len(key))]


def setpoint_vector_family(key: str):
    return [setpoint_series(key, i, dashed=True) for i in range(4)]


def _max_info_vector_len(key: str):
    return {"levels": 4, "temps": 4}.get(key, 1)


def _plot_series_panels(rollouts: list[dict], panels: list[dict], path: str, title: str):
    width, height = 1200, 1120
    left, right = 90, 40
    panel_h, gap = 150, 48
    top = 95
    panel_boxes = [(left, top + i * (panel_h + gap), width - left - right, panel_h) for i in range(len(panels))]
    colors = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#475569", "#0f766e", "#be123c"]
    parts = [_svg_header(width, height), _svg_text(60, 45, title, size=24, weight="700")]

    for pidx, (panel, box) in enumerate(zip(panels, panel_boxes)):
        x, y, w, h = box
        parts.append(_svg_text(x, y - 20, panel["title"], size=15, weight="700"))
        parts.extend(_svg_axes(x, y, w, h))
        collected = []
        for artifact in rollouts:
            rows = artifact.get("rollout", [])
            if not rows:
                continue
            t = [float(row.get("time", i)) for i, row in enumerate(rows)]
            for spec in panel["series"]:
                ys = _extract_series(rows, spec)
                if any(v is not None for v in ys):
                    collected.append({
                        "controller": artifact.get("name", "controller"),
                        "label": spec["label"],
                        "x": t,
                        "y": [0.0 if v is None else float(v) for v in ys],
                        "dashed": spec.get("dashed", False),
                        "muted": spec.get("muted", False),
                    })
        xs = [v for series in collected for v in series["x"]]
        ys = [v for series in collected for v in series["y"]]
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
        for i, series in enumerate(collected):
            color = colors[i % len(colors)]
            if series["muted"]:
                color = "#94a3b8"
            dash = ' stroke-dasharray="6 5"' if series["dashed"] else ""
            points = _polyline_points(series["x"], series["y"], xlo, xhi, ylo, yhi, x, y, w, h)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        parts.append(_svg_text(x - 12, y + 8, _fmt(yhi), size=10, anchor="end", fill="#64748b"))
        parts.append(_svg_text(x - 12, y + h, _fmt(ylo), size=10, anchor="end", fill="#64748b"))
        if pidx == len(panel_boxes) - 1:
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


def _extract_series(rows: list[dict], spec: dict):
    if spec["kind"] == "state":
        return [_list_value(row.get("next_state"), spec["index"]) for row in rows]
    if spec["kind"] == "info_vector":
        return [_list_value(row.get("info", {}).get(spec["key"]), spec["index"]) for row in rows]
    if spec["kind"] == "setpoint_vector":
        return [_list_value(row.get("setpoint", {}).get(spec["key"]), spec["index"]) for row in rows]
    if spec["kind"] == "action":
        return [_list_value(row.get("action"), spec["index"]) for row in rows]
    if spec["kind"] == "metric":
        return [row.get("info", {}).get(spec["key"]) for row in rows]
    raise ValueError(f"unknown plot series kind {spec['kind']!r}")


def _list_value(value, index: int):
    if isinstance(value, list) and len(value) > index:
        return value[index]
    return None


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
    ap.add_argument("--scenario", default="cstr", choices=SCENARIOS)
    ap.add_argument("--objective", default="economic", choices=["economic", "tracking", "robustness", "safety", "kpi"])
    ap.add_argument("--controllers", default="pid,mpc")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--oracle-episodes", type=int, default=1)
    ap.add_argument("--episode-steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--seed-list", default=None, help="comma-separated fixed seeds; overrides --seed/--episodes")
    ap.add_argument("--control-dt", type=float, default=0.5)
    ap.add_argument("--sb3-path", default=None)
    ap.add_argument("--sb3-algo", default="sac", choices=["sac", "ppo", "td3"])
    ap.add_argument("--sb3-action-mode", default="setpoint", choices=["actuator", "setpoint"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-rollouts", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--rollout-steps", type=int, default=None)
    args = ap.parse_args()

    make_protocol = protocol_factory(args.objective)
    baseline_protocol = make_protocol(
        args.scenario,
        action_mode="actuator",
        episode_steps=args.episode_steps,
        control_dt=args.control_dt,
    )
    out_path = args.out or f"aiogym/runs/bench_{args.scenario}_controllers.json"
    specs = controller_specs(args, baseline_protocol)
    results = [run_spec(spec, args.scenario) for spec in specs]
    rows = [compact_row(r) for r in results]
    configs = [
        BenchmarkConfig.from_protocol(
            spec["protocol"],
            controller=spec["name"],
            seeds=spec["seed_list"],
            controller_config=spec.get("config") or {},
        ).metadata()
        for spec in specs
    ]

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "controller_benchmark",
        "scenario": args.scenario,
        "objective": args.objective,
        "controllers": [spec["name"] for spec in specs],
        "metric": metric_for_reward_mode(baseline_protocol.reward_mode),
        "protocol": baseline_protocol.metadata(),
        "configs": configs,
        "rows": rows,
        "results": results,
        "report": build_evaluation_report(results),
    }

    rollouts = []
    if args.save_rollouts or args.plot:
        rollouts = [rollout_spec(spec, args.scenario, max_steps=args.rollout_steps) for spec in specs]
        payload["rollouts"] = rollouts

    if args.plot:
        figures = figure_paths(out_path, args.scenario)
        plot_summary(rows, figures["summary"], args.scenario)
        plot_rollouts(rollouts, figures["rollout"], args.scenario)
        payload["figures"] = figures

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"saved {out_path}")
    if args.plot:
        for kind, path in payload["figures"].items():
            print(f"saved {kind} figure {path}")
    for row in payload["rows"]:
        metric = row["metric"]
        print(
            f"{row['name']:18s} {row['control_structure']:18s} "
            f"{metric}={row[metric]:9.2f} +/- {row[f'{metric}_std']:.2f} "
            f"kpi={row['kpi']:8.2f} profit={row['profit']:8.2f} "
            f"track={row['track']:8.2f} constraint={row['constraint']:8.2f} "
            f"safety={row.get('constraint_violation_count', 0):6.1f}"
        )


if __name__ == "__main__":
    main()
