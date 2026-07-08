"""SVG plotting helpers for benchmark artifacts."""
from __future__ import annotations

from pathlib import Path

from aiogym.models import make_model


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


def plot_leaderboard(board: list[dict], path: str, title: str) -> None:
    metric = board[0]["metric"] if board else "metric"
    values = [0.0 if row["metric_value"] is None else float(row["metric_value"]) for row in board]
    width = 980
    height = max(260, 130 + 54 * max(1, len(board)))
    left, top, bar_h = 220, 90, 26
    lo = min([0.0] + values)
    hi = max([0.0] + values)
    if lo == hi:
        hi = lo + 1.0
    parts = [_svg_header(width, height), _svg_text(42, 46, f"{title} leaderboard", size=22, weight="700")]
    parts.append(_svg_text(42, 74, f"Primary metric: {metric}", size=12, fill="#475569"))
    for i, row in enumerate(board):
        y = top + i * 54
        value = 0.0 if row["metric_value"] is None else float(row["metric_value"])
        x0 = left + _map_x(min(0.0, value), lo, hi, 0, 660)
        x1 = left + _map_x(max(0.0, value), lo, hi, 0, 660)
        parts.append(_svg_text(42, y + 19, f"{row['rank']}. {row['controller']}", size=13, fill="#0f172a"))
        parts.append(f'<rect x="{min(x0, x1):.2f}" y="{y:.2f}" width="{max(2.0, abs(x1 - x0)):.2f}" height="{bar_h}" fill="#2563eb"/>')
        parts.append(_svg_text(max(x0, x1) + 8, y + 18, _fmt(value), size=12, fill="#334155"))
        parts.append(_svg_text(left + 675, y + 18, str(row.get("status") or ""), size=11, fill="#64748b"))
    parts.append("</svg>")
    _write_text(path, "\n".join(parts))


def plot_constraint_timeline(rollouts: list[dict], path: str, scenario: str) -> None:
    width, height = 1100, 520
    left, top, w, h = 80, 82, 960, 330
    colors = ["#dc2626", "#2563eb", "#059669", "#d97706", "#7c3aed", "#475569"]
    series = []
    for artifact in rollouts:
        rows = artifact.get("rollout", [])
        if not rows:
            continue
        series.append({
            "name": artifact.get("name", "controller"),
            "x": [float(row.get("time", i)) for i, row in enumerate(rows)],
            "y": [float(row.get("constraint", row.get("info", {}).get("constraint", 0.0)) or 0.0) for row in rows],
        })
    xs = [v for row in series for v in row["x"]]
    ys = [v for row in series for v in row["y"]]
    xlo, xhi = (min(xs), max(xs)) if xs else (0.0, 1.0)
    ylo, yhi = 0.0, max([1.0] + ys)
    if xlo == xhi:
        xhi = xlo + 1.0
    parts = [_svg_header(width, height), _svg_text(42, 46, f"{scenario} constraint timeline", size=22, weight="700")]
    parts.append(f'<rect x="{left}" y="{top}" width="{w}" height="{h}" fill="#f8fafc" stroke="#cbd5e1"/>')
    for i, row in enumerate(series):
        points = []
        for xv, yv in zip(row["x"], row["y"]):
            px = left + (xv - xlo) / (xhi - xlo) * w
            py = top + h - (yv - ylo) / (yhi - ylo) * h
            points.append(f"{px:.2f},{py:.2f}")
        color = colors[i % len(colors)]
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        lx = left + (i % 4) * 210
        ly = top + h + 48 + (i // 4) * 24
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 30}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(_svg_text(lx + 38, ly + 4, row["name"], size=12, fill="#334155"))
    parts.append(_svg_text(left - 12, top + 10, _fmt(yhi), size=10, anchor="end", fill="#64748b"))
    parts.append(_svg_text(left + w / 2, top + h + 34, "Time", size=12, anchor="middle", fill="#334155"))
    parts.append("</svg>")
    _write_text(path, "\n".join(parts))


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


def _map_x(value: float, lo: float, hi: float, x: float, w: float) -> float:
    return x + (float(value) - lo) / (hi - lo) * w


def _fmt(value: float):
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _escape(text: str):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_text(path: str, text: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
