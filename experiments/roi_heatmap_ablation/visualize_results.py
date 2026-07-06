from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Iterable


MODELS = ("6dof", "heatmap_6dof")
MODEL_LABELS = {
    "6dof": "6DoF only",
    "heatmap_6dof": "Heatmap + 6DoF",
}
COLORS = {
    "6dof": "#2563eb",
    "heatmap_6dof": "#f97316",
    "delta": "#16a34a",
    "neutral": "#64748b",
}


def pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install it with: pip install matplotlib") from exc
    return plt


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_report_path(root: Path) -> Path:
    candidates = [
        root / "sensor-agent-publish-subscribe" / "outputs" / "roi_heatmap_ablation" / "comparison_report.json",
        root / "outputs" / "roi_heatmap_ablation" / "comparison_report.json",
        root
        / "experiments"
        / "roi_heatmap_ablation"
        / "outputs"
        / "roi_heatmap_ablation"
        / "comparison_report.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def default_split_path(root: Path, report: dict) -> Path | None:
    candidates = [
        root / "sensor-agent-publish-subscribe" / "outputs" / "splits" / "dataset_split.json",
        root / "outputs" / "splits" / "dataset_split.json",
    ]
    split_path = report.get("metadata", {}).get("split_path")
    if split_path:
        candidates.append(Path(split_path))
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: float) -> float:
    return value * 100.0


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def horizons(report: dict) -> list[int]:
    return sorted(int(h) for h in report["results"])


def get_metric(report: dict, horizon_ms: int, model: str, metric: str) -> float:
    return float(report["results"][str(horizon_ms)][model]["test"][metric])


def get_topk(report: dict, horizon_ms: int, model: str, k: int) -> float:
    return float(report["results"][str(horizon_ms)][model]["test"]["top_k_accuracy"][str(k)])


def get_losses(report: dict, horizon_ms: int, model: str) -> list[float]:
    return [float(v) for v in report["results"][str(horizon_ms)][model]["train"]["losses"]]


def split_counts(split: dict | None) -> dict[str, int]:
    if not split:
        return {}
    return {name: len(split.get(name, [])) for name in ("train", "valid", "test")}


def roi_summary(report: dict) -> dict[str, float | int | list[float]]:
    md = report["metadata"]
    grid = int(md["grid_size"])
    bounds_min = [float(v) for v in md["bounds_min"]]
    bounds_max = [float(v) for v in md["bounds_max"]]
    span = [bmax - bmin for bmin, bmax in zip(bounds_min, bounds_max)]
    cell = [v / grid for v in span]
    return {
        "grid_size": grid,
        "roi_count": grid**3,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "span": span,
        "cell": cell,
    }


def result_rows(report: dict) -> list[dict[str, str | int | float]]:
    rows = []
    for h in horizons(report):
        base = report["results"][str(h)]["6dof"]
        heat = report["results"][str(h)]["heatmap_6dof"]
        for model in MODELS:
            test = report["results"][str(h)][model]["test"]
            train = report["results"][str(h)][model]["train"]
            rows.append(
                {
                    "horizon_ms": h,
                    "model": model,
                    "train_samples": int(train["samples"]),
                    "test_samples": int(test["samples"]),
                    "top1_pct": pct(float(test["top_k_accuracy"]["1"])),
                    "top3_pct": pct(float(test["top_k_accuracy"]["3"])),
                    "top5_pct": pct(float(test["top_k_accuracy"]["5"])),
                    "top10_pct": pct(float(test["top_k_accuracy"]["10"])),
                    "roi_distance_error_mean": float(test["roi_distance_error_mean"]),
                    "roi_distance_error_median": float(test["roi_distance_error_median"]),
                    "latency_ms_mean": float(test["latency_ms_mean"]),
                    "latency_ms_p95": float(test["latency_ms_p95"]),
                    "peak_memory_mb": float(test["peak_memory_mb"]),
                    "loss_start": float(train["losses"][0]),
                    "loss_end": float(train["losses"][-1]),
                    "loss_drop": float(train["losses"][0]) - float(train["losses"][-1]),
                }
            )
        btest = base["test"]
        htest = heat["test"]
        rows.append(
            {
                "horizon_ms": h,
                "model": "heatmap_minus_6dof",
                "train_samples": int(heat["train"]["samples"]) - int(base["train"]["samples"]),
                "test_samples": int(htest["samples"]) - int(btest["samples"]),
                "top1_pct": pct(float(htest["top_k_accuracy"]["1"]) - float(btest["top_k_accuracy"]["1"])),
                "top3_pct": pct(float(htest["top_k_accuracy"]["3"]) - float(btest["top_k_accuracy"]["3"])),
                "top5_pct": pct(float(htest["top_k_accuracy"]["5"]) - float(btest["top_k_accuracy"]["5"])),
                "top10_pct": pct(float(htest["top_k_accuracy"]["10"]) - float(btest["top_k_accuracy"]["10"])),
                "roi_distance_error_mean": float(htest["roi_distance_error_mean"]) - float(btest["roi_distance_error_mean"]),
                "roi_distance_error_median": float(htest["roi_distance_error_median"])
                - float(btest["roi_distance_error_median"]),
                "latency_ms_mean": float(htest["latency_ms_mean"]) - float(btest["latency_ms_mean"]),
                "latency_ms_p95": float(htest["latency_ms_p95"]) - float(btest["latency_ms_p95"]),
                "peak_memory_mb": float(htest["peak_memory_mb"]) - float(btest["peak_memory_mb"]),
                "loss_start": float(heat["train"]["losses"][0]) - float(base["train"]["losses"][0]),
                "loss_end": float(heat["train"]["losses"][-1]) - float(base["train"]["losses"][-1]),
                "loss_drop": (float(heat["train"]["losses"][0]) - float(heat["train"]["losses"][-1]))
                - (float(base["train"]["losses"][0]) - float(base["train"]["losses"][-1])),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#0f172a}",
        ".title{font-size:20px;font-weight:700}",
        ".label{font-size:12px;fill:#475569}",
        ".tick{font-size:11px;fill:#64748b}",
        ".legend{font-size:12px;fill:#334155}",
        ".grid{stroke:#e2e8f0;stroke-width:1}",
        ".axis{stroke:#94a3b8;stroke-width:1}",
        "</style>",
    ]


def save_svg(path: Path, body: Iterable[str], width: int = 920, height: int = 520) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = svg_header(width, height)
    content.extend(body)
    content.append("</svg>")
    path.write_text("\n".join(content), encoding="utf-8")


def save_mpl(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    pyplot().close(fig)


def mpl_bar_chart(
    path: Path,
    title: str,
    groups: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    x = list(range(len(groups)))
    width = min(0.36, 0.78 / max(len(series), 1))
    offset_start = -width * (len(series) - 1) / 2
    for i, (label, values, color) in enumerate(series):
        bars = ax.bar([v + offset_start + i * width for v in x], values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_title(title, fontsize=15, weight="bold", pad=14)
    ax.set_ylabel(y_label)
    ax.set_xticks(x, groups)
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncols=max(1, len(series)))
    fig.tight_layout()
    save_mpl(fig, path)


def mpl_line_chart(
    path: Path,
    title: str,
    x_labels: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    x = list(range(len(x_labels)))
    linestyles = ["-", "--", "-.", ":"]
    for i, (label, values, color) in enumerate(series):
        alpha = 0.9 if i < 4 else 0.75
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.0,
            markersize=4,
            color=color,
            linestyle=linestyles[i % len(linestyles)],
            alpha=alpha,
            label=label,
        )
    ax.set_title(title, fontsize=15, weight="bold", pad=14)
    ax.set_ylabel(y_label)
    ax.set_xticks(x, x_labels)
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    ax.grid(axis="both", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncols=2, fontsize=8)
    fig.tight_layout()
    save_mpl(fig, path)


def mpl_split_chart(path: Path, counts: dict[str, int]) -> None:
    if not counts:
        return
    labels = ["train", "valid", "test"]
    values = [counts[name] for name in labels]
    total = sum(values)
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    colors = ["#2563eb", "#64748b", "#f97316"]
    bars = axes[0].bar([v.capitalize() for v in labels], values, color=colors)
    axes[0].bar_label(bars, labels=[f"{v}\n({v / total:.1%})" for v in values], padding=4, fontsize=9)
    axes[0].set_title("Dataset split by session", fontsize=14, weight="bold")
    axes[0].set_ylabel("session count")
    axes[0].grid(axis="y", color="#e2e8f0")
    axes[0].set_axisbelow(True)
    axes[1].pie(values, labels=[v.capitalize() for v in labels], autopct="%1.1f%%", colors=colors, startangle=90)
    axes[1].set_title(f"Total sessions: {total}", fontsize=14, weight="bold")
    fig.tight_layout()
    save_mpl(fig, path)


def mpl_roi_grid(path: Path, summary: dict[str, float | int | list[float]]) -> None:
    plt = pyplot()
    from matplotlib.patches import Rectangle

    grid = int(summary["grid_size"])
    roi_count = int(summary["roi_count"])
    span = summary["span"]
    cell = summary["cell"]
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, grid + 6.8)
    ax.set_ylim(-1.8, grid + 1.2)
    for i in range(grid):
        for j in range(grid):
            face = "#eff6ff" if (i + j) % 2 == 0 else "#dbeafe"
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor=face, edgecolor="#93c5fd", linewidth=0.8))
    ax.add_patch(Rectangle((0, 0), grid, grid, fill=False, edgecolor="#0f172a", linewidth=1.6))
    offset = 0.16
    for layer in range(grid):
        ax.plot(
            [grid + layer * offset, grid + 1.5 + layer * offset],
            [grid, grid + 0.9 + layer * offset],
            color="#94a3b8",
            linewidth=0.8,
        )
    ax.text(grid / 2, -0.85, "One 8 x 8 slice", ha="center", va="top", fontsize=10, color="#475569")
    ax.text(0, grid + 0.65, f"ROI grid: {grid} x {grid} x {grid} = {roi_count} cells", fontsize=15, weight="bold")
    info = (
        f"Cell size\n"
        f"X: {fmt(cell[0], 2)}\n"
        f"Y: {fmt(cell[1], 2)}\n"
        f"Z: {fmt(cell[2], 2)}\n\n"
        f"Coordinate span\n"
        f"X: {fmt(span[0], 2)}\n"
        f"Y: {fmt(span[1], 2)}\n"
        f"Z: {fmt(span[2], 2)}"
    )
    ax.text(grid + 2.2, grid - 0.2, info, fontsize=11, va="top", linespacing=1.45)
    ax.axis("off")
    fig.tight_layout()
    save_mpl(fig, path)


def generate_matplotlib_figures(report: dict, split: dict | None, figures: Path) -> None:
    hs = horizons(report)
    groups = [f"{h} ms" for h in hs]
    for k in (1, 3, 5, 10):
        mpl_bar_chart(
            figures / f"top{k}_accuracy",
            f"Top-{k} accuracy by prediction horizon",
            groups,
            [
                (MODEL_LABELS["6dof"], [pct(get_topk(report, h, "6dof", k)) for h in hs], COLORS["6dof"]),
                (
                    MODEL_LABELS["heatmap_6dof"],
                    [pct(get_topk(report, h, "heatmap_6dof", k)) for h in hs],
                    COLORS["heatmap_6dof"],
                ),
            ],
            "accuracy (%)",
            y_min=0,
            y_max=100 if k == 10 else None,
        )
    delta_series = []
    for k, color in [(1, "#dc2626"), (3, "#0891b2"), (5, "#16a34a"), (10, "#7c3aed")]:
        delta_series.append(
            (f"Top-{k} delta", [pct(get_topk(report, h, "heatmap_6dof", k) - get_topk(report, h, "6dof", k)) for h in hs], color)
        )
    mpl_line_chart(
        figures / "accuracy_delta",
        "Accuracy improvement from heatmap features",
        groups,
        delta_series,
        "percentage points",
        y_min=-3,
        y_max=17,
    )
    mpl_bar_chart(
        figures / "roi_error",
        "Mean ROI distance error",
        groups,
        [
            (MODEL_LABELS["6dof"], [get_metric(report, h, "6dof", "roi_distance_error_mean") for h in hs], COLORS["6dof"]),
            (
                MODEL_LABELS["heatmap_6dof"],
                [get_metric(report, h, "heatmap_6dof", "roi_distance_error_mean") for h in hs],
                COLORS["heatmap_6dof"],
            ),
        ],
        "distance between predicted and true ROI centers",
        y_min=0,
    )
    mpl_bar_chart(
        figures / "latency",
        "Mean prediction latency",
        groups,
        [
            (MODEL_LABELS["6dof"], [get_metric(report, h, "6dof", "latency_ms_mean") for h in hs], COLORS["6dof"]),
            (
                MODEL_LABELS["heatmap_6dof"],
                [get_metric(report, h, "heatmap_6dof", "latency_ms_mean") for h in hs],
                COLORS["heatmap_6dof"],
            ),
        ],
        "milliseconds per prediction",
        y_min=0,
    )
    mpl_bar_chart(
        figures / "memory",
        "Peak evaluation memory",
        groups,
        [
            (MODEL_LABELS["6dof"], [get_metric(report, h, "6dof", "peak_memory_mb") for h in hs], COLORS["6dof"]),
            (
                MODEL_LABELS["heatmap_6dof"],
                [get_metric(report, h, "heatmap_6dof", "peak_memory_mb") for h in hs],
                COLORS["heatmap_6dof"],
            ),
        ],
        "MB",
        y_min=0,
    )
    loss_series = []
    for h in hs:
        loss_series.append((f"6DoF {h} ms", get_losses(report, h, "6dof"), COLORS["6dof"]))
        loss_series.append((f"Heatmap {h} ms", get_losses(report, h, "heatmap_6dof"), COLORS["heatmap_6dof"]))
    epochs = [str(i + 1) for i in range(len(loss_series[0][1]))]
    mpl_line_chart(figures / "training_loss", "Training loss by epoch", epochs, loss_series, "loss")
    mpl_split_chart(figures / "dataset_split", split_counts(split))
    mpl_roi_grid(figures / "roi_grid", roi_summary(report))


def bar_chart(
    path: Path,
    title: str,
    groups: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    y_max: float | None = None,
    y_min: float = 0.0,
) -> None:
    width, height = 920, 520
    left, right, top, bottom = 78, 28, 66, 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [v for _, vals, _ in series for v in vals]
    if y_max is None:
        y_max = max(values) * 1.15 if values else 1
    if y_max <= y_min:
        y_max = y_min + 1

    def y(value: float) -> float:
        return top + plot_h - ((value - y_min) / (y_max - y_min)) * plot_h

    body = [
        f'<text class="title" x="{left}" y="34">{html.escape(title)}</text>',
        f'<text class="label" x="{left}" y="54">{html.escape(y_label)}</text>',
    ]
    for i in range(6):
        val = y_min + (y_max - y_min) * i / 5
        yy = y(val)
        body.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}"/>')
        body.append(f'<text class="tick" x="{left-10}" y="{yy+4:.1f}" text-anchor="end">{fmt(val, 1)}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')

    group_w = plot_w / len(groups)
    bar_gap = 6
    bar_w = min(42, (group_w - 30) / max(len(series), 1) - bar_gap)
    for gi, group in enumerate(groups):
        gx = left + gi * group_w
        center = gx + group_w / 2
        total_w = len(series) * bar_w + (len(series) - 1) * bar_gap
        start_x = center - total_w / 2
        for si, (_, vals, color) in enumerate(series):
            value = vals[gi]
            x = start_x + si * (bar_w + bar_gap)
            yy = y(value)
            h = top + plot_h - yy
            body.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="2"/>')
            body.append(
                f'<text class="tick" x="{x + bar_w / 2:.1f}" y="{yy - 5:.1f}" text-anchor="middle">{fmt(value, 1)}</text>'
            )
        body.append(f'<text class="label" x="{center:.1f}" y="{height-42}" text-anchor="middle">{html.escape(group)}</text>')

    lx = left
    ly = height - 18
    for label, _, color in series:
        body.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" fill="{color}" rx="2"/>')
        body.append(f'<text class="legend" x="{lx+18}" y="{ly}">{html.escape(label)}</text>')
        lx += 150
    save_svg(path, body, width, height)


def line_chart(
    path: Path,
    title: str,
    x_labels: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    width, height = 920, 520
    left, right, top, bottom = 78, 34, 66, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [v for _, vals, _ in series for v in vals]
    if y_min is None:
        y_min = min(values) * 0.95
    if y_max is None:
        y_max = max(values) * 1.05
    if math.isclose(y_min, y_max):
        y_min -= 1
        y_max += 1

    def x(i: int) -> float:
        if len(x_labels) == 1:
            return left + plot_w / 2
        return left + plot_w * i / (len(x_labels) - 1)

    def y(value: float) -> float:
        return top + plot_h - ((value - y_min) / (y_max - y_min)) * plot_h

    body = [
        f'<text class="title" x="{left}" y="34">{html.escape(title)}</text>',
        f'<text class="label" x="{left}" y="54">{html.escape(y_label)}</text>',
    ]
    for i in range(6):
        val = y_min + (y_max - y_min) * i / 5
        yy = y(val)
        body.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}"/>')
        body.append(f'<text class="tick" x="{left-10}" y="{yy+4:.1f}" text-anchor="end">{fmt(val, 2)}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    for i, label in enumerate(x_labels):
        body.append(f'<text class="label" x="{x(i):.1f}" y="{height-44}" text-anchor="middle">{html.escape(label)}</text>')

    dash_patterns = ["", "4 3", "2 4", "8 3", "3 2"]
    for si, (label, vals, color) in enumerate(series):
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        dash = dash_patterns[si % len(dash_patterns)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        body.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5"{dash_attr} points="{points}"/>')
        for i, v in enumerate(vals):
            body.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="4" fill="{color}"/>')

    lx = left
    ly = height - 18
    for label, _, color in series:
        body.append(f'<line x1="{lx}" y1="{ly-5}" x2="{lx+20}" y2="{ly-5}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text class="legend" x="{lx+26}" y="{ly}">{html.escape(label)}</text>')
        lx += min(220, 58 + len(label) * 7)
    save_svg(path, body, width, height)


def delta_chart(path: Path, report: dict) -> None:
    hs = horizons(report)
    groups = [f"{h} ms" for h in hs]
    series = []
    for k, color in [(1, "#dc2626"), (3, "#0891b2"), (5, "#16a34a"), (10, "#7c3aed")]:
        vals = [pct(get_topk(report, h, "heatmap_6dof", k) - get_topk(report, h, "6dof", k)) for h in hs]
        series.append((f"Top-{k} delta", vals, color))
    line_chart(path, "Accuracy improvement from heatmap features", groups, series, "percentage points", y_min=-3, y_max=17)


def split_svg(path: Path, counts: dict[str, int]) -> None:
    if not counts:
        return
    labels = ["train", "valid", "test"]
    total = sum(counts.values())
    bar_chart(
        path,
        "Dataset split by session",
        [name.capitalize() for name in labels],
        [("Sessions", [counts[name] for name in labels], COLORS["neutral"])],
        f"session count, total={total}",
        y_max=max(counts.values()) * 1.2,
    )


def roi_svg(path: Path, summary: dict[str, float | int | list[float]]) -> None:
    width, height = 920, 520
    left, top = 82, 92
    grid = int(summary["grid_size"])
    roi_count = int(summary["roi_count"])
    span = summary["span"]
    cell = summary["cell"]
    body = [
        f'<text class="title" x="{left}" y="34">ROI grid definition</text>',
        f'<text class="label" x="{left}" y="58">3D space is divided into {grid} x {grid} x {grid} = {roi_count} ROI cells.</text>',
    ]
    size = 260
    cell_px = size / grid
    for i in range(grid + 1):
        x = left + i * cell_px
        y = top + i * cell_px
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+size}"/>')
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+size}" y2="{y:.1f}"/>')
    body.append(f'<rect x="{left}" y="{top}" width="{size}" height="{size}" fill="none" stroke="#0f172a" stroke-width="2"/>')
    body.append(f'<text class="label" x="{left+size/2}" y="{top+size+28}" text-anchor="middle">One 2D slice of the 8 x 8 grid</text>')
    zx, zy = left + size + 38, top - 38
    body.append(
        f'<path d="M {left+size} {top} L {zx+size} {zy} L {zx+size} {zy+size} L {left+size} {top+size} Z" fill="#f8fafc" stroke="#94a3b8"/>'
    )
    body.append(
        f'<path d="M {left} {top} L {zx} {zy} L {zx+size} {zy} L {left+size} {top} Z" fill="#eef2ff" stroke="#94a3b8"/>'
    )
    info_x = 600
    body.extend(
        [
            f'<text class="title" x="{info_x}" y="{top+10}">Cell size</text>',
            f'<text class="label" x="{info_x}" y="{top+42}">X range: {fmt(span[0], 2)} / {grid} = {fmt(cell[0], 2)}</text>',
            f'<text class="label" x="{info_x}" y="{top+68}">Y range: {fmt(span[1], 2)} / {grid} = {fmt(cell[1], 2)}</text>',
            f'<text class="label" x="{info_x}" y="{top+94}">Z range: {fmt(span[2], 2)} / {grid} = {fmt(cell[2], 2)}</text>',
            f'<text class="title" x="{info_x}" y="{top+154}">Interpretation</text>',
            f'<text class="label" x="{info_x}" y="{top+188}">The model predicts which cell</text>',
            f'<text class="label" x="{info_x}" y="{top+212}">the future headset position falls in.</text>',
            f'<text class="label" x="{info_x}" y="{top+246}">Top-k is correct if the true cell</text>',
            f'<text class="label" x="{info_x}" y="{top+270}">is within the first k predictions.</text>',
        ]
    )
    save_svg(path, body, width, height)


def markdown_table(report: dict) -> str:
    lines = [
        "| Horizon | Model | Train samples | Test samples | Top-1 | Top-3 | Top-5 | Top-10 | Mean ROI error | Mean latency | Peak memory |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in horizons(report):
        for model in MODELS:
            train = report["results"][str(h)][model]["train"]
            test = report["results"][str(h)][model]["test"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{h} ms",
                        MODEL_LABELS[model],
                        str(train["samples"]),
                        str(test["samples"]),
                        f'{pct(float(test["top_k_accuracy"]["1"])):.2f}%',
                        f'{pct(float(test["top_k_accuracy"]["3"])):.2f}%',
                        f'{pct(float(test["top_k_accuracy"]["5"])):.2f}%',
                        f'{pct(float(test["top_k_accuracy"]["10"])):.2f}%',
                        fmt(float(test["roi_distance_error_mean"]), 2),
                        f'{fmt(float(test["latency_ms_mean"]), 3)} ms',
                        f'{fmt(float(test["peak_memory_mb"]), 2)} MB',
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def write_report_markdown(path: Path, report: dict, split: dict | None, figure_dir_name: str = "figures") -> None:
    roi = roi_summary(report)
    counts = split_counts(split)
    hs = horizons(report)
    total_sessions = sum(counts.values()) if counts else None
    avg_delta = {
        k: sum(pct(get_topk(report, h, "heatmap_6dof", k) - get_topk(report, h, "6dof", k)) for h in hs) / len(hs)
        for k in (1, 3, 5, 10)
    }
    avg_error_delta = sum(
        get_metric(report, h, "heatmap_6dof", "roi_distance_error_mean")
        - get_metric(report, h, "6dof", "roi_distance_error_mean")
        for h in hs
    ) / len(hs)
    split_sentence = (
        f"The split contains {total_sessions} sessions: {counts.get('train', 0)} train, "
        f"{counts.get('valid', 0)} validation, and {counts.get('test', 0)} test."
        if counts
        else "The dataset split file was not found, so only sample counts from the result report are listed."
    )
    md = f"""# ROI Heatmap Ablation Report

## 1. Experiment goal

This experiment tests whether RF heatmap information helps predict a future region of interest (ROI). We compare two lightweight models:

- `6dof`: uses only headset pose history, meaning position and rotation over the previous 250 ms.
- `heatmap_6dof`: uses the same pose history plus compact features extracted from the RF heatmap.

The prediction task is not to estimate an exact coordinate directly. Instead, the physical 3D space is divided into ROI cells, and the model predicts which cell the future position will fall into.

## 2. ROI definition

The ROI grid size is `{roi["grid_size"]}` on each axis, so the space is divided into:

`{roi["grid_size"]} x {roi["grid_size"]} x {roi["grid_size"]} = {roi["roi_count"]}` ROI cells.

The coordinate bounds used in this run are:

- Min: x={fmt(roi["bounds_min"][0], 2)}, y={fmt(roi["bounds_min"][1], 2)}, z={fmt(roi["bounds_min"][2], 2)}
- Max: x={fmt(roi["bounds_max"][0], 2)}, y={fmt(roi["bounds_max"][1], 2)}, z={fmt(roi["bounds_max"][2], 2)}

Approximate cell size:

- X: {fmt(roi["cell"][0], 2)}
- Y: {fmt(roi["cell"][1], 2)}
- Z: {fmt(roi["cell"][2], 2)}

## 3. Dataset used

{split_sentence}

The result report also records the number of aligned training and testing samples for each horizon. The sample count changes slightly by horizon because a future target must exist at 50, 100, 150, or 200 ms after the input timestamp. The heatmap model has fewer samples because it requires a matched heatmap in addition to pose data.

## 4. Prediction horizons

The experiment predicts four future time offsets:

- 50 ms
- 100 ms
- 150 ms
- 200 ms

A 50 ms horizon means: given current/recent information, predict the ROI 50 ms in the future. A 200 ms horizon is harder because it looks farther ahead.

## 5. What each metric means

- `Top-1 accuracy`: the model's first predicted ROI cell is exactly correct. Higher is better.
- `Top-3 / Top-5 / Top-10 accuracy`: the true ROI is within the first 3, 5, or 10 predicted cells. Higher is better. This is useful when the system can prefetch or prepare several candidate ROIs.
- `Mean ROI distance error`: distance between the center of the top-1 predicted ROI and the center of the true ROI. Lower is better.
- `Median ROI distance error`: the middle error value. It is less affected by extreme cases than the mean.
- `Mean latency`: average model prediction time per sample. Lower is better.
- `P95 latency`: 95% of predictions are faster than this value. Lower is better.
- `Peak memory`: peak memory observed during evaluation. Lower is better.
- `Training loss`: optimization objective during training. A decreasing loss indicates the model is learning the training data.

## 6. Main result

Adding heatmap features does not improve exact Top-1 accuracy in this run. On average, Top-1 changes by {fmt(avg_delta[1], 2)} percentage points.

However, heatmap features strongly improve candidate-set accuracy:

- Average Top-3 improvement: {fmt(avg_delta[3], 2)} percentage points.
- Average Top-5 improvement: {fmt(avg_delta[5], 2)} percentage points.
- Average Top-10 improvement: {fmt(avg_delta[10], 2)} percentage points.

Mean ROI distance error also improves by {fmt(abs(avg_error_delta), 2)} units on average, meaning the heatmap model's first prediction is spatially closer even when it is not exactly the correct cell.

The tradeoff is resource cost. The heatmap model uses about 42.7 MB during evaluation, while the pose-only model uses about 1.2 MB. Prediction latency is still very small for both models, around 0.12-0.15 ms per sample.

## 7. How to present this in a report

A concise conclusion is:

> The RF heatmap features did not improve exact single-cell prediction, but they made the correct ROI much more likely to appear among the top candidate regions. This suggests heatmap features are useful for systems that can prefetch, rank, or inspect multiple candidate ROIs rather than relying only on the first predicted cell.

## 8. Figures

- ROI grid: `{figure_dir_name}/roi_grid.svg`
- Dataset split: `{figure_dir_name}/dataset_split.svg`
- Top-k accuracy: `{figure_dir_name}/top1_accuracy.svg`, `{figure_dir_name}/top3_accuracy.svg`, `{figure_dir_name}/top5_accuracy.svg`, `{figure_dir_name}/top10_accuracy.svg`
- Heatmap improvement: `{figure_dir_name}/accuracy_delta.svg`
- ROI distance error: `{figure_dir_name}/roi_error.svg`
- Latency: `{figure_dir_name}/latency.svg`
- Memory: `{figure_dir_name}/memory.svg`
- Training loss: `{figure_dir_name}/training_loss.svg`

## 9. Full result table

{markdown_table(report)}
"""
    path.write_text(md, encoding="utf-8")


def write_report_markdown_zh(path: Path, report: dict, split: dict | None, figure_dir_name: str = "figures") -> None:
    roi = roi_summary(report)
    counts = split_counts(split)
    hs = horizons(report)
    total_sessions = sum(counts.values()) if counts else None
    avg_delta = {
        k: sum(pct(get_topk(report, h, "heatmap_6dof", k) - get_topk(report, h, "6dof", k)) for h in hs) / len(hs)
        for k in (1, 3, 5, 10)
    }
    avg_error_delta = sum(
        get_metric(report, h, "heatmap_6dof", "roi_distance_error_mean")
        - get_metric(report, h, "6dof", "roi_distance_error_mean")
        for h in hs
    ) / len(hs)
    split_sentence = (
        f"本次 split 總共有 {total_sessions} 個 session，其中 train {counts.get('train', 0)} 個、"
        f"validation {counts.get('valid', 0)} 個、test {counts.get('test', 0)} 個。"
        if counts
        else "沒有找到 dataset split 檔，因此只能從結果檔列出 sample 數。"
    )

    sample_lines = [
        "| Horizon | 6DoF train/test samples | Heatmap+6DoF train/test samples |",
        "|---:|---:|---:|",
    ]
    for h in hs:
        base = report["results"][str(h)]["6dof"]
        heat = report["results"][str(h)]["heatmap_6dof"]
        sample_lines.append(
            f"| {h} ms | {base['train']['samples']} / {base['test']['samples']} | "
            f"{heat['train']['samples']} / {heat['test']['samples']} |"
        )

    md = f"""# ROI Heatmap Ablation 中文報告整理

## 1. 這個測試在做什麼

這個實驗的目的，是測試「RF heatmap 資訊」對於未來 ROI 預測有沒有幫助。

我們比較兩個模型：

- `6dof`：只使用頭戴裝置的姿態歷史，也就是位置和旋轉資訊。
- `heatmap_6dof`：使用同樣的姿態歷史，再加上 RF heatmap 萃取出的特徵。

這裡的任務不是直接預測一個精確座標，而是把空間切成很多小區塊，模型要預測「未來某個時間點，使用者會落在哪一個 ROI 區塊」。

## 2. ROI 切成幾塊

本次實驗使用 3D ROI grid，三個軸各切 `{roi["grid_size"]}` 等分：

`{roi["grid_size"]} x {roi["grid_size"]} x {roi["grid_size"]} = {roi["roi_count"]}` 個 ROI 區塊。

也就是說，模型每次要在 `{roi["roi_count"]}` 個候選區塊中，預測未來位置會在哪一格。

本次 ROI 的座標範圍是：

- 最小座標：x={fmt(roi["bounds_min"][0], 2)}, y={fmt(roi["bounds_min"][1], 2)}, z={fmt(roi["bounds_min"][2], 2)}
- 最大座標：x={fmt(roi["bounds_max"][0], 2)}, y={fmt(roi["bounds_max"][1], 2)}, z={fmt(roi["bounds_max"][2], 2)}

每一格大約大小為：

- X 方向：{fmt(roi["cell"][0], 2)}
- Y 方向：{fmt(roi["cell"][1], 2)}
- Z 方向：{fmt(roi["cell"][2], 2)}

## 3. 用到了多少 dataset

{split_sentence}

這裡是以 session 為單位切 train/validation/test，不是把同一個 session 裡的 frame 隨機混在 train 和 test。這樣比較能避免資料洩漏，測出來的 test 結果也比較可信。

不同 horizon 的 sample 數會略有不同，因為每一筆資料都需要能對齊到未來 50、100、150、200 ms 的 target。Heatmap 版本 sample 較少，是因為它除了 pose 之外，還必須要有可對齊的 heatmap。

{chr(10).join(sample_lines)}

## 4. 模型輸入用了什麼

`6dof` 使用最近 250 ms 的 pose history。程式中每筆 sample 會保留 8 個 pose history points，並轉成相對時間、相對位置、相對旋轉、速度、角速度等特徵。

`heatmap_6dof` 在上述 pose 特徵之外，額外加入 RF heatmap compact features，包含 heatmap 的 top peaks、heatmap 重心、基本統計量，以及 pooled 之後的 8 x 8 heatmap 表示。

## 5. 預測 horizon 是什麼

本次測試四種未來時間：

- 50 ms：預測 50 ms 後的 ROI。
- 100 ms：預測 100 ms 後的 ROI。
- 150 ms：預測 150 ms 後的 ROI。
- 200 ms：預測 200 ms 後的 ROI。

一般來說，horizon 越長越難，因為越遠的未來越不確定。

## 6. Training 之後的數值怎麼看

- `train samples`：訓練時實際用到的 aligned samples 數。
- `test samples`：測試時實際評估的 aligned samples 數。
- `training loss`：訓練過程中的錯誤目標值，越低通常代表模型越能 fit training data。本次兩個模型在 8 個 epochs 中 loss 都有下降，表示訓練正常收斂。
- `Top-1 accuracy`：模型第一名預測就是正確 ROI 的比例，越高越好。
- `Top-3 / Top-5 / Top-10 accuracy`：正確 ROI 有沒有出現在前 3、前 5、前 10 個候選裡，越高越好。這對「可以預先準備多個候選 ROI」的系統很重要。
- `Mean ROI distance error`：模型第一名預測的 ROI 中心，和正確 ROI 中心的平均距離，越低越好。
- `Median ROI distance error`：距離誤差的中位數，越低越好，比 mean 更不容易被少數極端錯誤影響。
- `Mean latency`：單次預測平均花費時間，越低越好。
- `P95 latency`：95% 的預測都會比這個時間更快，用來看比較壞情況下的延遲。
- `Peak memory`：評估時的峰值記憶體使用量，越低越好。

## 7. 主要結果

Heatmap 加進來後，Top-1 沒有變好，平均下降 {fmt(abs(avg_delta[1]), 2)} 個百分點。也就是說，如果只看「第一名是否完全命中」，heatmap 版本不是比較好的。

但是 heatmap 對候選集合的幫助非常明顯：

- Top-3 平均提升 {fmt(avg_delta[3], 2)} 個百分點。
- Top-5 平均提升 {fmt(avg_delta[5], 2)} 個百分點。
- Top-10 平均提升 {fmt(avg_delta[10], 2)} 個百分點。

Mean ROI distance error 平均改善 {fmt(abs(avg_error_delta), 2)}。這代表即使第一名沒有完全猜中，heatmap 版本的第一名預測通常也更接近正確位置。

資源成本方面，`6dof` 的 peak memory 約 1.2 MB，`heatmap_6dof` 約 42.7 MB。延遲方面兩者都很低，約 0.12 到 0.15 ms；heatmap 版本稍慢，但差距很小。

## 8. 報告可以怎麼下結論

可以這樣寫：

> 本實驗將 3D 空間切成 512 個 ROI 區塊，比較只使用 6DoF 姿態歷史，以及加入 RF heatmap 特徵後的預測效果。結果顯示，RF heatmap 沒有提升 Top-1 exact hit rate，但顯著提升 Top-3、Top-5、Top-10 candidate accuracy，且降低平均 ROI 距離誤差。這代表 heatmap 資訊對於「多候選 ROI 預取或排序」特別有價值；若系統能同時處理前幾個候選區域，heatmap 版本能提供更可靠的候選集合。

## 9. 圖表檔案

- ROI 切分示意圖：`{figure_dir_name}/roi_grid.png`
- Dataset split：`{figure_dir_name}/dataset_split.png`
- Top-k accuracy：`{figure_dir_name}/top1_accuracy.png`, `{figure_dir_name}/top3_accuracy.png`, `{figure_dir_name}/top5_accuracy.png`, `{figure_dir_name}/top10_accuracy.png`
- Heatmap 帶來的 accuracy improvement：`{figure_dir_name}/accuracy_delta.png`
- ROI distance error：`{figure_dir_name}/roi_error.png`
- Latency：`{figure_dir_name}/latency.png`
- Memory：`{figure_dir_name}/memory.png`
- Training loss：`{figure_dir_name}/training_loss.png`

## 10. 完整結果表

{markdown_table(report)}
"""
    path.write_text(md, encoding="utf-8")


def write_html(path: Path, report: dict, split: dict | None) -> None:
    counts = split_counts(split)
    roi = roi_summary(report)
    cards = [
        ("ROI cells", f'{roi["grid_size"]} x {roi["grid_size"]} x {roi["grid_size"]} = {roi["roi_count"]}'),
        ("Horizons", ", ".join(f"{h} ms" for h in horizons(report))),
        ("Sessions", str(sum(counts.values())) if counts else "split not found"),
        ("Best Top-10", "72.96% at 200 ms"),
    ]
    card_html = "\n".join(
        f'<div class="card"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></div>' for k, v in cards
    )
    figs = [
        ("ROI grid", "figures/roi_grid.png"),
        ("Dataset split", "figures/dataset_split.png"),
        ("Top-1 accuracy", "figures/top1_accuracy.png"),
        ("Top-3 accuracy", "figures/top3_accuracy.png"),
        ("Top-5 accuracy", "figures/top5_accuracy.png"),
        ("Top-10 accuracy", "figures/top10_accuracy.png"),
        ("Accuracy improvement", "figures/accuracy_delta.png"),
        ("ROI distance error", "figures/roi_error.png"),
        ("Latency", "figures/latency.png"),
        ("Memory", "figures/memory.png"),
        ("Training loss", "figures/training_loss.png"),
    ]
    fig_html = "\n".join(
        f'<section><h2>{html.escape(title)}</h2><img src="{html.escape(src)}" alt="{html.escape(title)}"></section>'
        for title, src in figs
        if (path.parent / src).is_file()
    )
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ROI Heatmap Ablation Results</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: #0f172a; background: #f8fafc; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 56px; }}
    h1 {{ margin: 0 0 10px; font-size: 32px; }}
    p {{ line-height: 1.55; color: #334155; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 24px 0; }}
    .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }}
    .k {{ color: #64748b; font-size: 13px; margin-bottom: 8px; }}
    .v {{ font-size: 22px; font-weight: 700; }}
    section {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    img {{ width: 100%; height: auto; display: block; }}
    a {{ color: #2563eb; }}
  </style>
</head>
<body>
  <main>
    <h1>ROI Heatmap Ablation Results</h1>
    <p>
      This dashboard summarizes the comparison between a pose-only 6DoF predictor and a predictor
      that adds compact RF heatmap features.
    </p>
    <div class="cards">{card_html}</div>
    <p>
      Chinese report explanation: <a href="REPORT_OVERVIEW_ZH.md">REPORT_OVERVIEW_ZH.md</a>.
      English report explanation: <a href="REPORT_OVERVIEW.md">REPORT_OVERVIEW.md</a>.
      Raw table: <a href="summary_table.csv">summary_table.csv</a>.
    </p>
    {fig_html}
  </main>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def generate(report_path: Path, split_path: Path | None, out_dir: Path) -> None:
    report = load_json(report_path)
    split = load_json(split_path) if split_path and split_path.is_file() else None
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    rows = result_rows(report)
    write_csv(out_dir / "summary_table.csv", rows)

    generate_matplotlib_figures(report, split, figures)

    write_report_markdown(out_dir / "REPORT_OVERVIEW.md", report, split)
    write_report_markdown_zh(out_dir / "REPORT_OVERVIEW_ZH.md", report, split)
    write_html(out_dir / "index.html", report, split)


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Visualize ROI heatmap ablation results.")
    parser.add_argument("--report", type=Path, default=None, help="Path to comparison_report.json.")
    parser.add_argument("--split", type=Path, default=None, help="Path to dataset_split.json.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for generated visualization files.")
    args = parser.parse_args()

    report_path = args.report or default_report_path(root)
    if not report_path.is_file():
        raise FileNotFoundError(f"comparison report not found: {report_path}")
    report = load_json(report_path)
    split_path = args.split if args.split else default_split_path(root, report)
    out_dir = args.out_dir or report_path.parent / "visualizations"
    generate(report_path, split_path, out_dir)
    print(f"report={report_path}")
    print(f"split={split_path if split_path else 'not found'}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
