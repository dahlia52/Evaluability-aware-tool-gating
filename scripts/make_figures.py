#!/usr/bin/env python3
"""Reproduce paper Figure 2 from per-instance experiment outputs.

The x-axis is the no-intervention (B0) accuracy. The y-axis is the accuracy
change caused by removing rather than tagging BLOCKED tools under three-way
exposure control (M1b - M1a).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FAMILY = {
    "kanana-1.3b": "Kanana",
    "kanana-3b": "Kanana",
    "kanana-30b": "Kanana",
    "qwen35-2b": "Qwen",
    "qwen35-4b": "Qwen",
    "qwen35-9b": "Qwen",
}
LABEL = {
    "kanana-1.3b": "1.3B",
    "kanana-3b": "3B",
    "kanana-30b": "30B-A3B",
    "qwen35-2b": "2B",
    "qwen35-4b": "4B",
    "qwen35-9b": "9B",
}
MOE = {"kanana-30b"}


def load_accuracies(run_dirs: list[str]) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for directory in run_dirs:
        for path in sorted(Path(directory).glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError, KeyError):
                continue
            metrics = data["metrics"]
            key = (metrics["model"], metrics["condition"])
            for record in data.get("records", []):
                totals[key][0] += bool(record.get("valid"))
                totals[key][1] += 1
    return {key: passed / total for key, (passed, total) in totals.items() if total}


def average_ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = average_ranks(x), average_ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = (
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    ) ** 0.5
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=[str(ROOT / "out" / "runs")])
    parser.add_argument(
        "--out", default=str(ROOT / "out" / "figures" / "remove_vs_tag_accuracy.pdf")
    )
    args = parser.parse_args()

    accuracy = load_accuracies(args.runs)
    models = sorted(
        model
        for model in {model for model, _ in accuracy}
        if all((model, condition) in accuracy for condition in ("B0", "M1a", "M1b"))
    )
    if not models:
        raise SystemExit("No complete B0/M1a/M1b model results were found.")

    baseline = [accuracy[(model, "B0")] * 100 for model in models]
    delta = [
        (accuracy[(model, "M1b")] - accuracy[(model, "M1a")]) * 100
        for model in models
    ]
    rho = spearman(baseline, delta)

    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    styles = {"Kanana": ("o", "#2673b8"), "Qwen": ("s", "#d97706")}
    for family, (marker, color) in styles.items():
        selected = [i for i, model in enumerate(models) if FAMILY.get(model) == family]
        if selected:
            axis.scatter(
                [baseline[i] for i in selected],
                [delta[i] for i in selected],
                marker=marker,
                color=color,
                s=60,
                label=family,
                zorder=3,
            )

    for i, model in enumerate(models):
        if model in MOE:
            axis.scatter(
                baseline[i], delta[i], marker="D", facecolors="none",
                edgecolors="#333333", s=100, linewidths=1.4, label="MoE", zorder=4,
            )
        axis.annotate(
            LABEL.get(model, model), (baseline[i], delta[i]),
            xytext=(5, 5), textcoords="offset points", fontsize=9,
        )

    axis.axhline(0, color="#777777", linewidth=0.9)
    axis.grid(alpha=0.25)
    axis.set_xlabel("No-intervention accuracy (%)")
    axis.set_ylabel("Accuracy change: remove - tag (pp)")
    axis.text(0.03, 0.05, f"Spearman ρ = {rho:.3f}", transform=axis.transAxes)
    axis.legend(frameon=False)
    fig.tight_layout()

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    if destination.suffix.lower() == ".pdf":
        fig.savefig(destination.with_suffix(".png"), dpi=180)

    print(f"saved: {destination}")
    print(f"Spearman rho: {rho:.3f}")
    for model, x, y in zip(models, baseline, delta):
        print(f"  {model:<13} baseline={x:5.1f}%  remove-tag={y:+5.1f}pp")


if __name__ == "__main__":
    main()
