"""Collect the full Temporal BORA model and its ablations into one table.

Usage:
    python scripts/summarize_ablations.py [--config config/train_config.json] [--outputs outputs] [--out report.csv]

For the full model and every ablation preset, reads ``holdout/result.csv`` and
``cross_validation/summary_mean_std.csv`` under ``<outputs>/<experiment name>``
and prints each test metric with its difference from the full model. Missing
runs are skipped. Use scripts/compare_ablation.py for the paired McNemar test.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import ABLATIONS, TrainConfig, experiment_name

METRICS = (
    "test_accuracy",
    "test_f1_macro",
    "test_rank_mae",
    "test_qwk",
    "test_within_one_accuracy",
    "test_severe_error_rate",
)
VARIANTS = ("none", *ABLATIONS)
Result = Dict[str, Tuple[float, float]]


def _read_holdout(run_dir: Path) -> Optional[Result]:
    path = run_dir / "holdout" / "result.csv"
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    return {metric: (float(row[metric]), math.nan) for metric in METRICS}


def _read_cross_validation(run_dir: Path) -> Optional[Result]:
    path = run_dir / "cross_validation" / "summary_mean_std.csv"
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = {row["metric"]: row for row in csv.DictReader(file)}
    return {metric: (float(rows[metric]["mean"]), float(rows[metric]["std"])) for metric in METRICS}


READERS: Dict[str, Callable[[Path], Optional[Result]]] = {
    "holdout": _read_holdout,
    "cross_validation": _read_cross_validation,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config" / "train_config.json")
    parser.add_argument("--outputs", type=Path, default=None, help="Outputs root (default: the config's output_dir).")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV report path.")
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    runs: Dict[str, Path] = {}
    for variant in VARIANTS:
        cfg = TrainConfig.model_validate({**raw, "evaluation_mode": "holdout", "ablation": variant})
        outputs = args.outputs or Path(cfg.output_dir)
        if not outputs.is_absolute():
            outputs = PROJECT_DIR / outputs
        runs[variant] = outputs / experiment_name(cfg)

    report: List[Dict[str, object]] = []
    for mode, reader in READERS.items():
        results = {variant: result for variant, run_dir in runs.items() if (result := reader(run_dir)) is not None}
        if not results:
            print(f"\n== {mode}: no results found ==")
            continue
        full = results.get("none")
        print(f"\n== {mode} (delta vs full model in brackets) ==")
        print(f"{'variant':15s}" + "".join(f"{metric.removeprefix('test_'):>26s}" for metric in METRICS))
        for variant, result in results.items():
            cells = []
            for metric in METRICS:
                value, std = result[metric]
                delta = value - full[metric][0] if full is not None and variant != "none" else math.nan
                text = f"{value:.4f}" if math.isnan(std) else f"{value:.4f}+-{std:.4f}"
                if not math.isnan(delta):
                    text += f" ({delta:+.4f})"
                cells.append(f"{text:>26s}")
                report.append(
                    {
                        "mode": mode,
                        "variant": "full" if variant == "none" else variant,
                        "experiment": runs[variant].name,
                        "metric": metric,
                        "value": value,
                        "std": std,
                        "delta_vs_full": delta,
                    }
                )
            print(f"{'full' if variant == 'none' else variant:15s}" + "".join(cells))

    if args.out is not None and report:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(report[0]))
            writer.writeheader()
            writer.writerows(report)
        print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()
