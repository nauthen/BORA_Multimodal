"""Compare a Temporal BORA baseline run against a gate ablation run.

Usage:
    python scripts/compare_gate_ablation.py <baseline_holdout_dir> <ablation_holdout_dir> [--out report.csv]

Each directory must contain the trainer outputs ``result.csv`` and ``predictions.csv``.
Reports outcome metrics, per-boundary gate mechanics, and an exact McNemar test
on the paired test predictions.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.ordinal import BOUNDARY_NAMES

RESULT_KEYS = (
    "test_accuracy",
    "test_f1_macro",
    "test_rank_mae",
    "test_qwk",
    "test_within_one_accuracy",
    "test_severe_error_rate",
)
DEAD_GATE_BAND = (0.45, 0.55)


def _read_result(run_dir: Path) -> Dict[str, float]:
    with (run_dir / "result.csv").open("r", newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    return {key: float(row[key]) for key in RESULT_KEYS}


def _read_predictions(run_dir: Path) -> Dict[str, Dict[str, str]]:
    with (run_dir / "predictions.csv").open("r", newline="", encoding="utf-8") as file:
        return {row["sample_key"]: row for row in csv.DictReader(file)}


def _gate_mechanics(predictions: Dict[str, Dict[str, str]]) -> List[Dict[str, float]]:
    rows = list(predictions.values())
    correct = np.asarray([row["true_label"] == row["predicted_label"] for row in rows])
    mechanics = []
    for boundary in BOUNDARY_NAMES:
        video_gate = np.asarray([float(row[f"video_gate_{boundary}"]) for row in rows])
        audio_rel = np.asarray([float(row[f"audio_reliability_{boundary}"]) for row in rows])
        video_rel = np.asarray([float(row[f"video_reliability_{boundary}"]) for row in rows])
        low, high = DEAD_GATE_BAND
        separation = (
            abs(video_gate[correct].mean() - video_gate[~correct].mean())
            if correct.any() and (~correct).any()
            else float("nan")
        )
        mechanics.append(
            {
                "boundary": boundary,
                "video_gate_mean": float(video_gate.mean()),
                "video_gate_std": float(video_gate.std()),
                "dead_gate_fraction": float(((video_gate >= low) & (video_gate <= high)).mean()),
                "gate_separation": float(separation),
                "audio_reliability_mean": float(audio_rel.mean()),
                "audio_reliability_std": float(audio_rel.std()),
                "video_reliability_mean": float(video_rel.mean()),
                "video_reliability_std": float(video_rel.std()),
            }
        )
    return mechanics


def _mcnemar(baseline: Dict[str, Dict[str, str]], ablation: Dict[str, Dict[str, str]]) -> Dict[str, float]:
    if set(baseline) != set(ablation):
        raise ValueError("Runs were evaluated on different test samples; McNemar requires identical keys.")
    only_baseline = only_ablation = 0
    for key, row in baseline.items():
        base_ok = row["true_label"] == row["predicted_label"]
        abl_ok = ablation[key]["true_label"] == ablation[key]["predicted_label"]
        only_baseline += int(base_ok and not abl_ok)
        only_ablation += int(abl_ok and not base_ok)
    discordant = only_baseline + only_ablation
    p_value = binomtest(only_baseline, discordant, 0.5).pvalue if discordant else 1.0
    return {
        "only_baseline_correct": only_baseline,
        "only_ablation_correct": only_ablation,
        "p_value": float(p_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("ablation", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV report path.")
    args = parser.parse_args()

    runs = {"baseline": args.baseline, "ablation": args.ablation}
    results = {name: _read_result(path) for name, path in runs.items()}
    predictions = {name: _read_predictions(path) for name, path in runs.items()}
    mechanics = {name: _gate_mechanics(preds) for name, preds in predictions.items()}
    test = _mcnemar(predictions["baseline"], predictions["ablation"])

    print("\n== Outcome (test) ==")
    print(f"{'metric':28s}{'baseline':>12s}{'ablation':>12s}{'delta':>12s}")
    for key in RESULT_KEYS:
        base, abl = results["baseline"][key], results["ablation"][key]
        print(f"{key:28s}{base:12.5f}{abl:12.5f}{abl - base:+12.5f}")

    print("\n== Gate mechanics per boundary (video gate; separation = |mean correct - mean wrong|) ==")
    fields = [key for key in mechanics["baseline"][0] if key != "boundary"]
    for index, boundary in enumerate(BOUNDARY_NAMES):
        print(f"\n[{boundary}]")
        for field in fields:
            base = mechanics["baseline"][index][field]
            abl = mechanics["ablation"][index][field]
            print(f"  {field:26s}{base:12.5f}{abl:12.5f}")

    print("\n== Exact McNemar test on paired test predictions ==")
    print(
        f"  only baseline correct = {test['only_baseline_correct']}, "
        f"only ablation correct = {test['only_ablation_correct']}, p = {test['p_value']:.4g}"
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["section", "boundary", "metric", "baseline", "ablation"])
            for key in RESULT_KEYS:
                writer.writerow(["outcome", "", key, results["baseline"][key], results["ablation"][key]])
            for index, boundary in enumerate(BOUNDARY_NAMES):
                for field in fields:
                    writer.writerow(
                        [
                            "gate",
                            boundary,
                            field,
                            mechanics["baseline"][index][field],
                            mechanics["ablation"][index][field],
                        ]
                    )
            for key, value in test.items():
                writer.writerow(["mcnemar", "", key, value, ""])
        print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()
