from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd


# Locked three-view consensus configuration. Weights were selected on
# validation only and frozen before any test inference.
VIEW_WEIGHTS = {"original": 0.35, "center": 0.50, "quarter": 0.15}
VIEW_OFFSETS = {"original": None, "center": 0.5, "quarter": 0.25}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locked three-view temporal-consensus test evaluation over snapshot ensembles."
    )
    parser.add_argument(
        "--project-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="Directory containing scripts/evaluate_ordinal_ensemble.py.",
    )
    parser.add_argument("--config", type=pathlib.Path, required=True, help="holdout config_snapshot.json")
    parser.add_argument("--split-dir", type=pathlib.Path, required=True, help="directory with train/val/test.csv")
    parser.add_argument("--checkpoint", type=pathlib.Path, nargs="+", required=True, help="snapshot checkpoints")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=3,
        required=True,
        metavar=("NONE_WEAK", "WEAK_MEDIUM", "MEDIUM_STRONG"),
        help="Thresholds previously fitted on validation and locked before this test run.",
    )
    args = parser.parse_args()

    artifacts = args.output_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": "."}

    for name in VIEW_OFFSETS:
        out = artifacts / name
        cmd = [
            sys.executable,
            "scripts/evaluate_ordinal_ensemble.py",
            "--config", str(args.config),
            "--split-dir", str(args.split_dir),
            "--checkpoint", *[str(path) for path in args.checkpoint],
            "--output-dir", str(out),
            "--evaluate-test",
            "--fixed-thresholds", *[str(value) for value in args.thresholds],
        ]
        if VIEW_OFFSETS[name] is not None:
            cmd += ["--temporal-offset", str(VIEW_OFFSETS[name])]
        with (artifacts / f"{name}.log").open("w") as log:
            subprocess.run(cmd, cwd=args.project_root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    arrays: list[np.ndarray] = []
    labels: np.ndarray | None = None
    keys: np.ndarray | None = None
    for name, weight in VIEW_WEIGHTS.items():
        data = np.load(artifacts / name / "test_snapshot_probabilities.npz", allow_pickle=True)
        arrays.append(weight * data["probabilities"].mean(axis=0))
        if labels is None:
            labels = data["labels"]
            keys = data["sample_keys"]
        else:
            assert np.array_equal(labels, data["labels"]) and np.array_equal(keys, data["sample_keys"])
    probabilities = np.sum(arrays, axis=0)

    true_ranks = np.asarray([0, 3, 2, 1])[labels]
    raw_ranks = probabilities.argmax(axis=1)
    survival = np.stack(
        [
            probabilities[:, 1:].sum(axis=1),
            probabilities[:, 2:].sum(axis=1),
            probabilities[:, 3],
        ],
        axis=1,
    )
    above_0 = survival[:, 0] > args.thresholds[0]
    above_1 = above_0 & (survival[:, 1] > args.thresholds[1])
    above_2 = above_1 & (survival[:, 2] > args.thresholds[2])
    calibrated_ranks = above_0.astype(int) + above_1.astype(int) + above_2.astype(int)

    summary = {
        "protocol": "locked_from_validation_before_test",
        "snapshot_checkpoints": [str(path) for path in args.checkpoint],
        "view_weights": VIEW_WEIGHTS,
        "temporal_offsets": VIEW_OFFSETS,
        "thresholds": [float(value) for value in args.thresholds],
        "test_raw_correct": int((raw_ranks == true_ranks).sum()),
        "test_raw_accuracy": float((raw_ranks == true_ranks).mean()),
        "test_calibrated_correct": int((calibrated_ranks == true_ranks).sum()),
        "test_calibrated_accuracy": float((calibrated_ranks == true_ranks).mean()),
    }
    (artifacts / "locked_three_view_result.json").write_text(json.dumps(summary, indent=2))

    label_map = np.asarray([0, 3, 2, 1])
    pd.DataFrame(
        {
            "sample_key": keys,
            "true_label": labels,
            "true_rank": true_ranks,
            "raw_rank": raw_ranks,
            "raw_label": label_map[raw_ranks],
            "calibrated_rank": calibrated_ranks,
            "calibrated_label": label_map[calibrated_ranks],
            "p_none": probabilities[:, 0],
            "p_weak": probabilities[:, 1],
            "p_medium": probabilities[:, 2],
            "p_strong": probabilities[:, 3],
        }
    ).to_csv(artifacts / "locked_three_view_predictions.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
