#!/usr/bin/env bash
# Train the full Temporal BORA model and its leave-one-component-out ablations
# one after another (docs/ablation_design.md).
#
# Usage: bash scripts/run_ablations.sh [holdout|cross_validation] [variant ...]
#   variants: none (full model), no_motion, no_confidence, ordinal_only, nominal_only
#   CONFIG=path/to/config.json overrides the base config (default config/train_config.json).
set -u
cd "$(dirname "$0")/.."

mode="${1:-holdout}"
if [ $# -gt 0 ]; then
  shift
fi
variants=("$@")
if [ ${#variants[@]} -eq 0 ]; then
  variants=(none no_motion no_confidence ordinal_only nominal_only)
fi

failed=()
for variant in "${variants[@]}"; do
  echo "=== ${mode} / ${variant} ==="
  if ! python main.py --config "${CONFIG:-config/train_config.json}" --evaluation-mode "${mode}" --ablation "${variant}"; then
    failed+=("${variant}")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "Failed variants (${mode}): ${failed[*]}" >&2
  exit 1
fi
echo "All ${mode} variants finished. Summary: python scripts/summarize_ablations.py"
