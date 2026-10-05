#!/usr/bin/env bash
# Optional: train the full Temporal BORA model and its ablations back to back.
# Each run is `python main.py` on a copy of the config with "ablation" and
# "evaluation_mode" set, exactly as if they were edited by hand.
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
base="${CONFIG:-config/train_config.json}"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

failed=()
for variant in "${variants[@]}"; do
  echo "=== ${mode} / ${variant} ==="
  config="${tmp_dir}/${variant}.json"
  python - "${base}" "${config}" "${mode}" "${variant}" <<'PY'
import json
import sys

base, output, mode, ablation = sys.argv[1:]
with open(base, encoding="utf-8") as file:
    raw = json.load(file)
raw["evaluation_mode"] = mode
raw["ablation"] = ablation
with open(output, "w", encoding="utf-8") as file:
    json.dump(raw, file, indent=2)
PY
  if ! python main.py --config "${config}"; then
    failed+=("${variant}")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "Failed variants (${mode}): ${failed[*]}" >&2
  exit 1
fi
echo "All ${mode} variants finished. Summary: python scripts/summarize_ablations.py"
