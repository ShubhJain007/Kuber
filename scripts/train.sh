#!/usr/bin/env bash
# Train a Kuber baseline. Usage:
#   scripts/train.sh <data_dir> <splits.json> [geom_mode] [difficulty]
# geom_mode ∈ {none, sdf, dsdf, surface}   (default: surface)
# difficulty ∈ {easy, medium, hard}        (default: medium)
set -euo pipefail
DATA=${1:?data dir}; SPLITS=${2:?splits.json}
GEOM=${3:-surface}; DIFF=${4:-medium}

python -m kuber.train_simshift \
    --data "$DATA" --splits "$SPLITS" --difficulty "$DIFF" \
    --geom_mode "$GEOM" --out "outputs/${DIFF}_${GEOM}"
