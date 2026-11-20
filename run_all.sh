#!/usr/bin/env bash
# Regenerates every table and figure from a clean checkout (proposal Gate 4).
# Prerequisite: MegaDepth-1500 data in place (README step 3).
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"

echo "== Phase 0: inspect data, select pairs (commit pairs/pairs.json now) =="
python -m wbm.data --inspect
python -m wbm.select_pairs

echo "== Phase 1: unit tests + negative controls =="
python -m pytest -q tests
python -m wbm.controls --config configs/config.yaml --matchers sift_mnn splg loftr

echo "== Tune on the 200 DEV pairs and freeze =="
python -m wbm.sweep --config configs/config.yaml --split dev --freeze      # writes configs/frozen.yaml

echo "== Phase 2: reproduction on all 1,500 pairs (published protocol) =="
python -m wbm.run --config configs/frozen.yaml --protocol published --split full --matchers loftr splg

echo "== Phase 3: main experiment on the 100 TEST pairs =="
python -m wbm.run --config configs/frozen.yaml --protocol main --split test --matchers sift_mnn splg loftr
python -m wbm.run --config configs/frozen.yaml --protocol published --split test --matchers sift_mnn splg loftr sift_ratio
python -m wbm.run --config configs/frozen.yaml --protocol main --split test --matchers sift_ratio
python -m wbm.sweep --config configs/frozen.yaml --split test --matchers sift_mnn splg loftr    # sensitivity (O6)
python -m wbm.analyze --config configs/frozen.yaml
python -m wbm.qualitative --config configs/frozen.yaml

echo "== Phase 4 (stretch, drop first if short on time): in-plane rotation sweep =="
python -m wbm.rotation_sweep --config configs/frozen.yaml || true

echo "Done. Open results/summary.json, results/tables/, results/figures/ or start the web app (README step 7)."
