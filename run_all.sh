#!/usr/bin/env bash
# Regenerates every table and figure. Prerequisite: MegaDepth-1500 in data/ (README step 3).
# Usage:  bash run_all.sh                full run (hours on CPU)
#         bash run_all.sh --tables-only  rebuild tables/figures/report from committed per-pair results
#         RETUNE=1 bash run_all.sh       also redo dev tuning (overwrites configs/frozen.yaml)
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
mkdir -p logs
exec > >(tee -a logs/run_all.log) 2>&1
echo "== run_all.sh $(date -u +%FT%TZ) | RANSAC seeds: 0 1 2 =="

if [ "${1:-}" != "--tables-only" ]; then
  echo "== Phase 0: inspect data, select pairs =="
  python -m wbm.data --inspect
  python -m wbm.select_pairs

  echo "== Phase 1: unit tests + negative controls =="
  python -m pytest -q tests
  python -m wbm.controls --config configs/config.yaml --matchers sift_mnn splg loftr

  if [ "${RETUNE:-0}" = "1" ]; then
    echo "== Tune on the 200 DEV pairs and freeze =="
    python -m wbm.sweep --config configs/config.yaml --split dev --freeze
  else
    echo "== Using committed configs/frozen.yaml (set RETUNE=1 to redo dev tuning) =="
  fi

  echo "== Reproduction on all 1,500 pairs (published protocol) =="
  python -m wbm.run --config configs/frozen.yaml --protocol published --split full --matchers loftr splg

  echo "== Main experiment on the 100 TEST pairs =="
  python -m wbm.run --config configs/frozen.yaml --protocol main --split test --matchers sift_mnn splg loftr sift_ratio
  python -m wbm.sweep --config configs/frozen.yaml --split test --matchers sift_mnn splg loftr
fi

echo "== Analysis, figures, report tables =="
python -m wbm.analyze --config configs/frozen.yaml
[ "${1:-}" = "--tables-only" ] || python -m wbm.qualitative --config configs/frozen.yaml
python -m wbm.report
(cd report && latexmk -pdf main.tex) || echo "latexmk missing: upload report/ to Overleaf"
echo "Done. Main table: results/tables/table1_main.csv"