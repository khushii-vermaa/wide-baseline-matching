# Does the matcher with the most matches also recover the camera pose?
**SIFT+MNN vs SuperPoint+LightGlue vs LoFTR on wide-baseline MegaDepth pairs — UCS532 (P20)**

A complete, reproducible benchmark harness + a web app (FastAPI backend, single-page frontend) + Docker deployment.
All numbers in your report must come from experiments *you* run; this repo never ships results.

```
configs/config.yaml     all settings (frozen -> configs/frozen.yaml after dev tuning)
src/wbm/
  data.py               MegaDepth-1500 loader (+ --inspect sanity check)
  select_pairs.py       N1: 100 test / 200 dev / 3 tiers chosen by GT rotation ONLY
  matchers.py           SIFT+MNN, SIFT+ratio, SuperPoint+LightGlue, LoFTR (as released)
  pose.py, geometry.py  one identical RANSAC pose stage; pose error, AUC, epipolar maths
  run.py                run a matcher on a split, caches per-pair results
  sweep.py              dev tuning + freeze (N5); sensitivity arms on test (O6)
  controls.py           negative controls on dev pairs (N6)
  analyze.py            Table 1, bootstrap CIs, Arms A-D, Figs 2-5, summary.json
  qualitative.py        Fig 6, hardest pairs by pre-declared rule
  rotate.py, rotation_sweep.py   O7 stretch (in-plane rotation, GT pose updated)
  synth.py              FAKE data generator to try the pipeline/UI early
api/main.py             backend: /api/match, /api/matchers, /api/results
frontend/index.html     UI: live demo tab + benchmark results tab
tests/test_geometry.py  metric unit tests, random-match / symmetry / rotation-convention checks
run_all.sh              regenerates everything
Dockerfile              deployment image (port 7860)
```

## 1. Set up (local, Python 3.10–3.11 recommended)
```bash
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
# PyTorch first. CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# (NVIDIA GPU: pick the command for your CUDA version at pytorch.org instead)
pip install -r requirements.txt
pip install -e . --no-deps
python -m pytest -q tests          # all tests should pass before anything else
```
Needs `git` installed (LightGlue installs from GitHub). First use of LoFTR/LightGlue downloads weights (internet required once).

## 2. Try everything on FAKE data first (2 minutes, no dataset needed)
```bash
python -m wbm.synth --out results_demo
python -m wbm.analyze --config configs/config.yaml --results-root results_demo \
       --pairs results_demo/pairs.json --synthetic
RESULTS_ROOT=results_demo uvicorn api.main:app --port 8000      # open http://localhost:8000 -> "Benchmark results"
```
The page shows a "synthetic" banner. Never report these numbers.

## 3. Get MegaDepth-1500 (Phase 0)
1. From the LoFTR repository (`zju3dv/LoFTR`): take the folder `assets/megadepth_test_1500_scene_info/` (2 `.npz` files) and follow its README / `docs/TRAINING.md` to download the MegaDepth **test** images.
2. Arrange so that `config.yaml` points at them:
```
data/megadepth_test_1500_scene_info/*.npz
data/megadepth/...   (the folder containing Undistorted_SfM/<scene>/images/*.jpg, possibly under test/)
```
3. Verify — this is your Phase 0 audit:
```bash
python -m wbm.data --inspect
```
It prints pair/scene counts, rotation range, whether images are found, and the pose convention assumed (**world-to-camera**). If images are not found, only change `data_root` in `configs/config.yaml`. The loader also falls back to searching by file name.

## 4. Pair selection (commit the output before any matcher runs)
```bash
python -m wbm.select_pairs          # writes pairs/pairs.json + results/figures/fig1_pair_selection.png
git add pairs/pairs.json && git commit -m "Frozen pair lists"
```

## 5. Controls, tuning, freeze (Phases 1–2)
```bash
python -m wbm.controls --matchers sift_mnn splg loftr    # must print CONTROLS PASSED
python -m wbm.sweep --split dev --freeze                 # tunes on 200 dev pairs -> configs/frozen.yaml
git add configs/frozen.yaml && git commit -m "Freeze protocol"
```
If the reproduction check (below) is off, suspect resize / keypoints / RANSAC threshold / pose convention before the code.

## 6. Experiments (Phases 2–4)
```bash
bash run_all.sh        # everything, in order; or run the steps individually
```
Outputs: `results/summary.json`, `results/tables/*.csv` (Table 1, paired differences, tiers, sweeps), `results/figures/*.png` (Figs 1–6).
Fill `published_auc` in the config from the LoFTR / LightGlue papers and compare with your 1,500-pair reproduction (Table 2) — within a tolerance you state.

GPU time: minutes per matcher for 100 pairs; the 1,500-pair reproduction is the longest step. Free Colab works:
```python
!git clone <your-repo-url> && %cd <repo>
!pip install -q torch torchvision && pip install -q -r requirements.txt && pip install -q -e . --no-deps
# mount Drive with the data, symlink it into data/, then:
!bash run_all.sh
```

## 7. Web app (backend + frontend)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```
* **Try a pair**: upload two photos of the same scene, choose a matcher, see matches (green = consistent with estimated geometry, red = not), inlier ratio and the estimated camera rotation. Without a focal length it guesses 1.2 × long side, so the pose is approximate — the benchmark uses true MegaDepth intrinsics.
* **Benchmark results**: reads `results/summary.json`: Table 1 with confidence intervals, count-ranking vs pose-ranking, which paired differences are real, all figures.
* API docs: `http://localhost:8000/docs`.

## 8. Deployment
**Docker (any host):**
```bash
docker build -t wbm-demo .
docker run -p 7860:7860 -v "$(pwd)/results:/app/results" wbm-demo      # http://localhost:7860
```
**Hugging Face Spaces (free, easiest for a viva demo):**
1. New Space → SDK **Docker** (blank). Clone it, copy this project in, put the contents of `deploy/README_hf_space_header.md` at the very top of the Space's `README.md`.
2. Commit `results/summary.json` and `results/figures/` (no `matches/`). `git push`. The Space builds and serves on port 7860.
3. Free Spaces are CPU-only: keep resize ≈ 640 in the demo; LoFTR is slow on CPU but works.

**Render / Railway:** create a Web Service from the repo, runtime Docker; set env `PORT` if required. Use at least 2 GB RAM (PyTorch + models).

## 9. Honest limits
* The numeric core (pose error, AUC, bootstrap, analysis, SIFT path, in-plane rotation convention) is unit-tested. The SuperPoint+LightGlue and LoFTR wrappers and the real-data loader depend on your environment and the dataset layout: confirm them with `--inspect`, the controls, and the reproduction gate.
* AUC uses the trapezoid convention of the LoFTR/LightGlue evaluations (see `geometry.error_auc`).
* The published-protocol numbers in `config.yaml` are starting points to verify against the papers.
* 100 pairs from a few scenes: use the paired intervals; do not over-read small differences. Declare AI assistance as the challenge requires.
