# FEATURE MATCHING UNDER EXTREME VIEWPOINT CHANGE 
# Does the matcher with the most matches also recover the camera pose? 

SIFT+MNN vs SuperPoint+LightGlue vs LoFTR on wide-baseline MegaDepth pairs. UCS532 Computer Vision, Sessional Open Challenge 2026 (Project P20).
Authors: Khushi Verma (1024170065), Kashish Kannojia (1024170342).

## Problem
Papers and demos usually report match counts or inlier counts, which do not say whether the camera geometry was recovered. We compare three matchers by relative camera pose (pose AUC@5/10/20 degrees) on the 100 MegaDepth-1500 pairs with the largest ground-truth rotation (28.6 to 58.0 degrees), selected from geometry alone before any matcher ran. All matchers share one resize, keypoint cap and RANSAC stage, tuned on 200 separate dev pairs and frozen.

## Main result (100 test pairs, mean over RANSAC seeds 0, 1, 2)

| Matcher | AUC@5 | AUC@10 | AUC@20 | Matches (mean) |
|---|---|---|---|---|
| SIFT+MNN | 0.037 | 0.069 | 0.123 | 458 |
| SuperPoint+LightGlue | 0.484 | 0.653 | 0.778 | 364 |
| LoFTR | 0.650 | 0.787 | 0.878 | 1334 |

Source: `results/tables/table1_main.csv` (95% paired-bootstrap intervals are in the same file and in the report). Equal protocol: resize 1024 px, 2048 keypoints, RANSAC threshold 0.5 px. The ranking by pose (LoFTR > SP+LG > SIFT+MNN) differs from the ranking by match count (LoFTR > SIFT+MNN > SP+LG). Every paired difference in AUC@5 and AUC@10 has a 95% interval that excludes zero.

Harness check on all 1,500 pairs (Colab T4), our AUC@5/10/20 against the published values recorded in `configs/config.yaml`:

| Matcher | Ours | Published | Result |
|---|---|---|---|
| LoFTR | 51.2 / 68.0 / 80.3 | 52.8 / 69.2 / 81.2 | within the +-3 point tolerance |
| SuperPoint+LightGlue | 42.9 / 61.8 / 76.7 | 49.9 / 67.0 / 80.1 | 3.4 to 7.0 points lower; not reproduced within tolerance, reported as a limitation |

Details, negative controls, ablation, failure analysis and the changes from the proposal are in `report/main.pdf`.

## Data
- Pair lists: MegaDepth-1500 test scene info from the LoFTR repository (https://github.com/zju3dv/LoFTR, folder `assets/megadepth_test_1500_scene_info/`).
- Images: MegaDepth (Li and Snavely, CVPR 2018) test scenes, downloaded by following the data instructions in the LoFTR repository.
- Subset used: all 1,500 pairs for the harness check; 100 test pairs (largest rotation) and 200 dev pairs, listed in `pairs/pairs.json`. The pairs come from two landmarks (0015 and 0022).
- The data is not in this repository. Put it under `data/` as shown below, then run `python -m wbm.data --inspect` to check it.
```
data/megadepth_test_1500_scene_info/*.npz
data/megadepth/...   (the folder containing Undistorted_SfM/<scene>/images/*.jpg)
```
If images are not found, change only `data_root` in `configs/config.yaml`.

## Reproduce
Setup (Python 3.12 on the Windows laptop, 3.13 on Colab):
```bash
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
python -m pytest -q tests      # 9 tests should pass
```
On a GPU machine such as Colab, skip the torch line (Colab already has torch). Colab prints dependency-conflict warnings about its own preinstalled packages; they do not affect the run. Needs `git` (LightGlue installs from GitHub). LoFTR and LightGlue download weights on first use (internet needed once).

Two ways to run:
- **Fast path:** `bash run_all.sh --tables-only` rebuilds every table, figure and report fragment from the committed per-pair results. Measured: 17 seconds on the laptop. The regenerated tables are identical to the committed ones.
- **Full run:** `bash run_all.sh` runs everything from the raw data (needs `data/`). It uses the committed `configs/frozen.yaml`; set `RETUNE=1` to redo the dev tuning. The full run was not timed end to end. Mean matching time per pair on the laptop CPU (Intel Core i5-1340P), from `mean_time_s` in `results/tables/table1_main.csv`: SIFT+MNN 0.2 s, SuperPoint+LightGlue 8.2 s, LoFTR 10.6 s. The 1,500-pair reproduction is the longest step and was run on a Colab Tesla T4.

Seeds 0, 1 and 2 are fixed and printed at the start of `logs/run_all.log`. Every number in the report traces to a committed file listed in `results/TRACE.md`.

Main outputs: `results/tables/*.csv`, `results/summary.json`, `results/figures/*.png`, `logs/`.

## Environment used
| Item | Laptop (main experiment) | Colab (reproduction) |
|---|---|---|
| Python | 3.12 | 3.13 |
| PyTorch | 2.15.0 (CPU) | 2.14.1 (CUDA 13.0) |
| Kornia | 0.9.2 | 0.9.2 |
| LightGlue | commit 5a650c4 | GitHub master (commit not recorded) |
| OpenCV | 5.1.0 | 5.0.0.93 |
| NumPy | 2.5.4 | 2.4.6 |
| Hardware | Intel Core i5-1340P, CPU | Tesla T4 GPU |

## Repository layout
```
configs/config.yaml     all settings (frozen to configs/frozen.yaml after dev tuning)
pairs/pairs.json        frozen test/dev pair lists, chosen by ground-truth rotation only
src/wbm/
  data.py               MegaDepth-1500 loader (+ --inspect sanity check)
  select_pairs.py       100 test / 200 dev / 3 rotation tiers
  matchers.py           SIFT+MNN, SIFT+ratio, SuperPoint+LightGlue, LoFTR (as released)
  pose.py, geometry.py  one RANSAC pose stage; pose error, AUC, epipolar maths
  run.py                run a matcher on a split, caches per-pair results
  sweep.py              dev tuning + freeze; sensitivity arms on test
  controls.py           negative controls on dev pairs
  analyze.py            Table 1, bootstrap CIs, analysis arms, figures, summary.json
  qualitative.py        Fig. 6 panels by a pre-declared rule
  report.py             generates the report's table fragments
api/main.py, frontend/  web demo (FastAPI + single page)
tests/                  metric unit tests and negative-control checks
run_all.sh              regenerates everything
report/                 main.tex, main.pdf, figures, generated tables
results/, logs/         per-pair results, tables, figures, logs
```

## Demo
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000      # open http://localhost:8000
```
Upload two photos, choose a matcher, and see matches (green = consistent with the estimated geometry, red = not), the inlier ratio and the estimated rotation. Without a focal length it guesses 1.2 x long side, so the pose is approximate; the benchmark uses the true MegaDepth intrinsics. LoFTR and LightGlue need their weights cached first, so run the demo once with internet. The "Benchmark results" tab reads `results/summary.json`. The demo is not part of the reported results.

## Not run
- The in-plane rotation sweep (Arm F, stretch objective O7) was dropped first, as the proposal's drop order specifies. The code `wbm.rotation_sweep` exists, but no result is reported.
- The published-protocol settings were used only for the 1,500-pair reproduction, not as a sensitivity arm on the test pairs.
- The sensitivity arms did not vary the pose estimator.
- The Docker and cloud deployment files (`Dockerfile`, `deploy/`) are provided but were not tested and were not used for any reported result.

## Limitations
100 test pairs from two landmarks, a narrow rotation range (28.6 to 58.0 degrees), SfM ground truth, and OpenCV RANSAC rather than the estimators used in the published tables. The learned matchers were trained on MegaDepth data, which favours them over SIFT. See the report for the full list.

## Credits
- OpenCV: SIFT, RANSAC, five-point solver (Bradski, 2000).
- SuperPoint (DeTone et al., 2018) and LightGlue (Lindenberger et al., ICCV 2023), `cvg/LightGlue`, commit 5a650c4.
- LoFTR (Sun et al., CVPR 2021) through Kornia 0.9.2 (Riba et al., 2020).
- MegaDepth (Li and Snavely, CVPR 2018) and the MegaDepth-1500 pair lists from `zju3dv/LoFTR`.
- PyTorch, NumPy, SciPy, pandas, matplotlib, FastAPI.

All methods are used as released; none was proposed or retrained here. The contribution is the evaluation design: geometry-only pair selection, one shared protocol, paired statistics and negative controls.

## Contribution statement
Khushi Verma: geometry-only pair selection and pair lists, the evaluation harness (pose-error and AUC code with unit tests, RANSAC pose stage), the negative controls, the laptop CPU experiments on the 100 test and 200 dev pairs, the Colab 1,500-pair reproduction, and the repository and run scripts.
Kashish Kannojia: the three matcher wrappers (SIFT+MNN, SP+LG, LoFTR) and their pinned environments, the paired-bootstrap statistics, the analysis arms A-E (counts versus pose, tiers, landmarks, disagreement, sensitivity ablation), the figures and qualitative panels, and the web demo.
Both authors wrote and reviewed the report and checked every reported number against the logs.

## AI-assistance declaration
AI assistance (Claude, Anthropic) was used to help draft the code, the web demo and the report text, and to explain methods and debug errors. The authors ran every experiment, and every reported number comes from the authors' own runs.
