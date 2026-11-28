# Traceability: every reported number and where it comes from

Regenerate all tables with `bash run_all.sh --tables-only` (17 s). Seeds 0, 1, 2 are printed at the top of `logs/run_all.log`.

| Report item | Source file |
|---|---|
| Table 1: AUC@5/10/20, 95% CIs, median error, success rate, matches, inlier ratio, epipolar precision | `results/tables/table1_main.csv` |
| Paired differences (SIFT-SP+LG, SIFT-LoFTR, SP+LG-LoFTR) | `results/tables/paired_differences.csv` |
| Seed spread of AUC@10 | `results/tables/per_seed.csv` |
| Tier table: AUC@10 by rotation tier, D_ab | `results/tables/tiers.csv` |
| Landmark rows (0015: 30 pairs, 0022: 70 pairs) | `results/tables/per_landmark.csv`, `results/summary.json` (`scenes`) |
| Landmark mix per tier (9/25, 10/23, 11/22) | `results/summary.json` (`tier_scene`) |
| Sensitivity table and the 36/36 ranking check | `results/tables/sweep_test.csv` |
| Dev tuning grid (resize 640/1024, 2048 keypoints, threshold 0.5/1/2) | `results/tables/sweep_dev.csv`, `logs/sweep_dev.txt` |
| SIFT ratio test vs MNN (0.069 vs 0.118) | `report/generated/table_ratio.tex` |
| Ratio-vs-MNN paired difference 0.048 [0.01, 0.09] | `logs/ratio_vs_mnn.txt` |
| Negative controls: random AUC@10 = 0, LoFTR 18/20 pairs, 100% agreement, 0.81 deg, five small-rotation errors, SIFT diagnostic | `logs/controls.txt` |
| First symmetry check (before redefinition) | Output not saved; the report gives no numbers for it |
| Harness reproduction on 1,500 pairs, ours (LoFTR 51.2/68.0/80.3; SP+LG 42.9/61.8/76.7) | `results/published/full/` per-pair JSONs, `report/generated/table_repro.tex` |
| Published values (LoFTR 52.8/69.2/81.2; SP+LG 49.9/67.0/80.1) | LoFTR (CVPR 2021) and LightGlue (ICCV 2023) papers; also `configs/config.yaml` |
| Match-count, inlier-count and precision rankings | `results/summary.json`, `results/tables/table1_main.csv` |
| Spearman 0.64 / 0.63 / 0.64 | `results/summary.json` (`inlier_ratio_vs_pose_spearman`) |
| Best-of-three oracle AUC@5/10/20 = 0.691 / 0.815 / 0.896 | `results/summary.json` (`best_of_three`) |
| Solved-by-k counts per tier (tier 3: 0/0/4 by exactly one, 3 by all) | `results/summary.json` (per-tier entries) |
| Rotation range 28.6-58.0 deg (test), overall 0.3-58.0 and median 10.5 | `logs/session_1.txt` (line 128), `pairs/pairs.json`, `results/figures/fig1_pair_selection.png` |
| Tier sizes 34/33/33 | `pairs/pairs.json` |
| Failure-panel counts and fractions (452/158/485; 692/319/609) | printed in the headers of `results/figures/fig6_*.png` |
| Environment table | `results/env_colab.txt` (Colab); README (laptop) |

Not from this repo: the published AUC values (papers) and the proposal date and commit (18 October; commit 66c2a11).
Not run: in-plane rotation sweep (Arm F); published-protocol arm on the test pairs.
