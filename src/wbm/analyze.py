"""Tables, figures and summary.json from cached results (Arms A-D, Table 1, Figs 2-5)."""
import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import load_config, ensure_dir
from .geometry import auc
from .matchers import LABELS
from .results_io import load_matcher
from .stats import bootstrap, ci, success_matrix, disagreement, auc_seeds

MAIN = ["sift_mnn", "splg", "loftr"]
SHORT = {"sift_mnn": "SIFT+MNN", "splg": "SP+LG", "loftr": "LoFTR"}
COL = {"sift_mnn": "#8a6d3b", "splg": "#1f7a8c", "loftr": "#c0392b"}


def rank(values, higher_better=True):
    order = sorted(values, key=lambda k: -values[k] if higher_better else values[k])
    return {k: order.index(k) + 1 for k in values}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/frozen.yaml")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--protocol", default="main")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = Path(a.results_root or cfg["results_root"])
    pairs = json.load(open(a.pairs or cfg["pairs_file"]))
    ids = pairs["test"]
    tier = np.array([pairs["tier"][i] for i in ids])
    ntier = int(tier.max())
    fig_dir, tab_dir = ensure_dir(root / "figures"), ensure_dir(root / "tables")

    D = {m: load_matcher(root, a.protocol, "test", m, ids) for m in MAIN}
    errs = {m: D[m]["err"] for m in MAIN}
    S = len(cfg["seeds"])

    # ---------- Table 1 (Arm B) with paired bootstrap CIs ----------
    boot = bootstrap(errs, (5, 10, 20), B=1000)
    rows = []
    for m in MAIN:
        d = D[m]
        med = np.median(d["err"], axis=1)
        ir = np.where(d["n_matches"][:, None] > 0, d["n_inliers"] / np.maximum(d["n_matches"][:, None], 1), 0).mean(1)
        r = dict(matcher=LABELS[m], key=m)
        for t in (5, 10, 20):
            lo, hi = ci(boot[(m, t)])
            r[f"AUC@{t}"] = auc_seeds(d["err"], t)
            r[f"AUC@{t}_lo"], r[f"AUC@{t}_hi"] = lo, hi
        seed_aucs = [auc(d["err"][:, s], 10) for s in range(S)]
        r.update(AUC10_seed_std=float(np.std(seed_aucs)), median_err=float(np.median(med)),
                 success_rate=float(np.mean(med < 10)), mean_matches=float(d["n_matches"].mean()),
                 mean_inliers=float(d["n_inliers"].mean()), inlier_ratio=float(ir.mean()),
                 epi_precision=float(d["epi"].mean()), mean_time_s=float(np.nanmean(d["time"])))
        rows.append(r)
    T1 = pd.DataFrame(rows)
    T1.to_csv(tab_dir / "table1_main.csv", index=False)

    # ---------- paired differences ----------
    diffs = []
    for x, y in itertools.combinations(MAIN, 2):
        for t in (5, 10):
            smp = boot[(x, t)] - boot[(y, t)]
            lo, hi = ci(smp)
            diffs.append(dict(a=LABELS[x], b=LABELS[y], tau=t, diff=float(np.mean(smp)), lo=lo, hi=hi,
                              significant=bool(lo > 0 or hi < 0)))
    pd.DataFrame(diffs).to_csv(tab_dir / "paired_differences.csv", index=False)

    # ---------- Arms A/B/C: counts vs pose ----------
    def rank_table(mask):
        v = {k: {"matches": D[k]["n_matches"][mask].mean(), "inliers": D[k]["n_inliers"][mask].mean(),
                 "epi": D[k]["epi"][mask].mean(), "auc10": auc_seeds(D[k]["err"][mask], 10)} for k in MAIN}
        rk = {c: rank({k: v[k][c] for k in MAIN}) for c in ("matches", "inliers", "epi", "auc10")}
        return {c: {"ranks": rk[c], "agrees_with_auc": rk[c] == rk["auc10"]} for c in ("matches", "inliers", "epi")} | {"auc_ranks": rk["auc10"]}
    ranks = {"overall": rank_table(np.ones(len(ids), bool))}
    for t in range(1, ntier + 1):
        ranks[f"tier{t}"] = rank_table(tier == t)
    corr = {}
    for m in MAIN:
        d = D[m]
        ir = (d["n_inliers"] / np.maximum(d["n_matches"][:, None], 1)).mean(1)
        e = np.log10(np.clip(np.median(d["err"], axis=1), 0.01, 1e3))
        corr[m] = float(spearmanr(ir, -e).statistic) if np.std(ir) > 0 else float("nan")

    # ---------- Arm D: tiers, disagreement, complementarity ----------
    succ = success_matrix(errs)
    tiers = []
    for t in range(1, ntier + 1):
        mk = tier == t
        row = dict(tier=t, n=int(mk.sum()), rot_min=float(np.min([pairs["rot"][i] for i, k in zip(ids, mk) if k])),
                   rot_max=float(np.max([pairs["rot"][i] for i, k in zip(ids, mk) if k])))
        tb = bootstrap({m: errs[m][mk] for m in MAIN}, (10,), B=500, seed=t)
        for m in MAIN:
            row[f"auc10_{m}"] = auc_seeds(errs[m][mk], 10)
            row[f"auc10_{m}_lo"], row[f"auc10_{m}_hi"] = ci(tb[(m, 10)])
        for x, y in itertools.combinations(MAIN, 2):
            row[f"D_{x}_{y}"] = disagreement(succ[x][mk], succ[y][mk])
        k = sum(succ[m][mk].astype(int) for m in MAIN)
        row.update(solved_by_none=int((k == 0).sum()), solved_by_one=int((k == 1).sum()),
                   solved_by_two=int((k == 2).sum()), solved_by_all=int((k == 3).sum()),
                   **{f"only_{m}": int((succ[m][mk] & (k == 1)).sum()) for m in MAIN})
        tiers.append(row)
    pd.DataFrame(tiers).to_csv(tab_dir / "tiers.csv", index=False)

    # ---------- per-landmark (section 7.5 ii) and per-seed (7.5 iii) ----------
    lm = np.array([pairs["scene"][i][:4] for i in ids])     # '0015_0.1_0.3' -> landmark '0015'
    scenes = []
    for sc in sorted(set(lm)):
        mk = lm == sc
        row = dict(scene=sc, n=int(mk.sum()))
        sb = bootstrap({m: errs[m][mk] for m in MAIN}, (5, 10), B=500, seed=11)
        for m in MAIN:
            for t in (5, 10):
                row[f"auc{t}_{m}"] = auc_seeds(errs[m][mk], t)
                row[f"auc{t}_{m}_lo"], row[f"auc{t}_{m}_hi"] = ci(sb[(m, t)])
        scenes.append(row)
    pd.DataFrame(scenes).to_csv(tab_dir / "per_landmark.csv", index=False)
    tier_scene = {int(t): {sc: int(((tier == t) & (lm == sc)).sum()) for sc in sorted(set(lm))} for t in range(1, ntier + 1)}
    per_seed = {m: {f"auc{t}": [auc(errs[m][:, k], t) for k in range(S)] for t in (5, 10)} for m in MAIN}
    pd.DataFrame([dict(matcher=m, seed=k, auc5=per_seed[m]["auc5"][k], auc10=per_seed[m]["auc10"][k]) for m in MAIN for k in range(S)]).to_csv(tab_dir / "per_seed.csv", index=False)
    best_err = np.min(np.stack([errs[m] for m in MAIN]), axis=0)
    best_of_three = {f"AUC@{t}": auc_seeds(best_err, t) for t in (5, 10, 20)}

    # ---------- Figures ----------
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.linspace(0, 20, 400)
    for m in MAIN:
        med = np.median(errs[m], axis=1)
        ax.plot(xs, [(med <= x).mean() for x in xs], color=COL[m], label=f"{LABELS[m]} (AUC@10 {auc_seeds(errs[m], 10):.2f})")
    for c in (5, 10):
        ax.axvline(c, color="grey", ls=":")
    ax.set_xlabel("pose error (deg)"); ax.set_ylabel("fraction of pairs <= error"); ax.legend(); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_pose_error_cdf.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for m in MAIN:
        y = [r[f"auc10_{m}"] for r in tiers]
        ax.plot(range(1, ntier + 1), y, "-o", color=COL[m], label=LABELS[m])
        ax.fill_between(range(1, ntier + 1), [r[f"auc10_{m}_lo"] for r in tiers], [r[f"auc10_{m}_hi"] for r in tiers], color=COL[m], alpha=.15)
    ax.set_xticks(range(1, ntier + 1)); ax.set_xticklabels([f"tier {r['tier']}\n{r['rot_min']:.0f}-{r['rot_max']:.0f} deg" for r in tiers])
    ax.set_ylabel("AUC@10"); ax.legend(); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(fig_dir / "fig3_auc_by_tier.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    cnt = {m: D[m]["n_matches"].mean() for m in MAIN}
    aucs = {m: auc_seeds(errs[m], 10) for m in MAIN}
    for ax, vals, title in [(axs[0], cnt, "mean match count"), (axs[1], aucs, "AUC@10")]:
        ax.bar([SHORT[m] for m in MAIN], [vals[m] for m in MAIN], color=[COL[m] for m in MAIN]); ax.set_title(title)
        ax.tick_params(axis="x", labelsize=9)
    for m in MAIN:
        d = D[m]
        ir = (d["n_inliers"] / np.maximum(d["n_matches"][:, None], 1)).mean(1)
        axs[2].scatter(ir, np.clip(np.median(d["err"], axis=1), 0.1, 180), s=10, alpha=.6, color=COL[m], label=LABELS[m])
    axs[2].set_yscale("log"); axs[2].set_xlabel("inlier ratio"); axs[2].set_ylabel("pose error (deg)"); axs[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_counts_vs_pose.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    cats = ["solved_by_none", "solved_by_one", "solved_by_two", "solved_by_all"]
    bottom = np.zeros(ntier)
    for c, colr in zip(cats, ["#b0b0b0", "#e0a31a", "#7fb7c4", "#1f7a8c"]):
        v = np.array([r[c] for r in tiers]); axs[0].bar(range(1, ntier + 1), v, bottom=bottom, color=colr, label=c.replace("solved_by_", "solved by ")); bottom += v
    axs[0].set_xlabel("tier"); axs[0].set_ylabel("pairs (e<10 deg)"); axs[0].legend(fontsize=7)
    for x, y in itertools.combinations(MAIN, 2):
        axs[1].plot(range(1, ntier + 1), [r[f"D_{x}_{y}"] for r in tiers], "-o", label=f"{LABELS[x]} vs {LABELS[y]}")
    axs[1].set_xlabel("tier"); axs[1].set_ylabel("disagreement D_ab"); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "fig5_complementarity.png", dpi=150); plt.close(fig)

    summary = dict(
        generated=datetime.now(timezone.utc).isoformat(), synthetic=bool(a.synthetic), protocol=a.protocol,
        n_pairs=len(ids), settings=cfg["main"], table1=T1.drop(columns=["key"]).to_dict("records"),
        paired_differences=diffs, ranks=ranks, inlier_ratio_vs_pose_spearman={LABELS[k]: v for k, v in corr.items()},
        tiers=tiers, best_of_three=best_of_three, scenes=scenes, tier_scene=tier_scene, per_seed=per_seed, seeds=list(cfg["seeds"]),
        figures=sorted(p.name for p in fig_dir.glob("*.png")),
        expectations_note="Pre-declared expectations (proposal 7.2) must be checked against these numbers; if one fails, report it as the finding.",
    )
    json.dump(summary, open(root / "summary.json", "w"), indent=1, default=float)
    print(T1[["matcher", "AUC@5", "AUC@10", "AUC@20", "median_err", "success_rate", "mean_matches", "inlier_ratio"]].round(3).to_string(index=False))
    print("paired differences:"); print(pd.DataFrame(diffs).round(3).to_string(index=False))
    print("wrote", root / "summary.json")


if __name__ == "__main__":
    main()
