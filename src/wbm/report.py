"""Generate the LaTeX fragments of the report from your own results (never invents numbers).
   python -m wbm.report                       # reads results/summary.json, writes report/generated/*.tex
   python -m wbm.report --placeholders        # writes TBD placeholders (report compiles before experiments exist)
"""
import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .geometry import auc
from .matchers import LABELS
from .results_io import load_matcher

MAIN = ["sift_mnn", "splg", "loftr"]
SHORT = {"sift_mnn": "SIFT+MNN", "splg": "SP+LG", "loftr": "LoFTR", "sift_ratio": "SIFT+ratio"}
INV = {v: k for k, v in LABELS.items()}
DEG = r"$^\circ$"
TBD = "TBD"


def esc(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def n(x, d=3, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    return s.replace("-", "$-$")


def ci_str(v, lo, hi, d=3):
    return f"{n(v, d)} [{n(lo, 2)}, {n(hi, 2)}]"


def write(p, s):
    Path(p).write_text(s)


def tabular(cols, rows, align=None):
    align = align or "l" + "r" * (len(cols) - 1)
    out = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(cols) + r" \\", "\\midrule"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def placeholders(gen):
    write(gen / "macros.tex", "\\newcommand{\\nTest}{TBD}\\newcommand{\\rsz}{TBD}\\newcommand{\\mkp}{TBD}\\newcommand{\\rthr}{TBD}\n"
                              "\\newcommand{\\abstractfindings}{\\textbf{[Results pending: run \\texttt{python -m wbm.report} after the experiments.]}}\n")
    main_cols = ["Matcher", "AUC@5", "AUC@10", "AUC@20", "Med. err.", "Succ.", "Matches", "Inl. ratio", "Epi. prec."]
    write(gen / "table_main.tex", tabular(main_cols, [[SHORT[m]] + [TBD] * 8 for m in MAIN]))
    write(gen / "table_diffs.tex", tabular(["Pair", "$\\Delta$AUC@5", "$\\Delta$AUC@10"], [[f"{SHORT[a]} $-$ {SHORT[b]}", TBD, TBD] for a, b in itertools.combinations(MAIN, 2)]))
    write(gen / "table_tiers.tex", tabular(["Tier", "Rot. ($^\\circ$)"] + [SHORT[m] for m in MAIN] + ["$D$ S/L", "$D$ S/F", "$D$ L/F"], [[str(t), TBD] + [TBD] * 6 for t in (1, 2, 3)]))
    write(gen / "table_repro.tex", tabular(["Matcher", "", "AUC@5", "AUC@10", "AUC@20"], [[SHORT[m], s, TBD, TBD, TBD] for m in ("loftr", "splg") for s in ("ours", "paper")]))
    write(gen / "table_sens.tex", tabular(["Setting"] + [SHORT[m] for m in MAIN], [[TBD] + [TBD] * 3]))
    write(gen / "table_ratio.tex", tabular(["Variant", "AUC@5", "AUC@10", "AUC@20"], [["SIFT+MNN", TBD, TBD, TBD], ["SIFT+ratio", TBD, TBD, TBD]]))
    write(gen / "table_scene.tex", tabular(["Landmark", "$n$"] + [SHORT[m] for m in MAIN], [[TBD, TBD] + [TBD] * 3]))
    write(gen / "table_seed.tex", tabular(["Matcher", "Seed 0", "Seed 1", "Seed 2"], [[SHORT[m]] + [TBD] * 3 for m in MAIN]))
    write(gen / "findings.tex", "\\todo{Run the experiments, then \\texttt{python -m wbm.report}; this section is generated from \\texttt{results/summary.json}.}\n")


def order(d):   # keys sorted by value desc
    return sorted(d, key=lambda k: -d[k])


def findings(S):
    T = {INV[r["matcher"]]: r for r in S["table1"]}
    out = []
    a10 = {m: T[m]["AUC@10"] for m in MAIN}
    a5 = {m: T[m]["AUC@5"] for m in MAIN}
    o10 = order(a10)
    out.append("\\paragraph{Pose ranking.} On the %d test pairs, ranking by AUC@10 gives %s (%s). By AUC@5 the order is %s. Intervals are 95\\%% paired-bootstrap intervals over pairs."
               % (S["n_pairs"], " $>$ ".join(SHORT[m] for m in o10),
                  "; ".join(f"{SHORT[m]} {ci_str(a10[m], T[m]['AUC@10_lo'], T[m]['AUC@10_hi'])}" for m in o10),
                  " $>$ ".join(SHORT[m] for m in order(a5))))
    sig, ns = [], []
    for d in S["paired_differences"]:
        if d["tau"] != 10:
            continue
        a, b = INV[d["a"]], INV[d["b"]]
        s = f"{SHORT[a]} $-$ {SHORT[b]}: {n(d['diff'], 3, True)} [{n(d['lo'], 3, True)}, {n(d['hi'], 3, True)}]"
        (sig if d["significant"] else ns).append(s)
    out.append("\\paragraph{Which differences are real.} AUC@10 differences whose interval excludes zero: %s. Differences inside the interval, which this benchmark cannot separate at $n=%d$: %s."
               % ("; ".join(sig) or "none", S["n_pairs"], "; ".join(ns) or "none"))
    R = S["ranks"]
    o = R["overall"]
    cnt = " $>$ ".join(SHORT[m] for m in sorted(o["matches"]["ranks"], key=o["matches"]["ranks"].get))
    inl = " $>$ ".join(SHORT[m] for m in sorted(o["inliers"]["ranks"], key=o["inliers"]["ranks"].get))
    epi = " $>$ ".join(SHORT[m] for m in sorted(o["epi"]["ranks"], key=o["epi"]["ranks"].get))
    tier_agree = {k: R[k]["matches"]["agrees_with_auc"] for k in R if k.startswith("tier")}
    corr = S["inlier_ratio_vs_pose_spearman"]
    out.append("\\paragraph{Do counts predict pose?} Ranking by mean match count: %s; by mean inlier count: %s; by epipolar precision: %s; by AUC@10: %s. The match-count ranking %s the pose ranking overall; by tier: %s. "
               "(Dense and sparse match counts count different things and are not compared as if equal.) Spearman correlation between per-pair inlier ratio and pose accuracy: %s."
               % (cnt, inl, epi, " $>$ ".join(SHORT[m] for m in sorted(o["auc_ranks"], key=o["auc_ranks"].get)),
                  "agrees with" if o["matches"]["agrees_with_auc"] else "does not agree with",
                  ", ".join(f"tier {k[4:]} {'agrees' if v else 'differs'}" for k, v in sorted(tier_agree.items())),
                  "; ".join(f"{esc(k)} {n(v, 2)}" for k, v in corr.items())))
    tiers = S["tiers"]
    tier_orders = {t["tier"]: order({m: t[f"auc10_{m}"] for m in MAIN}) for t in tiers}
    changes = [t for t in tier_orders if tier_orders[t] != o10]
    out.append("\\paragraph{Rotation tiers.} The tiers split the test pairs by ground-truth rotation only (tier 1 = smallest rotation of the test set), so they are rotation tiers, not a complete difficulty scale. AUC@10 by tier: %s. %s"
               % ("; ".join(f"tier {t['tier']} ({t['rot_min']:.0f}--{t['rot_max']:.0f}{DEG}): " + ", ".join(f"{SHORT[m]} {n(t[f'auc10_{m}'], 2)}" for m in MAIN) for t in tiers),
                  ("The ranking differs from the overall ranking in tier %s; this is reported, not averaged away." % ", ".join(map(str, changes))) if changes
                  else "The ranking is the same in every tier."))
    last, first = tiers[-1], tiers[0]
    best = S["best_of_three"]
    pairs3 = list(itertools.combinations(MAIN, 2))
    def dser(t):
        return "; ".join(f"{SHORT[x]}/{SHORT[y]}: " + ", ".join(f"{n(tt[f'D_{x}_{y}'], 2)} ({round(tt[f'D_{x}_{y}'] * tt['n'])}/{tt['n']})" for tt in tiers) for x, y in pairs3)
    best = S["best_of_three"]
    out.append("\\paragraph{Disagreement and complementarity.} Here a pair counts as solved by a matcher if its median pose error over the RANSAC seeds is below $10^\\circ$, and $D_{ab}$ is the fraction of the pairs in a tier on which exactly one of the two matchers solves it (denominator: all pairs in the tier). "
               "$D_{ab}$ by tier, from the smallest to the largest rotation (pairs in parentheses): %s. In the largest-rotation tier, the number of pairs solved by no matcher is %d, by exactly one %d, by two %d and by all three %d (pairs solved only by SIFT+MNN / SP+LG / LoFTR: %d / %d / %d). "
               "A best-of-three oracle, which picks the best matcher per pair using the ground truth, reaches AUC@5 %s, AUC@10 %s, AUC@20 %s."
               % (dser(None), last["solved_by_none"], last["solved_by_one"], last["solved_by_two"], last["solved_by_all"], last["only_sift_mnn"], last["only_splg"], last["only_loftr"],
                  n(best["AUC@5"], 3), n(best["AUC@10"], 3), n(best["AUC@20"], 3)))
    if S.get("tier_scene"):
        out.append("\\paragraph{Landmarks per tier.} Pairs per tier by landmark: %s." % "; ".join(
            f"tier {t}: " + ", ".join(f"{sc} {c}" for sc, c in v.items()) for t, v in sorted(S["tier_scene"].items(), key=lambda kv: int(kv[0]))))
    # pre-declared expectations (proposal 7.2)
    e1 = o10[-1] == "sift_mnn"
    mism = [k for k in tier_agree if not tier_agree[k]]
    most = max(MAIN, key=lambda m: T[m]["mean_matches"])
    e2 = bool(mism) or not o["matches"]["agrees_with_auc"]
    seq = lambda x, y: [tt[f"D_{x}_{y}"] for tt in tiers]
    rises = sum(seq(x, y)[-1] > seq(x, y)[0] for x, y in pairs3)
    strict = sum(all(b > a for a, b in zip(seq(x, y), seq(x, y)[1:])) for x, y in pairs3)
    if strict == 3:
        e3 = "held (strictly increasing across all tiers for all three matcher pairs)"
    elif rises == 0:
        e3 = "did not hold"
    else:
        e3 = ("partly held: the rate is higher in the largest-rotation tier than in the smallest for %d of 3 matcher pairs, but strictly increasing across all tiers for %d of 3; "
              "SIFT+MNN pairs are dominated by SIFT+MNN failing in every tier" % (rises, strict))
    dd = {(d["a"], d["b"], d["tau"]): d for d in S["paired_differences"]}
    d_ll = dd.get((LABELS["splg"], LABELS["loftr"], 10))
    out.append("\\paragraph{Pre-declared expectations (Section~\\ref{sec:expect}).} "
               "(1) SIFT+MNN lowest AUC@10: %s. (2) Count ranking differs from the AUC ranking in at least one tier or overall: %s (the matcher with most matches is %s). "
               "(3) $D_{ab}$ rises from easiest to hardest tier: %s. (4) SP+LG versus LoFTR at AUC@10: %s, which is %s."
               % ("held" if e1 else "did not hold", "held" if e2 else "did not hold", SHORT[most], e3,
                  n(d_ll["diff"], 3, True) + f" [{n(d_ll['lo'], 3, True)}, {n(d_ll['hi'], 3, True)}]" if d_ll else "--",
                  "a difference we claim" if d_ll and d_ll["significant"] else "not claimed as a difference (interval includes zero)"))
    return "\n\n".join(out) + "\n"


def scene_findings(S):
    T = {INV[r["matcher"]]: r for r in S["table1"]}
    overall = order({m: T[m]["AUC@10"] for m in MAIN})
    parts, rev = [], []
    for sc in S["scenes"]:
        o = order({m: sc[f"auc10_{m}"] for m in MAIN})
        parts.append(f"landmark {sc['scene']} ($n={sc['n']}$): " + " $>$ ".join(SHORT[m] for m in o))
        if o != overall:
            rev.append(sc["scene"])
    ps = S["per_seed"]
    spread = max(max(ps[m]["auc10"]) - min(ps[m]["auc10"]) for m in MAIN)
    return ("\\paragraph{Per landmark and per seed.} AUC@10 ranking by landmark: %s. %s Across the %d RANSAC seeds the largest spread in AUC@10 for any matcher is %s."
            % ("; ".join(parts), ("The ranking differs from the overall ranking for landmark %s; this is stated, not averaged away." % ", ".join(rev)) if rev
               else "The ranking is the same in every landmark.", len(S["seeds"]), n(spread, 3))) + "\n"


def abstract(S):
    T = {INV[r["matcher"]]: r for r in S["table1"]}
    o10 = order({m: T[m]["AUC@10"] for m in MAIN})
    o = S["ranks"]["overall"]
    return ("On %d geometry-selected wide-baseline pairs the pose ranking by AUC@10 is %s; the match-count ranking %s it."
            % (S["n_pairs"], " $>$ ".join(f"{SHORT[m]} ({n(T[m]['AUC@10'], 2)})" for m in o10),
               "agrees with" if o["matches"]["agrees_with_auc"] else "does not agree with"))


def make(S, root, cfg, gen):
    st = S["settings"]
    write(gen / "macros.tex", "\\newcommand{\\nTest}{%d}\\newcommand{\\rsz}{%s}\\newcommand{\\mkp}{%s}\\newcommand{\\rthr}{%s}\n\\newcommand{\\abstractfindings}{%s}\n"
          % (S["n_pairs"], st["resize"], st["max_kp"], st["ransac_thr"], abstract(S)))
    rows = []
    for r in S["table1"]:
        rows.append([SHORT[INV[r["matcher"]]], ci_str(r["AUC@5"], r["AUC@5_lo"], r["AUC@5_hi"]), ci_str(r["AUC@10"], r["AUC@10_lo"], r["AUC@10_hi"]), n(r["AUC@20"]),
                     n(r["median_err"], 1) + DEG, n(100 * r["success_rate"], 0) + r"\%", n(r["mean_matches"], 0), n(r["inlier_ratio"], 2), n(r["epi_precision"], 2)])
    write(gen / "table_main.tex", tabular(["Matcher", "AUC@5", "AUC@10", "AUC@20", "Med. err.", "Succ.", "Matches", "Inl. ratio", "Epi. prec."], rows))
    dd = {}
    for d in S["paired_differences"]:
        dd.setdefault((d["a"], d["b"]), {})[d["tau"]] = d
    rows = []
    for (a, b), v in dd.items():
        cell = lambda d: f"{n(d['diff'], 3, True)} [{n(d['lo'], 3, True)}, {n(d['hi'], 3, True)}]" + ("*" if d["significant"] else "")
        rows.append([f"{SHORT[INV[a]]} $-$ {SHORT[INV[b]]}", cell(v[5]), cell(v[10])])
    write(gen / "table_diffs.tex", tabular(["Pair", "$\\Delta$AUC@5", "$\\Delta$AUC@10"], rows))
    rows = []
    for t in S["tiers"]:
        rows.append([str(t["tier"]), f"{t['rot_min']:.0f}--{t['rot_max']:.0f}"] + [ci_str(t[f"auc10_{m}"], t[f"auc10_{m}_lo"], t[f"auc10_{m}_hi"], 2) for m in MAIN]
                    + [n(t["D_sift_mnn_splg"], 2), n(t["D_sift_mnn_loftr"], 2), n(t["D_splg_loftr"], 2)])
    write(gen / "table_tiers.tex", tabular(["Tier", "Rot. ($^\\circ$)"] + [SHORT[m] for m in MAIN] + ["$D$ S/L", "$D$ S/F", "$D$ L/F"], rows))
    rows = []
    for sc in S["scenes"]:
        rows.append([sc["scene"], str(sc["n"])] + [ci_str(sc[f"auc10_{m}"], sc[f"auc10_{m}_lo"], sc[f"auc10_{m}_hi"], 2) for m in MAIN])
    write(gen / "table_scene.tex", tabular(["Landmark", "$n$"] + [SHORT[m] + " AUC@10" for m in MAIN], rows))
    ps = S["per_seed"]
    ns = len(S["seeds"])
    rows = [[SHORT[m]] + [f"{n(ps[m]['auc5'][k])} / {n(ps[m]['auc10'][k])}" for k in range(ns)] for m in MAIN]
    write(gen / "table_seed.tex", tabular(["Matcher (AUC@5 / AUC@10)"] + [f"Seed {k}" for k in S["seeds"]], rows))
    write(gen / "findings.tex", findings(S) + scene_findings(S))
    pub = cfg.get("published_auc", {})
    pairs = json.load(open(root / "pairs.json")) if (root / "pairs.json").exists() else json.load(open(cfg["pairs_file"]))
    ids = list(pairs["rot"].keys())
    rows = []
    for m in ("loftr", "splg"):
        ours = [TBD] * 3
        d = root / "published" / "full" / m
        if d.exists() and any(d.glob("*.json")):
            have = [i for i in ids if (d / f"{i}.json").exists()]
            e = load_matcher(root, "published", "full", m, have)["err"]
            ours = [n(np.mean([auc(e[:, s], t) for s in range(e.shape[1])])) for t in (5, 10, 20)]
            ours[0] += f" ({len(have)} pairs)" if len(have) != 1500 else ""
        p = pub.get(m) or {}
        rows.append([SHORT[m], "ours"] + ours)
        rows.append(["", "paper"] + [n(p.get(k)) if p.get(k) is not None else TBD for k in ("auc5", "auc10", "auc20")])
    write(gen / "table_repro.tex", tabular(["Matcher", "", "AUC@5", "AUC@10", "AUC@20"], rows))
    # sift ratio ablation
    ids_t = pairs["test"]
    rows = []
    for m in ("sift_mnn", "sift_ratio"):
        try:
            e = load_matcher(root, "main", "test", m, ids_t)["err"]
            rows.append([SHORT[m]] + [n(np.mean([auc(e[:, s], t) for s in range(e.shape[1])])) for t in (5, 10, 20)])
        except Exception:
            rows.append([SHORT[m], TBD, TBD, TBD])
    write(gen / "table_ratio.tex", tabular(["Variant", "AUC@5", "AUC@10", "AUC@20"], rows))
    sw = root / "tables" / "sweep_test.csv"
    if sw.exists():
        df = pd.read_csv(sw)
        rows = []
        for col, lab in (("resize", "resize"), ("max_kp", "keypoints"), ("ransac_thr", "RANSAC thr.")):
            for v in sorted(df[col].dropna().unique()):
                row = [f"{lab} {v:g}"]
                for m in MAIN:
                    g = df[(df.matcher == m) & (df[col] == v)]
                    row.append(n(g.auc10.mean(), 3) if len(g) else "--")
                rows.append(row)
        write(gen / "table_sens.tex", tabular(["Setting (AUC@10, mean over other settings)"] + [SHORT[m] for m in MAIN], rows))
    else:
        write(gen / "table_sens.tex", tabular(["Setting"] + [SHORT[m] for m in MAIN], [[TBD] * 4]))
    rt = root / "tables" / "rotation_sweep.csv"
    if rt.exists():
        df = pd.read_csv(rt)
        rows = [[f"{a:g}{DEG}"] + [n(df[(df.matcher == m) & (df.angle == a)].auc10.mean()) for m in MAIN] for a in sorted(df.angle.unique())]
        write(gen / "table_rotation.tex", tabular(["Rotation"] + [SHORT[m] for m in MAIN], rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/frozen.yaml")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--out", default="report")
    ap.add_argument("--placeholders", action="store_true")
    a = ap.parse_args()
    cfgp = a.config if Path(a.config).exists() else "configs/config.yaml"
    cfg = load_config(cfgp)
    root = Path(a.results_root or cfg["results_root"])
    out = Path(a.out)
    gen = out / "generated"; gen.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if a.placeholders or not (root / "summary.json").exists():
        placeholders(gen)
        print("wrote PLACEHOLDER fragments (no summary.json found)" if not a.placeholders else "wrote placeholders")
        return
    S = json.load(open(root / "summary.json"))
    if S.get("synthetic"):
        print("WARNING: summary.json is SYNTHETIC. Do not submit a report built from it.")
    make(S, root, cfg, gen)
    fdir = root / "figures"
    if fdir.exists():
        for p in fdir.glob("*.png"):
            shutil.copy(p, out / "figures" / p.name)
    print("report fragments written to", gen, "-> cd report && latexmk -pdf main.tex")


if __name__ == "__main__":
    main()
