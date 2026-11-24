"""N6 / section 7.4 negative controls on DEV pairs (never the test pairs).
  1. random-match baseline -> AUC near 0     2. pair-order symmetry     3. zero-baseline sanity"""
import argparse
import json

import numpy as np

from .config import load_config, settings_for
from .core import eval_matches
from .data import load_table, read_pair
from .geometry import auc
from .matchers import build_matcher, get_device


def symmetry_stats(e_ab, e_ba, thr=10.0):
    """Consistency of pose errors when the image order is swapped, on pairs where both directions succeed."""
    e_ab, e_ba = np.asarray(e_ab, float), np.asarray(e_ba, float)
    both = (e_ab < thr) & (e_ba < thr)
    diffs = np.abs(e_ab[both] - e_ba[both])
    return dict(n_both=int(both.sum()), agree=float(np.mean((e_ab < thr) == (e_ba < thr))),
                med=float(np.median(diffs)) if both.any() else float("nan"))


def swap(row):
    return np.linalg.inv(row["T01"]), row["K1"], row["K0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-sym", type=int, default=20, help="dev pairs used for the symmetry check")
    ap.add_argument("--matchers", nargs="+", default=["sift_mnn"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    table = load_table(cfg)
    pairs = json.load(open(cfg["pairs_file"]))
    dev = pairs["dev"][: a.n]
    rng = np.random.RandomState(0)
    ok = True

    # 1. random correspondences, 1000 per pair
    errs = []
    for pid in dev:
        row = table[pid]; im0, im1 = read_pair(cfg, row)
        h0, w0 = im0.shape[:2]; h1, w1 = im1.shape[:2]
        p0 = rng.rand(1000, 2) * [w0, h0]; p1 = rng.rand(1000, 2) * [w1, h1]
        rec = eval_matches(p0, p1, row["K0"], row["K1"], row["T01"], cfg["seeds"][:1], cfg["main"]["ransac_thr"], cfg["epi_thr"])
        errs.append(rec["seeds"][0]["err"])
    a10 = auc(errs, 10)
    print(f"[1] random-match AUC@10 = {a10:.4f}  (must be ~0)")
    ok &= a10 < 0.05

    # 2. symmetry, 3. zero-baseline, per requested matcher
    # Symmetry is measured on pairs where BOTH directions succeed (error < 10 deg). On failed pairs a matcher's
    # error is an arbitrary large number, so including them measures matcher failure, not harness consistency.
    # Only learned matchers gate pass/fail; SIFT is reported as a diagnostic (it often fails at wide baseline).
    easy = sorted((i for i in table if i not in set(pairs["test"])), key=lambda i: table[i]["rot_deg"])[:5]
    for m in a.matchers:
        s = settings_for(cfg, "main", m)
        matcher = build_matcher(m, s, get_device())
        gated = m != "sift_mnn"
        e_ab_l, e_ba_l = [], []
        for pid in dev[:a.n_sym]:
            row = table[pid]; im0, im1 = read_pair(cfg, row)
            e_ab_l.append(eval_matches(*matcher.match(im0, im1), row["K0"], row["K1"], row["T01"], [0], s["ransac_thr"], cfg["epi_thr"])["seeds"][0]["err"])
            T10, K1, K0 = swap(row)
            e_ba_l.append(eval_matches(*matcher.match(im1, im0), K1, K0, T10, [0], s["ransac_thr"], cfg["epi_thr"])["seeds"][0]["err"])
        st = symmetry_stats(e_ab_l, e_ba_l)
        tag = "GATED" if gated else "diagnostic only"
        print(f"[2] {m} ({tag}): both directions succeed on {st['n_both']}/{len(e_ab_l)} pairs; "
              f"success status agrees on {100*st['agree']:.0f}%; median |err(A,B)-err(B,A)| over successes = {st['med']:.2f} deg")
        if gated:
            if st["n_both"] < 5:
                print(f"    only {st['n_both']} pairs succeeded in both directions: too few to judge -> raise --n-sym or check the matcher")
                ok = False
            else:
                ok &= st["med"] < 5
        e0 = []
        for pid in easy:
            row = table[pid]; im0, im1 = read_pair(cfg, row)
            e0.append(eval_matches(*matcher.match(im0, im1), row["K0"], row["K1"], row["T01"], [0], s["ransac_thr"], cfg["epi_thr"])["seeds"][0]["err"])
        print(f"[3] {m} ({tag}): errors on 5 smallest-rotation non-test pairs = {np.round(e0, 2).tolist()}  (learned matchers: median must be < 10)")
        if gated:
            ok &= float(np.median(e0)) < 10
    print("CONTROLS PASSED" if ok else "CONTROLS FAILED -> fix the harness before any comparison")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
