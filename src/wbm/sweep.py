"""Grid over (resize, max_kp, ransac_thr). Matches are computed once per (resize, max_kp); RANSAC threshold is cheap.
  dev  + --freeze : tune on the 200 dev pairs and write configs/frozen.yaml     (N5)
  test            : sensitivity arms (Arm E / O6) on the 100 test pairs, after freezing
"""
import argparse
import copy
import itertools
import json
from pathlib import Path

import pandas as pd
import yaml

from .config import load_config, ensure_dir, progress
from .core import eval_matches
from .data import load_table, read_pair
from .geometry import auc
from .matchers import build_matcher, get_device
from .run import pair_ids


def sweep(cfg, split, grid, matchers, device):
    table = load_table(cfg)
    ids = pair_ids(cfg, table, split)
    imgs = {}
    rows = []
    for m in matchers:
        kps = [None] if m == "loftr" else grid["max_kp"]
        for resize, kp in itertools.product(grid["resize"], kps):
            s = dict(resize=resize, max_kp=kp or 2048)
            matcher = build_matcher(m, s, device)
            matches = {}
            for pid in progress(ids, desc=f"{m} r={resize} kp={kp}"):
                row = table[pid]
                if pid not in imgs:
                    imgs[pid] = read_pair(cfg, row)
                matches[pid] = matcher.match(*imgs[pid])
            for thr in grid["ransac_thr"]:
                errs = []
                for pid in ids:
                    row = table[pid]
                    rec = eval_matches(*matches[pid], row["K0"], row["K1"], row["T01"], cfg["seeds"], thr, cfg["epi_thr"])
                    errs.append([x["err"] for x in rec["seeds"]])
                import numpy as np
                e = np.array(errs)
                rows.append(dict(matcher=m, resize=resize, max_kp=kp, ransac_thr=thr,
                                 **{f"auc{t}": float(np.mean([auc(e[:, j], t) for j in range(e.shape[1])])) for t in (5, 10, 20)}))
    return pd.DataFrame(rows)


def choose(df):
    """One common setting for all matchers: maximise mean AUC@10 over matchers. LoFTR ignores max_kp."""
    best, best_score = None, -1
    sparse = df[df.matcher != "loftr"]
    for (resize, kp, thr), g in sparse.groupby(["resize", "max_kp", "ransac_thr"]):
        scores = list(g.auc10)
        lo = df[(df.matcher == "loftr") & (df.resize == resize) & (df.ransac_thr == thr)]
        scores += list(lo.auc10)
        sc = sum(scores) / len(scores)
        if sc > best_score + 1e-9:
            best, best_score = dict(resize=int(resize), max_kp=int(kp), ransac_thr=float(thr)), sc
    return best, best_score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--matchers", nargs="+", default=["sift_mnn", "splg", "loftr"])
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.split == "test" and not cfg.get("frozen"):
        raise SystemExit("Test-split sweeps (sensitivity arms) must use the frozen config.")
    grid = cfg["tuning_grid"] if a.split == "dev" else cfg["sensitivity_grid"]
    df = sweep(cfg, a.split, grid, a.matchers, get_device())
    out = ensure_dir(Path(cfg["results_root"]) / "tables")
    df.to_csv(out / f"sweep_{a.split}.csv", index=False)
    print(df.to_string(index=False))
    if a.freeze:
        best, score = choose(df)
        new = copy.deepcopy({k: v for k, v in cfg.items() if k != "_path"})
        new["main"] = best
        new["frozen"] = True
        new["tuned_on"] = "dev split; mean AUC@10 over matchers = %.4f" % score
        with open("configs/frozen.yaml", "w") as f:
            yaml.safe_dump(new, f, sort_keys=False)
        print("FROZEN ->", best, " written to configs/frozen.yaml  (commit it)")
