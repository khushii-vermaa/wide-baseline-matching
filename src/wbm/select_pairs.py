"""N1: choose test/dev pairs and tiers from ground-truth rotation ONLY (no matcher output)."""
import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config, ensure_dir
from .data import load_table


def select(rot, ids, n_test, n_dev, n_tiers, seed):
    rng = np.random.RandomState(seed)
    jitter = rng.rand(len(ids))
    order = np.lexsort((jitter, -np.asarray(rot)))      # primary: -rot (descending), tie: jitter
    test = order[:n_test]
    dev = order[n_test:n_test + n_dev]                  # same rule applied within the remainder
    test_asc = test[::-1]                               # ascending rotation
    tiers = {}
    for ti, chunk in enumerate(np.array_split(test_asc, n_tiers), start=1):   # tier 1 = smallest rotation
        for i in chunk:
            tiers[ids[i]] = ti
    return [ids[i] for i in test], [ids[i] for i in dev], tiers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    table = load_table(cfg)
    ids = list(table.keys())
    rot = [table[i]["rot_deg"] for i in ids]
    s = cfg["selection"]
    test, dev, tiers = select(rot, ids, s["n_test"], s["n_dev"], s["n_tiers"], s["seed"])
    out = dict(
        selection_rule="top-N pairs by ground-truth relative rotation; ties by fixed seed",
        test=test, dev=dev, tier=tiers,
        rot={i: table[i]["rot_deg"] for i in ids},
        scene={i: table[i]["scene"] for i in ids},
        all_rot=rot,
    )
    ensure_dir(Path(cfg["pairs_file"]).parent)
    json.dump(out, open(cfg["pairs_file"], "w"))
    tr = np.array([table[i]["rot_deg"] for i in test])
    print(f"test rotation: min {tr.min():.1f} max {tr.max():.1f} deg | overall median {np.median(rot):.1f}")
    print("If the top-100 are not 'wide' enough you may relax the rule ONCE, in writing (proposal section 4).")
    print(f"wrote {cfg['pairs_file']}  -> COMMIT THIS FILE before running any matcher.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.hist(rot, bins=60, color="#9fb8c4")
        ax.axvline(tr.min(), color="#1f7a8c", label=f"test (top {len(test)}) >= {tr.min():.0f} deg")
        dv = np.array([table[i]["rot_deg"] for i in dev])
        ax.axvline(dv.min(), color="#e0a31a", label=f"dev >= {dv.min():.0f} deg")
        ax.set_xlabel("ground-truth relative rotation (deg)"); ax.set_ylabel("pairs"); ax.legend()
        figdir = ensure_dir(Path(cfg["results_root"]) / "figures")
        fig.tight_layout(); fig.savefig(figdir / "fig1_pair_selection.png", dpi=150)
    except Exception as e:   # figure is optional
        print("fig1 skipped:", e)


if __name__ == "__main__":
    main()
