"""Fig 6: hardest pairs by a PRE-DECLARED rule: 4 largest-rotation test pairs + 4 pairs where matchers disagree
(exactly one matcher succeeds; ordered by pair_id for determinism). Matches coloured by epipolar error."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .config import load_config, ensure_dir
from .data import load_table, read_pair
from .geometry import sampson_sq
from .matchers import LABELS
from .results_io import load_matcher
from .stats import success_matrix

MAIN = ["sift_mnn", "splg", "loftr"]


def draw(im0, im1, p0, p1, good, H=360, max_lines=150, label=""):
    def rs(im):
        s = H / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * s), H)), s
    a, sa = rs(im0); b, sb = rs(im1)
    canvas = np.hstack([a, b])
    rng = np.random.RandomState(0)
    sel = rng.permutation(len(p0))[:max_lines]
    for i in sel:
        x0, y0 = (p0[i] * sa).astype(int); x1, y1 = (p1[i] * sb).astype(int)
        col = (80, 170, 60) if good[i] else (50, 50, 220)
        cv2.line(canvas, (x0, y0), (x1 + a.shape[1], y1), col, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (30, 30, 30), -1)
    cv2.putText(canvas, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/frozen.yaml")
    ap.add_argument("--protocol", default="main")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = Path(cfg["results_root"])
    pairs = json.load(open(cfg["pairs_file"]))
    ids = pairs["test"]
    table = load_table(cfg)
    errs = {m: load_matcher(root, a.protocol, "test", m, ids)["err"] for m in MAIN}
    succ = success_matrix(errs)
    hardest = sorted(ids, key=lambda i: -pairs["rot"][i])[:4]
    k = sum(succ[m].astype(int) for m in MAIN)
    disagree = sorted(i for i, kk in zip(ids, k) if kk in (1, 2))[:4]
    out = ensure_dir(root / "figures")
    for kind, lst in (("hardest", hardest), ("disagree", disagree)):
        for pid in lst:
            row = table[pid]; im0, im1 = read_pair(cfg, row)
            R, t = row["T01"][:3, :3], row["T01"][:3, 3]
            rows = []
            for m in MAIN:
                z = np.load(root / a.protocol / "test" / m / "matches" / f"{pid}.npz")
                e = sampson_sq(z["p0"], z["p1"], row["K0"], row["K1"], R, t)
                good = e < cfg["epi_thr"]
                rows.append(draw(im0, im1, z["p0"], z["p1"], good,
                                 label=f"{LABELS[m]}: {len(good)} matches, {100*good.mean() if len(good) else 0:.0f}% epipolar-consistent"))
            w = max(r.shape[1] for r in rows)
            rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
            cv2.imwrite(str(out / f"fig6_{kind}_{pid}.png"), np.vstack(rows))
    print("wrote qualitative panels to", out)


if __name__ == "__main__":
    main()
