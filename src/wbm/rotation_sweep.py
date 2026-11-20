"""O7 / Arm F: in-plane rotation of image 1 by 0,45,90,135,180 degrees with GT pose updated."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, settings_for, ensure_dir, progress
from .core import eval_matches
from .data import load_table, read_pair
from .geometry import auc
from .matchers import build_matcher, get_device
from .rotate import rotate_image, rotate_pose
from .run import pair_ids

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/frozen.yaml")
    ap.add_argument("--matchers", nargs="+", default=["sift_mnn", "splg", "loftr"])
    ap.add_argument("--angles", nargs="+", type=float, default=[0, 45, 90, 135, 180])
    ap.add_argument("--n-pairs", type=int, default=100)
    a = ap.parse_args()
    cfg = load_config(a.config)
    assert cfg.get("frozen"), "use the frozen config"
    table = load_table(cfg)
    ids = pair_ids(cfg, table, "test")[: a.n_pairs]
    rows = []
    for m in a.matchers:
        s = settings_for(cfg, "main", m)
        matcher = build_matcher(m, s, get_device())
        for ang in a.angles:
            errs = []
            for pid in progress(ids, desc=f"{m} rot={ang}"):
                row = table[pid]
                im0, im1 = read_pair(cfg, row)
                im1r, K1 = rotate_image(im1, row["K1"], ang)
                T = rotate_pose(row["T01"], ang)
                p0, p1 = matcher.match(im0, im1r)
                rec = eval_matches(p0, p1, row["K0"], K1, T, cfg["seeds"], s["ransac_thr"], cfg["epi_thr"])
                errs.append([x["err"] for x in rec["seeds"]])
            e = np.array(errs)
            rows.append(dict(matcher=m, angle=ang, auc10=float(np.mean([auc(e[:, j], 10) for j in range(e.shape[1])]))))
    df = pd.DataFrame(rows)
    out = ensure_dir(Path(cfg["results_root"]) / "tables")
    df.to_csv(out / "rotation_sweep.csv", index=False)
    print(df.pivot(index="angle", columns="matcher", values="auc10"))
