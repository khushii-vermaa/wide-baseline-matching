import json
from pathlib import Path
import numpy as np


def load_matcher(root, protocol, split, matcher, ids):
    d = Path(root) / protocol / split / matcher
    err, nm, ni, epi, tt = [], [], [], [], []
    for pid in ids:
        r = json.load(open(d / f"{pid}.json"))
        err.append([s["err"] for s in r["seeds"]])
        ni.append([s["n_inliers"] for s in r["seeds"]])
        nm.append(r["n_matches"]); epi.append(r["epi_precision"]); tt.append(r.get("time_s", np.nan))
    return dict(err=np.array(err, float), n_inliers=np.array(ni, float), n_matches=np.array(nm, float),
                epi=np.array(epi, float), time=np.array(tt, float))
