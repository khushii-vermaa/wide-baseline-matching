"""Evaluate a set of matches for one pair: pose error per seed + epipolar precision."""
import numpy as np
from .geometry import pose_error, epipolar_precision
from .pose import estimate_pose

INF = float("inf")


def eval_matches(p0, p1, K0, K1, T01, seeds, ransac_thr, epi_thr):
    R_gt, t_gt = T01[:3, :3], T01[:3, 3]
    rec = {
        "n_matches": int(len(p0)),
        "epi_precision": epipolar_precision(p0, p1, K0, K1, R_gt, t_gt, epi_thr) if len(p0) else 0.0,
        "seeds": [],
    }
    for s in seeds:
        est = estimate_pose(p0, p1, K0, K1, ransac_thr, seed=s)
        if est is None:
            rec["seeds"].append(dict(seed=s, err=INF, err_R=INF, err_t=INF, n_inliers=0))
        else:
            e, eR, et = pose_error(est["R"], est["t"], R_gt, t_gt)
            rec["seeds"].append(dict(seed=s, err=e, err_R=eR, err_t=et, n_inliers=est["n_inliers"]))
    return rec
