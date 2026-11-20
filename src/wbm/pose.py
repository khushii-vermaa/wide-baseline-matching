"""One identical RANSAC pose stage for every matcher."""
import cv2
import numpy as np


def estimate_pose(p0, p1, K0, K1, thr_px, seed=0, conf=0.99999):
    """Essential matrix with known intrinsics + RANSAC (OpenCV), then pose recovery.
    Returns dict(R, t, n_inliers, inliers) or None if no valid pose."""
    if len(p0) < 5:
        return None
    rs = np.random.RandomState(seed)
    perm = rs.permutation(len(p0))          # seed also changes sample order
    p0, p1 = p0[perm], p1[perm]
    cv2.setRNGSeed(int(seed))
    f = np.mean([K0[0, 0], K0[1, 1], K1[0, 0], K1[1, 1]])
    thr = thr_px / f
    n0 = (p0 - K0[[0, 1], [2, 2]]) / K0[[0, 1], [0, 1]]
    n1 = (p1 - K1[[0, 1], [2, 2]]) / K1[[0, 1], [0, 1]]
    E, mask = cv2.findEssentialMat(n0, n1, np.eye(3), method=cv2.RANSAC, prob=conf, threshold=thr)
    if E is None or mask is None or E.shape[0] % 3 != 0 or E.shape[0] == 0:
        return None
    best = None
    for Ei in np.split(E, E.shape[0] // 3):
        n, R, t, _ = cv2.recoverPose(Ei, n0, n1, np.eye(3), 1e9, mask=mask.copy())
        if best is None or n > best[0]:
            best = (n, R, t[:, 0], mask.ravel() > 0)
    inl = np.zeros(len(perm), bool)
    inl[perm] = best[3]                     # undo permutation -> original match order
    return {"R": best[1], "t": best[2], "n_inliers": int(best[0]), "inliers": inl}
