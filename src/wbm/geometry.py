"""Pose-error, AUC and epipolar geometry. Pure NumPy; unit-tested in tests/."""
import numpy as np


def relative_pose(T0, T1):
    """T0, T1: 4x4 world-to-camera. Returns T_0to1 (maps camera-0 coords to camera-1 coords)."""
    return T1 @ np.linalg.inv(T0)


def rotation_angle_deg(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def pose_error(R_est, t_est, R_gt, t_gt):
    """Eq. (1): e = max(eps_R, eps_t). Translation compared by direction only (up to sign)."""
    t_est = np.asarray(t_est, float).reshape(3)
    t_gt = np.asarray(t_gt, float).reshape(3)
    n = np.linalg.norm(t_est) * np.linalg.norm(t_gt)
    if n < 1e-12:
        eps_t = 180.0
    else:
        eps_t = float(np.degrees(np.arccos(np.clip(abs(t_est @ t_gt) / n, -1.0, 1.0))))
    eps_R = rotation_angle_deg(R_gt.T @ R_est)
    return max(eps_R, eps_t), eps_R, eps_t


def _trapz(y, x):
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x=x)


def error_auc(errors, thresholds):
    """Eq. (2): area under the recall-vs-error curve up to tau, divided by tau.
    Uses the trapezoid rule over the sorted-error points (the LoFTR / LightGlue evaluation convention),
    so numbers are comparable with published MegaDepth-1500 tables. (An exact step-function integral
    would differ slightly on small samples.) inf / NaN errors (no valid pose) count as never recalled."""
    errors = np.asarray(errors, dtype=float)
    errors = np.where(np.isnan(errors), np.inf, errors)
    errs = np.sort(errors)
    recall = (np.arange(len(errs)) + 1) / len(errs)
    errs = np.r_[0.0, errs]
    recall = np.r_[0.0, recall]
    out = {}
    for t in thresholds:
        idx = np.searchsorted(errs, t)
        r = np.r_[recall[:idx], recall[idx - 1]]
        e = np.r_[errs[:idx], t]
        out[t] = float(_trapz(r, e) / t)
    return out


def auc(errors, tau):
    return error_auc(errors, [tau])[tau]


def skew(t):
    t = np.asarray(t, float).reshape(3)
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def normalise(p, K):
    return (p - K[[0, 1], [2, 2]]) / K[[0, 1], [0, 1]]


def sampson_sq(p0, p1, K0, K1, R, t):
    """Squared Sampson epipolar distance in normalised coordinates, per match."""
    if len(p0) == 0:
        return np.zeros(0)
    t = np.asarray(t, float).reshape(3)
    t = t / (np.linalg.norm(t) + 1e-12)
    E = skew(t) @ R
    x0, x1 = normalise(p0, K0), normalise(p1, K1)
    h0 = np.c_[x0, np.ones(len(x0))]
    h1 = np.c_[x1, np.ones(len(x1))]
    Ex0 = h0 @ E.T
    Etx1 = h1 @ E
    num = np.sum(h1 * Ex0, axis=1) ** 2
    den = Ex0[:, 0] ** 2 + Ex0[:, 1] ** 2 + Etx1[:, 0] ** 2 + Etx1[:, 1] ** 2
    return num / np.maximum(den, 1e-18)


def epipolar_precision(p0, p1, K0, K1, R, t, thr):
    if len(p0) == 0:
        return 0.0
    return float(np.mean(sampson_sq(p0, p1, K0, K1, R, t) < thr))
