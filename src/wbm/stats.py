"""Paired bootstrap over pairs (N4)."""
import numpy as np
from .geometry import auc


def auc_seeds(err, tau):
    """err: (n_pairs, n_seeds) -> mean over seeds of AUC(tau)."""
    return float(np.mean([auc(err[:, s], tau) for s in range(err.shape[1])]))


def bootstrap(errs, taus=(5, 10, 20), B=1000, seed=0):
    """errs: {name: (n, S)}. Same resampled pair indices for every matcher (paired).
    Returns {(name, tau): array(B)}."""
    n = next(iter(errs.values())).shape[0]
    idx = np.random.RandomState(seed).randint(0, n, size=(B, n))
    out = {}
    for name, e in errs.items():
        for tau in taus:
            out[(name, tau)] = np.array([auc_seeds(e[ix], tau) for ix in idx])
    return out


def ci(samples, level=0.95):
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))


def success_matrix(errs, thr=10.0):
    """Per-pair success (median error over seeds < thr). Returns {name: bool array}."""
    return {k: np.median(v, axis=1) < thr for k, v in errs.items()}


def disagreement(sa, sb):
    return float(np.mean(sa != sb))
