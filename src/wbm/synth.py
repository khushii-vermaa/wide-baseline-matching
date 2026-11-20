"""Generate SYNTHETIC (fake) results so you can try analyze.py and the web UI before the real data is ready.
Everything it writes is flagged synthetic=true. NEVER report these numbers."""
import argparse
import json
from pathlib import Path

import numpy as np

from .config import ensure_dir

SKILL = {"sift_mnn": 0.55, "splg": 0.8, "loftr": 0.85}      # arbitrary fake skill levels
MATCHES = {"sift_mnn": 300, "splg": 600, "loftr": 2500}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_demo")
    a = ap.parse_args()
    rng = np.random.RandomState(0)
    out = ensure_dir(a.out)
    ids = [f"demo_{i:04d}" for i in range(100)]
    rot = np.sort(rng.uniform(40, 120, 100))
    tiers = {pid: int(i * 3 // 100) + 1 for i, pid in enumerate(ids)}
    json.dump(dict(test=ids, dev=[], tier=tiers, rot=dict(zip(ids, rot.tolist())), scene={i: "demo" for i in ids}, all_rot=rot.tolist()),
              open(out / "pairs.json", "w"))
    for m, sk in SKILL.items():
        d = ensure_dir(out / "main" / "test" / m)
        for pid, r in zip(ids, rot):
            p_ok = np.clip(sk - (r - 40) / 140, 0.02, 0.98)
            seeds, nm = [], int(MATCHES[m] * rng.uniform(.4, 1.3))
            hit = rng.rand() < p_ok
            for s in range(3):
                e = float(rng.gamma(2, 1.5)) if (hit and rng.rand() < .92) else float(rng.uniform(15, 120))
                seeds.append(dict(seed=s, err=e, err_R=e, err_t=e, n_inliers=int(nm * (0.5 if hit else 0.12))))
            json.dump(dict(pair_id=pid, matcher=m, n_matches=nm, epi_precision=float(rng.uniform(.3, .9)), seeds=seeds, time_s=float(rng.uniform(.1, 1.5))),
                      open(d / f"{pid}.json", "w"))
    print("synthetic results in", out, "-> python -m wbm.analyze --config configs/config.yaml --results-root", out, "--pairs", out / "pairs.json", "--synthetic")


if __name__ == "__main__":
    main()
