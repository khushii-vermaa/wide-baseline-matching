"""Run matchers on a split; cache per-pair results (a lost session costs nothing)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import load_config, settings_for, ensure_dir, progress
from .core import eval_matches
from .data import load_table, read_pair
from .matchers import build_matcher, get_device


def pair_ids(cfg, table, split):
    if split == "full":
        return list(table.keys())
    return json.load(open(cfg["pairs_file"]))[split]


def run(cfg, protocol, split, matchers, device, root, overwrite=False):
    if split == "test" and protocol == "main" and not cfg.get("frozen", False):
        raise SystemExit("Refusing to run TEST pairs with an unfrozen config. "
                         "Run `python -m wbm.sweep --split dev --freeze` and pass --config configs/frozen.yaml.")
    table = load_table(cfg)
    ids = pair_ids(cfg, table, split)
    for m in matchers:
        s = settings_for(cfg, protocol, m)
        out = ensure_dir(Path(root) / protocol / split / m)
        ensure_dir(out / "matches")
        matcher = build_matcher(m, s, device)
        print(f"[{m}] {protocol}/{split}: {len(ids)} pairs, settings={s}")
        for pid in progress(ids, desc=m):
            f = out / f"{pid}.json"
            if f.exists() and not overwrite:
                continue
            row = table[pid]
            im0, im1 = read_pair(cfg, row)
            t0 = time.time()
            p0, p1 = matcher.match(im0, im1)
            dt = time.time() - t0
            rec = eval_matches(p0, p1, row["K0"], row["K1"], row["T01"], cfg["seeds"], s["ransac_thr"], cfg["epi_thr"])
            rec.update(pair_id=pid, matcher=m, time_s=dt, settings=s, rot_deg=row["rot_deg"])
            json.dump(rec, open(f, "w"))
            np.savez_compressed(out / "matches" / f"{pid}.npz", p0=np.asarray(p0, np.float32), p1=np.asarray(p1, np.float32))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/frozen.yaml")
    ap.add_argument("--protocol", default="main", choices=["main", "published"])
    ap.add_argument("--split", default="test", choices=["test", "dev", "full"])
    ap.add_argument("--matchers", nargs="+", default=["sift_mnn", "splg", "loftr"])
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    run(cfg, a.protocol, a.split, a.matchers, get_device(), a.results_root or cfg["results_root"], a.overwrite)
