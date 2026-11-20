"""MegaDepth-1500 loader.

Expected input: the LoFTR scene-info files (assets/megadepth_test_1500_scene_info/*.npz) with keys
  pair_infos, image_paths, intrinsics, poses   (poses are 4x4 WORLD-TO-CAMERA)
and the MegaDepth test images. CONFIRM in Phase 0 with:  python -m wbm.data --inspect
"""
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from .config import load_config, ensure_dir
from .geometry import relative_pose, rotation_angle_deg

_BASENAME_INDEX = {}


def resolve_image(root, rel):
    root = Path(root)
    for cand in (root / rel, root / "test" / rel, root / "Undistorted_SfM" / Path(rel).name):
        if cand.exists():
            return str(cand)
    key = str(root)
    if key not in _BASENAME_INDEX:   # last resort: index every jpg under root once
        _BASENAME_INDEX[key] = {p.name: str(p) for p in root.rglob("*.jpg")}
    return _BASENAME_INDEX[key].get(Path(rel).name)


def build_table(cfg):
    info_dir = Path(cfg["scene_info_dir"])
    files = sorted(info_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz scene-info files in {info_dir}. See README, Phase 0.")
    table = {}
    for f in files:
        scene = f.stem
        si = np.load(f, allow_pickle=True)
        for k, pinfo in enumerate(si["pair_infos"]):
            (i0, i1) = pinfo[0]
            T01 = relative_pose(np.asarray(si["poses"][i0], float), np.asarray(si["poses"][i1], float))
            pid = f"{scene}_{k:04d}"
            table[pid] = dict(
                pair_id=pid, scene=scene,
                img0=str(si["image_paths"][i0]), img1=str(si["image_paths"][i1]),
                K0=np.asarray(si["intrinsics"][i0], float), K1=np.asarray(si["intrinsics"][i1], float),
                T01=T01, overlap=float(pinfo[1]), rot_deg=rotation_angle_deg(T01[:3, :3]),
            )
    return table


def load_table(cfg):
    cache = ensure_dir(cfg["cache_dir"]) / "pair_table.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    table = build_table(cfg)
    pickle.dump(table, open(cache, "wb"))
    return table


def read_pair(cfg, row):
    out = []
    for key in ("img0", "img1"):
        p = resolve_image(cfg["data_root"], row[key])
        if p is None:
            raise FileNotFoundError(f"Image not found under {cfg['data_root']}: {row[key]}")
        out.append(cv2.imread(p, cv2.IMREAD_COLOR))
    return out


def inspect(cfg):
    t = build_table(cfg)
    rot = np.array([r["rot_deg"] for r in t.values()])
    scenes = sorted({r["scene"] for r in t.values()})
    print(f"pairs: {len(t)}   scenes: {scenes}")
    print(f"rotation deg  min {rot.min():.1f}  median {np.median(rot):.1f}  max {rot.max():.1f}")
    first = next(iter(t.values()))
    print("sample pair:", first["pair_id"], first["img0"])
    print("K0=\n", first["K0"])
    miss = sum(resolve_image(cfg["data_root"], r["img0"]) is None for r in list(t.values())[:50])
    print(f"images missing in first 50 pairs: {miss}")
    print("Pose convention assumed: world-to-camera (T_0to1 = T1 @ inv(T0)). "
          "The random-match / symmetry / reproduction checks will catch a wrong assumption.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.inspect:
        inspect(cfg)
