"""FastAPI backend: live matching demo + benchmark results for the frontend."""
import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from wbm.geometry import rotation_angle_deg
from wbm.matchers import LABELS, available, build_matcher, get_device
from wbm.pose import estimate_pose

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("RESULTS_ROOT", ROOT / "results"))
MAX_BYTES = 12 * 1024 * 1024
app = FastAPI(title="Wide-baseline matching demo", version="1.0")
_cache = {}


def get_matcher(name, resize, max_kp):
    key = (name, resize, max_kp)
    if key not in _cache:
        if len(_cache) >= 4:                       # keep memory bounded on small servers
            _cache.pop(next(iter(_cache)))
        _cache[key] = build_matcher(name, {"resize": resize, "max_kp": max_kp}, get_device())
    return _cache[key]


def decode(upload: UploadFile):
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 12 MB.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"Could not read {upload.filename} as an image.")
    return img


def guess_K(img, focal):
    h, w = img.shape[:2]
    f = focal if focal > 0 else 1.2 * max(h, w)   # rough prior when real intrinsics are unknown
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def draw(im0, im1, p0, p1, inl, H=480, max_lines=300):
    def rs(im):
        s = H / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * s), H)), s
    a, sa = rs(im0); b, sb = rs(im1)
    canvas = np.hstack([a, b])
    idx = np.random.RandomState(0).permutation(len(p0))[:max_lines]
    for i in sorted(idx, key=lambda k: inl[k]):     # draw inliers last (on top)
        x0, y0 = (p0[i] * sa).astype(int); x1, y1 = (p1[i] * sb).astype(int)
        col = (90, 190, 70) if inl[i] else (60, 60, 220)
        cv2.line(canvas, (x0, y0), (x1 + a.shape[1], y1), col, 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


@app.get("/api/health")
def health():
    return {"status": "ok", "device": get_device()}


@app.get("/api/matchers")
def matchers():
    av = available()
    return [{"id": k, "label": LABELS[k], "available": av[k]} for k in ["sift_mnn", "sift_ratio", "splg", "loftr"]]


@app.post("/api/match")
def match(img0: UploadFile = File(...), img1: UploadFile = File(...), matcher: str = Form("sift_mnn"),
          resize: int = Form(640), max_kp: int = Form(2048), ransac_thr: float = Form(1.0), focal: float = Form(0.0)):
    if matcher not in LABELS:
        raise HTTPException(400, "Unknown matcher.")
    if not available().get(matcher):
        raise HTTPException(400, f"{LABELS[matcher]} is not installed on this server (install torch, kornia, lightglue).")
    resize = int(np.clip(resize, 256, 1600)); max_kp = int(np.clip(max_kp, 256, 8192))
    a, b = decode(img0), decode(img1)
    K0, K1 = guess_K(a, focal), guess_K(b, focal)
    t0 = time.time()
    p0, p1 = get_matcher(matcher, resize, max_kp).match(a, b)
    est = estimate_pose(p0, p1, K0, K1, ransac_thr, seed=0)
    dt = time.time() - t0
    if est is None:
        return {"matcher": LABELS[matcher], "n_matches": int(len(p0)), "n_inliers": 0, "pose": None, "time_s": dt,
                "image": draw(a, b, p0, p1, np.zeros(len(p0), bool)) if len(p0) else None,
                "message": "Not enough consistent matches to estimate a camera pose."}
    return {
        "matcher": LABELS[matcher], "n_matches": int(len(p0)), "n_inliers": est["n_inliers"],
        "inlier_ratio": est["n_inliers"] / max(len(p0), 1), "time_s": dt,
        "pose": {"rotation_deg": rotation_angle_deg(est["R"]), "translation_dir": (est["t"] / np.linalg.norm(est["t"])).tolist()},
        "image": draw(a, b, p0, p1, est["inliers"]),
        "message": "Camera intrinsics are guessed (focal = 1.2 x long side) unless you set a focal length, so the pose is approximate.",
    }


@app.get("/api/results")
def results():
    f = RESULTS / "summary.json"
    if not f.exists():
        return {"available": False}
    s = json.load(open(f)); s["available"] = True
    return s


RESULTS.mkdir(parents=True, exist_ok=True)
app.mount("/results-files", StaticFiles(directory=str(RESULTS)), name="results")
app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")
