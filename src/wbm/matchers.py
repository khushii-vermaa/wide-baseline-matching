"""The three matchers, used as released. Each returns matched pixel coords in ORIGINAL image coordinates."""
import cv2
import numpy as np

NAMES = ["sift_mnn", "sift_ratio", "splg", "loftr"]
LABELS = {"sift_mnn": "SIFT + MNN", "sift_ratio": "SIFT + ratio test", "splg": "SuperPoint + LightGlue", "loftr": "LoFTR"}


def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resize_long(img, long_side, multiple=1):
    """Resize so the long side == long_side (optionally rounding dims to a multiple). Returns img, (sx, sy) = orig/new."""
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    if multiple > 1:
        nw, nh = max(multiple, nw // multiple * multiple), max(multiple, nh // multiple * multiple)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp), (w / nw, h / nh)


class SiftMatcher:
    def __init__(self, resize, max_kp, ratio=None, **_):
        self.resize, self.ratio = resize, ratio
        self.sift = cv2.SIFT_create(nfeatures=int(max_kp))

    def _feat(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g, sc = resize_long(g, self.resize)
        kp, des = self.sift.detectAndCompute(g, None)
        pts = np.array([k.pt for k in kp], np.float32).reshape(-1, 2) * np.array(sc, np.float32)
        return pts, des

    def match(self, img0, img1):
        (p0, d0), (p1, d1) = self._feat(img0), self._feat(img1)
        if d0 is None or d1 is None or len(d0) < 2 or len(d1) < 2:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        if self.ratio is None:                                   # mutual nearest neighbour
            m = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(d0, d1)
            i0 = np.array([x.queryIdx for x in m], int); i1 = np.array([x.trainIdx for x in m], int)
        else:                                                    # Lowe ratio test
            knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0, d1, k=2)
            good = [a for a, b in (x for x in knn if len(x) == 2) if a.distance < self.ratio * b.distance]
            i0 = np.array([x.queryIdx for x in good], int); i1 = np.array([x.trainIdx for x in good], int)
        if len(i0) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        return p0[i0], p1[i1]


class SuperPointLightGlue:
    def __init__(self, resize, max_kp, device=None, **_):
        import torch
        from lightglue import LightGlue, SuperPoint
        self.torch, self.resize = torch, int(resize)
        self.device = device or get_device()
        self.ext = SuperPoint(max_num_keypoints=int(max_kp)).eval().to(self.device)
        self.lg = LightGlue(features="superpoint").eval().to(self.device)

    def _tensor(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)

    def match(self, img0, img1):
        from lightglue.utils import rbd
        with self.torch.no_grad():
            f0 = self.ext.extract(self._tensor(img0), resize=self.resize)
            f1 = self.ext.extract(self._tensor(img1), resize=self.resize)
            m01 = self.lg({"image0": f0, "image1": f1})
            f0, f1, m01 = [rbd(x) for x in (f0, f1, m01)]
            m = m01["matches"]
            return (f0["keypoints"][m[..., 0]].cpu().numpy().astype(np.float32),
                    f1["keypoints"][m[..., 1]].cpu().numpy().astype(np.float32))


class LoFTRMatcher:
    def __init__(self, resize, device=None, pretrained="outdoor", **_):
        import torch
        import kornia.feature as KF
        self.torch, self.resize = torch, int(resize)
        self.device = device or get_device()
        self.model = KF.LoFTR(pretrained=pretrained).eval().to(self.device)

    def _prep(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g, sc = resize_long(g, self.resize, multiple=8)
        t = self.torch.from_numpy(g)[None, None].float().div(255).to(self.device)
        return t, np.array(sc, np.float32)

    def match(self, img0, img1):
        (t0, s0), (t1, s1) = self._prep(img0), self._prep(img1)
        with self.torch.no_grad():
            out = self.model({"image0": t0, "image1": t1})
        return out["keypoints0"].cpu().numpy() * s0, out["keypoints1"].cpu().numpy() * s1


def build_matcher(name, s, device=None):
    if name == "sift_mnn":
        return SiftMatcher(s["resize"], s["max_kp"], ratio=None)
    if name == "sift_ratio":
        return SiftMatcher(s["resize"], s["max_kp"], ratio=0.8)
    if name == "splg":
        return SuperPointLightGlue(s["resize"], s["max_kp"], device)
    if name == "loftr":
        return LoFTRMatcher(s["resize"], device)
    raise ValueError(f"unknown matcher {name}")


def available():
    """Which matchers can run in this environment (used by the API)."""
    out = {"sift_mnn": True, "sift_ratio": True}
    try:
        import torch, lightglue  # noqa
        out["splg"] = True
    except Exception:
        out["splg"] = False
    try:
        import torch, kornia.feature  # noqa
        out["loftr"] = True
    except Exception:
        out["loftr"] = False
    return out
