"""Negative control: metric unit tests with hand-computed answers + synthetic-geometry end-to-end checks."""
import numpy as np
import cv2
from wbm.geometry import pose_error, error_auc, rotation_angle_deg, sampson_sq, relative_pose
from wbm.pose import estimate_pose
from wbm.rotate import rotate_pose, rotate_points


def Rz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_rotation_angle():
    assert abs(rotation_angle_deg(Rz(30)) - 30) < 1e-6
    assert abs(rotation_angle_deg(np.eye(3))) < 1e-6


def test_pose_error_known():
    t = np.array([1.0, 0, 0])
    e, eR, et = pose_error(Rz(7), t, np.eye(3), t)
    assert abs(eR - 7) < 1e-6 and abs(et) < 1e-6 and abs(e - 7) < 1e-6
    e, eR, et = pose_error(np.eye(3), -t, np.eye(3), t)          # sign ambiguity must not matter
    assert abs(et) < 1e-6
    e, eR, et = pose_error(np.eye(3), np.array([0, 1.0, 0]), np.eye(3), t)
    assert abs(et - 90) < 1e-6


def test_auc_hand_computed():
    # Trapezoid convention (as in LoFTR/LightGlue eval). errors [2, 4, 100], tau = 5.
    # points (err, recall): (0,0) (2,1/3) (4,2/3) (5,2/3)
    # areas: 0.5*(0+1/3)*2 = 1/3 ; 0.5*(1/3+2/3)*2 = 1 ; (5-4)*2/3 = 2/3  -> total 2 -> AUC@5 = 2/5 = 0.4
    a = error_auc([2, 4, 100], [5])[5]
    assert abs(a - 0.4) < 1e-9
    assert error_auc([np.inf, np.inf], [5, 10]) == {5: 0.0, 10: 0.0}
    assert abs(error_auc([0.0, 0.0], [10])[10] - 1.0) < 1e-9


def make_scene(seed=0, n=300):
    rs = np.random.RandomState(seed)
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.0]])
    X = np.c_[rs.uniform(-2, 2, n), rs.uniform(-1.5, 1.5, n), rs.uniform(4, 8, n)]
    R = cv2.Rodrigues(np.radians([5, 20, 3]))[0]
    t = np.array([0.8, 0.1, 0.2])
    T01 = np.eye(4); T01[:3, :3] = R; T01[:3, 3] = t
    def proj(Xc): u = (K @ Xc.T).T; return u[:, :2] / u[:, 2:]
    return K, X, T01, proj(X), proj(X @ R.T + t)


def test_estimate_pose_recovers_gt():
    K, X, T01, p0, p1 = make_scene()
    est = estimate_pose(p0, p1, K, K, 0.5, seed=0)
    e, eR, et = pose_error(est["R"], est["t"], T01[:3, :3], T01[:3, 3])
    assert e < 1.0, e


def test_random_matches_fail():
    K, X, T01, p0, p1 = make_scene()
    rs = np.random.RandomState(1)
    bad = []
    for s in range(10):
        q0 = rs.rand(300, 2) * [640, 480]; q1 = rs.rand(300, 2) * [640, 480]
        est = estimate_pose(q0, q1, K, K, 1.0, seed=s)
        bad.append(pose_error(est["R"], est["t"], T01[:3, :3], T01[:3, 3])[0] if est else np.inf)
    assert np.mean(np.array(bad) < 10) < 0.2


def test_sampson_zero_on_true_matches():
    K, X, T01, p0, p1 = make_scene()
    assert np.max(sampson_sq(p0, p1, K, K, T01[:3, :3], T01[:3, 3])) < 1e-12


def test_pair_order_symmetry():
    K, X, T01, p0, p1 = make_scene()
    est_ab = estimate_pose(p0, p1, K, K, 0.5, 0)
    est_ba = estimate_pose(p1, p0, K, K, 0.5, 0)
    T10 = np.linalg.inv(T01)
    e1 = pose_error(est_ab["R"], est_ab["t"], T01[:3, :3], T01[:3, 3])[0]
    e2 = pose_error(est_ba["R"], est_ba["t"], T10[:3, :3], T10[:3, 3])[0]
    assert e1 < 1 and e2 < 1


def test_relative_pose_convention():
    # world-to-camera poses: X_c = R X_w + t
    T0 = np.eye(4); T1 = np.eye(4); T1[:3, :3] = Rz(10); T1[:3, 3] = [1, 0, 0]
    T01 = relative_pose(T0, T1)
    assert np.allclose(T01, T1)


def test_inplane_rotation_updates_gt_pose():
    """Rotating image-1 pixels about the principal point must keep GT epipolar error ~0 under the UPDATED pose."""
    K, X, T01, p0, p1 = make_scene()
    for ang in (45, 90, 135, 180, -30):
        q1 = rotate_points(p1, K, ang)
        T = rotate_pose(T01, ang)
        assert np.max(sampson_sq(p0, q1, K, K, T[:3, :3], T[:3, 3])) < 1e-10, ang
        wrong = rotate_pose(T01, -ang) if ang % 180 else T01 * 0 + np.eye(4)
        if ang % 180:
            assert np.max(sampson_sq(p0, q1, K, K, wrong[:3, :3], wrong[:3, 3])) > 1e-6
