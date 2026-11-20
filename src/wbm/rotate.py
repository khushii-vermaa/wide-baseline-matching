"""O7 stretch: rotate image 1 in-plane and update the ground-truth pose accordingly."""
import cv2
import numpy as np


def rotate_pose(T01, angle_cv_deg):
    """angle_cv_deg uses cv2.getRotationMatrix2D's convention (positive = counter-clockwise on screen).
    In (x right, y down) coordinates that is a 2-D rotation by theta = -angle. Image-1 camera rays rotate by
    Rz(theta) (valid for fx ~= fy and principal point at the rotation centre), so T01' = Rz @ T01."""
    th = np.radians(-angle_cv_deg)
    c, s = np.cos(th), np.sin(th)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = Rz @ T01[:3, :3]
    T[:3, 3] = Rz @ T01[:3, 3]
    return T


def rotate_image(img, K, angle_cv_deg):
    """Rotate about the principal point, keeping canvas size (corners are cropped). Returns img, K (unchanged)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((float(K[0, 2]), float(K[1, 2])), angle_cv_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0), K


def rotate_points(p, K, angle_cv_deg):
    M = cv2.getRotationMatrix2D((float(K[0, 2]), float(K[1, 2])), angle_cv_deg, 1.0)
    return p @ M[:, :2].T + M[:, 2]
