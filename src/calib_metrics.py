"""Các metric đo độ khớp LiDAR-camera, dùng chung cho demo, sweep và failure case.

Hai nhóm metric:
1. Cần label (dùng để đo trong thí nghiệm):
   - box hit rate: lấy điểm LiDAR nằm trong 3D box GT (xác định bằng calib ĐÚNG), chiếu bằng calib
     ĐANG KIỂM TRA, đếm tỉ lệ điểm rơi vào 2D box GT của chính object đó.
   - pixel shift: độ dịch (pixel) của cùng một điểm giữa calib đúng và calib đang kiểm tra.
2. Không cần label (dùng được khi xe chạy thật):
   - edge alignment score: tỉ lệ điểm "biên độ sâu" của LiDAR (điểm tiền cảnh nằm cạnh điểm xa hơn hẳn)
     rơi gần biên Canny của ảnh. Calib đúng thì biên vật thể trong LiDAR trùng biên trong ảnh.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import cam_to_image, perturb_extrinsic, velo_to_cam

DIST_BINS = (("near", 0.0, 15.0), ("mid", 15.0, 30.0), ("far", 30.0, np.inf))


# ---------------------------------------------------------------- cấu hình perturb

@dataclass(frozen=True)
class Perturb:
    """Một mức lệch calibration, chỉ đổi đúng 1 yếu tố so với baseline."""
    axis: str            # baseline / roll / pitch / yaw / tx / ty / tz
    level: float         # độ (roll/pitch/yaw) hoặc mét (tx/ty/tz)

    @property
    def name(self) -> str:
        if self.axis == "baseline":
            return "baseline"
        unit = "deg" if self.axis in ("roll", "pitch", "yaw") else "cm"
        val = self.level if unit == "deg" else self.level * 100
        return f"{self.axis}_{val:g}{unit}"

    def apply(self, calib: KittiCalib) -> KittiCalib:
        if self.axis == "baseline":
            return calib
        rot = {k: self.level if self.axis == k else 0.0 for k in ("roll", "pitch", "yaw")}
        t = tuple(self.level if self.axis == k else 0.0 for k in ("tx", "ty", "tz"))
        return perturb_extrinsic(calib, rot["roll"], rot["pitch"], rot["yaw"], t)


def default_grid(rot_levels_deg=(0.5, 1.0, 2.0, 3.0), trans_levels_m=(0.02, 0.05, 0.10)) -> list[Perturb]:
    grid = [Perturb("baseline", 0.0)]
    grid += [Perturb(a, lv) for a in ("yaw", "pitch", "roll") for lv in rot_levels_deg]
    grid += [Perturb(a, lv) for a in ("tx", "ty", "tz") for lv in trans_levels_m]
    return grid


# ---------------------------------------------------------------- metric dùng label

def points_in_box3d(points_cam: np.ndarray, obj: KittiObject, ground_margin_m: float = 0.1) -> np.ndarray:
    """Mask (N,) các điểm (rectified camera frame) nằm trong 3D box KITTI.

    Đổi điểm sang hệ toạ độ của box (ngược phép xoay trong `box3d_corners_cam`):
    local = R^T (p - location), rồi so với nửa kích thước. Bỏ `ground_margin_m` sát đáy box
    để không đếm nhầm điểm mặt đường.
    """
    h, w, l = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    with np.errstate(invalid="ignore"):
        local = (points_cam - obj.location) @ R          # hàng: (R^T v)^T = v^T R
        return ((np.abs(local[:, 0]) <= l / 2) & (np.abs(local[:, 2]) <= w / 2) &
                (local[:, 1] <= -ground_margin_m) & (local[:, 1] >= -h))


def dist_bin(depth_m: float) -> str:
    return next(name for name, lo, hi in DIST_BINS if lo <= depth_m < hi)


def select_objects(points_cam_true: np.ndarray, labels: list[KittiObject], min_points: int = 10,
                   max_truncated: float = 0.3) -> list[tuple[int, KittiObject, np.ndarray]]:
    """Các object đủ điểm LiDAR để đo: trả về (chỉ số label, object, mask điểm)."""
    out = []
    for i, obj in enumerate(labels):
        x1, y1, x2, y2 = obj.bbox
        if obj.truncated > max_truncated or x2 <= x1 or y2 <= y1:
            continue
        m = points_in_box3d(points_cam_true, obj)
        if m.sum() >= min_points:
            out.append((i, obj, m))
    return out


def inside_bbox(uv: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)


def object_hits(points_xyz: np.ndarray, obj: KittiObject, calib_test: KittiCalib,
                image_shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chiếu điểm của một object bằng calib đang kiểm tra.
    Trả về (uv, valid, hit): valid = chiếu được vào ảnh, hit = valid và nằm trong 2D box GT."""
    uv_valid, _, valid = cam_to_image(velo_to_cam(points_xyz, calib_test), calib_test.P2, image_shape)
    uv = np.full((len(points_xyz), 2), np.nan)
    uv[valid] = uv_valid
    hit = valid.copy()
    hit[valid] = inside_bbox(uv_valid, obj.bbox)
    return uv, valid, hit


def pixel_shift(points_xyz: np.ndarray, calib_true: KittiCalib, calib_test: KittiCalib,
                image_shape) -> np.ndarray:
    """Khoảng cách pixel giữa vị trí chiếu bằng calib đúng và calib lệch (chỉ điểm chiếu được ở cả hai)."""
    uv0, _, m0 = cam_to_image(velo_to_cam(points_xyz, calib_true), calib_true.P2, image_shape)
    uv1, _, m1 = cam_to_image(velo_to_cam(points_xyz, calib_test), calib_test.P2, image_shape)
    full0 = np.full((len(points_xyz), 2), np.nan)
    full1 = np.full((len(points_xyz), 2), np.nan)
    full0[m0], full1[m1] = uv0, uv1
    both = m0 & m1
    return np.linalg.norm(full1[both] - full0[both], axis=1)


# ---------------------------------------------------------------- metric không cần label

def deg_to_px(calib: KittiCalib, deg: float) -> int:
    """Đổi một góc nhỏ sang pixel theo tiêu cự fx của camera (KITTI fx≈721, nuScenes fx≈1266)."""
    return max(1, int(round(calib.P2[0, 0] * np.tan(np.radians(deg)))))


def image_edge_distance(image: np.ndarray, canny_lo: int = 50, canny_hi: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Biên Canny của ảnh và distance transform: dt[v, u] = khoảng cách (px) tới pixel biên gần nhất."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    dt = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return edges, dt


def depth_edge_mask(uv: np.ndarray, depth: np.ndarray, image_shape, win_px: int,
                    min_jump_m: float = 1.0, rel_jump: float = 0.1) -> np.ndarray:
    """Đánh dấu điểm LiDAR nằm ở biên che khuất (tiền cảnh cạnh một điểm xa hơn hẳn).

    Dựng ảnh độ sâu thưa (max depth mỗi pixel), lấy max trong cửa sổ NGANG 3 x (2*win_px+1)
    quanh mỗi điểm. Điểm là biên nếu hàng xóm xa hơn nó > max(min_jump_m, rel_jump * depth).
    Chỉ tìm theo chiều ngang vì hai điểm cùng ring nằm cạnh nhau theo chiều ngang; theo chiều dọc
    hai ring kề nhau trên mặt đường cũng chênh độ sâu lớn và sẽ bị nhận nhầm là biên.
    """
    H, W = image_shape[:2]
    ui, vi = uv[:, 0].astype(int), uv[:, 1].astype(int)
    far = np.zeros((H, W), np.float32)
    np.maximum.at(far, (vi, ui), depth.astype(np.float32))
    neigh = cv2.dilate(far, np.ones((3, 2 * win_px + 1), np.uint8))
    return (neigh[vi, ui] - depth) > np.maximum(min_jump_m, rel_jump * depth)


def edge_alignment(points_xyz: np.ndarray, calib: KittiCalib, image_shape, dt: np.ndarray,
                   win_deg: float = 0.6, tol_deg: float = 0.25) -> dict[str, float]:
    """Edge alignment score của một calib trên một frame.

    score = tỉ lệ điểm biên độ sâu rơi trong tol (px) của biên Canny.
    lift  = score / tỉ lệ pixel nằm trong tol của biên Canny (trong vùng hàng mà điểm biên phủ),
            tức là so với việc rải điểm ngẫu nhiên: lift≈1 nghĩa là không khớp hơn ngẫu nhiên.
    """
    uv, depth, _ = cam_to_image(velo_to_cam(points_xyz, calib), calib.P2, image_shape)
    edge = depth_edge_mask(uv, depth, image_shape, deg_to_px(calib, win_deg))
    tol = deg_to_px(calib, tol_deg)
    ue, ve = uv[edge, 0].astype(int), uv[edge, 1].astype(int)
    if len(ue) < 20:
        return {"edge_points": int(len(ue)), "edge_score": np.nan, "edge_lift": np.nan}
    score = float((dt[ve, ue] <= tol).mean())
    lo, hi = np.percentile(ve, [5, 95]).astype(int)
    p_rand = float((dt[lo:hi + 1] <= tol).mean())
    return {"edge_points": int(len(ue)), "edge_score": score, "edge_lift": score / max(p_rand, 1e-6)}
