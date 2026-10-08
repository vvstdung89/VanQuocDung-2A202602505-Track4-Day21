"""Tự kiểm tra 2 hàm TODO(CP2). Chạy từ gốc repo: python -m src.test_projection

Nguồn: khung test lấy từ codelab Day 6 (VLearn, Phần 04 mục 4.3), thêm kiểm tra điểm Inf.
"""
import sys

import numpy as np

from starter.datasets import load_frame
from starter.projection import cam_to_image, velo_to_cam

sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252) không in được ✅
np.set_printoptions(suppress=True, precision=2)
fr = load_frame("data/synthetic", "000000")
calib, shape = fr["calib"], fr["image"].shape

pts = np.array([
    [10.0, 0.0, 0.0],     # 10 m phía trước -> hợp lệ
    [np.nan, 0.0, 0.0],   # điểm lỗi NaN -> loại
    [-10.0, 0.0, 0.0],    # 10 m phía sau xe -> loại (z_cam < 0)
    [10.0, 50.0, 0.0],    # 50 m bên trái -> loại (ngoài khung hình)
    [np.inf, 0.0, 0.0],   # điểm lỗi Inf -> loại
])
cam = velo_to_cam(pts, calib)
uv, depth, mask = cam_to_image(cam, calib.P2, shape)

print("camera frame:\n", np.round(cam, 2))
print("uv:", np.round(uv, 1), " depth:", np.round(depth, 2), " mask:", mask)

assert cam.shape == (5, 3), f"velo_to_cam phải trả về (N, 3), đang là {cam.shape}"
assert abs(cam[0, 2] - 9.73) < 0.01, f"z_cam của (10,0,0) phải ≈ 9.73, đang là {cam[0, 2]}"
assert cam[2, 2] < 0, "điểm phía sau xe phải có z_cam âm"
assert mask.tolist() == [True, False, False, False, False], f"mask sai: {mask}"
assert uv.shape == (1, 2) and depth.shape == (1,), "uv phải (M, 2), depth phải (M,)"
assert np.allclose(uv[0], [614, 175], atol=1), f"pixel của (10,0,0) phải ≈ (614, 175), đang là {uv[0]}"
print("✅ CP2 self-test passed")
