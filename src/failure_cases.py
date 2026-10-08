"""CP4: ảnh minh hoạ failure case (results/figures/fail_*.png).

    python -m src.failure_cases
    python -m src.failure_cases --help

fail_01_ped_clutter_metric.png    calib ĐÚNG nhưng box hit rate của người đi bộ chỉ 82–84%: 3D box GT rộng hơn
                                  thân người nên ôm cả vật sát người (xe đạp dựng cạnh, tay, túi) mà 2D box
                                  bó sát thân không chứa (lớp Metric)
fail_02_trans_10cm_undetected.png lệch ty = 10 cm làm điểm của người ở 4.7 m dịch ~16 px, nhưng edge-score
                                  detector (cửa sổ 10 frame) vẫn coi calib là đúng (lớp Geometry + Metric)
fail_03_nusc_no_ego_motion.png    nuScenes, calib ĐÚNG nhưng bỏ bù chuyển động 35 ms giữa LiDAR và camera:
                                  điểm lệch khỏi vật khi xe đang rẽ (lớp Time)
Mỗi ảnh: xanh lá = điểm của object chiếu bằng calib đúng / rơi trong box, đỏ = chiếu bằng cấu hình lỗi / rơi ra ngoài,
khung xanh dương = 2D box GT.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.calib_metrics import Perturb, object_hits, pixel_shift, select_objects
from starter.datasets import load_frame
from starter.projection import cam_to_image, draw_box2d, velo_to_cam

GREEN, RED, BOX = (0, 200, 0), (0, 0, 255), (255, 200, 0)


def crop_object(img: np.ndarray, bbox, pad: int, scale: float) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox).astype(int)
    H, W = img.shape[:2]
    c = img[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
    return cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def draw_uv(img: np.ndarray, uv: np.ndarray, color, r: int = 2) -> None:
    for u, v in uv.astype(int):
        cv2.circle(img, (int(u), int(v)), r, color, -1)


def find_obj(fr: dict, obj_idx: int):
    for i, obj, m in select_objects(velo_to_cam(fr["points"][:, :3], fr["calib"]), fr["labels"]):
        if i == obj_idx:
            return obj, m
    raise KeyError(f"{fr['frame_id']}: object {obj_idx} không đủ điểm để đo")


def hit_panel(root: str, fid: str, obj_idx: int, pad: int = 50, scale: float = 2.5) -> tuple[np.ndarray, float]:
    """Điểm của object chiếu bằng calib đúng: xanh = trong 2D box GT, đỏ = ngoài."""
    fr = load_frame(root, fid)
    obj, m = find_obj(fr, obj_idx)
    uv, valid, hit = object_hits(fr["points"][m, :3], obj, fr["calib"], fr["image"].shape)
    vis = draw_box2d(fr["image"], obj.bbox, BOX)
    draw_uv(vis, uv[valid & hit], GREEN)
    draw_uv(vis, uv[valid & ~hit], RED)
    return crop_object(vis, obj.bbox, pad, scale), float(hit.mean())


def shift_panel(fr: dict, obj_idx: int, calib_bad, pad: int, scale: float) -> tuple[np.ndarray, float, float]:
    """Điểm của object chiếu bằng calib đúng (xanh) và calib lỗi (đỏ). Trả về (ảnh, hit lỗi, shift px)."""
    obj, m = find_obj(fr, obj_idx)
    pts, shape = fr["points"][m, :3], fr["image"].shape
    uv_ok, _, _ = cam_to_image(velo_to_cam(pts, fr["calib"]), fr["calib"].P2, shape)
    uv_bad, valid, hit = object_hits(pts, obj, calib_bad, shape)
    vis = draw_box2d(fr["image"], obj.bbox, BOX)
    draw_uv(vis, uv_ok, GREEN)
    draw_uv(vis, uv_bad[valid], RED)
    shift = float(np.median(pixel_shift(pts, fr["calib"], calib_bad, shape)))
    return crop_object(vis, obj.bbox, pad, scale), float(hit.mean()), shift


def hstack(images: list[np.ndarray]) -> np.ndarray:
    h = max(im.shape[0] for im in images)
    padded = [cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
              for im in images]
    return np.hstack(padded)


def caption(img: np.ndarray, lines: list[str], scale: float = 0.6, line_h: int = 26) -> np.ndarray:
    """Thêm thanh chữ phía trên; nới ảnh sang phải (nền trắng) nếu chữ dài hơn ảnh để không bị cắt."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    W = max(img.shape[1], max(cv2.getTextSize(s, font, scale, 1)[0][0] for s in lines) + 16)
    img = cv2.copyMakeBorder(img, 0, 0, 0, W - img.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    bar = np.full((line_h * len(lines), W, 3), 30, np.uint8)
    for k, s in enumerate(lines):
        cv2.putText(bar, s, (8, line_h * (k + 1) - 8), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def rotation_deg(Tr_a: np.ndarray, Tr_b: np.ndarray) -> float:
    """Góc (độ) giữa hai phép quay 3x3 trong hai extrinsic."""
    R = Tr_a[:, :3] @ Tr_b[:, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Tạo ảnh failure case cho topic A")
    ap.add_argument("--out-dir", default="results/figures", help="thư mục lưu ảnh")
    ap.add_argument("--results", default="results", help="thư mục CSV (đọc kết quả drift_detect cho fail_02)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # fail_01: metric box-hit báo nhầm với người đi bộ khi calib đúng
    panels = []
    for fid, idx in [("000048", 2), ("000015", 3)]:
        crop, hit = hit_panel("data/kitti_mini", fid, idx)
        panels.append(caption(crop, [f"KITTI {fid} ped #{idx}: hit {hit:.0%}"]))
        print(f"fail_01 {fid} ped #{idx}: hit (calib dung) = {hit:.1%}")
    img = caption(hstack(panels), ["fail_01: calib DUNG nhung hit rate cua nguoi di bo chi ~82%",
                                   "do = diem trong 3D box GT, cung do sau voi nguoi, cao 0.7-1.35 m,",
                                   "lech ra ngoai mep than 2-15 cm (xe dap dung sat / tay / tui):",
                                   "3D box rong 0.9 m bao ca phan nay, 2D box chi bo sat than nguoi"])
    cv2.imwrite(str(out / "fail_01_ped_clutter_metric.png"), img)

    # fail_02: lệch tịnh tiến 10 cm không bị detector phát hiện
    fr = load_frame("data/kitti_mini", "000049")
    crop, hit, shift = shift_panel(fr, 6, Perturb("ty", 0.10).apply(fr["calib"]), pad=40, scale=2.0)
    win = pd.read_csv(Path(args.results) / "drift_detect_kitti_windows.csv")
    w = win[win.config == "ty_10cm"]
    est = ", ".join(f"{r.best_axis} {r.best_delta_deg:+g}" for r in w.itertuples())
    flagged = "CO" if w.flag.any() else "KHONG"
    img = caption(crop, [f"fail_02: KITTI 000049, ty lech 10 cm",
                         f"nguoi o 4.7 m: diem dich {shift:.0f} px, hit {hit:.0%}",
                         f"edge-score detector (2 cua so 10 frame):",
                         f"goc bu tot nhat {est} deg -> {flagged} bao drift",
                         "xanh = calib dung, do = calib lech 10 cm"])
    cv2.imwrite(str(out / "fail_02_trans_10cm_undetected.png"), img)
    print(f"fail_02: shift {shift:.1f} px, hit {hit:.1%}, detector windows: {est}, flagged={flagged}")

    # fail_03: nuScenes bỏ bù ego-motion (lỗi Time) ở frame xe đang rẽ
    fid = "scene-1094_014"
    fr = load_frame("data/nuscenes_mini_subset", fid)
    calib_noego = load_frame("data/nuscenes_mini_subset", fid, use_ego_motion=False)["calib"]
    dt_ms = abs(fr["timestamp_camera_us"] - fr["timestamp_lidar_us"]) / 1000
    turn = rotation_deg(fr["calib"].Tr_velo_to_cam, calib_noego.Tr_velo_to_cam)   # xe quay bao nhiêu trong dt
    objs = select_objects(velo_to_cam(fr["points"][:, :3], fr["calib"]), fr["labels"])
    shifts = {i: np.median(pixel_shift(fr["points"][m, :3], fr["calib"], calib_noego, fr["image"].shape))
              for i, _, m in objs}
    worst = sorted(shifts, key=shifts.get, reverse=True)[:2]
    panels = []
    for i in worst:
        crop, hit, shift = shift_panel(fr, i, calib_noego, pad=40, scale=2.0)
        panels.append(caption(crop, [f"{fr['labels'][i].type} {fr['labels'][i].location[2]:.0f} m: "
                                     f"dich {shift:.0f} px, hit van {hit:.0%}"]))
        print(f"fail_03 {fid} obj {i}: shift {shift:.1f} px, hit {hit:.1%}")
    img = caption(hstack(panels), [f"fail_03: nuScenes {fid} (ban dem), calib DUNG nhung bo bu ego-motion",
                                   f"camera chup lech LiDAR {dt_ms:.1f} ms, trong luc do xe quay {turn:.2f} deg "
                                   f"({turn / dt_ms * 1000:.0f} deg/s) -> giong yaw drift ~{turn:.1f} deg",
                                   "xanh = co bu chuyen dong, do = khong bu. 2D box nuScenes rong nen hit van ~100%"])
    cv2.imwrite(str(out / "fail_03_nusc_no_ego_motion.png"), img)
    print(f"fail_03: dt {dt_ms:.1f} ms, ego turn {turn:.2f} deg")
    print("->", out / "fail_0*.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252) không in được tiếng Việt
    main()
