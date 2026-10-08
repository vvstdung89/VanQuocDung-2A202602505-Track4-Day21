"""CP2 demo: overlay điểm LiDAR lên ảnh ở 3 khoảng cách, và ảnh so sánh calib đúng / calib lệch.

    python -m src.demo_overlays                       # tạo tất cả ảnh demo mặc định
    python -m src.demo_overlays --help

Ảnh tạo ra (results/figures/):
  demo_distance_synthetic.png  synthetic 000000/000002/000004: xe ở 8 m, 18 m, 35 m
  demo_distance_kitti.png      KITTI 000031 (xe 8 m), 000009 (xe 24 m), 000004 (xe 38 m và 51 m)
  demo_yaw_drift_kitti.png     KITTI 000019 với yaw 0°, 1°, 3°: điểm của object xanh = trong 2D box, đỏ = rơi ra ngoài
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.calib_metrics import Perturb, object_hits, select_objects
from starter.datasets import load_frame
from starter.projection import draw_box2d, overlay_points, project_velo_to_image, velo_to_cam


def header(img: np.ndarray, text: str, height: int = 30) -> np.ndarray:
    bar = np.full((height, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def render(fr: dict, perturb: Perturb, title: str, show_hits: bool = False) -> tuple[np.ndarray, dict]:
    """Overlay toàn bộ điểm (màu theo độ sâu) + 2D box GT kèm khoảng cách.
    show_hits: tô lại điểm của từng object (xanh lá = rơi trong 2D box GT, đỏ = rơi ra ngoài)."""
    calib_true, img = fr["calib"], fr["image"]
    calib = perturb.apply(calib_true)
    uv, depth, _ = project_velo_to_image(fr["points"], calib, img.shape)
    vis = overlay_points(img, uv, depth, radius=1 if show_hits else 2)

    stats = {}
    objs = select_objects(velo_to_cam(fr["points"][:, :3], calib_true), fr["labels"])
    for i, obj, m in objs:
        lbl = f"{obj.type} {obj.location[2]:.0f}m"
        if show_hits:
            uv_o, valid, hit = object_hits(fr["points"][m, :3], obj, calib, img.shape)
            for (u, v), h in zip(uv_o[valid].astype(int), hit[valid]):
                cv2.circle(vis, (int(u), int(v)), 2, (0, 220, 0) if h else (0, 0, 255), -1)
            stats[lbl] = hit.mean()
        vis = draw_box2d(vis, obj.bbox, color=(255, 255, 0) if show_hits else (0, 255, 0), label=lbl)
    if show_hits:
        title += " | hit: " + ", ".join(f"{k} {v:.0%}" for k, v in stats.items())
    return header(vis, title), stats


def stack(images: list[np.ndarray]) -> np.ndarray:
    w = min(im.shape[1] for im in images)
    return np.vstack([cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1]))) if im.shape[1] != w else im
                      for im in images])


def main() -> None:
    ap = argparse.ArgumentParser(description="Tạo ảnh demo overlay LiDAR-camera cho topic A")
    ap.add_argument("--out-dir", default="results/figures", help="thư mục lưu ảnh")
    ap.add_argument("--drift-frame", default="000019", help="frame KITTI dùng cho ảnh so sánh yaw drift")
    ap.add_argument("--drift-yaws", type=float, nargs="+", default=[0.0, 1.0, 3.0], help="các mức yaw (độ)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Perturb("baseline", 0.0)

    panels = {
        "demo_distance_synthetic.png": ("data/synthetic", [("000000", "xe 8 m"), ("000002", "xe 18 m"), ("000004", "xe 35 m")]),
        "demo_distance_kitti.png": ("data/kitti_mini", [("000031", "xe gan 8 m"), ("000009", "xe 24 m"), ("000004", "xe xa 38 m, 51 m")]),
    }
    for fname, (root, frames) in panels.items():
        imgs = [render(load_frame(root, f), base, f"{Path(root).name} {f}: {desc}, calib goc")[0] for f, desc in frames]
        cv2.imwrite(str(out / fname), stack(imgs))
        print("->", out / fname)

    fr = load_frame("data/kitti_mini", args.drift_frame)
    imgs = []
    for yaw in args.drift_yaws:
        p = Perturb("yaw", yaw) if yaw else base
        img, stats = render(fr, p, f"kitti_mini {args.drift_frame} yaw {yaw:g} deg (xanh=trong box, do=ngoai)",
                            show_hits=True)
        imgs.append(img)
        print(f"yaw={yaw:g}: hit rate theo object = " + ", ".join(f"{k} {v:.0%}" for k, v in stats.items()))
    cv2.imwrite(str(out / "demo_yaw_drift_kitti.png"), stack(imgs))
    print("->", out / "demo_yaw_drift_kitti.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252) không in được tiếng Việt
    main()
