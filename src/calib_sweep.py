"""CP3: sweep calibration drift, đo projection mismatch trên mọi frame của một dataset.

    python -m src.calib_sweep --data-root data/kitti_mini --out results/calib_sweep_kitti.csv
    python -m src.calib_sweep --data-root data/nuscenes_mini_subset --out results/calib_sweep_nusc.csv
    python -m src.calib_sweep --help

Mỗi cấu hình chỉ đổi ĐÚNG MỘT yếu tố so với calib gốc: yaw/pitch/roll (độ) hoặc tx/ty/tz (mét, trong
LiDAR frame). Frame, object, ngưỡng và tham số metric giữ nguyên. Không có phép ngẫu nhiên nào nên chạy
lại luôn ra cùng số. Với nuScenes, thêm một cấu hình "no_ego_motion": calib đúng nhưng bỏ bù chuyển động
xe giữa thời điểm chụp LiDAR và camera (lỗi lớp Time).

Ghi ra 2 file:
  <out>.csv          mỗi dòng = (frame, cấu hình): fov_pct, hit_pct, shift_px_p50, edge_score
  <out>_objects.csv  mỗi dòng = (frame, cấu hình, object): khoảng cách, số điểm, hit_pct, shift_px_p50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.calib_metrics import (Perturb, default_grid, dist_bin, edge_score, image_edge_distance,
                               lidar_depth_edges, object_hits, pixel_shift, select_objects)
from starter.datasets import dataset_type, list_frames, load_frame
from starter.kitti_io import KittiCalib
from starter.projection import project_velo_to_image, velo_to_cam


def eval_config(fr: dict, pts: np.ndarray, objs, dt: np.ndarray, calib: KittiCalib,
                cfg: dict) -> tuple[dict, list[dict]]:
    """Đo mọi metric của một calib (calib đang kiểm tra) trên một frame."""
    calib0, shape = fr["calib"], fr["image"].shape
    _, _, in_img = project_velo_to_image(pts, calib, shape)
    shift_all = pixel_shift(pts, calib0, calib, shape)

    obj_rows = []
    for i, obj, m in objs:
        _, _, hit = object_hits(pts[m], obj, calib, shape)
        shift = pixel_shift(pts[m], calib0, calib, shape)
        depth = float(obj.location[2])
        obj_rows.append({**cfg, "obj_idx": i, "type": obj.type, "depth_m": round(depth, 2),
                         "dist_bin": dist_bin(depth), "n_points": int(m.sum()),
                         "hit_pct": 100 * hit.mean(),
                         "shift_px_p50": float(np.median(shift)) if len(shift) else np.nan})

    row = {**cfg, "n_points": len(pts), "fov_pct": 100 * in_img.mean(), "n_objects": len(objs),
           "hit_pct": float(np.mean([r["hit_pct"] for r in obj_rows])) if obj_rows else np.nan,
           "shift_px_p50": float(np.median(shift_all)) if len(shift_all) else np.nan,
           "edge_score": edge_score(lidar_depth_edges(pts, calib, shape), calib, shape, dt)}
    return row, obj_rows


def sweep(data_root: str, frames: list[str], grid: list[Perturb], time_check: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = Path(data_root).name
    is_nusc = dataset_type(data_root) == "nuscenes"
    rows, obj_rows = [], []
    for k, fid in enumerate(frames):
        fr = load_frame(data_root, fid)
        pts = fr["points"][:, :3]
        objs = select_objects(velo_to_cam(pts, fr["calib"]), fr["labels"])  # điểm của object: luôn theo calib ĐÚNG
        _, dt = image_edge_distance(fr["image"])                              # biên ảnh không phụ thuộc calib

        configs = [({"axis": p.axis, "level": p.level, "config": p.name}, p.apply(fr["calib"])) for p in grid]
        if is_nusc and time_check:
            dt_ms = abs(fr["timestamp_camera_us"] - fr["timestamp_lidar_us"]) / 1000
            calib_noego = load_frame(data_root, fid, use_ego_motion=False)["calib"]
            configs.append(({"axis": "time", "level": round(dt_ms, 1), "config": "no_ego_motion"}, calib_noego))

        for cfg, calib in configs:
            cfg = {"dataset": ds, "frame_id": fid, **cfg}
            r, o = eval_config(fr, pts, objs, dt, calib, cfg)
            rows.append(r)
            obj_rows.extend(o)
        print(f"[{k + 1}/{len(frames)}] {fid}: {len(objs)} object, {len(configs)} cấu hình")
    return pd.DataFrame(rows), pd.DataFrame(obj_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep calibration drift (yaw/pitch/roll/tx/ty/tz) và đo projection mismatch")
    ap.add_argument("--data-root", default="data/kitti_mini", help="data/kitti_mini, data/nuscenes_mini_subset hoặc data/synthetic")
    ap.add_argument("--frames", nargs="*", help="danh sách frame (mặc định: tất cả frame của dataset)")
    ap.add_argument("--out", default="results/calib_sweep_kitti.csv", help="CSV theo frame; CSV theo object thêm hậu tố _objects")
    ap.add_argument("--rot-levels", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0], help="các mức xoay (độ)")
    ap.add_argument("--trans-levels", type=float, nargs="+", default=[0.02, 0.05, 0.10], help="các mức tịnh tiến (mét)")
    ap.add_argument("--no-time-check", action="store_true", help="nuScenes: không chạy thêm cấu hình no_ego_motion")
    args = ap.parse_args()

    frames = args.frames or list_frames(args.data_root)
    grid = default_grid(tuple(args.rot_levels), tuple(args.trans_levels))
    t0 = time.perf_counter()
    df, df_obj = sweep(args.data_root, frames, grid, not args.no_time_check)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out, index=False)
    df_obj.round(4).to_csv(out.with_name(out.stem + "_objects.csv"), index=False)
    print(f"{len(frames)} frame x {df['config'].nunique()} cấu hình, {len(df_obj)} dòng object, "
          f"{time.perf_counter() - t0:.1f}s -> {out}")
    summary = df.groupby("config", sort=False)[["fov_pct", "hit_pct", "shift_px_p50", "edge_score"]].mean()
    print(summary.round(2).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252) không in được tiếng Việt
    main()
