"""Đo latency p50/p95 của các bước QA calibration (CPU), bỏ lần chạy đầu, lặp >= 20 lần.

    python -m src.bench_latency
    python -m src.bench_latency --repeats 50 --out results/latency.csv

Các bước đo trên một frame:
  project      chiếu toàn bộ point cloud lên ảnh (velo_to_cam + cam_to_image)
  image_edges  Canny + distance transform của ảnh (mỗi frame một lần)
  edge_score   tách điểm biên độ sâu + edge alignment score cho calib hiện tại
  drift_search edge score khi xoay thử 3 trục x 25 góc (bước kiểm tra drift đầy đủ của drift_detect)
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.calib_metrics import edge_score, image_edge_distance, lidar_depth_edges, score_curves
from starter.datasets import load_frame
from starter.projection import project_velo_to_image


def bench(fn, repeats: int) -> np.ndarray:
    fn()                                  # lần đầu chậm hơn (cache, cấp phát) -> bỏ
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Đo latency p50/p95 các bước QA calibration")
    ap.add_argument("--repeats", type=int, default=30, help="số lần đo (sau khi bỏ lần đầu), tối thiểu 20")
    ap.add_argument("--out", default="results/latency.csv", help="CSV kết quả")
    args = ap.parse_args()
    repeats = max(20, args.repeats)

    rows = []
    for root, fid in [("data/kitti_mini", "000011"), ("data/nuscenes_mini_subset", "scene-0103_010")]:
        fr = load_frame(root, fid)
        pts, calib, img, shape = fr["points"], fr["calib"], fr["image"], fr["image"].shape
        _, dt = image_edge_distance(img)
        edges = lidar_depth_edges(pts[:, :3], calib, shape)
        steps = {
            "project": lambda: project_velo_to_image(pts, calib, shape),
            "image_edges": lambda: image_edge_distance(img),
            "edge_score": lambda: edge_score(lidar_depth_edges(pts[:, :3], calib, shape), calib, shape, dt),
            "drift_search": lambda: score_curves(edges, calib, shape, dt),
        }
        for name, fn in steps.items():
            t = bench(fn, repeats)
            rows.append({"dataset": Path(root).name, "frame_id": fid, "n_points": len(pts),
                         "image": f"{shape[1]}x{shape[0]}", "step": name, "repeats": repeats,
                         "p50_ms": np.percentile(t, 50), "p95_ms": np.percentile(t, 95), "max_ms": t.max(),
                         "cpu": platform.processor(), "cpu_count": os.cpu_count(),
                         "python": platform.python_version(), "numpy": np.__version__})
            print(f"{Path(root).name:22s} {name:12s} p50={rows[-1]['p50_ms']:7.2f} ms  p95={rows[-1]['p95_ms']:7.2f} ms")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(3).to_csv(out, index=False)
    print(f"CPU: {platform.processor()} ({os.cpu_count()} luồng) -> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252) không in được tiếng Việt
    main()
