"""Advanced: tự phát hiện calibration drift KHÔNG cần label, bằng edge alignment score.

    python -m src.drift_detect --data-root data/kitti_mini --out results/drift_detect_kitti.csv
    python -m src.drift_detect --data-root data/nuscenes_mini_subset --out results/drift_detect_nusc.csv
    python -m src.drift_detect --help

Ý tưởng: edge score tuyệt đối thay đổi rất mạnh theo cảnh (ảnh nhiều cây hay ít biên), nên không đặt
ngưỡng tuyệt đối được. Thay vào đó, với calib đang chạy, xoay thử thêm -3°..+3° quanh yaw/pitch/roll
và xem calib hiện tại có phải "đỉnh" của edge score không:
    gain = max score khi xoay thử - score(calib hiện tại)
Calib đúng thì gain ≈ 0 và góc bù tốt nhất ≈ 0. Calib lệch thì có góc bù cho score cao hơn hẳn.

Hai mức quyết định:
  - từng frame:          cờ drift nếu gain > ngưỡng (ngưỡng = phân vị 95% của gain lúc calib đúng, tức 5% báo nhầm)
  - cửa sổ N frame liền: trung bình edge score của N frame rồi tìm góc bù tốt nhất; cờ drift nếu |góc bù| >= --min-delta

Ghi ra <out>.csv (mỗi dòng = frame x cấu hình) và <out>_windows.csv (mỗi dòng = cửa sổ x cấu hình).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.calib_metrics import Perturb, consistency, image_edge_distance, lidar_depth_edges, score_curves
from starter.datasets import list_frames, load_frame

DETECT_GRID = [Perturb("baseline", 0.0)] + [Perturb(a, lv) for a in ("yaw", "pitch") for lv in (0.5, 1.0, 2.0)] \
    + [Perturb("roll", 1.0), Perturb("roll", 2.0), Perturb("ty", 0.10), Perturb("tz", 0.10)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Phát hiện calibration drift bằng edge alignment score (không cần label)")
    ap.add_argument("--data-root", default="data/kitti_mini", help="thư mục KITTI hoặc nuScenes")
    ap.add_argument("--frames", nargs="*", help="danh sách frame (mặc định: tất cả)")
    ap.add_argument("--out", default="results/drift_detect_kitti.csv", help="CSV theo frame; CSV cửa sổ thêm hậu tố _windows")
    ap.add_argument("--window", type=int, default=10, help="số frame liền nhau gộp lại cho quyết định theo cửa sổ")
    ap.add_argument("--fa-rate", type=float, default=0.05, help="tỉ lệ báo nhầm cho phép khi đặt ngưỡng gain từng frame")
    ap.add_argument("--min-delta", type=float, default=0.5, help="cửa sổ: cờ drift nếu |góc bù tốt nhất| >= giá trị này (độ)")
    args = ap.parse_args()

    ds = Path(args.data_root).name
    frames = args.frames or list_frames(args.data_root)
    rows, curves = [], {}   # curves[(config, frame)] = {axis: score theo SEARCH_DEG}
    for k, fid in enumerate(frames):
        fr = load_frame(args.data_root, fid)
        pts, shape = fr["points"][:, :3], fr["image"].shape
        _, dt = image_edge_distance(fr["image"])
        for p in DETECT_GRID:
            calib = p.apply(fr["calib"])
            edges = lidar_depth_edges(pts, calib, shape)
            c = score_curves(edges, calib, shape, dt)
            curves[(p.name, fid)] = c
            rows.append({"dataset": ds, "frame_id": fid, "config": p.name, "axis": p.axis, "level": p.level,
                         "edge_points": len(edges[0]), **consistency(c)})
        print(f"[{k + 1}/{len(frames)}] {fid}")

    df = pd.DataFrame(rows)
    thr = float(np.nanpercentile(df.loc[df.config == "baseline", "gain"], 100 * (1 - args.fa_rate)))
    df["gain_thr"] = thr
    df["flag"] = df["gain"] > thr       # frame quá ít điểm biên (gain = NaN) không bị gắn cờ

    win_rows = []
    for p in DETECT_GRID:
        for w0 in range(0, len(frames) - args.window + 1, args.window):
            fids = frames[w0:w0 + args.window]
            mean_curve = {a: np.nanmean([curves[(p.name, f)][a] for f in fids], axis=0) for a in curves[(p.name, fids[0])]}
            c = consistency(mean_curve)
            win_rows.append({"dataset": ds, "config": p.name, "axis": p.axis, "level": p.level,
                             "frames": f"{fids[0]}..{fids[-1]}", **c,
                             "flag": abs(c["best_delta_deg"]) >= args.min_delta})
    dw = pd.DataFrame(win_rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out, index=False)
    dw.round(4).to_csv(out.with_name(out.stem + "_windows.csv"), index=False)

    print(f"\nNgưỡng gain từng frame (báo nhầm {args.fa_rate:.0%} lúc calib đúng): {thr:.3f}")
    per_frame = df.groupby("config", sort=False).agg(gain_mean=("gain", "mean"), detect_rate=("flag", "mean"))
    per_win = dw.groupby("config", sort=False).agg(
        win_detect_rate=("flag", "mean"),
        win_best=("best_delta_deg", lambda s: ", ".join(f"{v:+g}" for v in s)),
        win_axis=("best_axis", lambda s: ", ".join(s)))
    print(per_frame.join(per_win).round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252) không in được tiếng Việt
    main()
