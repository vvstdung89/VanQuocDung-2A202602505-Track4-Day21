"""CP3: gộp CSV của calib_sweep và drift_detect thành bảng tóm tắt + biểu đồ.

    python -m src.plot_results
    python -m src.plot_results --help

Đọc  results/calib_sweep_{kitti,nusc}[_objects].csv và results/drift_detect_{kitti,nusc}[_windows].csv
Ghi  results/calib_sweep_summary.csv          mỗi dòng = dataset x cấu hình
     results/figures/calib_sweep_hit.png      hit % theo mức xoay / tịnh tiến, KITTI và nuScenes
     results/figures/yaw_hit_by_distance.png  hit % theo yaw, tách theo khoảng cách object
     results/figures/drift_detect.png         tỉ lệ phát hiện drift (từng frame) + ước lượng drift (cửa sổ 10 frame)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua: luôn theo thứ tự này, không xoay vòng
DATASETS = {"kitti": "KITTI (64 beam, 1242x375)", "nusc": "nuScenes (32 beam, 1600x900)"}
ROT, TRANS, BINS = ("yaw", "pitch", "roll"), ("tx", "ty", "tz"), ("near", "mid", "far")
BIN_LABEL = {"near": "gần < 15 m", "mid": "15–30 m", "far": "xa ≥ 30 m"}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
    })


def line(ax, x, y, i: int, label: str) -> None:
    ax.plot(x, y, color=SERIES[i], lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, label=label,
            solid_capstyle="round", solid_joinstyle="round")


def load(results: Path, key: str) -> dict[str, pd.DataFrame]:
    rd = lambda name: pd.read_csv(results / name, dtype={"frame_id": str})  # noqa: E731
    return {"frames": rd(f"calib_sweep_{key}.csv"), "objects": rd(f"calib_sweep_{key}_objects.csv"),
            "detect": rd(f"drift_detect_{key}.csv"), "windows": rd(f"drift_detect_{key}_windows.csv")}


def with_baseline(df: pd.DataFrame, axis: str, col: str) -> pd.DataFrame:
    """Mean của `col` theo mức lệch của một trục, thêm điểm 0 = baseline."""
    base = df[df.axis == "baseline"][col].mean()
    s = df[df.axis == axis].groupby("level")[col].mean()
    return pd.concat([pd.Series({0.0: base}), s]).sort_index()


def summarize(d: dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    f, o, det, win = d["frames"], d["objects"], d["detect"], d["windows"]
    gf, go = f.groupby("config", sort=False), o.groupby("config", sort=False)
    s = pd.DataFrame({"dataset": key, "axis": gf["axis"].first(), "level": gf["level"].first(),
                      "n_frames": gf["frame_id"].nunique(), "fov_pct": gf["fov_pct"].mean(),
                      "n_objects": go.size(), "hit_pct": go["hit_pct"].mean()})
    s["miss_increase_pp"] = s.loc["baseline", "hit_pct"] - s["hit_pct"]
    for b in BINS:
        ob = o[o.dist_bin == b].groupby("config", sort=False)
        s[f"hit_{b}"], s[f"n_{b}"] = ob.hit_pct.mean(), ob.size()
    s["shift_px_p50"] = gf.shift_px_p50.median()
    s["edge_score"] = gf.edge_score.mean()
    s["detect_rate_frame"] = det.groupby("config", sort=False).flag.mean()
    s["detect_rate_window"] = win.groupby("config", sort=False).flag.mean()
    s["est_drift_window"] = win.groupby("config", sort=False).apply(
        lambda w: ", ".join(f"{r.best_axis} {-r.best_delta_deg:+g}" for r in w.itertuples()), include_groups=False)
    return s.reset_index()


def plot_hit(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    for c, (key, d) in enumerate(data.items()):
        o = d["objects"]
        for r, (axes_names, unit, scale) in enumerate([(ROT, "độ", 1), (TRANS, "cm", 100)]):
            ax = axes[r, c]
            for i, a in enumerate(axes_names):
                s = with_baseline(o, a, "hit_pct")
                line(ax, s.index * scale, s.values, i, a)
            ax.set_ylim(0, 105)
            ax.set_xlabel(f"mức lệch calibration ({unit})")
            ax.set_title(f"{DATASETS[key]}: lệch {'xoay' if r == 0 else 'tịnh tiến'}", loc="left", fontsize=10)
            ax.legend(loc="lower left")
        axes[1, c].text(0.5, 0.55, "tx, ty, tz gần như trùng nhau: lệch 10 cm chỉ mất 0–3 điểm %",
                        transform=axes[1, c].transAxes, ha="center", color=INK2, fontsize=9)
        axes[0, c].set_ylabel("% điểm của object rơi vào 2D box GT")
        axes[1, c].set_ylabel("% điểm của object rơi vào 2D box GT")
    fig.suptitle("Hit rate trung bình theo object khi chỉ đổi 1 tham số calibration", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_distance(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (key, d) in zip(axes, data.items()):
        o = d["objects"]
        for i, b in enumerate(BINS):
            ob = o[o.dist_bin == b]
            n = ob[ob.config == "baseline"].shape[0]
            base = ob[ob.config == "baseline"].hit_pct.mean()
            s = ob[ob.config.str.startswith("yaw_")].groupby(
                ob.config.str.extract(r"yaw_([\d.]+)deg")[0].astype(float)).hit_pct.mean()
            s = pd.concat([pd.Series({0.0: base}), s]).sort_index()
            line(ax, s.index, s.values, i, f"{BIN_LABEL[b]} (n={n})")
        ax.set_ylim(0, 105)
        ax.set_xlabel("yaw drift (độ)")
        ax.set_title(DATASETS[key], loc="left", fontsize=10)
        ax.legend(loc="lower left")
    axes[0].set_ylabel("% điểm của object rơi vào 2D box GT")
    fig.suptitle("Cùng một yaw drift, vật càng xa càng lệch khỏi box (box nhỏ hơn, độ dịch pixel như nhau)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_detect(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (key, d) in zip(axes[:2], data.items()):
        det = d["detect"]
        thr = det.gain_thr.iloc[0]
        for i, a in enumerate(ROT):
            s = with_baseline(det.assign(flag=det.flag.astype(float) * 100), a, "flag")
            line(ax, s.index, s.values, i, a)
        ax.axhline(5, color=INK2, lw=1, label="5% báo nhầm (mức đặt ngưỡng)")
        ax.set_ylim(0, 105)
        ax.set_xlabel("drift (độ)")
        ax.set_ylabel("% frame bị gắn cờ drift")
        ax.set_title(f"{DATASETS[key]}\ntừng frame: cờ nếu gain > {thr:.3f}", loc="left", fontsize=10)
        ax.legend(loc="upper left")

    ax = axes[2]
    lim = 2.4
    ax.plot([-0.2, lim], [-0.2, lim], color=GRID, lw=1.5, zorder=0)
    markers = {"kitti": "o", "nusc": "s"}
    win = {k: d["detect"].frame_id.nunique() // (d["windows"].config == "baseline").sum() for k, d in data.items()}
    for key, d in data.items():
        w = d["windows"]
        w = w[w.axis.isin(ROT + ("baseline",))]
        for i, a in enumerate(ROT):
            sub = w[(w.axis == a) | (w.axis == "baseline")]
            est = [-r.best_delta_deg if r.best_axis == a else 0.0 for r in sub.itertuples()]
            ax.scatter(sub.level, est, s=55, color=SERIES[i], marker=markers[key], edgecolors=SURFACE,
                       linewidths=1.5, label=f"{a} – {key}", zorder=3)
    ax.set_xlim(-0.2, lim)
    ax.set_ylim(-0.2, lim)
    ax.set_xlabel("drift thật (độ)")
    ax.set_ylabel("drift ước lượng (độ)")
    ax.set_title(f"Gộp cửa sổ KITTI {win['kitti']} frame, nuScenes {win['nusc']} frame\n"
                 "đường xám = ước lượng đúng; ra trục khác vẽ ở 0", loc="left", fontsize=10)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.suptitle("Phát hiện drift không cần label bằng edge alignment score", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tổng hợp kết quả sweep thành bảng CSV và biểu đồ")
    ap.add_argument("--results", default="results", help="thư mục chứa CSV của calib_sweep và drift_detect")
    args = ap.parse_args()
    results = Path(args.results)
    figs = results / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    style()

    data = {key: load(results, key) for key in DATASETS}
    summary = pd.concat([summarize(d, key) for key, d in data.items()], ignore_index=True)
    summary.round(2).to_csv(results / "calib_sweep_summary.csv", index=False)
    plot_hit(data, figs / "calib_sweep_hit.png")
    plot_distance(data, figs / "yaw_hit_by_distance.png")
    plot_detect(data, figs / "drift_detect.png")

    cols = ["dataset", "config", "fov_pct", "hit_pct", "hit_near", "hit_mid", "hit_far", "shift_px_p50",
            "edge_score", "detect_rate_frame", "detect_rate_window"]
    print(summary[cols].round(2).to_string(index=False))
    print(f"-> {results / 'calib_sweep_summary.csv'}, {figs}/calib_sweep_hit.png, yaw_hit_by_distance.png, drift_detect.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252) không in được tiếng Việt
    main()
