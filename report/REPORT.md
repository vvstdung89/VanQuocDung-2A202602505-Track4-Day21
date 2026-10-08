# Báo cáo Day 6: Độ nhạy của LiDAR-camera projection với calibration drift

> Thay **mọi** ô có chữ ĐIỀN nằm trong ngoặc vuông bằng nội dung của bạn, xoá luôn cả dấu ngoặc vuông. Lệnh `python tools/check_submission.py` sẽ báo FAIL nếu còn sót bất kỳ chỗ nào.

- **Họ tên:** Văn Quốc Dũng
- **MSSV:** 2A202602505
- **Lớp:** K4 Track4
- **Link repo:** https://github.com/vvstdung89/VanQuocDung-2A202602505-Track4-Day21
- **Topic:** A — LiDAR-camera projection QA
- **Dataset:** data/kitti_mini (thí nghiệm chính), data/nuscenes_mini_subset (so sánh sensor), data/synthetic (debug, demo theo khoảng cách)
- **Các frame đã dùng:** toàn bộ 20 frame KITTI mini (000001 … 000061), toàn bộ 80 keyframe nuScenes (scene-0103_000 … 039, scene-1094_000 … 039), synthetic 000000 … 000004

> Hãy viết ngắn: mỗi mục từ 3 đến 8 dòng, ưu tiên số liệu và hình ảnh.

## 1. Claim

Một câu khẳng định kỹ thuật có thể kiểm chứng. Ví dụ: *"Lệch yaw 1° làm 12% điểm LiDAR rơi ra khỏi vật thể ở 30 m, phát hiện được bằng edge-alignment score với ngưỡng X."*

**Claim nháp (CP1):** Trên KITTI mini, lệch yaw 1° làm hơn 10% điểm LiDAR của object (điểm nằm trong 3D box GT) rơi ra ngoài 2D box của chính object đó. Ảnh hưởng tăng theo khoảng cách: vật xa hơn 30 m bị nặng hơn vật gần hơn 15 m. Lệch tịnh tiến 5 cm thì ngược lại, ảnh hưởng chủ yếu vật gần.

## 2. Evidence

Bảng hoặc plot số liệu, kèm ảnh/video demo. Ghi rõ đường dẫn file trong `results/`.

| Cấu hình / mức perturb | Metric 1 | Metric 2 | Ghi chú |
|---|---|---|---|
| [ĐIỀN] | | | |

![demo](../results/figures/[ĐIỀN].png)

## 3. Failure case

Nêu khi nào hệ thống hoặc phương pháp fail, vì sao fail, và liên hệ tới lớp nào trong 6 lớp debug: I/O, Geometry, Time, Preprocess, Model, Metric.

![failure](../results/figures/fail_[ĐIỀN].png)

[ĐIỀN]

## 4. Khuyến nghị nếu triển khai thật

Use-case cụ thể (ADAS / robot / drone), trade-off và bước tiếp theo.

[ĐIỀN]

## 5. Cách chạy lại

Các lệnh tái tạo lại toàn bộ kết quả từ repo sạch.

```bash
# CP2: overlay calib gốc trên 3 dataset (dùng 2 hàm TODO trong starter/projection.py)
python -m starter.projection --data-root data/synthetic --frame 000000
python -m starter.projection --data-root data/kitti_mini --frame 000011
python -m starter.projection --data-root data/nuscenes_mini_subset --frame scene-0103_010
# CP2: ảnh 3 khoảng cách + ảnh so sánh yaw 0°/1°/3°
python -m src.demo_overlays
```

## 6. Khai báo sử dụng AI

Ghi rõ đã dùng công cụ AI nào, dùng vào việc gì, và bạn đã tự kiểm chứng kết quả đó bằng cách nào. Nếu không dùng AI, ghi "Không sử dụng". Xem quy định ở `RULES.md` mục 2.

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| [ĐIỀN] | | |
