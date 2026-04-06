import json
from fusion import process_frame, KITTICalib, MultiObjectTracker, fuse, lidar_in_camera_fov, CAMERA_FOV_DEG

# 路径配置
YOLO_JSON = "./record_output/record_output5/detection_results.json"
PVRCNN_JSON = "./record_output/record_output5/lidar_detection_results.json"
OUTPUT_JSON = "./record_output/record_output5/fusion_results.json"
USE_TRACKING = False  # 评估融合质量时建议关闭，避免跟踪器影响检测指标


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    yolo_data = load_json(YOLO_JSON)
    pvrcnn_data = load_json(PVRCNN_JSON)

    yolo_frames = yolo_data.get("frames", [])
    pvrcnn_frames = pvrcnn_data.get("frames", [])

    if len(yolo_frames) != len(pvrcnn_frames):
        print(f"[WARNING] 帧数不匹配: YOLO={len(yolo_frames)}, PV-RCNN={len(pvrcnn_frames)}")

    calib = KITTICalib("calib.txt")
    mot = MultiObjectTracker()

    all_results = []
    stats = {
        "yolo_total": 0,
        "pvrcnn_total": 0,
        "pvrcnn_in_fov": 0,
        "fused_total": 0,
        "matched_2d_total": 0,
        "yolo_only_total": 0,
        "per_class": {
            "car": {"pvrcnn": 0, "fused": 0},
            "truck": {"pvrcnn": 0, "fused": 0},
            "bus": {"pvrcnn": 0, "fused": 0},
            "pedestrian": {"pvrcnn": 0, "fused": 0},
            "cyclist": {"pvrcnn": 0, "fused": 0},
        },
    }

    total_frames = max(len(yolo_frames), len(pvrcnn_frames))
    for idx in range(total_frames):
        yolo_frame = yolo_frames[idx] if idx < len(yolo_frames) else {}
        pvrcnn_frame = pvrcnn_frames[idx] if idx < len(pvrcnn_frames) else {}
        yolo_dets = yolo_frame.get("detections", [])
        pvrcnn_dets = pvrcnn_frame.get("detections", [])
        stats["yolo_total"] += len(yolo_dets)
        stats["pvrcnn_total"] += len(pvrcnn_dets)
        for d in pvrcnn_dets:
            cls = d.get("class_label", "").lower()
            if cls in stats["per_class"]:
                stats["per_class"][cls]["pvrcnn"] += 1
        stats["pvrcnn_in_fov"] += sum(
            1 for d in pvrcnn_dets
            if lidar_in_camera_fov(d["box"]["center"], CAMERA_FOV_DEG)
        )

        if pvrcnn_dets:
            timestamp = pvrcnn_dets[0].get("timestamp", "")
        elif yolo_dets:
            timestamp = yolo_dets[0].get("timestamp", "")
        else:
            timestamp = ""

        fused = fuse(
            pvrcnn_dets,
            yolo_dets,
            calib,
            # 与当前融合策略保持一致：YOLO 仅辅助，不输出 YOLO-only 目标
            include_unmatched_yolo=False,
            camera_fov_only=False,
        )
        stats["fused_total"] += len(fused)
        stats["matched_2d_total"] += sum(1 for d in fused if d.get("matched_2d"))
        for d in fused:
            cls = d.get("label", "").lower()
            if cls in stats["per_class"]:
                stats["per_class"][cls]["fused"] += 1
        stats["yolo_only_total"] += sum(
            1 for d in fused
            if (not d.get("matched_2d")) and d.get("center") == [0, 0, 0]
        )

        if fused:
            tracked = mot.update(fused, timestamp) if USE_TRACKING else fused
            all_results.append({
                "frame_id": idx,
                "filename": pvrcnn_frame.get("filename", yolo_frame.get("filename", "")),
                "detections": tracked
            })

    output = {
        "meta": {
            "total_frames": len(all_results),
            "fusion_mode": "calib" if calib.use_calib else "theta",
            "use_tracking": USE_TRACKING,
            "stats": stats,
        },
        "frames": all_results
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== 类别保留统计（PVRCNN -> Fused） ===")
    for cls, v in stats["per_class"].items():
        pv_n = v["pvrcnn"]
        fu_n = v["fused"]
        keep = (fu_n / pv_n) if pv_n > 0 else 0.0
        print(f"{cls:10s}: {pv_n:5d} -> {fu_n:5d}  keep={keep:.3f}")

    print(f"\n✅ 融合完成: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
