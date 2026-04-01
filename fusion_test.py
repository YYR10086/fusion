import json
from fusion import process_frame, KITTICalib, MultiObjectTracker, fuse

# 路径配置
YOLO_JSON = "./record_output/record_output5/detection_results.json"
PVRCNN_JSON = "./record_output/record_output5/lidar_detection_results.json"
OUTPUT_JSON = "./record_output/record_output5/fusion_results.json"


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

    for idx in range(min(len(yolo_frames), len(pvrcnn_frames))):
        yolo_dets = yolo_frames[idx].get("detections", [])
        pvrcnn_dets = pvrcnn_frames[idx].get("detections", [])

        if not pvrcnn_dets:
            continue

        timestamp = pvrcnn_dets[0].get("timestamp", "")
        fused = fuse(
            pvrcnn_dets,
            yolo_dets,
            calib,
            include_unmatched_yolo=True,
            camera_fov_only=True,
            unmatched_yolo_min_score=0.2,
        )

        if fused:
            tracked = mot.update(fused, timestamp)
            all_results.append({
                "frame_id": idx,
                "filename": pvrcnn_frames[idx].get("filename", ""),
                "detections": tracked
            })

    output = {
        "meta": {
            "total_frames": len(all_results),
            "fusion_mode": "calib" if calib.use_calib else "theta"
        },
        "frames": all_results
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 融合完成: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
