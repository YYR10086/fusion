import json
from fusion import (
    KITTICalib,
    MultiObjectTracker,
    fuse,
    lidar_in_camera_fov,
    CAMERA_FOV_DEG,
)

# 路径配置
YOLO_JSON = "./record_output/record_output5/detection_results.json"
PVRCNN_JSON = "./record_output/record_output5/lidar_detection_results.json"
OUTPUT_JSON = "./record_output/record_output5/fusion_results.json"
USE_TRACKING = False  # 输出层是否启用跟踪平滑
USE_MOTION_PRIOR = True  # 融合层是否启用卡尔曼预测先验
ENABLE_DUAL_CONF_RECOVERY = True  # 双置信度筛选：检测置信度 + 轨迹连续性置信度
DET_CONF_KEEP_THRESH = 0.30        # 低于该检测置信度时，进入“轨迹恢复”判定
DUAL_CONF_ALPHA = 0.55             # 联合分数 = alpha*det_conf + (1-alpha)*track_conf
DUAL_CONF_RECOVER_THRESH = 0.42    # 联合分数阈值
TRACK_CONF_DECAY = 0.90            # 轨迹置信度衰减
TRACK_CONF_BOOST_MATCHED = 0.18    # 与图像匹配时提升轨迹置信度
TRACK_CONF_BOOST_MOTION = 0.12     # motion prior 较高时提升轨迹置信度
ENABLE_TRACK_YOLO_RECOVERY = True  # 交由 fusion.py 内核执行轨迹+YOLO补检
RECOVERY_MIN_YOLO_SCORE = 0.55
RECOVERY_MAX_THETA_DIFF = 9.0
RANGE_FILTER_BY_LABEL = {"car": 42.0}  # car 超距过滤，降低远距误检


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def dual_confidence_filter(detections, track_conf_state):
    """
    双置信度筛选：
    1) 检测置信度 det_conf：来自融合结果 fused_score；
    2) 轨迹置信度 track_conf：由历史连续性（matched_2d / motion_prior）递推。
    低 det_conf 目标可在高 track_conf 下被恢复保留，降低短时遮挡导致的漏检。
    """
    if not ENABLE_DUAL_CONF_RECOVERY:
        return detections

    filtered = []
    for det in detections:
        track_id = det.get("track_id")
        det_conf = clamp01(det.get("fused_score", det.get("score", 0.0)))
        prev_track_conf = clamp01(track_conf_state.get(track_id, det_conf))
        track_conf = prev_track_conf * TRACK_CONF_DECAY

        if det.get("matched_2d"):
            track_conf += TRACK_CONF_BOOST_MATCHED
        if det.get("motion_prior", 0.0) >= 0.35:
            track_conf += TRACK_CONF_BOOST_MOTION
        track_conf = clamp01(track_conf)
        track_conf_state[track_id] = track_conf

        combined_conf = DUAL_CONF_ALPHA * det_conf + (1.0 - DUAL_CONF_ALPHA) * track_conf
        keep = (det_conf >= DET_CONF_KEEP_THRESH) or (combined_conf >= DUAL_CONF_RECOVER_THRESH)
        if not keep:
            continue

        det_out = dict(det)
        det_out["det_conf"] = round(det_conf, 4)
        det_out["track_conf"] = round(track_conf, 4)
        det_out["combined_conf"] = round(combined_conf, 4)
        det_out["recovered_by_dual_conf"] = bool(
            det_conf < DET_CONF_KEEP_THRESH and combined_conf >= DUAL_CONF_RECOVER_THRESH
        )
        filtered.append(det_out)

    return filtered


def main():
    yolo_data = load_json(YOLO_JSON)
    pvrcnn_data = load_json(PVRCNN_JSON)

    yolo_frames = yolo_data.get("frames", [])
    pvrcnn_frames = pvrcnn_data.get("frames", [])

    if len(yolo_frames) != len(pvrcnn_frames):
        print(f"[WARNING] 帧数不匹配: YOLO={len(yolo_frames)}, PV-RCNN={len(pvrcnn_frames)}")

    calib = KITTICalib("calib.txt")
    mot = MultiObjectTracker()
    track_conf_state = {}

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
            motion_predictions=(
                mot.predict_states(timestamp) if (USE_MOTION_PRIOR and timestamp) else None
            ),
            # 与当前融合策略保持一致：YOLO 仅辅助，不输出 YOLO-only 目标
            include_unmatched_yolo=ENABLE_TRACK_YOLO_RECOVERY,
            unmatched_yolo_min_score=RECOVERY_MIN_YOLO_SCORE,
            camera_fov_only=False,
            range_filter_by_label=RANGE_FILTER_BY_LABEL,
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
            tracked = mot.update(fused, timestamp) if timestamp else fused
            output_dets = tracked if USE_TRACKING else fused
            output_dets = dual_confidence_filter(output_dets, track_conf_state)
            if not output_dets:
                continue
            all_results.append({
                "frame_id": idx,
                "filename": pvrcnn_frame.get("filename", yolo_frame.get("filename", "")),
                "detections": output_dets
            })

    output = {
        "meta": {
            "total_frames": len(all_results),
            "fusion_mode": "calib" if calib.use_calib else "theta",
            "use_tracking": USE_TRACKING,
            "use_motion_prior": USE_MOTION_PRIOR,
            "enable_dual_conf_recovery": ENABLE_DUAL_CONF_RECOVERY,
            "dual_conf_params": {
                "det_conf_keep_thresh": DET_CONF_KEEP_THRESH,
                "dual_conf_alpha": DUAL_CONF_ALPHA,
                "dual_conf_recover_thresh": DUAL_CONF_RECOVER_THRESH,
                "track_conf_decay": TRACK_CONF_DECAY,
                "track_conf_boost_matched": TRACK_CONF_BOOST_MATCHED,
                "track_conf_boost_motion": TRACK_CONF_BOOST_MOTION,
            },
            "range_filter_by_label": RANGE_FILTER_BY_LABEL,
            "track_yolo_recover_params": {
                "enable_track_yolo_recovery": ENABLE_TRACK_YOLO_RECOVERY,
                "recovery_min_yolo_score": RECOVERY_MIN_YOLO_SCORE,
                "recovery_max_theta_diff": RECOVERY_MAX_THETA_DIFF,
            },
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
