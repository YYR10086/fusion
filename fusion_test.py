import json
import math
from datetime import datetime
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
RANGE_FILTER_BY_LABEL = {"car": 42.0}  # 保留 42m 距离过滤，主要面向 car
ENABLE_TRACK_PREDICTION_RECOVERY = True  # 使用卡尔曼预测补框，缓解短时漏检
MAX_RECOVERY_STEP_M = 1.8  # 单帧恢复位置最大步长，抑制“闪现”


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def angle_diff_rad(a, b):
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def parse_ts(ts):
    if not ts:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            continue
    return None


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
        if track_id is None:
            track_conf = det_conf
        else:
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


def apply_track_strength_recovery(observed_dets, mot, track_state, dt_s=0.1):
    """
    按“0.5 -> 1.0 -> 0.5 -> 删除”规则进行跟踪补框：
    - 新轨迹默认强度 0.5；
    - 当前帧有同类图像确认（matched_2d=True）时，强度提到 1.0；
    - 缺失一帧：若上次为 1.0 则降到 0.5 并保留预测框；若上次为 0.5 则直接删除。
    """
    out = []
    observed_ids = set()

    for det in observed_dets:
        tid = det.get("track_id")
        if tid is None:
            out.append(det)
            continue
        st = track_state.get(tid, {
            "strength": 0.5,
            "last_dimensions": det.get("dimensions", [4.0, 1.8, 1.6]),
            "last_heading": det.get("heading", 0.0),
            "label": det.get("label", ""),
            "last_center": list(det.get("center", [0.0, 0.0, 0.0])),
            "last_speed": 0.0,
        })
        # 只要本帧被观测到（尤其是 cyclist/pedestrian 的 PVRCNN 观测），就应提升轨迹强度，
        # 否则会出现“从未进入可预测状态”的问题。
        if det.get("matched_2d") or det.get("source") == "pvrcnn":
            st["strength"] = 1.0
        else:
            st["strength"] = max(st.get("strength", 0.5), 0.7)
        st["last_dimensions"] = det.get("dimensions", st["last_dimensions"])
        st["last_heading"] = det.get("heading", st["last_heading"])
        st["label"] = det.get("label", st.get("label", ""))
        st["last_center"] = list(det.get("center", st.get("last_center", [0.0, 0.0, 0.0])))
        vx0, vy0 = det.get("velocity", [0.0, 0.0])
        st["last_speed"] = float(math.hypot(vx0, vy0))
        track_state[tid] = st
        observed_ids.add(tid)

        det_out = dict(det)
        det_out["track_strength"] = round(st["strength"], 2)
        out.append(det_out)

    if not ENABLE_TRACK_PREDICTION_RECOVERY:
        return out

    # 对“当前帧缺失但轨迹还连续”的目标进行补框
    for trk in mot.tracks:
        tid = trk.track_id
        if tid in observed_ids:
            continue
        st = track_state.get(tid)
        if st is None:
            continue
        # 最多只对首个丢失帧做恢复，防止长期残留框
        if trk.miss_count > 1:
            track_state.pop(tid, None)
            continue

        # 若同类目标已在附近被新轨迹观测到，判为ID切换，删除旧轨迹避免幽灵框
        lx, ly, _ = st.get("last_center", [0.0, 0.0, 0.0])
        duplicate_nearby = any(
            od.get("label") == st.get("label", trk.label) and
            od.get("track_id") is not None and
            od.get("track_id") != tid and
            math.hypot(od.get("center", [0.0, 0.0, 0.0])[0] - lx, od.get("center", [0.0, 0.0, 0.0])[1] - ly) <= 4.0
            for od in observed_dets
        )
        if duplicate_nearby:
            track_state.pop(tid, None)
            continue

        prev_strength = float(st.get("strength", 0.5))
        # 放宽缺失衰减门限：>=0.7 也允许进入一次预测恢复
        new_strength = 0.5 if prev_strength >= 0.7 else 0.0
        if new_strength <= 0.0:
            track_state.pop(tid, None)
            continue

        vx, vy = trk.get_velocity()
        speed = math.hypot(vx, vy)
        last_heading = float(st.get("last_heading", 0.0))
        if speed >= 0.1:
            vel_heading = float(math.atan2(vy, vx))
            # 朝向辅助：若速度方向与历史朝向差异过大，沿历史朝向收缩速度分量
            if angle_diff_rad(vel_heading, last_heading) > math.radians(60.0):
                align = math.cos(vel_heading - last_heading)
                projected_speed = max(0.0, speed * max(align, 0.2))
                vx = projected_speed * math.cos(last_heading)
                vy = projected_speed * math.sin(last_heading)
                speed = projected_speed
                vel_heading = last_heading
            heading = vel_heading
        else:
            heading = last_heading

        # 当前位置使用“当前帧预测状态 + 一小步速度前推”，并限制最大步长抑制跳变
        px = float(trk.x[0] + vx * dt_s)
        py = float(trk.x[1] + vy * dt_s)
        lx, ly, lz = st.get("last_center", [px, py, 0.0])
        step = math.hypot(px - lx, py - ly)
        if step > MAX_RECOVERY_STEP_M:
            scale = MAX_RECOVERY_STEP_M / max(step, 1e-6)
            px = float(lx + (px - lx) * scale)
            py = float(ly + (py - ly) * scale)
        rec = {
            "label": st.get("label", trk.label),
            "camera_label": "",
            "score": round(0.3 * new_strength, 4),
            "fused_score": round(0.3 * new_strength, 4),
            "center": [px, py, float(lz)],
            "dimensions": st.get("last_dimensions", [4.0, 1.8, 1.6]),
            "heading": heading,
            "proj_bbox": None,
            "matched_2d": False,
            "yolo_assisted": False,
            "yolo_conf": 0.0,
            "match_quality": 0.0,
            "motion_prior": 1.0,
            "source": "track_strength_recover",
            "track_id": tid,
            "velocity": [round(vx, 3), round(vy, 3)],
            "track_strength": round(new_strength, 2),
            "recovered_by_track_strength": True,
            "prediction_dt_s": round(float(dt_s), 3),
        }
        st["strength"] = new_strength
        st["last_heading"] = heading
        st["last_center"] = [px, py, float(lz)]
        st["last_speed"] = float(speed)
        track_state[tid] = st
        out.append(rec)

    return out


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
    track_strength_state = {}
    prev_timestamp = None

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
            include_track_predictions=False,  # 关闭旧补框逻辑，改用下方强度状态机
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

        tracked = mot.update(fused, timestamp) if timestamp else fused
        dt_s = 0.1
        cur_ts = parse_ts(timestamp)
        if cur_ts is not None and prev_timestamp is not None:
            dt_s = (cur_ts - prev_timestamp).total_seconds()
            dt_s = dt_s if dt_s > 1e-3 else 0.1
        if cur_ts is not None:
            prev_timestamp = cur_ts
        output_dets = tracked if USE_TRACKING else tracked
        output_dets = apply_track_strength_recovery(output_dets, mot, track_strength_state, dt_s=dt_s)
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
            "enable_track_prediction_recovery": ENABLE_TRACK_PREDICTION_RECOVERY,
            "track_strength_rule": {
                "initial_strength": 0.5,
                "match_raise_to": 1.0,
                "first_miss_drop_to": 0.5,
                "second_miss_drop": "remove",
            },
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
