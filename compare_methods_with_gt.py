import json
import math
from pathlib import Path

import pandas as pd

try:
    from shapely.geometry import Polygon
except ImportError:
    raise ImportError("Please install shapely first: pip install shapely")


# ============================================================
# 1. 路径配置区：直接改这里
# ============================================================

# GT 是一个文件夹，里面是逐帧 JSON
GT_DIR = r"/home/tl/OpenPCDet/tools/gt_json"

# 下面三个是大 JSON，内部带 frames
PVRCNN_PATH = r"./record_output/record_output2/lidar_detection_results.json"
YOLO_PATH = r"./record_output/record_output2/detection_results.json"
FUSION_PATH = r"./record_output/record_output2/fusion_results.json"

# 输出目录
OUTPUT_DIR = r"/home/tl/OpenPCDet/tools/compare_output"

# 只评估前 60 帧
MAX_FRAMES = 60

# 只统计三类
TARGET_CLASSES = ["Car", "Pedestrian", "Cyclist"]

# PVRCNN / Fusion 的 BEV IoU 阈值
BEV_IOU_THRESH = 0.25

# YOLO 与 GT 的 theta 匹配阈值（度）
YOLO_THETA_THRESH_DEG = 10.0

# 相机水平视场角
CAMERA_FOV_DEG = 150.0

# 是否允许类别组匹配
ALLOW_GROUP_MATCH = False


# ============================================================
# 2. 类别映射
# ============================================================

CLASS_MAP = {
    "0": "Car",
    "1": "Pedestrian",
    "2": "Cyclist",
    "vehicle": "Car",
    "car": "Car",
    "bus": "Bus",
    "truck": "Truck",
    "van": "Van",
    "bicyclist": "Cyclist",
    "bike": "Cyclist",
    "tricycle": "Cyclist",
    "person": "Pedestrian",
    "pedestrian": "Pedestrian",
    "cyclist": "Cyclist",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
    "rider": "Cyclist",
}

CATEGORY_GROUPS = [
    {"Car", "Bus", "Truck", "Van"},
    {"Pedestrian", "Cyclist"},
]


def normalize_label(label: str) -> str:
    if not label:
        return "Unknown"
    s = label.strip()
    mapped = CLASS_MAP.get(s.lower())
    return mapped if mapped is not None else s


def labels_match(pred_label: str, gt_label: str, allow_group_match=False) -> bool:
    p = normalize_label(pred_label)
    g = normalize_label(gt_label)

    if p == g:
        return True

    if allow_group_match:
        for group in CATEGORY_GROUPS:
            if p in group and g in group:
                return True

    return False


# ============================================================
# 3. JSON 读取
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_frames_from_json(path):
    data = load_json(path)

    # 多帧大 JSON
    if "frames" in data:
        return data["frames"]

    # 单帧 JSON
    if "detections" in data or "annotations" in data:
        return [data]

    raise ValueError(f"Unsupported JSON format: {path}")


def canonicalize_frame_token(token):
    if token is None:
        return None

    s = str(token).strip()
    if not s:
        return None

    stem = Path(s).stem
    low = stem.lower()

    if low.startswith("frame_"):
        low = low[6:]
    elif low.startswith("frame"):
        low = low[5:]

    if low.isdigit():
        return str(int(low))

    return low


def frame_lookup_keys(frame):
    keys = []

    for field in ("filename", "image_path", "image_file", "frame_name"):
        v = frame.get(field)
        k = canonicalize_frame_token(v)
        if k is not None:
            keys.append(k)

    frame_id = frame.get("frame_id")
    if frame_id is not None:
        keys.append(str(frame_id))
        keys.append(canonicalize_frame_token(frame_id))

    # 保序去重
    dedup = []
    seen = set()
    for k in keys:
        if k is None:
            continue
        if k not in seen:
            seen.add(k)
            dedup.append(k)
    return dedup


def build_frame_map_from_big_json(path, max_frames=60):
    frames = load_frames_from_json(path)
    out = {}

    for frame in frames:
        frame_id = frame.get("frame_id", None)
        if frame_id is not None and frame_id >= max_frames:
            continue

        keys = frame_lookup_keys(frame)
        if not keys:
            filename = frame.get("filename", f"frame_{frame.get('frame_id', 0)}")
            keys = [canonicalize_frame_token(filename)]

        for key in keys:
            if key is not None and key not in out:
                out[key] = frame

    return out


def build_frame_map_from_dir(dir_path, max_frames=60):
    dir_path = Path(dir_path)
    out = {}

    if not dir_path.exists():
        raise FileNotFoundError(f"GT directory not found: {dir_path}")

    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in GT directory: {dir_path}")

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            frame = json.load(f)

        frame_id = frame.get("frame_id", None)
        if frame_id is not None and frame_id >= max_frames:
            continue

        keys = [canonicalize_frame_token(json_file.stem)]
        keys.extend(frame_lookup_keys(frame))

        for key in keys:
            if key is not None and key not in out:
                out[key] = frame

    return out


# ============================================================
# 4. GT / 检测对象提取
# ============================================================

def extract_gt_objects(frame):
    return frame.get("annotations", [])


def extract_pred_objects(frame):
    return frame.get("detections", [])


def extract_box_from_gt(obj):
    label = obj.get("class_label", obj.get("label", "Unknown"))
    box = obj["box"]
    center = box["center"]
    dims = box["dimensions"]
    yaw = float(box["rotation"])
    return center, dims, yaw, label


def extract_box_from_pred(obj):
    # 兼容 PVRCNN / Fusion
    if "box" in obj:
        label = obj.get("class_label", obj.get("label", "Unknown"))
        center = obj["box"]["center"]
        dims = obj["box"]["dimensions"]
        yaw = float(obj["box"]["rotation"])
        score = float(obj.get("score", 1.0))
    else:
        label = obj.get("label", obj.get("class_label", "Unknown"))
        center = obj["center"]
        dims = obj["dimensions"]
        yaw = float(obj.get("heading", 0.0))
        score = float(obj.get("fused_score", obj.get("score", 1.0)))
    return center, dims, yaw, label, score


# ============================================================
# 5. BEV IoU 工具
# ============================================================

def get_bev_corners(center, dims, yaw):
    x, y = center[0], center[1]
    l, w = dims[0], dims[1]
    hl, hw = l / 2.0, w / 2.0

    pts = [
        ( hl,  hw),
        ( hl, -hw),
        (-hl, -hw),
        (-hl,  hw),
    ]

    c = math.cos(yaw)
    s = math.sin(yaw)

    corners = []
    for px, py in pts:
        rx = c * px - s * py + x
        ry = s * px + c * py + y
        corners.append((rx, ry))
    return corners


def bev_iou(center1, dims1, yaw1, center2, dims2, yaw2):
    poly1 = Polygon(get_bev_corners(center1, dims1, yaw1))
    poly2 = Polygon(get_bev_corners(center2, dims2, yaw2))

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    if union <= 1e-9:
        return 0.0

    return inter / union


# ============================================================
# 6. YOLO theta 工具
# ============================================================

def lidar_center_to_theta_deg(center):
    x, y = center[0], center[1]
    return math.degrees(math.atan2(y, x))


def in_camera_fov(center, fov_deg=CAMERA_FOV_DEG):
    theta = lidar_center_to_theta_deg(center)
    return abs(theta) <= fov_deg / 2.0


def angle_diff_deg(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def extract_yolo_label(pred):
    for field in ("label", "class_label", "class_name", "name"):
        v = pred.get(field)
        if v is not None:
            return str(v)
    return "Unknown"


def extract_yolo_theta_candidates_deg(pred):
    """
    返回候选角度（单位：度）。
    - 明确 *_deg 字段：按度处理
    - 明确 *_rad 字段：按弧度转度
    - 模糊字段（theta/angle/alpha）：同时尝试“原值即度”和“弧度转度”
      以避免小角度度值被误判成弧度导致匹配失败。
    """
    for field in ("theta_deg", "angle_deg", "alpha_deg"):
        if field in pred:
            return [float(pred[field])]

    for field in ("theta_rad", "angle_rad", "alpha_rad"):
        if field in pred:
            return [math.degrees(float(pred[field]))]

    for field in ("theta", "angle", "alpha"):
        if field in pred:
            raw = float(pred[field])
            return [raw, math.degrees(raw)]

    return []


# ============================================================
# 7. 评估 3D 方法：PVRCNN / Fusion
# ============================================================

def evaluate_3d_method(gt_map, pred_map, target_classes):
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in target_classes}

    common_keys = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    print(f"[3D Eval] Common frames: {len(common_keys)}")

    for key in common_keys:
        gt_frame = gt_map[key]
        pred_frame = pred_map[key]

        gts = [
            g for g in extract_gt_objects(gt_frame)
            if normalize_label(g["class_label"]) in target_classes
        ]

        preds = [
            p for p in extract_pred_objects(pred_frame)
            if normalize_label(p.get("label", p.get("class_label", "Unknown"))) in target_classes
        ]

        matched_gt = set()

        preds_sorted = sorted(preds, key=lambda x: extract_box_from_pred(x)[4], reverse=True)

        for pred in preds_sorted:
            p_center, p_dims, p_yaw, p_label, _ = extract_box_from_pred(pred)
            p_label = normalize_label(p_label)

            best_iou = -1.0
            best_gt_idx = None

            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt:
                    continue

                g_center, g_dims, g_yaw, g_label = extract_box_from_gt(gt)
                g_label = normalize_label(g_label)

                if not labels_match(p_label, g_label, allow_group_match=ALLOW_GROUP_MATCH):
                    continue

                iou = bev_iou(p_center, p_dims, p_yaw, g_center, g_dims, g_yaw)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx is not None and best_iou >= BEV_IOU_THRESH:
                stats[p_label]["tp"] += 1
                matched_gt.add(best_gt_idx)
            else:
                stats[p_label]["fp"] += 1

        for gt_idx, gt in enumerate(gts):
            if gt_idx not in matched_gt:
                _, _, _, g_label = extract_box_from_gt(gt)
                g_label = normalize_label(g_label)
                stats[g_label]["fn"] += 1

    return stats


# ============================================================
# 8. 评估 YOLO：用点云 GT 的 theta 比较
# ============================================================

def evaluate_yolo_method(gt_map, yolo_map, target_classes):
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in target_classes}

    common_keys = sorted(set(gt_map.keys()) & set(yolo_map.keys()))
    print(f"[YOLO Eval] Common frames: {len(common_keys)}")

    for key in common_keys:
        gt_frame = gt_map[key]
        yolo_frame = yolo_map[key]

        all_gts = extract_gt_objects(gt_frame)

        # 只保留目标类别 + 在相机FOV内的GT
        gts = []
        for g in all_gts:
            g_center, _, _, g_label = extract_box_from_gt(g)
            g_label = normalize_label(g_label)

            if g_label in target_classes and in_camera_fov(g_center):
                gts.append(g)

        preds = []
        for p in extract_pred_objects(yolo_frame):
            p_label = normalize_label(extract_yolo_label(p))
            if p_label in target_classes:
                preds.append(p)

        matched_gt = set()
        preds_sorted = sorted(preds, key=lambda x: float(x.get("score", 1.0)), reverse=True)

        for pred in preds_sorted:
            p_label = normalize_label(extract_yolo_label(pred))
            p_theta_candidates = extract_yolo_theta_candidates_deg(pred)
            if not p_theta_candidates:
                # 无方向信息无法参与 theta 匹配
                stats[p_label]["fp"] += 1
                continue

            best_diff = 1e9
            best_gt_idx = None

            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt:
                    continue

                g_center, _, _, g_label = extract_box_from_gt(gt)
                g_label = normalize_label(g_label)

                if not labels_match(p_label, g_label, allow_group_match=ALLOW_GROUP_MATCH):
                    continue

                g_theta = lidar_center_to_theta_deg(g_center)
                diff = min(angle_diff_deg(p_theta, g_theta) for p_theta in p_theta_candidates)

                if diff < best_diff:
                    best_diff = diff
                    best_gt_idx = gt_idx

            if best_gt_idx is not None and best_diff <= YOLO_THETA_THRESH_DEG:
                stats[p_label]["tp"] += 1
                matched_gt.add(best_gt_idx)
            else:
                stats[p_label]["fp"] += 1

        for gt_idx, gt in enumerate(gts):
            if gt_idx not in matched_gt:
                _, _, _, g_label = extract_box_from_gt(gt)
                g_label = normalize_label(g_label)
                stats[g_label]["fn"] += 1

    return stats


# ============================================================
# 9. 指标计算
# ============================================================

def calc_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def stats_to_rows(method_name, stats):
    rows = []
    for cls in TARGET_CLASSES:
        tp = stats[cls]["tp"]
        fp = stats[cls]["fp"]
        fn = stats[cls]["fn"]
        precision, recall, f1 = calc_metrics(tp, fp, fn)

        rows.append({
            "Method": method_name,
            "Class": cls,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
        })
    return rows


# ============================================================
# 10. 主程序
# ============================================================

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading frame maps...")
    gt_map = build_frame_map_from_dir(GT_DIR, max_frames=MAX_FRAMES)
    pv_map = build_frame_map_from_big_json(PVRCNN_PATH, max_frames=MAX_FRAMES)
    yolo_map = build_frame_map_from_big_json(YOLO_PATH, max_frames=MAX_FRAMES)
    fusion_map = build_frame_map_from_big_json(FUSION_PATH, max_frames=MAX_FRAMES)

    print(f"GT frames     : {len(gt_map)}")
    print(f"PVRCNN frames : {len(pv_map)}")
    print(f"YOLO frames   : {len(yolo_map)}")
    print(f"Fusion frames : {len(fusion_map)}")

    pv_stats = evaluate_3d_method(gt_map, pv_map, TARGET_CLASSES)
    yolo_stats = evaluate_yolo_method(gt_map, yolo_map, TARGET_CLASSES)
    fusion_stats = evaluate_3d_method(gt_map, fusion_map, TARGET_CLASSES)

    rows = []
    rows += stats_to_rows("PVRCNN", pv_stats)
    rows += stats_to_rows("YOLO", yolo_stats)
    rows += stats_to_rows("Fusion", fusion_stats)

    df = pd.DataFrame(rows)

    print(f"\n===== Detailed Results (First {MAX_FRAMES} Frames Only) =====")
    print(df.to_string(index=False))

    precision_table = df.pivot(index="Class", columns="Method", values="Precision")
    recall_table = df.pivot(index="Class", columns="Method", values="Recall")
    f1_table = df.pivot(index="Class", columns="Method", values="F1")

    print("\n===== Precision Table =====")
    print(precision_table)

    print("\n===== Recall Table =====")
    print(recall_table)

    print("\n===== F1 Table =====")
    print(f1_table)

    detail_csv = output_dir / "comparison_metrics_detail.csv"
    precision_csv = output_dir / "comparison_precision_table.csv"
    recall_csv = output_dir / "comparison_recall_table.csv"
    f1_csv = output_dir / "comparison_f1_table.csv"

    df.to_csv(detail_csv, index=False)
    precision_table.to_csv(precision_csv)
    recall_table.to_csv(recall_csv)
    f1_table.to_csv(f1_csv)

    print("\nSaved files:")
    print(f"  {detail_csv}")
    print(f"  {precision_csv}")
    print(f"  {recall_csv}")
    print(f"  {f1_csv}")


if __name__ == "__main__":
    main()
