import os
import json
import math
import glob
from datetime import datetime
from pathlib import Path

# 安装依赖: pip install ultralytics
from ultralytics import YOLO

#─────────────────────────────────────────────
# 配置区
# ─────────────────────────────────────────────
IMAGE_DIR    = "./record_output/record_output5/images"          # 图像文件夹路径
IMAGE_PATTERN = "image_*.jpg"      # 匹配文件名模式
MODEL_PATH= "yolo11n.pt"        # YOLO11 模型权重（自动下载）
OUTPUT_JSON  = "./record_output/record_output5/detection_results.json"
CONF_THRESH  = 0.80# 置信度阈值
FOV          = 150.0               # 摄像头水平视角（度）
# ─────────────────────────────────────────────

# 将 YOLO 原始细分类合并为统一大类，便于后续融合
YOLO_LABEL_MERGE = {
    # 车辆类
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "van": "car",
    "train": "truck",
    # 行人类
    "person": "pedestrian",
    # 骑行类
    "bicycle": "cyclist",
    "motorcycle": "cyclist",
    "rider": "cyclist",
}

TARGET_CLASSES = {"car", "bus", "truck", "pedestrian", "cyclist"}


def normalize_yolo_label(raw_label: str) -> str:
    """将 YOLO 原始标签映射为统一标签，不在目标集合中的返回空字符串。"""
    merged = YOLO_LABEL_MERGE.get(raw_label.lower(), "")
    return merged if merged in TARGET_CLASSES else ""

def compute_theta(bbox: list, img_width: int, fov: float = 150.0) -> float:
    """
    计算目标中心点偏离摄像头中轴线的角度theta。

    参数:
        bbox       : [x1, y1, x2, y2] 像素坐标
        img_width  : 图像宽度（像素）
        fov        : 摄像头水平视角（度），默认 150°

    返回:
        theta (float): 范围 [-75°, 75°]，负值=左偏，正值=右偏
    """
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2.0                # 目标水平中心像素
    img_cx = img_width / 2.0# 图像水平中心像素

    # 线性映射：像素偏移 → 角度偏移
    half_fov = fov / 2.0
    theta = ((cx - img_cx) / img_cx) * half_fov  # [-75, 75]
    theta = max(-half_fov, min(half_fov, theta))  # 截断保护
    return round(theta, 4)

def get_timestamp_from_file(filepath: str) -> str:
    """
    优先使用文件修改时间作为时间戳；
    若无法获取则使用当前时间。
    """
    try:
        mtime = os.path.getmtime(filepath)
        dt = datetime.fromtimestamp(mtime)
    except Exception:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def run_detection(
    image_dir: str,
    image_pattern: str,
    model_path: str,
    conf_thresh: float,
    fov: float,
    output_json: str,
) -> None:
    """
    主流程：
    1. 加载 YOLO11 模型
    2.遍历所有图像并推理
    3. 按指定数据结构保存 JSON
    """

    # ── 1. 加载模型 ──────────────────────────────
    print(f"[INFO] 加载模型: {model_path}")
    model = YOLO(model_path)

    # ── 2. 获取图像列表（按文件名排序）──────────────
    search_path = os.path.join(image_dir, image_pattern)
    image_files = sorted(glob.glob(search_path))

    if not image_files:
        raise FileNotFoundError(f"未找到图像文件，路径: {search_path}")

    print(f"[INFO] 共找到 {len(image_files)} 张图像，开始推理 ...")

    # ── 3.逐帧推理 ──────────────────────────────
    all_results = []   # 最终写入 JSON 的数据列表

    for idx, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        timestamp = get_timestamp_from_file(img_path)

        # YOLO 推理（单张图）
        results = model(img_path, conf=conf_thresh, verbose=False)
        result= results[0]                     # 取第一个（也是唯一一个）结果

        img_h, img_w = result.orig_shape[:2]     # 原始图像尺寸

        frame_detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                # 像素坐标 [x1, y1, x2, y2]
                xyxy  = box.xyxy[0].tolist()
                bbox  = [round(v, 2) for v in xyxy]

                # 置信度
                score = round(float(box.conf[0]), 4)

                # 类别名称
                cls_id = int(box.cls[0])
                raw_label = model.names[cls_id]
                label = normalize_yolo_label(raw_label)
                if not label:
                    continue

                # 偏角 theta
                theta  = compute_theta(bbox, img_w, fov)

                detection = {
                    "label": label,
                    "score"       : score,
                    "bbox_camera" : bbox,         # [x1, y1, x2, y2]
                    "theta"       : theta,        # 单位：度
                    "timestamp"   : timestamp,
                }
                frame_detections.append(detection)

        frame_result = {
            "frame_id"   : idx,                # 帧序号（0-based）
            "filename"   : filename,
            "image_size" : [img_w, img_h],        # [width, height]
            "detections" : frame_detections,
        }

        all_results.append(frame_result)
        print(f"  [{idx+1:>3}/{len(image_files)}] {filename}"
              f"  → 检测到 {len(frame_detections)} 个目标")

    # ── 4. 保存 JSON ─────────────────────────────
    output = {
        "meta": {
            "total_frames" : len(all_results),
            "model": model_path,
            "conf_thresh"  : conf_thresh,
            "fov_deg"      : fov,
            "generated_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "frames": all_results,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] 结果已保存至: {output_json}")

# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_detection(
        image_dir     = IMAGE_DIR,
        image_pattern = IMAGE_PATTERN,
        model_path= MODEL_PATH,
        conf_thresh   = CONF_THRESH,
        fov           = FOV,
        output_json   = OUTPUT_JSON,
    )
