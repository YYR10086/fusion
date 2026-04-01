import os
import json
import glob
import torch
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu

# ─────────────────────────────────────────────
# 配置区
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = str(BASE_DIR / "pv_rcnn.yaml")
CHECKPOINT = str(BASE_DIR / "large120.pth")
LIDAR_DIR = str(BASE_DIR / "record_output" / "record_output1" / "lidar_bin")
OUTPUT_JSON = str(BASE_DIR / "record_output" / "record_output1" / "lidar_detection_results.json")
LIDAR_PATTERN = "lidar_*.bin"
CONF_THRESH = 0.80
NUM_FEATURES = 4  # 点云特征数：[x, y, z, intensity]


# ─────────────────────────────────────────────

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None, ext='.bin',
                 num_features=4):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list
        self.num_features = num_features

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        read_out = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, self.num_features)

        if self.num_features == 3:
            points = np.zeros((read_out.shape[0], 4))
            points[:, :-1] = read_out
        else:
            if read_out[:, 3].max() > 1:
                read_out[:, 3] = read_out[:, 3] / 255
            points = read_out[:, :4]

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def create_logger():
    logger = logging.getLogger('pcdet')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger


def get_timestamp_from_file(filepath: str) -> str:
    try:
        dt = datetime.fromtimestamp(os.path.getmtime(filepath))
    except Exception:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def run_detection():
    if not os.path.isdir(LIDAR_DIR):
        raise FileNotFoundError(f"[ERROR] 点云目录不存在: {LIDAR_DIR}")

    print(f"[INFO] 点云目录  : {LIDAR_DIR}")
    print(f"[INFO] 配置文件  : {CONFIG_FILE}")
    print(f"[INFO] 权重文件  : {CHECKPOINT}")
    print(f"[INFO] 输出 JSON : {OUTPUT_JSON}\n")

    # 加载配置和模型
    logger = create_logger()
    cfg_from_yaml_file(CONFIG_FILE, cfg)

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(LIDAR_DIR),
        ext='.bin',
        logger=logger,
        num_features=NUM_FEATURES
    )

    print(f"[INFO] 共找到 {len(demo_dataset)} 个点云文件")

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=CHECKPOINT, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    print(f"[INFO] 类别列表: {cfg.CLASS_NAMES}\n")

    # 逐帧推理
    all_results = []

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            filename = os.path.basename(demo_dataset.sample_file_list[idx])
            timestamp = get_timestamp_from_file(demo_dataset.sample_file_list[idx])

            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pred_dict = pred_dicts[0]
            frame_detections = []

            if 'pred_boxes' in pred_dict and len(pred_dict['pred_boxes']) > 0:
                pred_boxes = pred_dict['pred_boxes'].cpu().numpy()
                pred_scores = pred_dict['pred_scores'].cpu().numpy()
                pred_labels = pred_dict['pred_labels'].cpu().numpy()

                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    if score < CONF_THRESH:
                        continue

                    x, y, z, dx, dy, dz, heading = box
                    class_label = cfg.CLASS_NAMES[int(label) - 1]

                    detection = {
                        "class_label": class_label,
                        "location": [round(float(x), 4), round(float(y), 4), round(float(z), 4)],
                        "box": {
                            "center": [round(float(x), 4), round(float(y), 4), round(float(z), 4)],
                            "dimensions": [round(float(dx), 4), round(float(dy), 4), round(float(dz), 4)],
                            "rotation": round(float(heading), 4)
                        },
                        "score": round(float(score), 4),
                        "timestamp": timestamp
                    }
                    frame_detections.append(detection)

            all_results.append({
                "frame_id": idx,
                "filename": filename,
                "num_points": len(data_dict['points']),
                "detections": frame_detections
            })

            print(f"  [{idx + 1:>3}/{len(demo_dataset)}] {filename}  →  检测到 {len(frame_detections)} 个目标")

    # 保存 JSON
    output = {
        "meta": {
            "total_frames": len(all_results),
            "model": "PV-RCNN",
            "config": CONFIG_FILE,
            "checkpoint": CHECKPOINT,
            "conf_thresh": CONF_THRESH,
            "class_names": cfg.CLASS_NAMES,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "frames": all_results
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] ✅ 检测完成！结果已保存至:\n       {OUTPUT_JSON}")


if __name__ == "__main__":
    run_detection()