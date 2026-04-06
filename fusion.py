import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Fusion")

# ============================================================
# PART 1  输入数据接口定义# YOLO 检测结果接口
# label        : 类别字符串，如 "person", "car", "bicycle"
# score        : 置信度 [0.0, 1.0]
# bbox_camera  : 图像像素坐标 [x1, y1, x2, y2]
# theta        : 物体偏离摄像头中轴线的角度，范围 [-75°, 75°]（摄像头FOV=150°）
# timestamp    : 时间戳字符串，格式 "YYYY-MM-DD HH:MM:SS"
# ============================================================
# class_label  : 类别字符串，如 "Car", "Pedestrian", "Cyclist"
# location     : 3D 中心坐标 [x, y, z]，激光雷达坐标系（单位：米）
# box.center      : 同 location
# box.dimensions  : [长, 宽, 高]（单位：米）
# box.rotation    : 偏航角 heading（弧度），激光雷达坐标系
# score        : 置信度 [0.0, 1.0]
# timestamp    : 时间戳字符串，格式 "YYYY-MM-DD HH:MM:SS"


# ============================================================
# PART 2  超参数
# ============================================================
CAMERA_FOV_DEG  = 150.0
CAMERA_FOV_MARGIN_DEG = 8.0      # 视场边缘安全裕量，防止边界误检
CAMERA_MAX_RANGE_M = 45.0        # 超过该距离默认不参与相机融合
MIN_VISIBLE_PIXEL_W = 12.0       # 预计投影过窄时视为相机不可见
IMAGE_WIDTH     = 640
IMAGE_HEIGHT    = 480
IOU_THRESH      = 0.1
STRICT_IOU_MATCH_THRESH = 0.2      # 提高精度：跨模态匹配需更高 IoU
W_PVRCNN        = 0.85
W_YOLO          = 0.15
FUSED_THRESH    = 0.1
THETA_MATCH_DEG = 10.0
THETA_SIGMA_DEG = 6.0
MIN_THETA_SIM   = 0.35
STRICT_THETA_SIM_THRESH = 0.5      # 提高精度：theta 模式匹配需更高相似度
UNMATCHED_YOLO_MIN_SCORE = 0.25
YOLO_HIGH_CONF_THRESH = 0.75
PVRCNN_HIGH_CONF_KEEP = 0.8
DEFAULT_INCLUDE_UNMATCHED_YOLO = False
CAMERA_ONLY_CLASS_CFG: Dict[str, Dict[str, float]] = {
    "car": {"min_score": 0.68, "min_streak": 2},        # 车辆宽松
    "truck": {"min_score": 0.78, "min_streak": 2},
    "bus": {"min_score": 0.82, "min_streak": 2},
    "pedestrian": {"min_score": 0.90, "min_streak": 3}, # 行人严格
    "cyclist": {"min_score": 0.90, "min_streak": 3},    # 骑行者严格
}
CAMERA_ONLY_MAX_THETA_GAP_DEG = 4.0
TIMESTAMP_TOL_S = 1
MAX_MISS_FRAMES = 2
OUTPUT_DIR      = "./fusion_output"
CALIB_PATH      = "calib.txt"# kitti格式标定文件路径

CANONICAL_LABELS = {"car", "bus", "truck", "pedestrian", "cyclist"}

# YOLO 原始标签 -> 统一标签
YOLO_TO_CANONICAL: Dict[str, str] = {
    # 已经规范化过的标签（来自 data_yolo_test.py）
    "pedestrian": "pedestrian",
    "cyclist": "cyclist",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "van": "car",
    "train": "truck",
    "motorcycle": "cyclist",
    "bicycle": "cyclist",
    "person": "pedestrian",
    "rider": "cyclist",
}

# PVRCNN 原始标签 -> 统一标签
PVRCNN_TO_CANONICAL: Dict[str, str] = {
    "car": "car",
    "van": "car",
    "bus": "bus",
    "truck": "truck",
    "pedestrian": "pedestrian",
    "cyclist": "cyclist",
}

CLASS_SIZE_PRIOR: Dict[str, List[float]] = {
    "car": [4.2, 1.8, 1.6],
    "bus": [11.0, 2.5, 3.2],
    "truck": [8.0, 2.5, 3.2],
    "pedestrian": [0.8, 0.8, 1.75],
    "cyclist": [1.8, 0.7, 1.7],
}

def normalize_yolo_label(label: str) -> str:
    """将 YOLO 标签标准化为五大类之一；未知返回空字符串。"""
    return YOLO_TO_CANONICAL.get(str(label).lower(), "")

def normalize_pvrcnn_label(label: str) -> str:
    """将 PVRCNN 标签标准化为五大类之一；未知返回空字符串。"""
    normalized = PVRCNN_TO_CANONICAL.get(str(label).lower(), "")
    return normalized if normalized in CANONICAL_LABELS else ""

def category_compatibility(yolo_label: str, pvrcnn_label: str) -> float:
    """
    类别兼容分数：
    - 1.0：强一致（映射后完全一致）
    - 0.0：不兼容
    """
    yolo_norm = normalize_yolo_label(yolo_label)
    pvrcnn_norm = normalize_pvrcnn_label(pvrcnn_label)
    if not yolo_norm or not pvrcnn_norm:
        return 0.0
    if yolo_norm == pvrcnn_norm:
        return 1.0
    return 0.0

# ============================================================
# PART 3  标定参数解析（读取失败自动降级到 theta 模式）
# ============================================================
class KITTICalib:
    """
    尝试从 calib_path 读取 KITTI 标定文件。
    读取成功：use_calib=True，使用完整投影链做匹配。
    读取失败：use_calib=False，降级到 theta 粗估模式，并打印警告。
    """

    def __init__(self, calib_path: str):
        self.use_calib = False
        self.P2             = None
        self.R0_rect        = None
        self.Tr_velo_to_cam = None
        self.K              = None

        try:
            data = self._parse(calib_path)
            self.P2             = data["P2"].reshape(3, 4)

            R0 = data["R0_rect"].reshape(3, 3)
            self.R0_rect        = np.eye(4)
            self.R0_rect[:3, :3] = R0

            Tr = data["Tr_velo_to_cam"].reshape(3, 4)
            self.Tr_velo_to_cam = np.eye(4)
            self.Tr_velo_to_cam[:3, :] = Tr

            self.K         = self.P2[:3, :3]
            self.use_calib = True
            logger.info(f"标定文件加载成功: {calib_path}")

        except FileNotFoundError:
            logger.warning(f"标定文件未找到: {calib_path}，降级到 theta 粗估模式")
        except KeyError as e:
            logger.warning(f"标定文件缺少字段 {e}: {calib_path}，降级到 theta 粗估模式")
        except Exception as e:
            logger.warning(f"标定文件解析失败 ({e})，降级到 theta 粗估模式")

    @staticmethod
    def _parse(path: str) -> Dict[str, np.ndarray]:
        data = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                data[key.strip()] = np.array(
                    [float(x) for x in val.strip().split()]
                )
        return data

    def lidar_to_img(self, pts_lidar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(N,3) 激光雷达点 → 图像像素坐标 (N,2)，同时返回有效掩码"""
        N    = pts_lidar.shape[0]
        pts_h   = np.hstack([pts_lidar, np.ones((N, 1))])
        pts_cam = (self.R0_rect @ self.Tr_velo_to_cam @ pts_h.T).T

        in_front = pts_cam[:, 2] > 0

        pts_img        = (self.P2 @ pts_cam.T).T
        pts_img[:, 0] /= pts_img[:, 2]
        pts_img[:, 1] /= pts_img[:, 2]
        uv = pts_img[:, :2]

        in_image = (
            (uv[:, 0] >= 0) & (uv[:, 0] < IMAGE_WIDTH) &
            (uv[:, 1] >= 0) & (uv[:, 1] < IMAGE_HEIGHT)
        )
        return uv, in_front & in_image

    def box3d_to_bbox2d(self, center: List[float], dimensions: List[float],
                         heading: float) -> Optional[List[float]]:
        """3D 框 8 角点投影 → 图像 2D bbox [x1,y1,x2,y2]，不可见返回 None"""
        cx, cy, cz = center
        l, w, h    = dimensions
        hl, hw, hh = l/2, w/2, h/2

        corners_local = np.array([
            [ hl,  hw, -hh], [ hl, -hw, -hh],
            [-hl, -hw, -hh], [-hl,  hw, -hh],
            [ hl,  hw,  hh], [ hl, -hw,  hh],
            [-hl, -hw,  hh], [-hl,  hw,  hh],
        ])
        c, s = np.cos(heading), np.sin(heading)
        rot  = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        corners_lidar = (rot @ corners_local.T).T + np.array([cx, cy, cz])

        uv, valid = self.lidar_to_img(corners_lidar)
        if valid.sum() == 0:
            return None

        uv_v = uv[valid]
        x1 = float(np.clip(uv_v[:, 0].min(), 0, IMAGE_WIDTH  - 1))
        y1 = float(np.clip(uv_v[:, 1].min(), 0, IMAGE_HEIGHT - 1))
        x2 = float(np.clip(uv_v[:, 0].max(), 0, IMAGE_WIDTH  - 1))
        y2 = float(np.clip(uv_v[:, 1].max(), 0, IMAGE_HEIGHT - 1))

        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return None
        return [x1, y1, x2, y2]

# ============================================================
# PART 4  工具函数
# ============================================================
def parse_timestamp(ts: str) -> Optional[datetime]:
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def timestamps_aligned(ts1: str, ts2: str, tol_s: float = TIMESTAMP_TOL_S) -> bool:
    t1, t2 = parse_timestamp(ts1), parse_timestamp(ts2)
    if t1 is None or t2 is None:
        return True
    return abs((t1 - t2).total_seconds()) <= tol_s

def compute_iou_2d(a: List[float], b: List[float]) -> float:
    u1 = max(a[0], b[0]); v1 = max(a[1], b[1])
    u2 = min(a[2], b[2]); v2 = min(a[3], b[3])
    inter  = max(0.0, u2 - u1) * max(0.0, v2 - v1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-6))

def pvrcnn_to_unified(d: Dict) -> Dict:
    cx, cy, cz = d["box"]["center"]
    dx, dy, dz = d["box"]["dimensions"]
    return {
        "label"     : normalize_pvrcnn_label(d["class_label"]),
        "score"     : float(d["score"]),
        "center"    : [cx, cy, cz],
        "dimensions": [dx, dy, dz],
        "heading"   : d["box"]["rotation"],
        "timestamp" : d.get("timestamp", ""),
    }

def yolo_to_unified(d: Dict) -> Dict:
    return {
        "label"    : normalize_yolo_label(d["label"]),
        "score"    : float(d["score"]),
        "bbox"     : d["bbox_camera"],
        "theta"    : float(d.get("theta", 0.0)),
        "timestamp": d.get("timestamp", ""),
    }

def lidar_center_to_theta(center: List[float], fov: float = CAMERA_FOV_DEG,
                          clip_to_fov: bool = True) -> float:
    """
    将激光雷达 3D 中心点 [x, y, z] 转换为摄像头视角 theta（度）。
    激光雷达坐标系：x 向前，y 向左。
    摄像头中轴线沿 x 轴，theta = arctan2(y, x) 转换为角度。
    """
    x, y, _ = center
    theta_rad = np.arctan2(y, x)  # 水平偏角（弧度）
    theta_deg = np.degrees(theta_rad)
    if clip_to_fov:
        theta_deg = np.clip(theta_deg, -fov / 2, fov / 2)
    return float(theta_deg)

def lidar_in_camera_fov(center: List[float], fov: float = CAMERA_FOV_DEG) -> bool:
    """判断激光雷达目标中心是否落在相机水平视场范围内。"""
    theta = lidar_center_to_theta(center, fov=fov, clip_to_fov=False)
    return abs(theta) <= (fov / 2.0)

def lidar_in_strict_camera_view(center: List[float], fov: float = CAMERA_FOV_DEG,
                                max_range_m: float = CAMERA_MAX_RANGE_M,
                                margin_deg: float = CAMERA_FOV_MARGIN_DEG) -> bool:
    """更严格的可见性约束：同时满足角度和距离。"""
    x, y, _ = center
    dist_xy = float(np.hypot(x, y))
    if dist_xy > max_range_m:
        return False
    theta = lidar_center_to_theta(center, fov=fov, clip_to_fov=False)
    return abs(theta) <= max((fov / 2.0) - margin_deg, 1.0)

def lidar_in_strict_camera_view(center: List[float], fov: float = CAMERA_FOV_DEG,
                                max_range_m: float = CAMERA_MAX_RANGE_M,
                                margin_deg: float = CAMERA_FOV_MARGIN_DEG) -> bool:
    """更严格的可见性约束：同时满足角度和距离。"""
    x, y, _ = center
    dist_xy = float(np.hypot(x, y))
    if dist_xy > max_range_m:
        return False
    theta = lidar_center_to_theta(center, fov=fov, clip_to_fov=False)
    return abs(theta) <= max((fov / 2.0) - margin_deg, 1.0)

def lidar_in_strict_camera_view(center: List[float], fov: float = CAMERA_FOV_DEG,
                                max_range_m: float = CAMERA_MAX_RANGE_M,
                                margin_deg: float = CAMERA_FOV_MARGIN_DEG) -> bool:
    """更严格的可见性约束：同时满足角度和距离。"""
    x, y, _ = center
    dist_xy = float(np.hypot(x, y))
    if dist_xy > max_range_m:
        return False
    theta = lidar_center_to_theta(center, fov=fov, clip_to_fov=False)
    return abs(theta) <= max((fov / 2.0) - margin_deg, 1.0)

def lidar_in_strict_camera_view(center: List[float], fov: float = CAMERA_FOV_DEG,
                                max_range_m: float = CAMERA_MAX_RANGE_M,
                                margin_deg: float = CAMERA_FOV_MARGIN_DEG) -> bool:
    """更严格的可见性约束：同时满足角度和距离。"""
    x, y, _ = center
    dist_xy = float(np.hypot(x, y))
    if dist_xy > max_range_m:
        return False
    theta = lidar_center_to_theta(center, fov=fov, clip_to_fov=False)
    return abs(theta) <= max((fov / 2.0) - margin_deg, 1.0)

def lidar_in_strict_camera_view(center: List[float], fov: float = CAMERA_FOV_DEG,
                                max_range_m: float = CAMERA_MAX_RANGE_M,
                                margin_deg: float = CAMERA_FOV_MARGIN_DEG) -> bool:
    """更严格的可见性约束：同时满足角度和距离。"""
    x, y, _ = center
    dist_xy = float(np.hypot(x, y))
    if dist_xy > max_range_m:
        return False
    theta = lidar_center_to_theta(center, fov=fov, clip_to_fov=False)
    return abs(theta) <= max((fov / 2.0) - margin_deg, 1.0)


def angle_diff_deg(a: float, b: float) -> float:
    """返回两个角度（度）之间的最小差值。"""
    d = abs(a - b) % 360.0
    return float(min(d, 360.0 - d))

def wrap_angle_deg(a: float) -> float:
    """角度归一化到 [-180, 180)。"""
    return float((a + 180.0) % 360.0 - 180.0)

def yolo_theta_to_fusion(theta_deg: float) -> float:
    """
    将 YOLO/theta（相机平面）角度转换到融合坐标系角度。
    当前标定关系：相机结果相对融合坐标逆时针旋转 90°。
    """
    return wrap_angle_deg(float(theta_deg) + 90.0)


def theta_to_proj_bbox(det3d: Dict, img_width: int = IMAGE_WIDTH,
                       img_height: int = IMAGE_HEIGHT,
                       fov: float = CAMERA_FOV_DEG) -> List[float]:
    """
    theta 降级模式：
    - 用激光雷达 3D 中心计算其对应摄像头的水平角 theta_lidar
    - 将 theta_lidar 映射到图像 x 坐标作为 3D 框投影中心
    - y 范围沿用 YOLO 的 bbox（垂直方向无 theta 信息）
    - 宽度估算：用激光雷达目标宽度/距离做透视近似，或回落到 YOLO bbox 宽度
    """
    theta_lidar = lidar_center_to_theta(det3d["center"], fov)
    x_center    = (theta_lidar / fov + 0.5) * img_width

    # 估算投影宽度：利用激光雷达宽度和距离做简单透视
    dist_xy = max(np.sqrt(det3d["center"][0]**2 + det3d["center"][1]**2), 0.1)
    # 粗略焦距 f ≈ img_width / (2 * tan(fov/2))
    fov_rad  = np.radians(fov)
    f_approx = img_width / (2.0 * np.tan(fov_rad / 2.0))
    proj_w   = f_approx * det3d["dimensions"][1] / dist_xy  # 用激光雷达宽度

    half_w  = max(proj_w / 2.0, 8.0)  # 至少 8px 防止退化

    # theta 模式下缺少可靠垂直信息，使用一个保守的默认高度占位
    box_h = max(proj_w * 1.4, 20.0)
    y2 = img_height * 0.85
    y1 = y2 - box_h

    return [x_center - half_w, y1, x_center + half_w, y2]

def find_lidar_hint_by_theta(yolo_det: Dict, det3d: List[Dict],
                             max_theta_diff: float = THETA_MATCH_DEG) -> Optional[Dict]:
    """基于 YOLO 角度在激光雷达结果中查找最近目标，作为提示信息。"""
    best = None
    best_diff = float("inf")
    for d3 in det3d:
        if d3["label"] != yolo_det["label"]:
            continue
        theta_lidar = lidar_center_to_theta(d3["center"])
        diff = angle_diff_deg(theta_lidar, yolo_theta_to_fusion(yolo_det.get("theta", 0.0)))
        if diff < best_diff:
            best_diff = diff
            best = d3
    if best is None or best_diff > max_theta_diff:
        return None
    return {"det3d": best, "theta_diff": best_diff}

def estimate_yolo_only_3d(yolo_det: Dict) -> Dict:
    """
    当 YOLO 置信度很高且当前无匹配激光雷达时，估计一个保守 3D 目标，
    仅作为提示，不覆盖激光雷达结果。
    """
    label = yolo_det["label"]
    dims = CLASS_SIZE_PRIOR.get(label, [4.0, 1.8, 1.6])
    bbox = yolo_det["bbox"]
    pix_h = max(bbox[3] - bbox[1], 1.0)
    # 经验距离估计：高像素高度 -> 近；低像素高度 -> 远
    est_range = float(np.clip(800.0 / pix_h, 6.0, CAMERA_MAX_RANGE_M))
    theta_deg = yolo_det.get("theta", 0.0)
    theta_rad = np.radians(theta_deg)
    # 相机角度坐标与当前融合坐标系存在 90° 逆时针旋转差
    # 原始相机平面坐标: (x_cam, y_cam) = (r*cosθ, r*sinθ)
    # 旋转后映射到融合坐标: (x_fuse, y_fuse) = (-y_cam, x_cam)
    x_cam = est_range * np.cos(theta_rad)
    y_cam = est_range * np.sin(theta_rad)
    x = -y_cam
    y = x_cam
    heading_fuse = float(theta_rad + np.pi / 2.0)
    return {
        "center": [round(float(x), 3), round(float(y), 3), 0.0],
        "dimensions": dims,
        "heading": heading_fuse,
    }

def find_lidar_hint_by_theta(yolo_det: Dict, det3d: List[Dict],
                             max_theta_diff: float = THETA_MATCH_DEG) -> Optional[Dict]:
    """基于 YOLO 角度在激光雷达结果中查找最近目标，作为提示信息。"""
    best = None
    best_diff = float("inf")
    for d3 in det3d:
        if d3["label"] != yolo_det["label"]:
            continue
        theta_lidar = lidar_center_to_theta(d3["center"])
        diff = angle_diff_deg(theta_lidar, yolo_theta_to_fusion(yolo_det.get("theta", 0.0)))
        if diff < best_diff:
            best_diff = diff
            best = d3
    if best is None or best_diff > max_theta_diff:
        return None
    return {"det3d": best, "theta_diff": best_diff}

def estimate_yolo_only_3d(yolo_det: Dict) -> Dict:
    """
    当 YOLO 置信度很高且当前无匹配激光雷达时，估计一个保守 3D 目标，
    仅作为提示，不覆盖激光雷达结果。
    """
    label = yolo_det["label"]
    dims = CLASS_SIZE_PRIOR.get(label, [4.0, 1.8, 1.6])
    bbox = yolo_det["bbox"]
    pix_h = max(bbox[3] - bbox[1], 1.0)
    # 经验距离估计：高像素高度 -> 近；低像素高度 -> 远
    est_range = float(np.clip(800.0 / pix_h, 6.0, CAMERA_MAX_RANGE_M))
    theta_deg = yolo_det.get("theta", 0.0)
    theta_rad = np.radians(theta_deg)

    # 相机角度坐标与当前融合坐标系存在 90° 逆时针旋转差
    # 原始相机平面坐标: (x_cam, y_cam) = (r*cosθ, r*sinθ)
    # 旋转后映射到融合坐标: (x_fuse, y_fuse) = (-y_cam, x_cam)
    x_cam = est_range * np.cos(theta_rad)
    y_cam = est_range * np.sin(theta_rad)
    x = -y_cam
    y = x_cam
    heading_fuse = float(theta_rad + np.pi / 2.0)
    return {
        "center": [round(float(x), 3), round(float(y), 3), 0.0],
        "dimensions": dims,
        "heading": heading_fuse,
    }

def find_lidar_hint_by_theta(yolo_det: Dict, det3d: List[Dict],
                             max_theta_diff: float = THETA_MATCH_DEG) -> Optional[Dict]:
    """基于 YOLO 角度在激光雷达结果中查找最近目标，作为提示信息。"""
    best = None
    best_diff = float("inf")
    for d3 in det3d:
        if d3["label"] != yolo_det["label"]:
            continue
        theta_lidar = lidar_center_to_theta(d3["center"])
        diff = angle_diff_deg(theta_lidar, yolo_theta_to_fusion(yolo_det.get("theta", 0.0)))
        if diff < best_diff:
            best_diff = diff
            best = d3
    if best is None or best_diff > max_theta_diff:
        return None
    return {"det3d": best, "theta_diff": best_diff}

def estimate_yolo_only_3d(yolo_det: Dict) -> Dict:
    """
    当 YOLO 置信度很高且当前无匹配激光雷达时，估计一个保守 3D 目标，
    仅作为提示，不覆盖激光雷达结果。
    """
    label = yolo_det["label"]
    dims = CLASS_SIZE_PRIOR.get(label, [4.0, 1.8, 1.6])
    bbox = yolo_det["bbox"]
    pix_h = max(bbox[3] - bbox[1], 1.0)
    # 经验距离估计：高像素高度 -> 近；低像素高度 -> 远
    est_range = float(np.clip(800.0 / pix_h, 6.0, CAMERA_MAX_RANGE_M))
    theta_deg = yolo_det.get("theta", 0.0)
    theta_rad = np.radians(theta_deg)

    # 相机角度坐标与当前融合坐标系存在 90° 逆时针旋转差
    # 原始相机平面坐标: (x_cam, y_cam) = (r*cosθ, r*sinθ)
    # 旋转后映射到融合坐标: (x_fuse, y_fuse) = (-y_cam, x_cam)
    x_cam = est_range * np.cos(theta_rad)
    y_cam = est_range * np.sin(theta_rad)
    x = -y_cam
    y = x_cam
    heading_fuse = float(theta_rad + np.pi / 2.0)
    return {
        "center": [round(float(x), 3), round(float(y), 3), 0.0],
        "dimensions": dims,
        "heading": heading_fuse,
    }

def find_lidar_hint_by_theta(yolo_det: Dict, det3d: List[Dict],
                             max_theta_diff: float = THETA_MATCH_DEG) -> Optional[Dict]:
    """基于 YOLO 角度在激光雷达结果中查找最近目标，作为提示信息。"""
    best = None
    best_diff = float("inf")
    for d3 in det3d:
        if d3["label"] != yolo_det["label"]:
            continue
        theta_lidar = lidar_center_to_theta(d3["center"])
        diff = angle_diff_deg(theta_lidar, yolo_theta_to_fusion(yolo_det.get("theta", 0.0)))
        if diff < best_diff:
            best_diff = diff
            best = d3
    if best is None or best_diff > max_theta_diff:
        return None
    return {"det3d": best, "theta_diff": best_diff}

def estimate_yolo_only_3d(yolo_det: Dict) -> Dict:
    """
    当 YOLO 置信度很高且当前无匹配激光雷达时，估计一个保守 3D 目标，
    仅作为提示，不覆盖激光雷达结果。
    """
    label = yolo_det["label"]
    dims = CLASS_SIZE_PRIOR.get(label, [4.0, 1.8, 1.6])
    bbox = yolo_det["bbox"]
    pix_h = max(bbox[3] - bbox[1], 1.0)
    # 经验距离估计：高像素高度 -> 近；低像素高度 -> 远
    est_range = float(np.clip(800.0 / pix_h, 6.0, CAMERA_MAX_RANGE_M))
    theta_deg = yolo_det.get("theta", 0.0)
    theta_rad = np.radians(theta_deg)

    # 相机角度坐标与当前融合坐标系存在 90° 逆时针旋转差
    # 原始相机平面坐标: (x_cam, y_cam) = (r*cosθ, r*sinθ)
    # 旋转后映射到融合坐标: (x_fuse, y_fuse) = (-y_cam, x_cam)
    x_cam = est_range * np.cos(theta_rad)
    y_cam = est_range * np.sin(theta_rad)
    x = -y_cam
    y = x_cam
    heading_fuse = float(theta_rad + np.pi / 2.0)
    return {
        "center": [round(float(x), 3), round(float(y), 3), 0.0],
        "dimensions": dims,
        "heading": heading_fuse,
    }

def find_lidar_hint_by_theta(yolo_det: Dict, det3d: List[Dict],
                             max_theta_diff: float = THETA_MATCH_DEG) -> Optional[Dict]:
    """基于 YOLO 角度在激光雷达结果中查找最近目标，作为提示信息。"""
    best = None
    best_diff = float("inf")
    for d3 in det3d:
        if d3["label"] != yolo_det["label"]:
            continue
        theta_lidar = lidar_center_to_theta(d3["center"])
        diff = angle_diff_deg(theta_lidar, yolo_theta_to_fusion(yolo_det.get("theta", 0.0)))
        if diff < best_diff:
            best_diff = diff
            best = d3
    if best is None or best_diff > max_theta_diff:
        return None
    return {"det3d": best, "theta_diff": best_diff}

def estimate_yolo_only_3d(yolo_det: Dict) -> Dict:
    """
    当 YOLO 置信度很高且当前无匹配激光雷达时，估计一个保守 3D 目标，
    仅作为提示，不覆盖激光雷达结果。
    """
    label = yolo_det["label"]
    dims = CLASS_SIZE_PRIOR.get(label, [4.0, 1.8, 1.6])
    bbox = yolo_det["bbox"]
    pix_h = max(bbox[3] - bbox[1], 1.0)
    # 经验距离估计：高像素高度 -> 近；低像素高度 -> 远
    est_range = float(np.clip(800.0 / pix_h, 6.0, CAMERA_MAX_RANGE_M))
    theta_deg = yolo_det.get("theta", 0.0)
    theta_rad = np.radians(theta_deg)

    # 相机角度坐标与当前融合坐标系存在 90° 逆时针旋转差
    # 原始相机平面坐标: (x_cam, y_cam) = (r*cosθ, r*sinθ)
    # 旋转后映射到融合坐标: (x_fuse, y_fuse) = (-y_cam, x_cam)
    x_cam = est_range * np.cos(theta_rad)
    y_cam = est_range * np.sin(theta_rad)
    x = -y_cam
    y = x_cam
    heading_fuse = float(theta_rad + np.pi / 2.0)
    return {
        "center": [round(float(x), 3), round(float(y), 3), 0.0],
        "dimensions": dims,
        "heading": heading_fuse,
    }

# ============================================================
# PART 5  卡尔曼滤波器（单目标）
# ============================================================
class KalmanTracker:
    _id_counter = 0

    def __init__(self, x: float, y: float, label: str, timestamp: str):
        KalmanTracker._id_counter += 1
        self.track_id   = KalmanTracker._id_counter
        self.label      = label
        self.miss_count = 0
        self.last_ts    = timestamp
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 100.0, 100.0])
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        self.R = np.diag([1.0, 1.0])

    def _F(self, dt):
        return np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float64)

    def _Q(self, dt):
        s, dt2, dt3, dt4 = 2.0, dt**2, dt**3, dt**4
        return s**2 * np.array([
            [dt4/4,0,dt3/2,0],[0,dt4/4,0,dt3/2],
            [dt3/2,0,dt2,0],  [0,dt3/2,0,dt2]
        ], dtype=np.float64)

    def predict(self, timestamp: str) -> np.ndarray:
        dt     = max(self._calc_dt(timestamp), 1e-3)
        F      = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        self.last_ts = timestamp
        return self.x[:2].copy()

    def update(self, x: float, y: float) -> np.ndarray:
        z      = np.array([x, y], dtype=np.float64)
        S      = self.H @ self.P @ self.H.T + self.R
        K      = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.miss_count = 0
        return self.x[:2].copy()

    def get_velocity(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    def _calc_dt(self, new_ts: str) -> float:
        t0 = parse_timestamp(self.last_ts)
        t1 = parse_timestamp(new_ts)
        if t0 is None or t1 is None:
            return 0.1
        return max((t1 - t0).total_seconds(), 0.0)

# ============================================================
# PART 6  多目标跟踪管理器
# ============================================================
class MultiObjectTracker:
    def __init__(self, max_miss: int = MAX_MISS_FRAMES, match_dist: float = 5.0):
        self.tracks     : List[KalmanTracker] = []
        self.max_miss   = max_miss
        self.match_dist = match_dist

    def update(self, detections: List[Dict], timestamp: str) -> List[Dict]:
        if not self.tracks:
            return [self._attach(dict(d), self._new_track(d, timestamp), [0.0, 0.0])
                    for d in detections]

        pred_pos = np.array([t.predict(timestamp) for t in self.tracks])
        det_pos  = np.array([[d["center"][0], d["center"][1]] for d in detections])
        dist_mat = np.linalg.norm(pred_pos[:, None] - det_pos[None, :], axis=2)

        row_ind, col_ind = linear_sum_assignment(dist_mat)
        matched_t, matched_d, pairs = set(), set(), {}
        for r, c in zip(row_ind, col_ind):
            if (dist_mat[r, c] <= self.match_dist and
                    self.tracks[r].label == detections[c]["label"]):
                pairs[r] = c; matched_t.add(r); matched_d.add(c)

        smoothed = []
        for t_idx, d_idx in pairs.items():
            d      = detections[d_idx]
            trk    = self.tracks[t_idx]
            sx, sy = trk.update(d["center"][0], d["center"][1])
            out    = dict(d)
            out["center"] = [sx, sy, d["center"][2]]
            smoothed.append(self._attach(out, trk, list(trk.get_velocity())))

        for i in range(len(self.tracks)):
            if i not in matched_t:
                self.tracks[i].miss_count += 1
        self.tracks = [t for t in self.tracks if t.miss_count <= self.max_miss]

        for j in range(len(detections)):
            if j not in matched_d:
                d   = detections[j]
                trk = self._new_track(d, timestamp)
                smoothed.append(self._attach(dict(d), trk, [0.0, 0.0]))

        smoothed.sort(key=lambda x: x["fused_score"], reverse=True)
        return smoothed

    def _new_track(self, d: Dict, ts: str) -> KalmanTracker:
        trk = KalmanTracker(d["center"][0], d["center"][1], d["label"], ts)
        self.tracks.append(trk)
        return trk

    @staticmethod
    def _attach(d: Dict, trk: KalmanTracker, vel: List[float]) -> Dict:
        d["track_id"] = trk.track_id
        d["velocity"] = [round(vel[0], 3), round(vel[1], 3)]
        return d

# ============================================================
# PART 7  融合
# ============================================================
def fuse(
        pvrcnn_raw  : List[Dict],
        yolo_raw    : List[Dict],
        calib       : KITTICalib,
        w_pvrcnn    : float = W_PVRCNN,
        w_yolo      : float = W_YOLO,
        fused_thresh: float = FUSED_THRESH,
        include_unmatched_yolo: bool = DEFAULT_INCLUDE_UNMATCHED_YOLO,
        camera_fov_only: bool = True,
        unmatched_yolo_min_score: float = UNMATCHED_YOLO_MIN_SCORE,
) -> List[Dict]:

    det3d_all = [pvrcnn_to_unified(d) for d in pvrcnn_raw]
    det3d = det3d_all
    det2d = [yolo_to_unified(d) for d in yolo_raw]
    det3d = [d for d in det3d if d["label"]]
    det2d = [d for d in det2d if d["label"]]
    if camera_fov_only:
        det3d_with_vis = []
        for d in det3d:
            visible = lidar_in_strict_camera_view(d["center"], CAMERA_FOV_DEG)
            if visible or d["score"] >= PVRCNN_HIGH_CONF_KEEP:
                d2 = dict(d)
                d2["camera_visible"] = visible
                det3d_with_vis.append(d2)
        logger.info(f"PVRCNN FOV过滤(保留高置信): {len(det3d)} -> {len(det3d_with_vis)}")
        det3d = det3d_with_vis
    else:
        det3d = [dict(d, camera_visible=True) for d in det3d]
    logger.info(f"YOLO 原始数量: {len(yolo_raw)}")
    logger.info(f"YOLO 转换后数量: {len(det2d)}")
    if not det3d:
        # 没有可用 LiDAR 时，直接返回 YOLO 提示结果（高置信可估计3D）
        yolo_only_fused = []
        for d2 in det2d:
            if d2["score"] < unmatched_yolo_min_score:
                continue
            if d2["score"] >= YOLO_HIGH_CONF_THRESH:
                est_3d = estimate_yolo_only_3d(d2)
                center = est_3d["center"]
                dims = est_3d["dimensions"]
                heading = est_3d["heading"]
                source = "yolo_estimated"
            else:
                center = [0, 0, 0]
                dims = [0, 0, 0]
                heading = 0
                source = "yolo_only"
            yolo_only_fused.append({
                "label": d2["label"],
                "score": round(d2["score"], 4),
                "fused_score": round(min(d2["score"] * 0.7, 0.69), 4),
                "center": center,
                "dimensions": dims,
                "heading": heading,
                "proj_bbox": d2["bbox"],
                "matched_2d": False,
                "yolo_conf": round(d2["score"], 4),
                "match_quality": 0.0,
                "fusion_mode": "yolo_only",
                "timestamp": d2["timestamp"],
                "source": source,
                "camera_visible": True,
            })
        yolo_only_fused.sort(key=lambda x: x["fused_score"], reverse=True)
        return yolo_only_fused

    ts3 = det3d[0]["timestamp"]
    ts2 = det2d[0]["timestamp"] if det2d else ts3
    allow_cross_sensor_match = True
    if not timestamps_aligned(ts3, ts2):
        logger.warning(f"时间戳差异过大: {ts3} vs {ts2}，仅保留单传感器结果，不做跨传感器匹配")
        allow_cross_sensor_match = False

    # 根据标定是否可用，选择投影方式
    if calib.use_calib:
        # 精确模式：3D 框投影到图像平面
        proj_bboxes = [
            (calib.box3d_to_bbox2d(d["center"], d["dimensions"], d["heading"])
             if d.get("camera_visible", True) else None)
            for d in det3d
        ]
        mode_tag = "标定投影"
    else:
        # 降级模式：theta 粗估
        proj_bboxes = [theta_to_proj_bbox(d) if d.get("camera_visible", True) else None for d in det3d]
        mode_tag = "theta 粗估"

    logger.info(f"融合模式: {mode_tag}")
    if camera_fov_only:
        vis_before = len(det3d)
        if calib.use_calib:
            pairs = [
                (d, b) for d, b in zip(det3d, proj_bboxes)
                if (not d.get("camera_visible", True)) or b is not None
            ]
            det3d = [d for d, _ in pairs]
            proj_bboxes = [b for _, b in pairs]
        else:
            # theta模式下剔除预计投影过窄/超边界目标，减少“图上看不到但融合出了目标”
            filtered = []
            filtered_bbox = []
            for d, box in zip(det3d, proj_bboxes):
                if not d.get("camera_visible", True):
                    filtered.append(d)
                    filtered_bbox.append(None)
                    continue
                box_w = box[2] - box[0]
                in_img = (box[2] > 0 and box[0] < IMAGE_WIDTH and box[3] > 0 and box[1] < IMAGE_HEIGHT)
                if box_w >= MIN_VISIBLE_PIXEL_W and in_img:
                    filtered.append(d)
                    filtered_bbox.append(box)
            det3d = filtered
            proj_bboxes = filtered_bbox
        logger.info(f"可见性过滤: {vis_before} -> {len(det3d)}")
        if not det3d:
            return []

    n3, n2  = len(det3d), len(det2d)
    iou_mat = np.zeros((n3, n2))
    sim_mat = np.zeros((n3, n2))
    theta_diff_mat = np.full((n3, n2), np.inf)
    for i, d3 in enumerate(det3d):
        for j, d2 in enumerate(det2d):
            if not allow_cross_sensor_match:
                continue
            if not d3.get("camera_visible", True):
                continue
            yolo_label = d2["label"]
            logger.info(f"YOLO: '{yolo_label}' | 激光雷达: '{d3['label']}'")
            compat = category_compatibility(d2["label"], d3["label"])
            if compat <= 0.0:
                continue
            if calib.use_calib:
                if proj_bboxes[i] is None:
                    continue
                ref_box = proj_bboxes[i]
                iou = compute_iou_2d(ref_box, d2["bbox"])
                iou_mat[i, j] = iou
                sim_mat[i, j] = iou * compat
            else:
                # theta 降级模式：优先使用角度一致性做匹配，避免“都融合进去”
                ref_box = proj_bboxes[i]
                theta_lidar = lidar_center_to_theta(d3["center"])
                theta_diff = angle_diff_deg(theta_lidar, yolo_theta_to_fusion(d2.get("theta", 0.0)))
                theta_diff_mat[i, j] = theta_diff
                if theta_diff > THETA_MATCH_DEG:
                    continue
                theta_sim = float(np.exp(-(theta_diff ** 2) / (2 * THETA_SIGMA_DEG ** 2)))

                # 再加入 x 方向中心对齐，抑制左右错配
                est_x = (theta_lidar / CAMERA_FOV_DEG + 0.5) * IMAGE_WIDTH
                yolo_x = (d2["bbox"][0] + d2["bbox"][2]) / 2.0
                x_gap_norm = min(abs(est_x - yolo_x) / (IMAGE_WIDTH / 2.0), 1.0)
                x_sim = 1.0 - x_gap_norm

                sim = (0.75 * theta_sim + 0.25 * x_sim) * compat
                sim_mat[i, j] = sim

            # 调试：打印具体数值
            logger.info(f"激光雷达 {i} ({d3['label']}): center={d3['center']}")
            logger.info(f"YOLO {j} ({d2['label']}): bbox={d2['bbox']}, theta={d2.get('theta', 0):.2f}")
            logger.info(f"估算投影框: {ref_box}")
            if calib.use_calib:
                logger.info(f"IoU: {iou_mat[i, j]:.3f}")
            else:
                logger.info(f"theta_diff={theta_diff_mat[i, j]:.2f}°, sim={sim_mat[i, j]:.3f}")

    match_mat = iou_mat if calib.use_calib else sim_mat
    row_ind, col_ind = linear_sum_assignment(-match_mat)
    matched_3d = {
        r: c for r, c in zip(row_ind, col_ind)
        if (iou_mat[r, c] >= STRICT_IOU_MATCH_THRESH
            if calib.use_calib else sim_mat[r, c] >= STRICT_THETA_SIM_THRESH)
    }
    logger.info(f"匹配详情: 激光雷达 {n3} 个，YOLO {n2} 个，成功匹配 {len(matched_3d)} 对")
    if iou_mat.size > 0:
        max_iou = iou_mat.max()
        avg_iou = iou_mat[iou_mat > 0].mean() if max_iou > 0 else 0
        avg_iou = 0 if np.isnan(avg_iou) else avg_iou
        logger.info(f"IoU 矩阵最大值: {max_iou:.3f}，平均值: {avg_iou:.3f}")
    else:
        logger.info(f"IoU 矩阵为空")

    fused = []
    for i, d3 in enumerate(det3d):
        if i in matched_3d:
            d2          = det2d[matched_3d[i]]
            yolo_conf   = d2["score"]
            match_quality = iou_mat[i, matched_3d[i]] if calib.use_calib else sim_mat[i, matched_3d[i]]
            # 激光雷达主导：高置信度 PVRCNN 不会被 YOLO 明显拉低
            fused_score = d3["score"] * w_pvrcnn + (yolo_conf * match_quality) * w_yolo
            matched     = True
        else:
            yolo_conf   = 0.0
            fused_score = d3["score"] * w_pvrcnn
            match_quality = 0.0
            matched     = False

        if fused_score < fused_thresh:
            continue

        fused.append({
            "label"      : d3["label"],  # 最终类别以 LiDAR 为主，避免视觉误分类拉偏
            "camera_label": (det2d[matched_3d[i]]["label"] if matched else ""),
            "score"      : round(d3["score"],  4),
            "fused_score": round(fused_score,  4),
            "center"     : d3["center"],
            "dimensions" : d3["dimensions"],
            "heading"    : d3["heading"],
            "proj_bbox"  : proj_bboxes[i],
            "matched_2d" : matched,
            "yolo_conf"  : round(yolo_conf, 4),
            "match_quality": round(match_quality, 4),
            "fusion_mode": mode_tag,
            "timestamp"  : d3["timestamp"],
            "source"     : "pvrcnn" if not matched else "pvrcnn+yolo",
            "camera_visible": d3.get("camera_visible", True),
        })

    # 仅在显式开启时添加未匹配 YOLO，并进行置信度过滤
    matched_2d = set(matched_3d.values())
    for j, d2 in enumerate(det2d):
        if j in matched_2d:
            continue
        # include_unmatched_yolo=False 时，仍保留高置信 YOLO 估计，避免漏检
        if (not include_unmatched_yolo) and d2["score"] < YOLO_HIGH_CONF_THRESH:
            continue
        if d2["score"] < unmatched_yolo_min_score:
            continue
        lidar_hint = find_lidar_hint_by_theta(d2, det3d)
        if lidar_hint is not None:
            # 有角度邻近激光雷达目标时，直接借用其3D位置，仅做标签提示
            hint_det = lidar_hint["det3d"]
            center = hint_det["center"]
            dims = hint_det["dimensions"]
            heading = hint_det["heading"]
            source = "yolo_hint_lidar"
        elif d2["score"] >= YOLO_HIGH_CONF_THRESH:
            est_3d = estimate_yolo_only_3d(d2)
            center = est_3d["center"]
            dims = est_3d["dimensions"]
            heading = est_3d["heading"]
            source = "yolo_estimated"
        else:
            center = [0, 0, 0]
            dims = [0, 0, 0]
            heading = 0
            source = "yolo_only"
        fused.append({
            "label"      : d2["label"],
            "score"      : round(d2["score"], 4),
            # YOLO 仅做提示：默认低于 PVRCNN 主结果
            "fused_score": round(min(d2["score"] * 0.7, 0.69), 4),
            "center"     : center,
            "dimensions" : dims,
            "heading"    : heading,
            "proj_bbox"  : d2["bbox"],
            "matched_2d" : False,
            "yolo_conf"  : round(d2["score"], 4),
            "match_quality": 0.0,
            "fusion_mode": mode_tag,
            "timestamp"  : d2["timestamp"],
            "source"     : source,
        })

    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused
# ============================================================
# PART 8  可视化
# ============================================================
def visualize_bev(
        results     : List[Dict],
        save_path   : str   = "bev_result.png",
        canvas_size : int   = 800,
        range_m     : float = 60.0,
) -> np.ndarray:
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    scale  = canvas_size / (2 * range_m)

    def w2c(x, y):
        u = int(canvas_size / 2 - y * scale)  # 注意 y 取负，因为车辆坐标系 y 右为负
        v = int(canvas_size - x * scale)  # x 向前为正，图像从下往上
        # 边界限制，防止数组越界损坏程序
        u = max(0, min(canvas_size - 1, u))
        v = max(0, min(canvas_size - 1, v))
        return (u, v)

    for d in results:
        cx, cy, _ = d["center"]
        l, w      = d["dimensions"][0], d["dimensions"][1]
        heading   = d["heading"]
        hl, hw    = l/2, w/2

        corners = np.array([[ hl, hw],[ hl,-hw],[-hl,-hw],[-hl, hw]])
        c, s    = np.cos(heading), np.sin(heading)
        corners = (np.array([[c,-s],[s,c]]) @ corners.T).T + [cx, cy]
        pts     = np.array([w2c(p[0], p[1]) for p in corners], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], (0, 0, 0))

        tid      = d.get("track_id", "?")
        vx, vy   = d.get("velocity", [0, 0])
        speed    = round((vx**2 + vy**2)**0.5, 2)
        mode_tag = "C" if d.get("fusion_mode") == "标定投影" else "T"  # C=Calib T=Theta
        lpos     = w2c(cx, cy)
        cv2.putText(canvas,
                    f"[{mode_tag}]{d['label']}#{tid} {d['fused_score']:.2f} {speed}m/s",
                    (lpos[0]-30, lpos[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (60, 60, 60), 1)

    origin = w2c(0, 0)
    cv2.circle(canvas, origin, 5, (0,0,0), -1)
    cv2.putText(canvas, "ego", (origin[0]+6, origin[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True)
    cv2.imwrite(save_path, canvas)
    logger.info(f"BEV 已保存: {save_path}")
    return canvas

# ============================================================
# PART 9  主函数
# ============================================================
mot = MultiObjectTracker()
camera_only_memory: List[Dict] = []

def _candidate_theta(det: Dict) -> float:
    center = det.get("center", [0, 0, 0])
    if isinstance(center, list) and len(center) >= 2 and (abs(center[0]) + abs(center[1]) > 1e-3):
        return lidar_center_to_theta(center, clip_to_fov=False)
    bbox = det.get("proj_bbox")
    if bbox:
        x_center = (bbox[0] + bbox[2]) / 2.0
        return float((x_center / IMAGE_WIDTH - 0.5) * CAMERA_FOV_DEG)
    return 0.0

def confirm_camera_only_detections(candidates: List[Dict], timestamp: str) -> List[Dict]:
    """
    Camera-only 目标确认：
    - 单帧不输出；
    - 连续多帧命中后输出；
    - 车辆阈值宽松，行人/骑行者阈值严格。
    """
    global camera_only_memory
    for m in camera_only_memory:
        m["updated"] = False

    for d in candidates:
        label = d["label"]
        cfg = CAMERA_ONLY_CLASS_CFG.get(label, {"min_score": YOLO_HIGH_CONF_THRESH, "min_streak": 2})
        if d.get("score", 0.0) < cfg["min_score"]:
            continue
        if d.get("center", [0, 0, 0]) == [0, 0, 0]:
            continue

        theta = _candidate_theta(d)
        best_idx = -1
        best_gap = float("inf")
        for idx, mem in enumerate(camera_only_memory):
            if mem["label"] != label:
                continue
            gap = angle_diff_deg(theta, mem["theta"])
            if gap <= CAMERA_ONLY_MAX_THETA_GAP_DEG and gap < best_gap:
                best_gap = gap
                best_idx = idx

        if best_idx >= 0:
            mem = camera_only_memory[best_idx]
            mem["theta"] = theta
            mem["streak"] += 1
            mem["updated"] = True
            mem["det"] = dict(d)
            mem["last_ts"] = timestamp
        else:
            camera_only_memory.append({
                "label": label,
                "theta": theta,
                "streak": 1,
                "updated": True,
                "det": dict(d),
                "last_ts": timestamp,
            })

    confirmed: List[Dict] = []
    kept: List[Dict] = []
    for mem in camera_only_memory:
        if not mem["updated"]:
            continue
        cfg = CAMERA_ONLY_CLASS_CFG.get(mem["label"], {"min_score": YOLO_HIGH_CONF_THRESH, "min_streak": 2})
        if mem["streak"] >= cfg["min_streak"]:
            out = dict(mem["det"])
            out["source"] = "camera_confirmed"
            out["fusion_mode"] = "camera_confirmed"
            confirmed.append(out)
        kept.append(mem)
    camera_only_memory = kept
    return confirmed

def process_frame(pvrcnn_raw: List[Dict], yolo_raw: List[Dict],
                  calib: KITTICalib) -> List[Dict]:
    timestamp = pvrcnn_raw[0].get("timestamp", "") if pvrcnn_raw else ""
    if (not timestamp) and yolo_raw:
        timestamp = yolo_raw[0].get("timestamp", "")

    # 高精度策略：仅输出 LiDAR 检测或成功跨模态匹配结果，禁用 yolo-only 最终输出
    fused = fuse(
        pvrcnn_raw,
        yolo_raw,
        calib,
        include_unmatched_yolo=False,
    )
    if not fused:
        return []

    strict_fused = []
    for d in fused:
        # 最终严格检查：若标记为跨模态匹配，则要求满足严格匹配质量
        if d.get("matched_2d"):
            min_quality = STRICT_IOU_MATCH_THRESH if d.get("fusion_mode") == "标定投影" else STRICT_THETA_SIM_THRESH
            if d.get("match_quality", 0.0) < min_quality:
                continue
        strict_fused.append(d)

    if not strict_fused:
        return []
    return mot.update(strict_fused, timestamp)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 尝试加载标定文件，失败自动降级到 theta 模式
    calib = KITTICalib(CALIB_PATH)

    frames = [
        (pvrcnn_results, yolo_results),
        (pvrcnn_results, yolo_results),
    ]

    for idx, (pvrcnn_raw, yolo_raw) in enumerate(frames):
        results = process_frame(pvrcnn_raw, yolo_raw, calib)
        visualize_bev(results,
                      save_path=os.path.join(OUTPUT_DIR, f"bev_frame_{idx+1:04d}.png"))

    logger.info(f"活跃 tracks: {len(mot.tracks)}")

if __name__ == "__main__":
    main()
