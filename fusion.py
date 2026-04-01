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
IMAGE_WIDTH     = 640
IMAGE_HEIGHT    = 480
IOU_THRESH      = 0.1
W_PVRCNN        = 0.6
W_YOLO          = 0.4
FUSED_THRESH    = 0.1
TIMESTAMP_TOL_S = 1
MAX_MISS_FRAMES = 2
OUTPUT_DIR      = "./fusion_output"
CALIB_PATH      = "calib.txt"# kitti格式标定文件路径

YOLO_TO_PVRCNN: Dict[str, str] = {
    "car"        : "Car",
    "truck"      : "Car",
    "bus"        : "Bus",
    "van"        : "Car",
    "motorcycle" : "Cyclist",
    "bicycle"    : "Cyclist",
    "person"     : "Pedestrian",
    "rider"      : "Cyclist",
}

# 大类分组：同一 group 内允许跨细分类别匹配
# 例如：激光雷达识别 Cyclist，摄像头识别 person，视为同一大类，允许匹配
CATEGORY_GROUPS: List[set] = [
    {"Car", "Bus", "Truck", "Van"},
    {"Cyclist", "Pedestrian"},
    {"person", "Pedestrian", "Cyclist"},   # YOLO person 可匹配激光雷达 Cyclist/Pedestrian
]

def same_category_group(yolo_label: str, pvrcnn_label: str) -> bool:
    """判断 YOLO 标签和 PVRCNN 标签是否属于同一可匹配大类"""
    mapped = YOLO_TO_PVRCNN.get(yolo_label.lower())
    if mapped == pvrcnn_label:
        return True
    # 大类模糊匹配
    for group in CATEGORY_GROUPS:
        if pvrcnn_label in group:
            if mapped in group or yolo_label in group:
                return True
    return False

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
        "label"     : d["class_label"],
        "score"     : float(d["score"]),
        "center"    : [cx, cy, cz],
        "dimensions": [dx, dy, dz],
        "heading"   : d["box"]["rotation"],
        "timestamp" : d.get("timestamp", ""),
    }

def yolo_to_unified(d: Dict) -> Dict:
    return {
        "label"    : d["label"],
        "score"    : float(d["score"]),
        "bbox"     : d["bbox_camera"],
        "theta"    : float(d.get("theta", 0.0)),
        "timestamp": d.get("timestamp", ""),
    }

def lidar_center_to_theta(center: List[float], fov: float = CAMERA_FOV_DEG) -> float:
    """
    将激光雷达 3D 中心点 [x, y, z] 转换为摄像头视角 theta（度）。
    激光雷达坐标系：x 向前，y 向左。
    摄像头中轴线沿 x 轴，theta = arctan2(y, x) 转换为角度。
    """
    x, y, _ = center
    theta_rad = np.arctan2(y, x)  # 水平偏角（弧度）
    theta_deg = np.degrees(theta_rad)
    # 限制在 FOV 范围内
    return float(np.clip(theta_deg, -fov / 2, fov / 2))


def theta_to_proj_bbox(det3d: Dict, det2d: Dict, img_width: int = IMAGE_WIDTH,
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

    # 取投影宽度和 YOLO bbox 宽度的平均，减少极端情况
    yolo_w  = det2d["bbox"][2] - det2d["bbox"][0]
    half_w  = max((proj_w + yolo_w) / 4.0, 5.0)  # 至少 5px 防止退化

    return [x_center - half_w, det2d["bbox"][1],
            x_center + half_w, det2d["bbox"][3]]

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
) -> List[Dict]:

    det3d = [pvrcnn_to_unified(d) for d in pvrcnn_raw]
    det2d = [yolo_to_unified(d)   for d in yolo_raw]
    logger.info(f"YOLO 原始数量: {len(yolo_raw)}")
    logger.info(f"YOLO 转换后数量: {len(det2d)}")
    if not det3d:
        return []

    ts3 = det3d[0]["timestamp"]
    ts2 = det2d[0]["timestamp"] if det2d else ts3
    if not timestamps_aligned(ts3, ts2):
        logger.warning(f"时间戳差异过大: {ts3} vs {ts2}，跳过 YOLO 匹配")
        det2d = []

    # 根据标定是否可用，选择投影方式
    if calib.use_calib:
        # 精确模式：3D 框投影到图像平面
        proj_bboxes = [
            calib.box3d_to_bbox2d(d["center"], d["dimensions"], d["heading"])
            for d in det3d
        ]
        mode_tag = "标定投影"
    else:
        # 降级模式：theta 粗估
        proj_bboxes = [None] * len(det3d)   # 占位，匹配时用 theta_to_proj_bbox
        mode_tag = "theta 粗估"

    logger.info(f"融合模式: {mode_tag}")

    n3, n2  = len(det3d), len(det2d)
    iou_mat = np.zeros((n3, n2))
    for i, d3 in enumerate(det3d):
        for j, d2 in enumerate(det2d):
            yolo_label = d2["label"]
            yolo_mapped = YOLO_TO_PVRCNN.get(yolo_label.lower(), "未映射")
            logger.info(f"YOLO: '{yolo_label}' -> '{yolo_mapped}' | 激光雷达: '{d3['label']}'")
            # if not same_category_group(d2["label"], d3["label"]):
            #     continue
            if calib.use_calib:
                if proj_bboxes[i] is None:
                    continue
                ref_box = proj_bboxes[i]
            else:
                # theta 降级模式：用激光雷达 center 计算投影角，再与 YOLO bbox 对比
                ref_box = theta_to_proj_bbox(d3, d2)

            # 调试：打印具体数值
            logger.info(f"激光雷达 {i} ({d3['label']}): center={d3['center']}")
            logger.info(f"YOLO {j} ({d2['label']}): bbox={d2['bbox']}, theta={d2.get('theta', 0):.2f}")
            logger.info(f"估算投影框: {ref_box}")

            iou_mat[i, j] = compute_iou_2d(ref_box, d2["bbox"])
            logger.info(f"IoU: {iou_mat[i, j]:.3f}")

    row_ind, col_ind = linear_sum_assignment(-iou_mat)
    matched_3d = {
        r: c for r, c in zip(row_ind, col_ind)
        if iou_mat[r, c] >= IOU_THRESH
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
            fused_score = w_pvrcnn * d3["score"] + w_yolo * yolo_conf
            matched     = True
        else:
            yolo_conf   = 0.0
            fused_score = w_pvrcnn * d3["score"]
            matched     = False

        if fused_score < fused_thresh:
            continue

        fused.append({
            "label"      : YOLO_TO_PVRCNN.get(det2d[matched_3d[i]]["label"].lower(), d3["label"]) if matched else d3["label"],
            "score"      : round(d3["score"],  4),
            "fused_score": round(fused_score,  4),
            "center"     : d3["center"],
            "dimensions" : d3["dimensions"],
            "heading"    : d3["heading"],
            "proj_bbox"  : proj_bboxes[i],
            "matched_2d" : matched,
            "yolo_conf"  : round(yolo_conf, 4),
            "fusion_mode": mode_tag,
            "timestamp"  : d3["timestamp"],
        })

    # 添加未匹配的 YOLO 目标
    matched_2d = set(matched_3d.values())
    for j, d2 in enumerate(det2d):
        if j not in matched_2d:
            fused.append({
                "label"      : YOLO_TO_PVRCNN.get(d2["label"].lower(), d2["label"]),
                "score"      : round(d2["score"], 4),
                "fused_score": round(w_yolo * d2["score"], 4),
                "center"     : [0, 0, 0],
                "dimensions" : [0, 0, 0],
                "heading"    : 0,
                "proj_bbox"  : d2["bbox"],
                "matched_2d" : False,
                "yolo_conf"  : round(d2["score"], 4),
                "fusion_mode": mode_tag,
                "timestamp"  : d2["timestamp"],
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

def process_frame(pvrcnn_raw: List[Dict], yolo_raw: List[Dict],
                  calib: KITTICalib) -> List[Dict]:
    timestamp = pvrcnn_raw[0].get("timestamp", "") if pvrcnn_raw else ""
    fused     = fuse(pvrcnn_raw, yolo_raw, calib)
    if not fused:
        return []
    return mot.update(fused, timestamp)

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