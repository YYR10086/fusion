import os
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Fusion")

# ============================================================
# PART 1  超参数
# ============================================================
CAMERA_FOV_DEG = 150.0
CAMERA_FOV_MARGIN_DEG = 8.0
CAMERA_MAX_RANGE_M = 45.0
MIN_VISIBLE_PIXEL_W = 12.0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# 跨模态匹配：提高 Precision/F1
STRICT_IOU_MATCH_THRESH = 0.20
STRICT_THETA_SIM_THRESH = 0.50
THETA_MATCH_DEG = 10.0
THETA_SIGMA_DEG = 6.0

# 融合分数（LiDAR 主导）
W_PVRCNN = 0.90
W_YOLO = 0.10
FUSED_THRESH = 0.10

# 其他
YOLO_HIGH_CONF_THRESH = 0.75
TIMESTAMP_TOL_S = 1.0
MAX_MISS_FRAMES = 2
OUTPUT_DIR = "./fusion_output"
CALIB_PATH = "calib.txt"

CANONICAL_LABELS = {"car", "bus", "truck", "pedestrian", "cyclist"}

YOLO_TO_CANONICAL: Dict[str, str] = {
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "cyclist": "cyclist",
    "bicycle": "cyclist",
    "motorcycle": "cyclist",
    "rider": "cyclist",
    "car": "car",
    "van": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "truck",
}

PVRCNN_TO_CANONICAL: Dict[str, str] = {
    "car": "car",
    "van": "car",
    "truck": "truck",
    "bus": "bus",
    "pedestrian": "pedestrian",
    "cyclist": "cyclist",
}

CLASS_SIZE_PRIOR: Dict[str, List[float]] = {
    "car": [4.2, 1.8, 1.6],
    "truck": [8.0, 2.5, 3.2],
    "bus": [11.0, 2.5, 3.2],
    "pedestrian": [0.8, 0.8, 1.75],
    "cyclist": [1.8, 0.7, 1.7],
}


# ============================================================
# PART 2  标签与时间
# ============================================================
def normalize_yolo_label(label: str) -> str:
    return YOLO_TO_CANONICAL.get(str(label).lower(), "")


def normalize_pvrcnn_label(label: str) -> str:
    x = PVRCNN_TO_CANONICAL.get(str(label).lower(), "")
    return x if x in CANONICAL_LABELS else ""


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


def wrap_angle_deg(a: float) -> float:
    return float((a + 180.0) % 360.0 - 180.0)


def angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return float(min(d, 360.0 - d))


# 视觉 theta 与融合坐标系差 90°（逆时针）
def yolo_theta_to_fusion(theta_deg: float) -> float:
    return wrap_angle_deg(float(theta_deg) + 90.0)


# ============================================================
# PART 3  标定
# ============================================================
class KITTICalib:
    def __init__(self, calib_path: str):
        self.use_calib = False
        self.P2 = None
        self.R0_rect = None
        self.Tr_velo_to_cam = None

        try:
            data = self._parse(calib_path)
            self.P2 = data["P2"].reshape(3, 4)

            R0 = data["R0_rect"].reshape(3, 3)
            self.R0_rect = np.eye(4)
            self.R0_rect[:3, :3] = R0

            Tr = data["Tr_velo_to_cam"].reshape(3, 4)
            self.Tr_velo_to_cam = np.eye(4)
            self.Tr_velo_to_cam[:3, :] = Tr

            self.use_calib = True
            logger.info(f"标定文件加载成功: {calib_path}")
        except Exception as e:
            logger.warning(f"标定读取失败({e})，降级到 theta 模式")

    @staticmethod
    def _parse(path: str) -> Dict[str, np.ndarray]:
        data = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                data[k.strip()] = np.array([float(x) for x in v.strip().split()])
        return data

    def lidar_to_img(self, pts_lidar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = pts_lidar.shape[0]
        pts_h = np.hstack([pts_lidar, np.ones((n, 1))])
        pts_cam = (self.R0_rect @ self.Tr_velo_to_cam @ pts_h.T).T
        in_front = pts_cam[:, 2] > 0

        pts_img = (self.P2 @ pts_cam.T).T
        pts_img[:, 0] /= pts_img[:, 2]
        pts_img[:, 1] /= pts_img[:, 2]
        uv = pts_img[:, :2]

        in_img = (
            (uv[:, 0] >= 0) & (uv[:, 0] < IMAGE_WIDTH) &
            (uv[:, 1] >= 0) & (uv[:, 1] < IMAGE_HEIGHT)
        )
        return uv, in_front & in_img

    def box3d_to_bbox2d(self, center: List[float], dimensions: List[float], heading: float) -> Optional[List[float]]:
        cx, cy, cz = center
        l, w, h = dimensions
        hl, hw, hh = l / 2, w / 2, h / 2

        corners_local = np.array([
            [hl, hw, -hh], [hl, -hw, -hh], [-hl, -hw, -hh], [-hl, hw, -hh],
            [hl, hw, hh], [hl, -hw, hh], [-hl, -hw, hh], [-hl, hw, hh],
        ])
        c, s = np.cos(heading), np.sin(heading)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        corners_lidar = (rot @ corners_local.T).T + np.array([cx, cy, cz])

        uv, valid = self.lidar_to_img(corners_lidar)
        if valid.sum() == 0:
            return None

        vv = uv[valid]
        x1 = float(np.clip(vv[:, 0].min(), 0, IMAGE_WIDTH - 1))
        y1 = float(np.clip(vv[:, 1].min(), 0, IMAGE_HEIGHT - 1))
        x2 = float(np.clip(vv[:, 0].max(), 0, IMAGE_WIDTH - 1))
        y2 = float(np.clip(vv[:, 1].max(), 0, IMAGE_HEIGHT - 1))

        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return None
        return [x1, y1, x2, y2]


# ============================================================
# PART 4  坐标与工具
# ============================================================
def lidar_center_to_theta(center: List[float], fov: float = CAMERA_FOV_DEG, clip_to_fov: bool = True) -> float:
    x, y, _ = center
    theta = np.degrees(np.arctan2(y, x))
    if clip_to_fov:
        theta = float(np.clip(theta, -fov / 2.0, fov / 2.0))
    return float(theta)


def lidar_in_camera_fov(center: List[float], fov: float = CAMERA_FOV_DEG) -> bool:
    return abs(lidar_center_to_theta(center, fov=fov, clip_to_fov=False)) <= (fov / 2.0)


def lidar_in_strict_camera_view(center: List[float], fov: float = CAMERA_FOV_DEG,
                                max_range_m: float = CAMERA_MAX_RANGE_M,
                                margin_deg: float = CAMERA_FOV_MARGIN_DEG) -> bool:
    x, y, _ = center
    if np.hypot(x, y) > max_range_m:
        return False
    return abs(lidar_center_to_theta(center, fov=fov, clip_to_fov=False)) <= max((fov / 2.0) - margin_deg, 1.0)


def compute_iou_2d(a: List[float], b: List[float]) -> float:
    u1, v1 = max(a[0], b[0]), max(a[1], b[1])
    u2, v2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, u2 - u1) * max(0.0, v2 - v1)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    return float(inter / (area_a + area_b - inter + 1e-6))


def pvrcnn_to_unified(d: Dict) -> Dict:
    cx, cy, cz = d["box"]["center"]
    dx, dy, dz = d["box"]["dimensions"]
    return {
        "label": normalize_pvrcnn_label(d["class_label"]),
        "score": float(d["score"]),
        "center": [float(cx), float(cy), float(cz)],
        "dimensions": [float(dx), float(dy), float(dz)],
        "heading": float(d["box"]["rotation"]),
        "timestamp": d.get("timestamp", ""),
    }


def yolo_to_unified(d: Dict) -> Dict:
    return {
        "label": normalize_yolo_label(d["label"]),
        "score": float(d["score"]),
        "bbox": d["bbox_camera"],
        "theta": float(d.get("theta", 0.0)),
        "timestamp": d.get("timestamp", ""),
    }


# ============================================================
# PART 5  跟踪
# ============================================================
class KalmanTracker:
    _id_counter = 0

    def __init__(self, x: float, y: float, label: str, timestamp: str):
        KalmanTracker._id_counter += 1
        self.track_id = KalmanTracker._id_counter
        self.label = label
        self.miss_count = 0
        self.last_ts = timestamp

        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 100.0, 100.0])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.R = np.diag([1.0, 1.0])

    def _F(self, dt: float) -> np.ndarray:
        return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)

    def _Q(self, dt: float) -> np.ndarray:
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        s2 = 4.0
        return s2 * np.array([
            [dt4 / 4, 0, dt3 / 2, 0], [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0], [0, dt3 / 2, 0, dt2],
        ], dtype=np.float64)

    def _calc_dt(self, ts: str) -> float:
        t0, t1 = parse_timestamp(self.last_ts), parse_timestamp(ts)
        if t0 is None or t1 is None:
            return 0.1
        return max((t1 - t0).total_seconds(), 0.0)

    def predict(self, timestamp: str) -> np.ndarray:
        dt = max(self._calc_dt(timestamp), 1e-3)
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        self.last_ts = timestamp
        return self.x[:2].copy()

    def update(self, x: float, y: float) -> np.ndarray:
        z = np.array([x, y], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.miss_count = 0
        return self.x[:2].copy()

    def get_velocity(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])


class MultiObjectTracker:
    def __init__(self, max_miss: int = MAX_MISS_FRAMES, match_dist: float = 5.0):
        self.tracks: List[KalmanTracker] = []
        self.max_miss = max_miss
        self.match_dist = match_dist

    def _new_track(self, d: Dict, ts: str) -> KalmanTracker:
        trk = KalmanTracker(d["center"][0], d["center"][1], d["label"], ts)
        self.tracks.append(trk)
        return trk

    @staticmethod
    def _attach(d: Dict, trk: KalmanTracker, vel: List[float]) -> Dict:
        d["track_id"] = trk.track_id
        d["velocity"] = [round(vel[0], 3), round(vel[1], 3)]
        return d

    def update(self, detections: List[Dict], timestamp: str) -> List[Dict]:
        if not detections:
            return []
        if not self.tracks:
            return [self._attach(dict(d), self._new_track(d, timestamp), [0.0, 0.0]) for d in detections]

        pred = np.array([t.predict(timestamp) for t in self.tracks])
        det = np.array([[d["center"][0], d["center"][1]] for d in detections])
        dist = np.linalg.norm(pred[:, None] - det[None, :], axis=2)

        rows, cols = linear_sum_assignment(dist)
        pairs, used_t, used_d = {}, set(), set()
        for r, c in zip(rows, cols):
            if dist[r, c] <= self.match_dist and self.tracks[r].label == detections[c]["label"]:
                pairs[r] = c
                used_t.add(r)
                used_d.add(c)

        out = []
        for ti, di in pairs.items():
            d = dict(detections[di])
            trk = self.tracks[ti]
            sx, sy = trk.update(d["center"][0], d["center"][1])
            d["center"] = [sx, sy, d["center"][2]]
            out.append(self._attach(d, trk, list(trk.get_velocity())))

        for i, t in enumerate(self.tracks):
            if i not in used_t:
                t.miss_count += 1
        self.tracks = [t for t in self.tracks if t.miss_count <= self.max_miss]

        for j, d in enumerate(detections):
            if j not in used_d:
                trk = self._new_track(d, timestamp)
                out.append(self._attach(dict(d), trk, [0.0, 0.0]))

        out.sort(key=lambda x: x.get("fused_score", 0.0), reverse=True)
        return out


# ============================================================
# PART 6  融合（重点）
# ============================================================
def _theta_mode_similarity(d3: Dict, d2: Dict) -> float:
    theta_lidar = lidar_center_to_theta(d3["center"], clip_to_fov=False)
    theta_yolo = yolo_theta_to_fusion(d2.get("theta", 0.0))
    diff = angle_diff_deg(theta_lidar, theta_yolo)
    if diff > THETA_MATCH_DEG:
        return 0.0
    theta_sim = float(np.exp(-(diff ** 2) / (2 * THETA_SIGMA_DEG ** 2)))

    est_x = (theta_lidar / CAMERA_FOV_DEG + 0.5) * IMAGE_WIDTH
    yolo_x = (d2["bbox"][0] + d2["bbox"][2]) / 2.0
    x_gap_norm = min(abs(est_x - yolo_x) / (IMAGE_WIDTH / 2.0), 1.0)
    x_sim = 1.0 - x_gap_norm
    return 0.75 * theta_sim + 0.25 * x_sim


def fuse(
    pvrcnn_raw: List[Dict],
    yolo_raw: List[Dict],
    calib: KITTICalib,
    w_pvrcnn: float = W_PVRCNN,
    w_yolo: float = W_YOLO,
    fused_thresh: float = FUSED_THRESH,
    include_unmatched_yolo: bool = False,
    camera_fov_only: bool = True,
) -> List[Dict]:
    del include_unmatched_yolo  # 精度优先版本：不输出 yolo-only 最终目标

    det3d = [pvrcnn_to_unified(d) for d in pvrcnn_raw]
    det2d = [yolo_to_unified(d) for d in yolo_raw]
    det3d = [d for d in det3d if d["label"]]
    det2d = [d for d in det2d if d["label"]]

    if camera_fov_only:
        vis = []
        for d in det3d:
            ok = lidar_in_strict_camera_view(d["center"])
            if ok or d["score"] >= 0.8:
                vis.append(dict(d, camera_visible=ok))
        det3d = vis
    else:
        det3d = [dict(d, camera_visible=True) for d in det3d]

    if not det3d:
        return []

    ts3 = det3d[0].get("timestamp", "")
    ts2 = det2d[0].get("timestamp", "") if det2d else ts3
    allow_match = timestamps_aligned(ts3, ts2)

    # 计算匹配矩阵
    n3, n2 = len(det3d), len(det2d)
    sim = np.zeros((n3, n2), dtype=np.float64)
    iou = np.zeros((n3, n2), dtype=np.float64)

    proj_bboxes: List[Optional[List[float]]] = []
    if calib.use_calib:
        for d in det3d:
            if not d.get("camera_visible", True):
                proj_bboxes.append(None)
            else:
                proj_bboxes.append(calib.box3d_to_bbox2d(d["center"], d["dimensions"], d["heading"]))
    else:
        for d in det3d:
            if not d.get("camera_visible", True):
                proj_bboxes.append(None)
            else:
                theta = lidar_center_to_theta(d["center"])
                x_center = (theta / CAMERA_FOV_DEG + 0.5) * IMAGE_WIDTH
                dist = max(np.hypot(d["center"][0], d["center"][1]), 0.1)
                f = IMAGE_WIDTH / (2.0 * np.tan(np.radians(CAMERA_FOV_DEG) / 2.0))
                pw = f * d["dimensions"][1] / dist
                hw = max(pw / 2.0, 8.0)
                ph = max(pw * 1.4, 20.0)
                y2 = IMAGE_HEIGHT * 0.85
                y1 = y2 - ph
                box = [x_center - hw, y1, x_center + hw, y2]
                if (box[2] - box[0]) < MIN_VISIBLE_PIXEL_W:
                    proj_bboxes.append(None)
                else:
                    proj_bboxes.append(box)

    for i, d3 in enumerate(det3d):
        for j, d2 in enumerate(det2d):
            if not allow_match:
                continue
            if d3["label"] != d2["label"]:
                continue
            if calib.use_calib:
                if proj_bboxes[i] is None:
                    continue
                ij = compute_iou_2d(proj_bboxes[i], d2["bbox"])
                iou[i, j] = ij
                sim[i, j] = ij
            else:
                s = _theta_mode_similarity(d3, d2)
                sim[i, j] = s

    rows, cols = linear_sum_assignment(-sim) if n2 > 0 else ([], [])
    matched: Dict[int, int] = {}
    for r, c in zip(rows, cols):
        if calib.use_calib:
            ok = iou[r, c] >= STRICT_IOU_MATCH_THRESH
        else:
            ok = sim[r, c] >= STRICT_THETA_SIM_THRESH
        if ok:
            matched[r] = c

    out = []
    for i, d3 in enumerate(det3d):
        if i in matched:
            j = matched[i]
            quality = iou[i, j] if calib.use_calib else sim[i, j]
            yconf = det2d[j]["score"]
            fused_score = d3["score"] * w_pvrcnn + (yconf * quality) * w_yolo
            source = "pvrcnn+yolo"
            cam_label = det2d[j]["label"]
            matched_2d = True
        else:
            quality = 0.0
            yconf = 0.0
            fused_score = d3["score"] * w_pvrcnn
            source = "pvrcnn"
            cam_label = ""
            matched_2d = False

        if fused_score < fused_thresh:
            continue

        out.append({
            "label": d3["label"],
            "camera_label": cam_label,
            "score": round(float(d3["score"]), 4),
            "fused_score": round(float(fused_score), 4),
            "center": d3["center"],
            "dimensions": d3["dimensions"],
            "heading": d3["heading"],
            "proj_bbox": proj_bboxes[i],
            "matched_2d": matched_2d,
            "yolo_conf": round(float(yconf), 4),
            "match_quality": round(float(quality), 4),
            "fusion_mode": "标定投影" if calib.use_calib else "theta 粗估",
            "timestamp": d3.get("timestamp", ""),
            "source": source,
            "camera_visible": d3.get("camera_visible", True),
        })

    out.sort(key=lambda x: x["fused_score"], reverse=True)
    return out


# ============================================================
# PART 7  可视化（无 OpenCV 依赖）
# ============================================================
def _save_ppm(path: str, img: np.ndarray) -> None:
    h, w, _ = img.shape
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(img.astype(np.uint8).tobytes())


def visualize_bev(results: List[Dict], save_path: str = "bev_result.ppm",
                  canvas_size: int = 800, range_m: float = 60.0) -> np.ndarray:
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    scale = canvas_size / (2 * range_m)

    def w2c(x: float, y: float) -> Tuple[int, int]:
        u = int(canvas_size / 2 - y * scale)
        v = int(canvas_size - x * scale)
        return max(0, min(canvas_size - 1, u)), max(0, min(canvas_size - 1, v))

    for d in results:
        cx, cy, _ = d["center"]
        u, v = w2c(cx, cy)
        canvas[max(v - 2, 0):min(v + 3, canvas_size), max(u - 2, 0):min(u + 3, canvas_size)] = [0, 0, 0]

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    if save_path.lower().endswith(".ppm"):
        _save_ppm(save_path, canvas)
    else:
        # 无额外图像库时，自动回退为 ppm
        base, _ = os.path.splitext(save_path)
        _save_ppm(base + ".ppm", canvas)
    return canvas


# ============================================================
# PART 8  主流程
# ============================================================
mot = MultiObjectTracker()


def process_frame(pvrcnn_raw: List[Dict], yolo_raw: List[Dict], calib: KITTICalib) -> List[Dict]:
    timestamp = ""
    if pvrcnn_raw:
        timestamp = pvrcnn_raw[0].get("timestamp", "")
    elif yolo_raw:
        timestamp = yolo_raw[0].get("timestamp", "")

    fused = fuse(
        pvrcnn_raw,
        yolo_raw,
        calib,
        include_unmatched_yolo=False,  # 仅 LiDAR 或成功跨模态匹配
    )
    if not fused:
        return []

    return mot.update(fused, timestamp)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    calib = KITTICalib(CALIB_PATH)

    # 这里只保留示例，真实调用请传入真实帧数据
    logger.info("Fusion module ready. Use process_frame(...) in your pipeline.")


if __name__ == "__main__":
    main()
