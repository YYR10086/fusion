import json
import pandas as pd
import numpy as np

YOLO_JSON = "./record_output/record_output5/detection_results.json"
PVRCNN_JSON = "./record_output/record_output5/lidar_detection_results.json"
FUSION_JSON = "./record_output/record_output5/fusion_results.json"
OUTPUT_EXCEL = "./record_output/record_output5/comparison_analysis.xlsx"


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_in_camera_fov(det, fov_angle=75):
    """判断是否在相机视野内 - 修正坐标系"""
    try:
        x = det['location'][0]
        y = det['location'][1]  # 前进方向
        z = det['location'][2]

        # 使用 y 作为前进方向，x 作为左右
        if y == 0:
            return False
        theta = np.degrees(np.arctan2(x, y))
        return abs(theta) <= fov_angle
    except:
        return True


def build_frame_dict(frames, filter_fov=False):
    """构建 {frame_id: count} 字典"""
    result = {}
    for i, frame in enumerate(frames):
        frame_id = frame.get("frame_id", i)
        dets = frame.get("detections", [])

        if filter_fov:
            filtered = [d for d in dets if is_in_camera_fov(d)]
            result[frame_id] = len(filtered)
        else:
            result[frame_id] = len(dets)

    return result


def main():
    yolo_data = load_json(YOLO_JSON)
    pvrcnn_data = load_json(PVRCNN_JSON)
    fusion_data = load_json(FUSION_JSON)

    yolo_frames = yolo_data.get("frames", [])
    pvrcnn_frames = pvrcnn_data.get("frames", [])
    fusion_frames = fusion_data.get("frames", [])

    yolo_dict = build_frame_dict(yolo_frames)
    pvrcnn_all_dict = build_frame_dict(pvrcnn_frames, filter_fov=False)
    pvrcnn_fov_dict = build_frame_dict(pvrcnn_frames, filter_fov=True)
    fusion_dict = build_frame_dict(fusion_frames, filter_fov=False)

    all_frame_ids = sorted(set(yolo_dict.keys()) | set(pvrcnn_all_dict.keys()) | set(fusion_dict.keys()))

    data = {
        "帧序号": all_frame_ids,
        "YOLO检测数": [yolo_dict.get(fid, 0) for fid in all_frame_ids],
        "激光雷达全视野检测数": [pvrcnn_all_dict.get(fid, 0) for fid in all_frame_ids],
        "激光雷达相机视野检测数": [pvrcnn_fov_dict.get(fid, 0) for fid in all_frame_ids],
        "融合算法检测数": [fusion_dict.get(fid, 0) for fid in all_frame_ids]
    }

    df = pd.DataFrame(data)

    summary = {
        "指标": ["平均检测数", "最大检测数", "最小检测数", "总检测数"],
        "YOLO": [
            round(df["YOLO检测数"].mean(), 2),
            df["YOLO检测数"].max(),
            df["YOLO检测数"].min(),
            df["YOLO检测数"].sum()
        ],
        "激光雷达(全视野)": [
            round(df["激光雷达全视野检测数"].mean(), 2),
            df["激光雷达全视野检测数"].max(),
            df["激光雷达全视野检测数"].min(),
            df["激光雷达全视野检测数"].sum()
        ],
        "激光雷达(相机视野)": [
            round(df["激光雷达相机视野检测数"].mean(), 2),
            df["激光雷达相机视野检测数"].max(),
            df["激光雷达相机视野检测数"].min(),
            df["激光雷达相机视野检测数"].sum()
        ],
        "融合算法": [
            round(df["融合算法检测数"].mean(), 2),
            df["融合算法检测数"].max(),
            df["融合算法检测数"].min(),
            df["融合算法检测数"].sum()
        ]
    }

    df_summary = pd.DataFrame(summary)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='逐帧对比', index=False)
        df_summary.to_excel(writer, sheet_name='统计摘要', index=False)

    print(f"\n✅ 对比分析完成！")
    print(f"   保存路径: {OUTPUT_EXCEL}")
    print(f"\n📊 统计摘要:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()