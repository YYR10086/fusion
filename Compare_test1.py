import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_RECORD_DIR = Path("./record_output")
DATASET_GLOB = "record_output*"

FRAME_CSV_NAME = "comparison_by_frame.csv"
SUMMARY_CSV_NAME = "comparison_summary.csv"
SUMMARY_TABLE_SVG_NAME = "comparison_summary.svg"

ALL_DATASETS_SUMMARY_CSV = BASE_RECORD_DIR / "comparison_all_datasets_summary.csv"
ALL_DATASETS_TABLE_SVG = BASE_RECORD_DIR / "comparison_all_datasets_summary.svg"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_in_camera_fov(det: dict, fov_angle: float = 75.0) -> bool:
    try:
        x = det["location"][0]
        y = det["location"][1]
        if y == 0:
            return False
        theta = math.degrees(math.atan2(x, y))
        return abs(theta) <= fov_angle
    except Exception:
        return True


def build_frame_count(frames: List[dict], filter_fov: bool = False) -> Dict[int, int]:
    frame_count: Dict[int, int] = {}
    for i, frame in enumerate(frames):
        frame_id = frame.get("frame_id", i)
        detections = frame.get("detections", [])
        if filter_fov:
            detections = [d for d in detections if is_in_camera_fov(d)]
        frame_count[frame_id] = len(detections)
    return frame_count


def compute_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"avg_count": 0.0, "max_count": 0, "min_count": 0, "total_count": 0}
    return {
        "avg_count": round(sum(values) / len(values), 2),
        "max_count": max(values),
        "min_count": min(values),
        "total_count": sum(values),
    }


def write_csv(path: Path, headers: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def save_summary_table_svg(headers: List[str], rows: List[List[object]], output_path: Path) -> None:
    cell_w = 220
    cell_h = 44
    margin = 20

    n_cols = len(headers)
    n_rows = len(rows) + 1

    width = n_cols * cell_w + margin * 2
    height = n_rows * cell_h + margin * 2

    def esc(text: object) -> str:
        s = str(text)
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{margin}" y="{margin}" width="{n_cols * cell_w}" height="{cell_h}" fill="#eaf4ff"/>',
    ]

    for r in range(n_rows + 1):
        y = margin + r * cell_h
        parts.append(
            f'<line x1="{margin}" y1="{y}" x2="{width - margin}" y2="{y}" stroke="#cfcfcf" stroke-width="1"/>'
        )
    for c in range(n_cols + 1):
        x = margin + c * cell_w
        parts.append(
            f'<line x1="{x}" y1="{margin}" x2="{x}" y2="{height - margin}" stroke="#cfcfcf" stroke-width="1"/>'
        )

    for c, header in enumerate(headers):
        x = margin + c * cell_w + cell_w / 2
        y = margin + cell_h / 2 + 6
        parts.append(
            f'<text x="{x}" y="{y}" text-anchor="middle" font-size="14" font-weight="bold" fill="#222">{esc(header)}</text>'
        )

    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            x = margin + c * cell_w + cell_w / 2
            y = margin + r * cell_h + cell_h / 2 + 6
            parts.append(
                f'<text x="{x}" y="{y}" text-anchor="middle" font-size="13" fill="#333">{esc(val)}</text>'
            )

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def make_summary_rows(
    yolo_values: List[int],
    pvrcnn_all_values: List[int],
    pvrcnn_camera_values: List[int],
    fusion_values: List[int],
) -> List[List[object]]:
    yolo_stats = compute_stats(yolo_values)
    pvrcnn_all_stats = compute_stats(pvrcnn_all_values)
    pvrcnn_camera_stats = compute_stats(pvrcnn_camera_values)
    fusion_stats = compute_stats(fusion_values)

    metrics = ["avg_count", "max_count", "min_count", "total_count"]
    return [
        [m, yolo_stats[m], pvrcnn_all_stats[m], pvrcnn_camera_stats[m], fusion_stats[m]]
        for m in metrics
    ]


def process_one_dataset(dataset_dir: Path) -> Optional[Tuple[str, List[int], List[int], List[int], List[int]]]:
    yolo_json = dataset_dir / "detection_results.json"
    pvrcnn_json = dataset_dir / "lidar_detection_results.json"
    fusion_json = dataset_dir / "fusion_results.json"

    missing = [p.name for p in [yolo_json, pvrcnn_json, fusion_json] if not p.exists()]
    if missing:
        print(f"⚠️ 跳过 {dataset_dir.name}: 缺少 {', '.join(missing)}")
        return None

    yolo_frames = load_json(yolo_json).get("frames", [])
    pvrcnn_frames = load_json(pvrcnn_json).get("frames", [])
    fusion_frames = load_json(fusion_json).get("frames", [])

    yolo_count = build_frame_count(yolo_frames)
    pvrcnn_all_count = build_frame_count(pvrcnn_frames, filter_fov=False)
    pvrcnn_camera_count = build_frame_count(pvrcnn_frames, filter_fov=True)
    fusion_count = build_frame_count(fusion_frames)

    all_frame_ids = sorted(
        set(yolo_count.keys())
        | set(pvrcnn_all_count.keys())
        | set(pvrcnn_camera_count.keys())
        | set(fusion_count.keys())
    )

    frame_headers = [
        "frame_id",
        "yolo_count",
        "pvrcnn_all_fov_count",
        "pvrcnn_camera_fov_count",
        "fusion_count",
    ]
    frame_rows: List[List[object]] = []
    for fid in all_frame_ids:
        frame_rows.append(
            [
                fid,
                yolo_count.get(fid, 0),
                pvrcnn_all_count.get(fid, 0),
                pvrcnn_camera_count.get(fid, 0),
                fusion_count.get(fid, 0),
            ]
        )

    frame_csv = dataset_dir / FRAME_CSV_NAME
    summary_csv = dataset_dir / SUMMARY_CSV_NAME
    summary_svg = dataset_dir / SUMMARY_TABLE_SVG_NAME

    write_csv(frame_csv, frame_headers, frame_rows)

    yolo_values = [row[1] for row in frame_rows]
    pvrcnn_all_values = [row[2] for row in frame_rows]
    pvrcnn_camera_values = [row[3] for row in frame_rows]
    fusion_values = [row[4] for row in frame_rows]

    summary_headers = ["metric", "yolo", "pvrcnn_all_fov", "pvrcnn_camera_fov", "fusion"]
    summary_rows = make_summary_rows(yolo_values, pvrcnn_all_values, pvrcnn_camera_values, fusion_values)

    write_csv(summary_csv, summary_headers, summary_rows)
    save_summary_table_svg(summary_headers, summary_rows, summary_svg)

    print(f"✅ {dataset_dir.name} 分析完成")
    print(f"   - 逐帧 CSV: {frame_csv}")
    print(f"   - 汇总 CSV: {summary_csv}")
    print(f"   - 汇总图片: {summary_svg}")

    return dataset_dir.name, yolo_values, pvrcnn_all_values, pvrcnn_camera_values, fusion_values


def main() -> None:
    dataset_dirs = sorted([p for p in BASE_RECORD_DIR.glob(DATASET_GLOB) if p.is_dir()])

    if not dataset_dirs:
        print(f"❌ 未找到数据目录: {BASE_RECORD_DIR / DATASET_GLOB}")
        return

    all_dataset_rows: List[List[object]] = []
    all_yolo: List[int] = []
    all_pvrcnn_all: List[int] = []
    all_pvrcnn_camera: List[int] = []
    all_fusion: List[int] = []

    for dataset_dir in dataset_dirs:
        result = process_one_dataset(dataset_dir)
        if result is None:
            continue

        dataset_name, yolo_values, pvrcnn_all_values, pvrcnn_camera_values, fusion_values = result

        all_yolo.extend(yolo_values)
        all_pvrcnn_all.extend(pvrcnn_all_values)
        all_pvrcnn_camera.extend(pvrcnn_camera_values)
        all_fusion.extend(fusion_values)

        ds_summary = make_summary_rows(yolo_values, pvrcnn_all_values, pvrcnn_camera_values, fusion_values)
        total_row = [row for row in ds_summary if row[0] == "total_count"][0]

        all_dataset_rows.append(
            [
                dataset_name,
                total_row[1],
                total_row[2],
                total_row[3],
                total_row[4],
            ]
        )

    if not all_dataset_rows:
        print("❌ 没有可汇总的数据集（缺少必要 JSON 文件）")
        return

    all_headers = [
        "dataset",
        "yolo_total_count",
        "pvrcnn_all_fov_total_count",
        "pvrcnn_camera_fov_total_count",
        "fusion_total_count",
    ]
    write_csv(ALL_DATASETS_SUMMARY_CSV, all_headers, all_dataset_rows)

    grand_headers = ["metric", "yolo", "pvrcnn_all_fov", "pvrcnn_camera_fov", "fusion"]
    grand_rows = make_summary_rows(all_yolo, all_pvrcnn_all, all_pvrcnn_camera, all_fusion)
    save_summary_table_svg(grand_headers, grand_rows, ALL_DATASETS_TABLE_SVG)

    print("\n✅ 全部数据集汇总完成")
    print(f"- 数据集总数对比 CSV: {ALL_DATASETS_SUMMARY_CSV}")
    print(f"- 全体帧汇总图片: {ALL_DATASETS_TABLE_SVG}")


if __name__ == "__main__":
    main()
