#!/usr/bin/env python3
"""离线 MOT 评估管线：mp4 → YOLO 推理 → DetectionTracker → 代理指标。

无需 Ground Truth，使用代理指标评估跟踪质量：
  - ID 切换次数（IoU+中心距链式推断同一物理目标的 track_id 变化）
  - 锁定成功率（连续 ≥N 帧同 ID 的 track 占比）
  - track 存活时长分布
  - 平均 miss 率
  - 总 track 数 / 峰值并发 track 数

用法：
  python tools/mot_eval.py --input recordings/slot_0_xxx.mp4 --strategy hungarian
  python tools/mot_eval.py --input recordings/slot_0_xxx.mp4 --strategy greedy|hungarian|cascade
  python tools/mot_eval.py --input recordings/slot_0_xxx.mp4 --compare  # 三策略对比
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TrackRecord:
    """单条 track 的生命周期记录。"""
    track_id: int
    label: str
    first_frame: int
    last_frame: int
    frames_visible: int = 0
    boxes: list = field(default_factory=list)  # [(frame, x1, y1, x2, y2), ...]


@dataclass
class EvalMetrics:
    """评估代理指标。"""
    total_frames: int = 0
    total_tracks: int = 0
    peak_concurrent: int = 0
    id_switches: int = 0
    lock_success_rate: float = 0.0       # 连续 ≥5 帧同 ID 的 track 占比
    avg_track_lifetime: float = 0.0      # 平均存活帧数
    median_track_lifetime: float = 0.0
    avg_miss_rate: float = 0.0           # track miss 帧占比
    inference_fps: float = 0.0

    def __str__(self) -> str:
        return (
            f"  总帧数: {self.total_frames}\n"
            f"  总 track 数: {self.total_tracks}\n"
            f"  峰值并发 track: {self.peak_concurrent}\n"
            f"  ID 切换次数: {self.id_switches}\n"
            f"  锁定成功率(≥5帧): {self.lock_success_rate:.1%}\n"
            f"  平均存活帧数: {self.avg_track_lifetime:.1f}\n"
            f"  中位存活帧数: {self.median_track_lifetime:.1f}\n"
            f"  平均 miss 率: {self.avg_miss_rate:.1%}\n"
            f"  推理 FPS: {self.inference_fps:.1f}"
        )


def compute_id_switches(track_records: list[TrackRecord]) -> int:
    """通过 IoU+中心距链式推断同一物理目标的 track_id 变化次数。

    如果两个不同 track_id 的 track 在相邻帧的 IoU > 0.3，
    认为是同一物理目标发生了 ID 切换。
    """
    if not track_records:
        return 0

    # 按首帧排序
    sorted_tracks = sorted(track_records, key=lambda t: t.first_frame)
    switches = 0
    used = set()

    for i, trk_a in enumerate(sorted_tracks):
        if i in used:
            continue
        # 找在 trk_a 最后一帧附近出现的其他 track
        for j, trk_b in enumerate(sorted_tracks):
            if j <= i or j in used:
                continue
            # 检查时间是否相邻（trk_b 首帧 ≈ trk_a 末帧 ±2）
            gap = trk_b.first_frame - trk_a.last_frame
            if abs(gap) > 3:
                continue
            # 检查空间重叠
            if not trk_a.boxes or not trk_b.boxes:
                continue
            last_box = trk_a.boxes[-1][1:]  # (x1, y1, x2, y2)
            first_box = trk_b.boxes[0][1:]
            iou = _box_iou(last_box, first_box)
            if iou > 0.3:
                switches += 1
                used.add(j)
    return switches


def _box_iou(a: tuple, b: tuple) -> float:
    """计算两个 [x1, y1, x2, y2] 框的 IoU。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-10)


def run_evaluation(
    video_path: str,
    model_path: str = "models/yolov8n.pt",
    strategy: str = "hungarian",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.25,
    max_misses: int = 4,
    high_conf_thresh: float = 0.5,
    max_frames: int = 0,
    verbose: bool = True,
) -> EvalMetrics:
    """对视频跑单次评估，返回代理指标。"""
    import av

    from ai.tracker import DetectionTracker
    from common.yolo_detector import YOLODetector

    # 初始化检测器
    detector = YOLODetector(model_path=model_path, device="cpu")
    detector.load_model(model_path, device="cpu", imgsz=640)

    # 初始化跟踪器
    tracker = DetectionTracker(
        iou_threshold=iou_threshold,
        max_misses=max_misses,
        matching_strategy=strategy,
        high_conf_thresh=high_conf_thresh,
        filter_type="ukf",
    )

    # 打开视频
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = 0
    frame_id = 0

    track_records: dict[int, TrackRecord] = {}
    concurrent_counts = []
    miss_frames = 0
    inference_times = []

    if verbose:
        print(f"  策略: {strategy} | 模型: {model_path} | conf={conf_threshold}")

    for frame_idx, frame in enumerate(container.decode(video=0)):
        if max_frames > 0 and frame_idx >= max_frames:
            break

        # PyAV frame → RGB ndarray
        img = frame.to_ndarray(format="rgb24")

        t0 = time.time()
        detections = detector.predict(img, conf=conf_threshold)
        t1 = time.time()
        inference_times.append(t1 - t0)

        # 跟踪
        results = tracker.update(detections, frame=img)
        total_frames += 1

        # 记录 track 状态
        active_ids = set()
        for r in results:
            tid = r["track_id"]
            active_ids.add(tid)
            box = (r["x1"], r["y1"], r["x2"], r["y2"])
            if tid not in track_records:
                track_records[tid] = TrackRecord(
                    track_id=tid,
                    label=r.get("label", ""),
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    frames_visible=1,
                )
            else:
                track_records[tid].last_frame = frame_idx
                track_records[tid].frames_visible += 1
            track_records[tid].boxes.append((frame_idx, *box))

        concurrent_counts.append(len(active_ids))
        if not active_ids:
            miss_frames += 1

        if verbose and frame_idx % 100 == 0:
            print(f"    帧 {frame_idx}: {len(results)} 检测, {len(track_records)} 累计 track")

    container.close()

    # 计算指标
    metrics = EvalMetrics()
    metrics.total_frames = total_frames
    metrics.total_tracks = len(track_records)
    metrics.peak_concurrent = max(concurrent_counts) if concurrent_counts else 0
    metrics.id_switches = compute_id_switches(list(track_records.values()))

    # 锁定成功率：存活 ≥5 帧的 track 占比
    if track_records:
        locked = sum(1 for t in track_records.values() if t.frames_visible >= 5)
        metrics.lock_success_rate = locked / len(track_records)
        lifetimes = [t.frames_visible for t in track_records.values()]
        metrics.avg_track_lifetime = float(np.mean(lifetimes))
        metrics.median_track_lifetime = float(np.median(lifetimes))

    metrics.avg_miss_rate = miss_frames / max(total_frames, 1)
    if inference_times:
        metrics.inference_fps = 1.0 / max(np.mean(inference_times), 1e-6)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="离线 MOT 评估管线")
    parser.add_argument("--input", "-i", required=True, help="输入视频路径")
    parser.add_argument("--model", "-m", default="models/yolov8n.pt", help="YOLO 模型路径")
    parser.add_argument("--strategy", "-s", default="hungarian",
                        choices=["greedy", "hungarian", "cascade"],
                        help="匹配策略")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="三策略对比模式")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=0.25, help="匹配 IoU 阈值")
    parser.add_argument("--max-frames", type=int, default=0, help="最大评估帧数(0=全部)")
    parser.add_argument("--max-misses", type=int, default=4, help="最大丢失帧数")
    parser.add_argument("--high-conf", type=float, default=0.5, help="级联高置信度阈值")
    args = parser.parse_args()

    video_path = str(PROJECT_ROOT / args.input)
    model_path = str(PROJECT_ROOT / args.model)

    if not Path(video_path).exists():
        print(f"错误: 视频文件不存在: {video_path}")
        sys.exit(1)

    if args.compare:
        # 三策略对比
        print(f"\n{'='*60}")
        print(f"  A/B 对比评估: {args.input}")
        print(f"{'='*60}")
        results = {}
        for strategy in ["greedy", "hungarian", "cascade"]:
            print(f"\n--- {strategy.upper()} ---")
            metrics = run_evaluation(
                video_path, model_path, strategy,
                conf_threshold=args.conf, iou_threshold=args.iou,
                max_misses=args.max_misses, high_conf_thresh=args.high_conf,
                max_frames=args.max_frames,
            )
            results[strategy] = metrics
            print(metrics)

        # 对比表
        print(f"\n{'='*60}")
        print("  对比总结")
        print(f"{'='*60}")
        print(f"{'指标':<20} {'greedy':>12} {'hungarian':>12} {'cascade':>12}")
        print("-" * 60)
        print(f"{'ID 切换次数':<20} {results['greedy'].id_switches:>12} {results['hungarian'].id_switches:>12} {results['cascade'].id_switches:>12}")
        print(f"{'锁定成功率':<20} {results['greedy'].lock_success_rate:>11.1%} {results['hungarian'].lock_success_rate:>11.1%} {results['cascade'].lock_success_rate:>11.1%}")
        print(f"{'总 track 数':<20} {results['greedy'].total_tracks:>12} {results['hungarian'].total_tracks:>12} {results['cascade'].total_tracks:>12}")
        print(f"{'平均存活帧数':<20} {results['greedy'].avg_track_lifetime:>12.1f} {results['hungarian'].avg_track_lifetime:>12.1f} {results['cascade'].avg_track_lifetime:>12.1f}")
        print(f"{'峰值并发':<20} {results['greedy'].peak_concurrent:>12} {results['hungarian'].peak_concurrent:>12} {results['cascade'].peak_concurrent:>12}")
        print(f"{'推理 FPS':<20} {results['greedy'].inference_fps:>12.1f} {results['hungarian'].inference_fps:>12.1f} {results['cascade'].inference_fps:>12.1f}")
    else:
        print(f"\n{'='*60}")
        print(f"  MOT 评估: {args.input}")
        print(f"  策略: {args.strategy}")
        print(f"{'='*60}")
        metrics = run_evaluation(
            video_path, model_path, args.strategy,
            conf_threshold=args.conf, iou_threshold=args.iou,
            max_misses=args.max_misses, high_conf_thresh=args.high_conf,
            max_frames=args.max_frames,
        )
        print(f"\n结果:")
        print(metrics)


if __name__ == "__main__":
    main()
