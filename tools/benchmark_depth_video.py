"""复现深度视频时域稳定性与后处理性能对比。

示例：
  venv\\Scripts\\python.exe tools\\benchmark_depth_video.py --camera "ASUS FHD webcam"
  venv\\Scripts\\python.exe tools\\benchmark_depth_video.py --input sample.mp4 --output depth.json

指标使用同一批原始帧：raw 与运动自适应 EMA 后的深度共享同一稳定量程，
分别执行 Turbo + disparity + gamma=0.5 着色，再比较相邻帧平均 RGB L1 变化。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.depth_worker import (
    DepthWorker,
    colorize_depth,
    filter_depth_ema,
    update_adaptive_range,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="输入视频文件")
    source.add_argument("--camera", help="Windows DirectShow 摄像头名称")
    parser.add_argument("--frames", type=int, default=120, help="计入统计的帧数")
    parser.add_argument("--warmup", type=int, default=5, help="建立时域状态的预热帧数")
    parser.add_argument("--model", help="默认 models/yolo26s-depth.pt")
    parser.add_argument("--device", help="例如 cpu；默认自动选择 CUDA/CPU")
    parser.add_argument("--output", help="可选 JSON 报告输出路径")
    return parser.parse_args()


def _frames(args: argparse.Namespace):
    if args.input:
        import cv2

        capture = cv2.VideoCapture(args.input)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开输入视频：{args.input}")
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
    else:
        import av

        container = av.open(
            f"video={args.camera}",
            format="dshow",
            options={
                "rtbufsize": "200M",
                "video_size": "640x480",
                "framerate": "30",
            },
        )
        try:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")
        finally:
            container.close()


def run(args: argparse.Namespace) -> dict:
    if args.frames < 2 or args.warmup < 0:
        raise ValueError("frames 必须 >=2，warmup 必须 >=0")

    device = args.device
    if device is not None and device.isdigit():
        device = int(device)
    worker = DepthWorker(model_path=args.model, device=device)

    started = time.perf_counter()
    worker._resolved_device = worker._resolve_device()
    worker._model = worker._load_model()
    load_seconds = time.perf_counter() - started

    previous_depth = None
    metric_range = None
    previous_raw_color = None
    previous_filtered_color = None
    raw_deltas: list[float] = []
    filtered_deltas: list[float] = []
    stages: dict[str, list[float]] = {
        "infer_ms": [],
        "filter_ms": [],
        "range_ms": [],
        "color_ms": [],
    }
    measured = 0
    measure_started = None

    for index, rgb in enumerate(_frames(args)):
        t0 = time.perf_counter()
        raw = worker._infer_depth(rgb)
        t1 = time.perf_counter()
        filtered = filter_depth_ema(previous_depth, raw)
        t2 = time.perf_counter()
        metric_range = update_adaptive_range(metric_range, filtered)
        t3 = time.perf_counter()
        kwargs = {
            "vmin": metric_range[0],
            "vmax": metric_range[1],
            "cmap": "turbo",
            "mode": "disparity",
            "gamma": 0.5,
        }
        raw_color = colorize_depth(raw, **kwargs)
        filtered_color = colorize_depth(filtered, **kwargs)
        t4 = time.perf_counter()

        if index < args.warmup:
            previous_depth = filtered
            previous_raw_color = None
            previous_filtered_color = None
            continue
        if measure_started is None:
            measure_started = t0
        if previous_raw_color is not None:
            raw_deltas.append(
                float(np.mean(np.abs(raw_color.astype(np.float32) - previous_raw_color.astype(np.float32))))
            )
            filtered_deltas.append(
                float(
                    np.mean(
                        np.abs(filtered_color.astype(np.float32) - previous_filtered_color.astype(np.float32))
                    )
                )
            )
        stages["infer_ms"].append((t1 - t0) * 1000)
        stages["filter_ms"].append((t2 - t1) * 1000)
        stages["range_ms"].append((t3 - t2) * 1000)
        # raw + filtered 两次着色都计入复核脚本开销；产品只执行后者。
        stages["color_ms"].append((t4 - t3) * 1000)
        previous_depth = filtered
        previous_raw_color = raw_color
        previous_filtered_color = filtered_color
        measured += 1
        if measured >= args.frames:
            break

    if measured < args.frames:
        raise RuntimeError(f"输入帧不足：需要 {args.frames}，实际 {measured}")
    raw_mean = float(np.mean(raw_deltas))
    filtered_mean = float(np.mean(filtered_deltas))
    elapsed = time.perf_counter() - measure_started
    return {
        "schema_version": 1,
        "source": {"input": args.input, "camera": args.camera, "resolution": [640, 480]},
        "model": worker._model_path,
        "device": worker._resolved_device,
        "platform": platform.platform(),
        "frames": measured,
        "warmup_frames": args.warmup,
        "load_seconds": load_seconds,
        "effective_fps_including_dual_colorize": measured / elapsed,
        "stage_median_ms": {
            name: float(np.median(values)) for name, values in stages.items()
        },
        "raw_mean_rgb_delta": raw_mean,
        "filtered_mean_rgb_delta": filtered_mean,
        "color_fluctuation_reduction_pct": 100.0 * (1.0 - filtered_mean / raw_mean),
        "mapping": "Turbo + disparity + gamma=0.5；raw/filtered 共享同一稳定米制量程",
    }


def main() -> None:
    args = _parse_args()
    report = run(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
