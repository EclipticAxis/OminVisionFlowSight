r"""Python DetectionTracker vs omintrack.RustTracker 墙钟基准。

合成负载：20 个匀速运动 track x 500 帧（每帧 ~20 检测，seeded 噪声 + 偶发漏检），
三种 filter_type (ukf/mcukf/manifold_ukf) x 三种 matching_strategy
(greedy/hungarian/cascade) 共 9 组，对比单线程墙钟时间并输出加速比。

用法：venv\Scripts\python.exe benchmark\benchmark_tracker_rust.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

if sys.platform == "win32":
    # Windows 中文系统默认 GBK 控制台：避免输出表头中文乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import omintrack  # noqa: E402
from ai.tracker import DetectionTracker  # noqa: E402

N_TRACKS = 20
N_FRAMES = 500
FILTER_TYPES = ["ukf", "mcukf", "manifold_ukf"]
MATCHING_STRATEGIES = ["greedy", "hungarian", "cascade"]


def build_sequence(seed: int = 20261118):
    """20 个 track 的确定性合成检测序列（归一化坐标）。"""
    rng = np.random.default_rng(seed)
    # 每 track 初始位置 / 速度 / 尺寸（保证 box 留在画面内）
    states = []
    for _ in range(N_TRACKS):
        cx = rng.uniform(0.15, 0.85)
        cy = rng.uniform(0.15, 0.85)
        vx = rng.uniform(-0.002, 0.002)
        vy = rng.uniform(-0.0015, 0.0015)
        w = rng.uniform(0.06, 0.14)
        h = rng.uniform(0.08, 0.18)
        states.append([cx, cy, vx, vy, w, h])
    seq = []
    for f in range(N_FRAMES):
        dets = []
        for cx, cy, vx, vy, w, h in states:
            # ~7% 概率漏检
            if rng.random() < 0.07:
                continue
            ncx = min(max(cx + vx * f + rng.normal(0, 0.001), 0.05), 0.95)
            ncy = min(max(cy + vy * f + rng.normal(0, 0.001), 0.05), 0.95)
            nw = w * (1.0 + rng.normal(0, 0.01))
            nh = h * (1.0 + rng.normal(0, 0.01))
            conf = float(np.clip(rng.normal(0.85, 0.08), 0.05, 0.99))
            dets.append({
                "x1": float(ncx - nw / 2), "y1": float(ncy - nh / 2),
                "x2": float(ncx + nw / 2), "y2": float(ncy + nh / 2),
                "confidence": conf, "label": "person",
            })
        seq.append(dets)
    return seq


def run_python(seq, filter_type: str, strategy: str) -> float:
    tracker = DetectionTracker(filter_type=filter_type, matching_strategy=strategy)
    t0 = time.perf_counter()
    for dets in seq:
        tracker.update(dets)
    return time.perf_counter() - t0


def run_rust(seq, filter_type: str, strategy: str) -> float:
    tracker = omintrack.RustTracker(filter_type=filter_type, matching_strategy=strategy)
    t0 = time.perf_counter()
    for dets in seq:
        tracker.update(dets)
    return time.perf_counter() - t0


def main() -> None:
    seq = build_sequence()
    n_dets = sum(len(f) for f in seq)
    print(f"合成负载: {N_TRACKS} tracks x {N_FRAMES} 帧, 共 {n_dets} 检测 "
          f"(~{n_dets / N_FRAMES:.1f}/帧)")
    print(f"{'filter_type':<15} {'strategy':<10} {'python(s)':>10} {'rust(s)':>10} {'speedup':>8}")
    print("-" * 58)
    speedups = []
    for ft in FILTER_TYPES:
        for ms in MATCHING_STRATEGIES:
            py_s = run_python(seq, ft, ms)   # 单次测量（负载确定，抖动 < 5%）
            rs_s = run_rust(seq, ft, ms)
            sp = py_s / rs_s
            speedups.append(sp)
            print(f"{ft:<15} {ms:<10} {py_s:>10.3f} {rs_s:>10.3f} {sp:>8.2f}x")
    print("-" * 58)
    print(f"{'mean':<26} {'':>10} {'':>10} {np.mean(speedups):>8.2f}x")


if __name__ == "__main__":
    main()
