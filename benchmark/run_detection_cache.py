#!/usr/bin/env python3
r"""私有检测离线缓存生成（MOT det 格式，版本化）。

用项目现行检测后端（ultralytics YOLOv8n，与 ai/inference.py 同一调用形态
``model(frame, verbose=False, conf=..., imgsz=...)``）对指定序列逐帧检测，
输出 MOT det 格式缓存：

    <cache_root>/<dataset>/<seq>/det_yolov8n_v1.txt   # frame,id,left,top,width,height,conf（id=-1）
    <cache_root>/<dataset>/<seq>/det_yolov8n_v1.meta.json

设计约定（docs/tracking-quality-benchmark.md §3）：
  - 缓存保存**像素坐标**；DetectionTracker.update() 要求**归一化**坐标，
    换算由 run_tracking_offline.py 负责（依 seqinfo.ini 的 imWidth/imHeight）。
  - 缓存版本化：--det-version（默认 det_yolov8n_v1）；检测器/阈值/imgsz 变更
    必须递增版本号，旧基线作废重测。meta.json 记录模型、conf、imgsz、
    生成时间、代码 revision，供审计缓存口径。
  - 逐帧检测（不跳帧）；跳帧由跟踪阶段以 --stride 实现，保证缓存可复用。

纯离线：不 import GUI / InferenceWorker；ultralytics 与 cv2 延迟导入，
模块 import 无副作用。Windows 中文/空格路径统一走 pathlib。
"""
from __future__ import annotations

import configparser
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

DEFAULT_ROOT = Path(r"G:\基准测试")
DEFAULT_CACHE_ROOT = None  # main 中解析为 <root>/_det_cache
DEFAULT_DET_VERSION = "det_yolov8n_v1"
PERSON_CLASS_ID = 0  # COCO person

# 数据集布局（与 prepare_mot_data.py 一致；按任务约定脚本间不互相 import，独立维护）
_DATASET_RELPATH = {
    "mot17": Path("MOTChallenge") / "MOT17",
    "mot20": Path("MOTChallenge") / "MOT20",
    "dancetrack": Path("DanceTrack"),
}
_DATASET_SPLITS = {
    "mot17": ("train", "test"),
    "mot20": ("train", "test"),
    "dancetrack": ("train", "val", "test"),
}


@dataclass(frozen=True)
class SeqInfo:
    name: str
    im_dir: str
    seq_length: int
    im_width: int
    im_height: int
    im_ext: str


def read_seqinfo(seq_dir: Path) -> SeqInfo:
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.is_file():
        raise FileNotFoundError(f"缺少 seqinfo.ini: {ini_path}")
    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")
    sec = parser["Sequence"]
    return SeqInfo(
        name=sec.get("name", seq_dir.name).strip(),
        im_dir=sec.get("imDir", "img1").strip(),
        seq_length=int(sec["seqLength"]),
        im_width=int(sec["imWidth"]),
        im_height=int(sec["imHeight"]),
        im_ext=sec.get("imExt", ".jpg").strip(),
    )


def discover_sequences(root: Path, dataset: str, split: str) -> list[Path]:
    sdir = root / _DATASET_RELPATH[dataset] / split
    if not sdir.is_dir():
        return []
    return sorted(
        (p for p in sdir.iterdir() if p.is_dir() and (p / "seqinfo.ini").is_file()),
        key=lambda p: p.name,
    )


def list_frames(seq_dir: Path, info: SeqInfo) -> list[Path]:
    """按文件名排序列出图像帧；数量与 seqLength 不一致时抛错（防止静默错位）。"""
    img_dir = seq_dir / info.im_dir
    if not img_dir.is_dir():
        raise FileNotFoundError(f"缺少图像目录: {img_dir}")
    frames = sorted(
        (p for p in img_dir.iterdir() if p.suffix.lower() == info.im_ext.lower()),
        key=lambda p: p.name,
    )
    if len(frames) != info.seq_length:
        raise ValueError(
            f"{seq_dir.name}: 图像帧数 {len(frames)} 与 seqinfo seqLength "
            f"{info.seq_length} 不一致（{img_dir}）"
        )
    return frames


def git_revision(cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def cache_paths(cache_root: Path, dataset: str, seq: str, version: str) -> tuple[Path, Path]:
    base = cache_root / dataset / seq
    return base / f"{version}.txt", base / f"{version}.meta.json"


def detect_sequence(
    model,
    frames: list[Path],
    *,
    conf: float,
    imgsz: int,
    device: str,
) -> list[tuple[int, float, float, float, float, float]]:
    """对单序列逐帧检测，返回 [(frame_1based, left, top, w, h, conf)]（像素坐标）。"""
    import cv2  # 延迟导入：保持模块 import 轻量、无副作用

    rows: list[tuple[int, float, float, float, float, float]] = []
    for idx, img_path in enumerate(frames, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        results = model(img, verbose=False, conf=conf, imgsz=imgsz, device=device)
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                if int(b.cls.item()) != PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                rows.append((
                    idx, x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1),
                    float(b.conf.item()),
                ))
    return rows


def write_det_cache(rows, out_path: Path) -> None:
    """写 MOT det 格式：frame,id,left,top,width,height,conf（私有检测约定 id=-1）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for frame, left, top, w, h, conf in rows:
            f.write(f"{frame},-1,{left:.2f},{top:.2f},{w:.2f},{h:.2f},{conf:.4f}\n")


def run_dataset(
    *,
    root: Path,
    dataset: str,
    split: str,
    seqs: list[str],
    model_name: str,
    conf: float,
    imgsz: int,
    device: str,
    cache_root: Path,
    det_version: str,
    force: bool,
    project_root: Path,
) -> dict:
    """对选定序列生成检测缓存，返回 summary dict。"""
    available = {p.name: p for p in discover_sequences(root, dataset, split)}
    if not available:
        raise FileNotFoundError(
            f"未找到序列: {root / _DATASET_RELPATH[dataset] / split}\n"
            f"请先运行 benchmark/prepare_mot_data.py check --dataset {dataset} 查看下载指引。"
        )
    selected = sorted(available) if seqs == ["all"] else seqs
    missing = [s for s in selected if s not in available]
    if missing:
        raise ValueError(f"序列不存在: {missing}（可选: {sorted(available)}）")

    # 延迟导入 ultralytics：--help 与模块 import 不依赖重型包
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "需要 ultralytics（pip install ultralytics>=8.0.0）才能生成私有检测缓存"
        ) from exc

    model = YOLO(model_name)
    revision = git_revision(project_root)
    summary: dict = {
        "dataset": dataset,
        "split": split,
        "model": model_name,
        "conf": conf,
        "imgsz": imgsz,
        "device": device,
        "det_version": det_version,
        "revision": revision,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sequences": {},
    }
    for name in selected:
        seq_dir = available[name]
        info = read_seqinfo(seq_dir)
        frames = list_frames(seq_dir, info)
        out_path, meta_path = cache_paths(cache_root, dataset, name, det_version)
        if out_path.is_file() and meta_path.is_file() and not force:
            print(f"[skip] {name}: 缓存已存在（--force 可重生成）")
            summary["sequences"][name] = {"skipped": True, "det": str(out_path)}
            continue
        t0 = time.perf_counter()
        rows = detect_sequence(model, frames, conf=conf, imgsz=imgsz, device=device)
        elapsed = time.perf_counter() - t0
        write_det_cache(rows, out_path)
        meta = {
            "model": model_name,
            "conf": conf,
            "imgsz": imgsz,
            "device": device,
            "classes": {"person": PERSON_CLASS_ID},
            "frames": len(frames),
            "detections": len(rows),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "revision": revision,
            "det_version": det_version,
            "coordinate_space": "pixel",
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        fps = len(frames) / elapsed if elapsed > 0 else 0.0
        print(f"[done] {name}: {len(frames)} 帧, {len(rows)} 检测, "
              f"{elapsed:.1f}s ({fps:.1f} fps) -> {out_path}")
        summary["sequences"][name] = {
            "skipped": False,
            "frames": len(frames),
            "detections": len(rows),
            "elapsed_sec": round(elapsed, 3),
            "det": str(out_path),
        }
    return summary


def build_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="私有检测离线缓存生成（MOT det 格式，版本化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=sorted(_DATASET_RELPATH))
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help=f"数据集根目录（默认 {DEFAULT_ROOT}）")
    p.add_argument("--split", required=True,
                   help="split（mot17/mot20: train|test；dancetrack: train|val|test）")
    p.add_argument("--seqs", default="all",
                   help="all 或逗号分隔序列名（默认 all）")
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO 权重（默认 yolov8n.pt，项目现行检测后端）")
    p.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值（默认 0.25）")
    p.add_argument("--imgsz", type=int, default=640, help="推理边长（默认 640，同项目默认）")
    p.add_argument("--device", default="cpu", help="推理设备（默认 cpu；GPU 机器可传 0）")
    p.add_argument("--cache-root", type=Path, default=None,
                   help="缓存根目录（默认 <root>/_det_cache）")
    p.add_argument("--det-version", default=DEFAULT_DET_VERSION,
                   help=f"缓存版本标签（默认 {DEFAULT_DET_VERSION}；口径变更须递增）")
    p.add_argument("--force", action="store_true", help="忽略已有缓存强制重生成")
    p.add_argument("--summary-out", type=Path, default=None,
                   help="summary.json 输出路径（默认 <cache_root>/<dataset>/det_cache_summary_<split>.json）")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    if sys.platform == "win32":
        # Windows 中文系统默认 GBK 控制台：避免输出中文乱码
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if args.split not in _DATASET_SPLITS[args.dataset]:
        print(f"错误: 数据集 {args.dataset} 无 split={args.split}"
              f"（可选: {_DATASET_SPLITS[args.dataset]}）", file=sys.stderr)
        return 2
    cache_root = args.cache_root or (args.root / "_det_cache")
    seqs = [s.strip() for s in args.seqs.split(",") if s.strip()] if args.seqs != "all" else ["all"]
    project_root = Path(__file__).resolve().parent.parent
    try:
        summary = run_dataset(
            root=args.root, dataset=args.dataset, split=args.split, seqs=seqs,
            model_name=args.model, conf=args.conf, imgsz=args.imgsz, device=args.device,
            cache_root=cache_root, det_version=args.det_version, force=args.force,
            project_root=project_root,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    summary_out = args.summary_out or (
        cache_root / args.dataset / f"det_cache_summary_{args.split}.json"
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary -> {summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
