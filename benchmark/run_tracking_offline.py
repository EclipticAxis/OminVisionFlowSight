#!/usr/bin/env python3
r"""离线跟踪 runner：det 缓存/公开检测 → DetectionTracker → TrackEval 兼容 MOT 结果。

职责（docs/tracking-quality-benchmark.md §5.2）：
  - 逐帧读取检测（私有 det 缓存 或 MOT17/MOT20 公开检测 det/det.txt，均只含 person），
    直接驱动 ai/tracker.py 的 DetectionTracker.update()，无 GUI/摄像头依赖；
  - --stride N 跳帧：被跳帧调用 predict_step()（仅推进卡尔曼、不增 misses、
    输出门控语义与线上 InferWorker 完全一致，见 docs/rust-tracker-core.md §3.2）；
  - 坐标约定：det 文件为**像素坐标**；DetectionTracker.update() 要求**归一化**
    [0,1] 坐标（ai/tracker.py `_clamp01` / ReID 按帧宽缩放可证），本脚本按
    seqinfo.ini 的 imWidth/imHeight 换算；输出再换回像素写 MOT 结果；
  - 输出 TrackEval 兼容 MOT 结果：<out>/<tracker-tag>/<seq>.txt
    （frame,id,left,top,width,height,conf,-1,-1,-1，frame 1 起）；
  - 运行时元信息写 <out>/<tracker-tag>/summary.json（每序列耗时/帧数 +
    协议口径字段，供 evaluate_trackeval.py 汇入报告）。

脚本间只通过文件与命令行参数耦合；模块 import 无副作用（ai.tracker 延迟导入）。
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
from typing import Iterable, Protocol

# 项目根目录入 sys.path，保证从任意 cwd 运行都能 import ai.tracker
# （与 benchmark/benchmark_tracker_rust.py 同款做法）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_ROOT = Path(r"G:\基准测试")
DEFAULT_DET_VERSION = "det_yolov8n_v1"

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

FILTER_TYPES = ("ukf", "mcukf", "manifold_ukf", "auto")
BACKENDS = ("python", "rust")
DET_SOURCES = ("private", "public-frcnn")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeqInfo:
    name: str
    frame_rate: float
    seq_length: int
    im_width: int
    im_height: int


@dataclass(frozen=True)
class DetRow:
    """一条 MOT det 记录（像素坐标，1 起帧号）。"""
    frame: int
    left: float
    top: float
    width: float
    height: float
    conf: float


@dataclass(frozen=True)
class ResultRow:
    """一条 MOT 结果记录（像素坐标，1 起帧号）。"""
    frame: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    conf: float


class TrackerLike(Protocol):
    """DetectionTracker / RustDetectionTracker 公开 API 子集（runner 只依赖这些）。"""

    def update(self, detections: list[dict]) -> list[dict]: ...
    def predict_step(self) -> list[dict]: ...
    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def read_seqinfo(seq_dir: Path) -> SeqInfo:
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.is_file():
        raise FileNotFoundError(f"缺少 seqinfo.ini: {ini_path}")
    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")
    sec = parser["Sequence"]
    return SeqInfo(
        name=sec.get("name", seq_dir.name).strip(),
        frame_rate=float(sec["frameRate"]),
        seq_length=int(sec["seqLength"]),
        im_width=int(sec["imWidth"]),
        im_height=int(sec["imHeight"]),
    )


def load_mot_detections(path: Path) -> dict[int, list[DetRow]]:
    """解析 MOT det 格式（7 列或 10 列，像素坐标）→ {frame: [DetRow]}。

    私有缓存（7 列 frame,id,l,t,w,h,conf）与 MOT17/MOT20 公开检测
    （10 列，尾部 x,y,z 忽略）共用同一解析。
    """
    if not path.is_file():
        raise FileNotFoundError(f"检测文件不存在: {path}")
    by_frame: dict[int, list[DetRow]] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                raise ValueError(f"{path}:{lineno}: 至少需要 7 列，实际 {len(parts)} 列: {line!r}")
            try:
                frame = int(float(parts[0]))
                row = DetRow(
                    frame=frame,
                    left=float(parts[2]),
                    top=float(parts[3]),
                    width=float(parts[4]),
                    height=float(parts[5]),
                    conf=float(parts[6]),
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: 数值解析失败: {line!r}") from exc
            if frame < 1:
                raise ValueError(f"{path}:{lineno}: 帧号必须 >= 1（MOT 1 起）: {line!r}")
            if row.width <= 0 or row.height <= 0:
                continue  # 退化框跳过，不视为格式错误
            by_frame.setdefault(frame, []).append(row)
    return by_frame


# ---------------------------------------------------------------------------
# 坐标换算（像素 <-> 归一化；DetectionTracker 输入必须归一化，见模块 docstring）
# ---------------------------------------------------------------------------

def det_to_tracker_input(row: DetRow, im_width: int, im_height: int) -> dict:
    """像素 DetRow → DetectionTracker.update() 的归一化 dict schema。"""
    return {
        "x1": row.left / im_width,
        "y1": row.top / im_height,
        "x2": (row.left + row.width) / im_width,
        "y2": (row.top + row.height) / im_height,
        "confidence": row.conf,
        "label": "person",
        "keypoints": [],
    }


def track_to_result_row(frame: int, track: dict, im_width: int, im_height: int) -> ResultRow:
    """tracker 输出（归一化）→ 像素 ResultRow，并裁剪到图像边界。"""
    x1 = min(max(float(track["x1"]) * im_width, 0.0), float(im_width))
    y1 = min(max(float(track["y1"]) * im_height, 0.0), float(im_height))
    x2 = min(max(float(track["x2"]) * im_width, 0.0), float(im_width))
    y2 = min(max(float(track["y2"]) * im_height, 0.0), float(im_height))
    return ResultRow(
        frame=frame,
        track_id=int(track["track_id"]),
        left=x1,
        top=y1,
        width=max(0.0, x2 - x1),
        height=max(0.0, y2 - y1),
        conf=float(track.get("confidence", 1.0)),
    )


# ---------------------------------------------------------------------------
# 序列运行（stride 跳帧语义与线上一致）
# ---------------------------------------------------------------------------

def run_sequence(
    tracker: TrackerLike,
    dets_by_frame: dict[int, list[DetRow]],
    seq_info: SeqInfo,
    *,
    stride: int = 1,
    min_conf: float = 0.0,
) -> list[ResultRow]:
    """逐帧驱动 tracker，返回全部输出帧的 ResultRow。

    stride 语义（与线上 InferWorker 跳帧路径一致）：
      - (frame-1) % stride == 0 的帧：tracker.update(dets)；
      - 其余帧：tracker.predict_step()（仅推进卡尔曼，不增 misses、不删 track，
        misses/max_output_misses/predicted_only 语义见 docs/rust-tracker-core.md §3.2）；
      - 两种帧都写输出（输出可见性由 tracker 内部 hits/misses 门控决定）。
    """
    if stride < 1:
        raise ValueError(f"stride 必须 >= 1，实际 {stride}")
    rows: list[ResultRow] = []
    for frame in range(1, seq_info.seq_length + 1):
        if (frame - 1) % stride == 0:
            dets = [
                det_to_tracker_input(r, seq_info.im_width, seq_info.im_height)
                for r in dets_by_frame.get(frame, [])
                if r.conf >= min_conf
            ]
            tracks = tracker.update(dets)
        else:
            tracks = tracker.predict_step()
        for trk in tracks:
            rows.append(track_to_result_row(frame, trk, seq_info.im_width, seq_info.im_height))
    return rows


def write_mot_results(path: Path, rows: list[ResultRow]) -> None:
    """写 TrackEval 兼容 MOT 结果（10 列，尾部 x,y,z 以 -1 占位）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(
                f"{r.frame},{r.track_id},{r.left:.2f},{r.top:.2f},"
                f"{r.width:.2f},{r.height:.2f},{r.conf:.4f},-1,-1,-1\n"
            )


# ---------------------------------------------------------------------------
# 检测来源解析
# ---------------------------------------------------------------------------

def resolve_det_file(
    *,
    dataset: str,
    root: Path,
    split: str,
    seq: str,
    det_source: str,
    cache_root: Path,
    det_version: str,
) -> Path:
    """按检测协议解析 det 文件路径（docs/tracking-quality-benchmark.md §3 两条线）。"""
    if det_source == "private":
        det = cache_root / dataset / seq / f"{det_version}.txt"
        if not det.is_file():
            raise FileNotFoundError(
                f"私有检测缓存不存在: {det}\n"
                f"请先运行 benchmark/run_detection_cache.py --dataset {dataset} "
                f"--split {split} 生成缓存（版本 {det_version}）。"
            )
        return det
    if det_source == "public-frcnn":
        if dataset == "dancetrack":
            raise ValueError("DanceTrack 无公开检测协议，只能用 --det-source private")
        det = root / _DATASET_RELPATH[dataset] / split / seq / "det" / "det.txt"
        if not det.is_file():
            raise FileNotFoundError(
                f"公开检测不存在: {det}\n"
                f"请运行 benchmark/prepare_mot_data.py check --dataset {dataset} 校验数据集。"
            )
        return det
    raise ValueError(f"未知 det_source: {det_source}（可选: {DET_SOURCES}）")


def make_tracker(*, filter_type: str, backend: str, max_output_misses: int,
                 matching_strategy: str, iou_threshold: float, max_misses: int,
                 min_hits_to_show: int,
                 nsa_enabled: bool = False, nsa_interval: float = 1.0,
                 nsa_min: float = 0.1, nsa_conf_thresh: float = 0.6,
                 nsa_interval_sec: float = 2.0, ocm_enabled: bool = False,
                 ocm_delta_t: int = 3, ocm_weight: float = 0.2,
                 hmiou_enabled: bool = False, oru_enabled: bool = False,
                 ocr_enabled: bool = False, ocr_threshold: float = 0.3,
                 tcm_enabled: bool = False, tcm_weight: float = 0.2) -> TrackerLike:
    """构造 DetectionTracker（backend 经 __new__ 分发 python/rust，显式 rust 不可用时抛错）。

    跟踪增强（docs/tracker-nsa-ocm-hmiou.md）默认全部关闭，经参数透传。
    """
    from ai.tracker import DetectionTracker  # 延迟导入：模块 import 不拉入 cv2/numpy 链

    return DetectionTracker(
        backend=backend,
        filter_type=filter_type,
        matching_strategy=matching_strategy,
        iou_threshold=iou_threshold,
        max_misses=max_misses,
        min_hits_to_show=min_hits_to_show,
        max_output_misses=max_output_misses,
        nsa_enabled=nsa_enabled,
        nsa_interval=nsa_interval,
        nsa_min=nsa_min,
        nsa_conf_thresh=nsa_conf_thresh,
        nsa_interval_sec=nsa_interval_sec,
        ocm_enabled=ocm_enabled,
        ocm_delta_t=ocm_delta_t,
        ocm_weight=ocm_weight,
        hmiou_enabled=hmiou_enabled,
        oru_enabled=oru_enabled,
        ocr_enabled=ocr_enabled,
        ocr_threshold=ocr_threshold,
        tcm_enabled=tcm_enabled,
        tcm_weight=tcm_weight,
    )


def git_revision(cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return "unknown"
        sha = out.stdout.strip()
        # 脏指纹：跟踪/评测代码有未提交改动时标记 -dirty（排除 results/模型/数据产物），
        # 使 summary.json 记录的 revision 能真实复现产出代码。
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--",
             "ai", "rust", "benchmark", ":(exclude)benchmark/results"],
            cwd=cwd, capture_output=True, text=True, timeout=10,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            return sha + "-dirty"
        return sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def default_tracker_tag(filter_type: str, backend: str, stride: int, det_source: str) -> str:
    det_tag = "priv" if det_source == "private" else "frcnn"
    return f"omintrack_{filter_type}_{backend}_s{stride}_{det_tag}"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(
    *,
    root: Path,
    dataset: str,
    split: str,
    seqs: list[str],
    det_source: str,
    cache_root: Path,
    det_version: str,
    filter_type: str,
    backend: str,
    stride: int,
    max_output_misses: int,
    matching_strategy: str,
    iou_threshold: float,
    max_misses: int,
    min_hits_to_show: int,
    min_conf: float,
    out_dir: Path,
    tracker_tag: str,
    project_root: Path,
    nsa_enabled: bool = False,
    nsa_interval: float = 1.0,
    nsa_min: float = 0.1,
    nsa_conf_thresh: float = 0.6,
    nsa_interval_sec: float = 2.0,
    ocm_enabled: bool = False,
    ocm_delta_t: int = 3,
    ocm_weight: float = 0.2,
    hmiou_enabled: bool = False,
    oru_enabled: bool = False,
    ocr_enabled: bool = False,
    ocr_threshold: float = 0.3,
    tcm_enabled: bool = False,
    tcm_weight: float = 0.2,
) -> dict:
    sdir = root / _DATASET_RELPATH[dataset] / split
    if not sdir.is_dir():
        raise FileNotFoundError(
            f"split 目录不存在: {sdir}\n"
            f"请先运行 benchmark/prepare_mot_data.py check --dataset {dataset} 查看下载指引。"
        )
    available = sorted(
        (p for p in sdir.iterdir() if p.is_dir() and (p / "seqinfo.ini").is_file()),
        key=lambda p: p.name,
    )
    if not available:
        raise FileNotFoundError(f"split 下无序列: {sdir}")
    names = [p.name for p in available]
    selected = names if seqs == ["all"] else seqs
    missing = [s for s in selected if s not in names]
    if missing:
        raise ValueError(f"序列不存在: {missing}（可选: {names}）")

    tracker_dir = out_dir / tracker_tag
    tracker_dir.mkdir(parents=True, exist_ok=True)
    revision = git_revision(project_root)
    summary: dict = {
        "tracker_tag": tracker_tag,
        "dataset": dataset,
        "split": split,
        "det_protocol": det_source,
        "det_cache_version": det_version if det_source == "private" else None,
        "filter_type": filter_type,
        "backend": backend,
        "stride": stride,
        "max_output_misses": max_output_misses,
        "matching_strategy": matching_strategy,
        "iou_threshold": iou_threshold,
        "max_misses": max_misses,
        "min_hits_to_show": min_hits_to_show,
        "min_conf": min_conf,
        "nsa_enabled": nsa_enabled,
        "nsa_interval": nsa_interval,
        "nsa_min": nsa_min,
        "nsa_conf_thresh": nsa_conf_thresh,
        "nsa_interval_sec": nsa_interval_sec,
        "ocm_enabled": ocm_enabled,
        "ocm_delta_t": ocm_delta_t,
        "ocm_weight": ocm_weight,
        "hmiou_enabled": hmiou_enabled,
        "oru_enabled": oru_enabled,
        "ocr_enabled": ocr_enabled,
        "ocr_threshold": ocr_threshold,
        "tcm_enabled": tcm_enabled,
        "tcm_weight": tcm_weight,
        "revision": revision,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sequences": {},
    }
    for name in selected:
        seq_dir = sdir / name
        info = read_seqinfo(seq_dir)
        det_path = resolve_det_file(
            dataset=dataset, root=root, split=split, seq=name,
            det_source=det_source, cache_root=cache_root, det_version=det_version,
        )
        dets_by_frame = load_mot_detections(det_path)
        tracker = make_tracker(
            filter_type=filter_type, backend=backend,
            max_output_misses=max_output_misses,
            matching_strategy=matching_strategy, iou_threshold=iou_threshold,
            max_misses=max_misses, min_hits_to_show=min_hits_to_show,
            nsa_enabled=nsa_enabled, nsa_interval=nsa_interval, nsa_min=nsa_min,
            nsa_conf_thresh=nsa_conf_thresh, nsa_interval_sec=nsa_interval_sec,
            ocm_enabled=ocm_enabled, ocm_delta_t=ocm_delta_t, ocm_weight=ocm_weight,
            hmiou_enabled=hmiou_enabled,
            oru_enabled=oru_enabled, ocr_enabled=ocr_enabled,
            ocr_threshold=ocr_threshold, tcm_enabled=tcm_enabled,
            tcm_weight=tcm_weight,
        )
        t0 = time.perf_counter()
        rows = run_sequence(tracker, dets_by_frame, info, stride=stride, min_conf=min_conf)
        elapsed = time.perf_counter() - t0
        out_path = tracker_dir / f"{name}.txt"
        write_mot_results(out_path, rows)
        fps = info.seq_length / elapsed if elapsed > 0 else 0.0
        print(f"[done] {name}: {info.seq_length} 帧, {len(rows)} 输出行, "
              f"{elapsed:.2f}s ({fps:.1f} fps) -> {out_path}")
        summary["sequences"][name] = {
            "frames": info.seq_length,
            "output_rows": len(rows),
            "det_frames": len(dets_by_frame),
            "elapsed_sec": round(elapsed, 3),
            "result": str(out_path),
        }
    summary_path = tracker_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")
    return summary


def build_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="离线跟踪 runner：det 缓存/公开检测 → DetectionTracker → TrackEval 兼容 MOT 结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=sorted(_DATASET_RELPATH))
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help=f"数据集根目录（默认 {DEFAULT_ROOT}）")
    p.add_argument("--split", required=True,
                   help="split（mot17/mot20: train|test；dancetrack: train|val|test）")
    p.add_argument("--seqs", default="all", help="all 或逗号分隔序列名（默认 all）")
    p.add_argument("--det-source", choices=DET_SOURCES, default="private",
                   help="检测协议：private=私有 YOLO 缓存；public-frcnn=MOT 官方公开检测（默认 private）")
    p.add_argument("--cache-root", type=Path, default=None,
                   help="私有检测缓存根（默认 <root>/_det_cache）")
    p.add_argument("--det-version", default=DEFAULT_DET_VERSION,
                   help=f"私有缓存版本（默认 {DEFAULT_DET_VERSION}）")
    p.add_argument("--filter", dest="filter_type", choices=FILTER_TYPES, default="ukf",
                   help="滤波器类型（默认 ukf）")
    p.add_argument("--backend", choices=BACKENDS, default="python",
                   help="跟踪后端（默认 python；rust 需先 maturin develop --release）")
    p.add_argument("--stride", type=int, default=1,
                   help="跳帧步长：每 N 帧一次 update，其余 predict_step（默认 1）")
    p.add_argument("--max-output-misses", type=int, default=0,
                   help="输出门控：misses 超过此值不输出（默认 0，同线上默认）")
    p.add_argument("--matching-strategy", choices=("greedy", "hungarian", "cascade"),
                   default="hungarian", help="匹配策略（默认 hungarian，同项目默认）")
    p.add_argument("--iou-threshold", type=float, default=0.25, help="IoU 门控（默认 0.25）")
    p.add_argument("--max-misses", type=int, default=4, help="内部生命周期 miss 上限（默认 4）")
    p.add_argument("--min-hits", type=int, default=1, help="输出所需最少命中（默认 1）")
    p.add_argument("--min-conf", type=float, default=0.0,
                   help="送入 tracker 的检测置信度下限（默认 0.0 不过滤）")
    # 跟踪增强（docs/tracker-nsa-ocm-hmiou.md，默认全部关闭）
    p.add_argument("--nsa", dest="nsa_enabled", action="store_true",
                   help="启用 NSA-KF（观测噪声随置信度缩放，默认关闭）")
    p.add_argument("--nsa-interval", type=float, default=1.0, help="NSA 缩放区间（默认 1.0）")
    p.add_argument("--nsa-min", type=float, default=0.1, help="NSA 高置信度分支乘子下界（默认 0.1）")
    p.add_argument("--nsa-conf-thresh", type=float, default=0.6,
                   help="NSA 分段阈值：conf 高于此值缩 R，否则放大 R（默认 0.6）")
    p.add_argument("--nsa-interval-sec", type=float, default=2.0,
                   help="NSA 低置信度分支 R 放大倍数（默认 2.0）")
    p.add_argument("--ocm", dest="ocm_enabled", action="store_true",
                   help="启用 OCM 方向代价（默认关闭）")
    p.add_argument("--ocm-delta-t", type=int, default=3, help="OCM 观测回溯帧数（默认 3）")
    p.add_argument("--ocm-weight", type=float, default=0.2, help="OCM 代价权重（默认 0.2）")
    p.add_argument("--hmiou", dest="hmiou_enabled", action="store_true",
                   help="启用 HMIoU（高度调制 IoU，默认关闭；门控更严格，对照需固定阈值）")
    # 批次 3（docs/tracker-oru-ocr-tcm.md，默认全部关闭）
    p.add_argument("--oru", dest="oru_enabled", action="store_true",
                   help="启用 ORU（丢失恢复时回滚重放，默认关闭）")
    p.add_argument("--ocr", dest="ocr_enabled", action="store_true",
                   help="启用 OCR（丢失态 track 用最后观测二次关联，默认关闭）")
    p.add_argument("--ocr-threshold", type=float, default=0.3,
                   help="OCR IoU 门控（默认 0.3）")
    p.add_argument("--tcm", dest="tcm_enabled", action="store_true",
                   help="启用 TCM（置信度一致性奖励，默认关闭）")
    p.add_argument("--tcm-weight", type=float, default=0.2,
                   help="TCM 权重（默认 0.2）")
    p.add_argument("--out", type=Path, default=None,
                   help="结果根目录（默认 benchmark/results/runs）")
    p.add_argument("--tracker-tag", default=None,
                   help="tracker 标签（默认按 filter/backend/stride/det 自动生成）")
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
    out_dir = args.out or (Path(__file__).resolve().parent / "results" / "runs")
    seqs = [s.strip() for s in args.seqs.split(",") if s.strip()] if args.seqs != "all" else ["all"]
    tag = args.tracker_tag or default_tracker_tag(
        args.filter_type, args.backend, args.stride, args.det_source,
    )
    try:
        run(
            root=args.root, dataset=args.dataset, split=args.split, seqs=seqs,
            det_source=args.det_source, cache_root=cache_root, det_version=args.det_version,
            filter_type=args.filter_type, backend=args.backend, stride=args.stride,
            max_output_misses=args.max_output_misses,
            matching_strategy=args.matching_strategy, iou_threshold=args.iou_threshold,
            max_misses=args.max_misses, min_hits_to_show=args.min_hits,
            min_conf=args.min_conf, out_dir=out_dir, tracker_tag=tag,
            project_root=Path(__file__).resolve().parent.parent,
            nsa_enabled=args.nsa_enabled, nsa_interval=args.nsa_interval,
            nsa_min=args.nsa_min, nsa_conf_thresh=args.nsa_conf_thresh,
            nsa_interval_sec=args.nsa_interval_sec, ocm_enabled=args.ocm_enabled,
            ocm_delta_t=args.ocm_delta_t, ocm_weight=args.ocm_weight,
            hmiou_enabled=args.hmiou_enabled,
            oru_enabled=args.oru_enabled, ocr_enabled=args.ocr_enabled,
            ocr_threshold=args.ocr_threshold, tcm_enabled=args.tcm_enabled,
            tcm_weight=args.tcm_weight,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
