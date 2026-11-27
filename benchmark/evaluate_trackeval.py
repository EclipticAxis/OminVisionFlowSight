#!/usr/bin/env python3
r"""TrackEval 评测：HOTA(含 AssA/DetA) / IDF1 / MOTA / IDSW，总表 + 每序列明细。

职责（docs/tracking-quality-benchmark.md §4/§5.2）：
  - 调用外部依赖 TrackEval（pip install git+https://github.com/JonathonLuiten/TrackEval.git，
    或 --trackeval-dir 指向本地 clone）；import 失败给出安装指引而非崩溃；
  - 输入 run_tracking_offline.py 的输出目录（<results_dir>/<tracker-tag>/<seq>.txt），
    自动读取其中的 summary.json 汇入协议元信息（det 协议/缓存版本/stride/filter/backend/revision），
    也可由命令行覆盖；
  - 输出 benchmark/results/<date>_<tag>.json（机读）+ 同名 .md（人读摘要）。

TrackEval 数据布局要求（MotChallenge2DBox）：
  GT_FOLDER 下每个序列一个目录，含 gt/gt.txt 与 seqinfo.ini ——
  MOT17/MOT20 train、DanceTrack val 的原始 split 目录即满足；
  也可用 prepare_mot_data.py export-trackeval-gt 生成的目录（--gt-dir 指定）。

模块 import 无副作用（trackeval 延迟导入）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

DEFAULT_ROOT = Path(r"G:\基准测试")

_DATASET_RELPATH = {
    "mot17": Path("MOTChallenge") / "MOT17",
    "mot20": Path("MOTChallenge") / "MOT20",
    "dancetrack": Path("DanceTrack"),
}
_BENCHMARK_NAME = {"mot17": "MOT17", "mot20": "MOT20", "dancetrack": "DanceTrack"}
_SPLITS = {
    "mot17": ("train", "test"),
    "mot20": ("train", "test"),
    "dancetrack": ("train", "val", "test"),
}

_INSTALL_HINT = (
    "TrackEval 未安装。安装方式（二选一）：\n"
    "  1) pip install git+https://github.com/JonathonLuiten/TrackEval.git\n"
    "  2) git clone https://github.com/JonathonLuiten/TrackEval.git <目录>，"
    "然后以 --trackeval-dir <目录> 传入"
)


def _patch_numpy_compat() -> None:
    """NumPy 2.x 兼容 shim：TrackEval（含上游 master）仍使用 np.float/np.int 等
    NumPy 1.x 别名，在 NumPy>=1.24 环境直接 AttributeError。此处仅在本离线评测
    进程内补回别名（等价 builtin 类型，行为安全），不修改 trackeval 包本身。
    """
    import numpy as np

    # 用 __dict__ 而非 hasattr：避免触发 NumPy 2.x __getattr__ 的 FutureWarning
    for name, pytype in (("float", float), ("int", int), ("bool", bool), ("object", object)):
        if name not in np.__dict__:
            setattr(np, name, pytype)


def import_trackeval(trackeval_dir: Path | None):
    """导入 trackeval；失败时抛 RuntimeError 并附安装指引。"""
    _patch_numpy_compat()
    if trackeval_dir is not None:
        if not (trackeval_dir / "trackeval").is_dir():
            raise RuntimeError(
                f"--trackeval-dir 下未找到 trackeval 包: {trackeval_dir}\n{_INSTALL_HINT}"
            )
        sys.path.insert(0, str(trackeval_dir))
    try:
        import trackeval  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"无法导入 trackeval（{exc}）\n{_INSTALL_HINT}") from exc
    return sys.modules["trackeval"]


def default_gt_dir(root: Path, dataset: str, split: str) -> Path:
    """默认 GT 根：原始 split 目录（MOTChallenge/DanceTrack 布局本身满足 TrackEval）。"""
    return root / _DATASET_RELPATH[dataset] / split


def gt_sequences(gt_dir: Path) -> list[str]:
    """GT 权威序列集合：含 gt/gt.txt 的序列子目录（排序）。"""
    return sorted(
        p.name for p in gt_dir.iterdir()
        if p.is_dir() and p.name != "seqmaps" and (p / "gt" / "gt.txt").is_file()
    )


def read_seqmap(seqmap: Path) -> list[str]:
    """读取 seqmap（首行 name 表头，其后每行一个序列名）。"""
    lines = seqmap.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines[1:] if ln.strip()]


def ensure_seqmap(
    gt_dir: Path,
    benchmark: str,
    split: str,
    expected: list[str],
    *,
    allow_rewrite: bool = False,
) -> tuple[Path, bool]:
    """保证 TrackEval 需要的 seqmap 存在且与期望评测序列集合一致。

    - 缺失：按期望集合生成（rewritten=False）；
    - 已存在且一致：直接使用（rewritten=False）；
    - 已存在但不一致：默认**报错（不覆写）**；仅 allow_rewrite=True 时覆写
      并返回 rewritten=True（报告 meta 记 seqmap_rewritten）。
    返回 (seqmap 路径, 是否覆写)。
    """
    seqmap = gt_dir / "seqmaps" / f"{benchmark}-{split}.txt"
    if seqmap.is_file():
        existing = read_seqmap(seqmap)
        if sorted(existing) == sorted(expected):
            return seqmap, False
        if not allow_rewrite:
            raise RuntimeError(
                f"seqmap 与期望评测序列不一致: {seqmap}\n"
                f"  seqmap 内容({len(existing)}): {sorted(existing)}\n"
                f"  期望集合({len(expected)}): {sorted(expected)}\n"
                f"  处理：用 --seqs 对齐期望集合，或人工修正 seqmap；"
                f"确需覆写请显式加 --allow-seqmap-rewrite。"
            )
        seqmap.write_text("name\n" + "\n".join(expected) + "\n", encoding="utf-8")
        print(f"[info] 已按 --allow-seqmap-rewrite 覆写 seqmap: {seqmap}（{len(expected)} 序列）")
        return seqmap, True
    if not expected:
        raise RuntimeError(f"期望评测序列为空，无法生成 seqmap: {gt_dir}")
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(expected) + "\n", encoding="utf-8")
    print(f"[info] 已生成 seqmap: {seqmap}（{len(expected)} 序列）")
    return seqmap, False


def trackeval_version() -> str:
    """TrackEval 版本（importlib.metadata；本地 clone 等非 dist 安装回退 unknown）。"""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - py3.11 恒有 importlib.metadata
        return "unknown"
    for dist in ("TrackEval", "trackeval"):
        try:
            return version(dist)
        except PackageNotFoundError:
            continue
    return "unknown"


def run_trackeval(
    *,
    trackeval,
    benchmark: str,
    gt_dir: Path,
    trackers_dir: Path,
    tracker_tags: list[str],
    split: str,
    output_folder: Path,
) -> dict:
    """执行评测，返回 TrackEval output_res（{dataset_name: {tracker: {...}}}）。"""
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["PRINT_CONFIG"] = False
    eval_config["PRINT_RESULTS"] = False
    eval_config["SHOW_PROGRESS"] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = str(gt_dir)
    dataset_config["TRACKERS_FOLDER"] = str(trackers_dir)
    dataset_config["OUTPUT_FOLDER"] = str(output_folder)
    dataset_config["BENCHMARK"] = benchmark
    dataset_config["SPLIT_TO_EVAL"] = split
    dataset_config["SKIP_SPLIT_FOL"] = True
    dataset_config["CLASSES_TO_EVAL"] = ["pedestrian"]
    dataset_config["TRACKERS_TO_EVAL"] = tracker_tags
    # run_tracking_offline.py 输出布局为 <tracker>/<seq>.txt（TrackEval 默认
    # TRACKER_SUB_FOLDER=data 即 <tracker>/data/<seq>.txt），置空对齐我们的布局
    dataset_config["TRACKER_SUB_FOLDER"] = ""
    dataset_config["PRINT_CONFIG"] = False

    def _quiet(metric_cls):
        # Identity/CLEAR 默认 PRINT_CONFIG=True，静默之；HOTA 无 config 直接构造
        get_cfg = getattr(metric_cls, "get_default_config", None)
        if get_cfg is None:
            return metric_cls()
        cfg = get_cfg()
        cfg["PRINT_CONFIG"] = False
        return metric_cls(cfg)

    metrics = [_quiet(trackeval.metrics.HOTA),
               _quiet(trackeval.metrics.Identity),
               _quiet(trackeval.metrics.CLEAR)]
    evaluator = trackeval.Evaluator(eval_config)
    result = evaluator.evaluate(
        [trackeval.datasets.MotChallenge2DBox(dataset_config)],
        metrics,
        [dataset_config],
    )
    # 不同 TrackEval 版本返回值个数不同：(res, msg, csv) 或 (res, msg)
    return result[0]


def _f(value) -> float | None:
    """TrackEval 指标值 → float（None/NaN 安全；HOTA 系字段为 19 个 alpha
    阈值的数组，按 TrackEval 官方口径取均值作为头条数字）。"""
    if value is None:
        return None
    mean = getattr(value, "mean", None)
    try:
        v = float(mean()) if callable(mean) else float(value)
    except (TypeError, ValueError):
        return None
    return None if v != v else v  # NaN 检查


def extract_metrics(tracker_res: dict, class_name: str = "pedestrian") -> dict:
    """从单 tracker 的 TrackEval 结果提取报告字段。

    tracker_res 结构（MotChallenge2DBox + HOTA/Identity/CLEAR）：
        {seq_name: {class_name: {metric: {field: value}}},
         "COMBINED_SEQ": {class_name: {...}}}
    """
    def one(res: dict) -> dict:
        cls_res = res.get(class_name, res)  # 兼容无类别嵌套的结构
        hota = cls_res.get("HOTA", {})
        ident = cls_res.get("Identity", {})
        clear = cls_res.get("CLEAR", {})
        return {
            "HOTA": _f(hota.get("HOTA")),
            "AssA": _f(hota.get("AssA")),
            "DetA": _f(hota.get("DetA")),
            "IDF1": _f(ident.get("IDF1")),
            "MOTA": _f(clear.get("MOTA")),
            "IDSW": _f(clear.get("IDSW")),
        }

    per_seq = {k: one(v) for k, v in tracker_res.items() if k != "COMBINED_SEQ"}
    combined = one(tracker_res["COMBINED_SEQ"]) if "COMBINED_SEQ" in tracker_res else None
    return {"combined": combined, "per_sequence": per_seq}


def git_revision(cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load_tracker_summary(trackers_dir: Path, tag: str) -> dict:
    """读取 run_tracking_offline.py 写入的 summary.json（协议元信息），缺失返回 {}。"""
    path = trackers_dir / tag / "summary.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def render_markdown(report: dict) -> str:
    """人读摘要：元信息头 + 总表 + 每序列明细。"""
    meta = report["meta"]
    lines = [
        f"# MOT 评测报告 — {meta['tag']}",
        "",
        f"- 日期: {meta['date']}",
        f"- 数据集/split: {meta['dataset']} / {meta['split']}",
        f"- 检测协议: {meta.get('det_protocol', '?')}"
        + (f"（缓存版本 {meta['det_cache_version']}）" if meta.get("det_cache_version") else ""),
        f"- stride: {meta.get('stride', '?')}  滤波器: {meta.get('filter_type', '?')}"
        f"  后端: {meta.get('backend', '?')}",
        f"- 代码 revision: {meta.get('revision', '?')}",
        f"- TrackEval: {meta.get('trackeval_version', '?')}  GT: {meta.get('gt_dir', '?')}",
        "",
        "## 总表（多序列合并）",
        "",
        "| tracker | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag, res in report["trackers"].items():
        c = res.get("combined") or {}
        def fmt(key):
            v = c.get(key)
            return f"{v:.4f}" if isinstance(v, float) and key != "IDSW" else (
                f"{v:.0f}" if isinstance(v, float) else "-"
            )
        lines.append(
            f"| {tag} | {fmt('HOTA')} | {fmt('AssA')} | {fmt('DetA')} "
            f"| {fmt('IDF1')} | {fmt('MOTA')} | {fmt('IDSW')} |"
        )
    for tag, res in report["trackers"].items():
        lines += ["", f"## 每序列明细 — {tag}", "",
                  "| sequence | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |",
                  "|---|---|---|---|---|---|---|"]
        for seq, m in sorted(res["per_sequence"].items()):
            def fmt2(key):
                v = m.get(key)
                return f"{v:.4f}" if isinstance(v, float) and key != "IDSW" else (
                    f"{v:.0f}" if isinstance(v, float) else "-"
                )
            lines.append(
                f"| {seq} | {fmt2('HOTA')} | {fmt2('AssA')} | {fmt2('DetA')} "
                f"| {fmt2('IDF1')} | {fmt2('MOTA')} | {fmt2('IDSW')} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="TrackEval 评测：HOTA(AssA/DetA)/IDF1/MOTA/IDSW 汇总报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=sorted(_DATASET_RELPATH))
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help=f"数据集根目录（默认 {DEFAULT_ROOT}）")
    p.add_argument("--split", required=True,
                   help="split（mot17/mot20: train；dancetrack: val）")
    p.add_argument("--results-dir", type=Path, required=True,
                   help="run_tracking_offline.py 输出根目录（其下每个子目录是一个 tracker-tag）")
    p.add_argument("--trackers", default=None,
                   help="只评指定 tracker-tag（逗号分隔；默认评 results-dir 下全部）")
    p.add_argument("--seqs", default="all",
                   help="评测序列范围：all=GT 权威全量（默认），或逗号分隔序列名")
    p.add_argument("--allow-seqmap-rewrite", action="store_true",
                   help="允许覆写与期望集合不一致的既有 seqmap（默认报错退出）")
    p.add_argument("--gt-dir", type=Path, default=None,
                   help="GT 根目录（默认 <root>/<数据集布局>/<split>）")
    p.add_argument("--trackeval-dir", type=Path, default=None,
                   help="本地 TrackEval clone 路径（未 pip 安装时使用）")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parent / "results",
                   help="报告输出目录（默认 benchmark/results）")
    p.add_argument("--tag", required=True,
                   help="报告标签，用于 <date>_<tag>.json/.md 文件名")
    # 协议元信息覆盖项（默认从各 tracker 的 summary.json 读取）
    p.add_argument("--det-protocol", default=None, help="覆盖元信息 det_protocol")
    p.add_argument("--stride", type=int, default=None, help="覆盖元信息 stride")
    p.add_argument("--filter", dest="filter_type", default=None, help="覆盖元信息 filter_type")
    p.add_argument("--backend", default=None, help="覆盖元信息 backend")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    if sys.platform == "win32":
        # Windows 中文系统默认 GBK 控制台：避免输出中文乱码
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if args.split not in _SPLITS[args.dataset]:
        print(f"错误: 数据集 {args.dataset} 无 split={args.split}"
              f"（可选: {_SPLITS[args.dataset]}）", file=sys.stderr)
        return 2

    trackers_dir = args.results_dir
    if not trackers_dir.is_dir():
        print(f"错误: 结果目录不存在: {trackers_dir}", file=sys.stderr)
        return 1
    if args.trackers:
        tags = [t.strip() for t in args.trackers.split(",") if t.strip()]
    else:
        tags = sorted(p.name for p in trackers_dir.iterdir()
                      if p.is_dir() and any(p.glob("*.txt")))
    if not tags:
        print(f"错误: {trackers_dir} 下没有可评测的 tracker 结果目录", file=sys.stderr)
        return 1
    missing = [t for t in tags if not (trackers_dir / t).is_dir()]
    if missing:
        print(f"错误: tracker 结果目录不存在: {missing}", file=sys.stderr)
        return 1

    gt_dir = args.gt_dir or default_gt_dir(args.root, args.dataset, args.split)
    if not gt_dir.is_dir():
        print(f"错误: GT 目录不存在: {gt_dir}\n"
              f"可运行 prepare_mot_data.py check 校验，或用 export-trackeval-gt 生成后 --gt-dir 指定。",
              file=sys.stderr)
        return 1

    # --- 评测口径护栏（P0-3）：期望序列 = GT 权威全量（∩ --seqs 若指定）---
    gt_seqs = gt_sequences(gt_dir)
    if not gt_seqs:
        print(f"错误: GT 目录下没有含 gt/gt.txt 的序列: {gt_dir}", file=sys.stderr)
        return 1
    if args.seqs != "all":
        requested = [s.strip() for s in args.seqs.split(",") if s.strip()]
        unknown = [s for s in requested if s not in gt_seqs]
        if unknown:
            print(f"错误: --seqs 中序列不在 GT 中: {unknown}（GT 序列: {gt_seqs}）",
                  file=sys.stderr)
            return 1
        expected = [s for s in gt_seqs if s in requested]
    else:
        expected = gt_seqs

    # 覆盖检查：任何被评 tracker 缺期望序列结果 → fail loudly（不再静默取交集收窄）
    coverage_errors: list[tuple[str, list[str]]] = []
    for t in tags:
        have = {p.stem for p in (trackers_dir / t).glob("*.txt")}
        missing_seqs = [s for s in expected if s not in have]
        if missing_seqs:
            coverage_errors.append((t, missing_seqs))
    if coverage_errors:
        print("错误: tracker 结果未覆盖期望评测序列（fail loudly，不静默收窄）:",
              file=sys.stderr)
        for t, ms in coverage_errors:
            print(f"  {t}: 缺失 {len(ms)} 个序列: {ms}", file=sys.stderr)
        print("处理: 补齐缺失序列的跟踪结果，或用 --seqs 显式限定评测范围。",
              file=sys.stderr)
        return 1

    # seqmap 护栏：缺失才生成；不一致默认报错（--allow-seqmap-rewrite 才覆写）
    try:
        _seqmap_path, seqmap_rewritten = ensure_seqmap(
            gt_dir, _BENCHMARK_NAME[args.dataset], args.split, expected,
            allow_rewrite=args.allow_seqmap_rewrite,
        )
    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    # 全部本地校验通过后再导入 trackeval（重依赖 + NumPy 兼容 shim）
    try:
        trackeval = import_trackeval(args.trackeval_dir)
    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    out_dir: Path = args.out_dir
    te_out = out_dir / "_trackeval_workspace"
    te_out.mkdir(parents=True, exist_ok=True)
    try:
        output_res = run_trackeval(
            trackeval=trackeval,
            benchmark=_BENCHMARK_NAME[args.dataset],
            gt_dir=gt_dir, trackers_dir=trackers_dir, tracker_tags=tags,
            split=args.split, output_folder=te_out,
        )
    except Exception as exc:  # TrackEval 内部异常类型多样，统一转为可读错误
        print(f"错误: TrackEval 评测失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # 元信息：以第一个 tracker 的 summary.json 为基础，命令行覆盖优先
    base_summary = load_tracker_summary(trackers_dir, tags[0])
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    meta = {
        "tag": args.tag,
        "date": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "gt_dir": str(gt_dir),
        "results_dir": str(trackers_dir),
        "det_protocol": args.det_protocol or base_summary.get("det_protocol"),
        "det_cache_version": base_summary.get("det_cache_version"),
        "stride": args.stride if args.stride is not None else base_summary.get("stride"),
        "filter_type": args.filter_type or base_summary.get("filter_type"),
        "backend": args.backend or base_summary.get("backend"),
        "revision": base_summary.get("revision") or git_revision(
            Path(__file__).resolve().parent.parent),
        "trackeval_version": trackeval_version(),
        "evaluated_sequences": sorted(expected),
        "seqmap_rewritten": seqmap_rewritten,
    }
    report = {
        "meta": meta,
        "trackers": {
            tag: extract_metrics(output_res[next(iter(output_res))][tag])
            for tag in tags
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{date_str}_{args.tag}.json"
    md_path = out_dir / f"{date_str}_{args.tag}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"报告 -> {json_path}")
    print(f"      {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
