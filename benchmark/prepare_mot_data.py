#!/usr/bin/env python3
r"""MOT 评测数据集准备与校验（MOT17 / MOT20 / DanceTrack）。

职责（docs/tracking-quality-benchmark.md §5.2）：
  - check               校验数据集目录结构，缺失时给出明确下载指引（不崩溃）；
  - list-seqs           解析 seqinfo.ini，输出序列清单（帧率/帧数/图像尺寸/图像目录）；
  - index-public-det    定位 MOT17/MOT20 公开检测 det/det.txt，输出统一访问层 JSON；
  - export-trackeval-gt 生成 TrackEval 所需 GT 布局（<out>/<seq>/gt/gt.txt + seqinfo.ini，
                        seqinfo.ini 缺失时按 img1 目录实际帧数生成）。

数据集根目录通过 --root 指定（默认 G:\基准测试，用户指定路径）：
  MOT17/MOT20:  <root>/MOTChallenge/<MOT17|MOT20>/<train|test>/<SEQ>/{seqinfo.ini,det/,gt/,img1/}
  DanceTrack:   <root>/DanceTrack/<train|val|test>/<SEQ>/{seqinfo.ini,gt/,img1/}

模块 import 无副作用；argparse 仅在 main() 中构建。
"""
from __future__ import annotations

import configparser
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

DEFAULT_ROOT = Path(r"G:\基准测试")

# ---------------------------------------------------------------------------
# 数据集布局约定
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "mot17": {
        "relpath": Path("MOTChallenge") / "MOT17",
        "splits": ("train", "test"),
        "gt_splits": ("train",),          # 本地有 GT 的 split
        "public_det": True,                # 序列自带 det/det.txt 公开检测
        "benchmark": "MOT17",
        "download": (
            "MOT17 官方压缩包（需 MOTChallenge 账号，免费注册）：\n"
            "  https://motchallenge.net/data/MOT17.zip\n"
            "下载后放入 <root>/_downloads/ 并解压，使目录为\n"
            "  <root>/MOTChallenge/MOT17/{train,test}/MOT17-XX-YYY/{seqinfo.ini,det,gt,img1}"
        ),
    },
    "mot20": {
        "relpath": Path("MOTChallenge") / "MOT20",
        "splits": ("train", "test"),
        "gt_splits": ("train",),
        "public_det": True,
        "benchmark": "MOT20",
        "download": (
            "MOT20 官方压缩包（需 MOTChallenge 账号，免费注册）：\n"
            "  https://motchallenge.net/data/MOT20.zip\n"
            "解压后使目录为 <root>/MOTChallenge/MOT20/{train,test}/MOT20-XX/{seqinfo.ini,det,gt,img1}"
        ),
    },
    "dancetrack": {
        "relpath": Path("DanceTrack"),
        "splits": ("train", "val", "test"),
        "gt_splits": ("train", "val"),     # test GT 不公开
        "public_det": False,
        "benchmark": "DanceTrack",
        "download": (
            "DanceTrack 官方发布（GitHub 页面含 Google Drive / Baidu 链接）：\n"
            "  https://github.com/DanceTrack/DanceTrack\n"
            "解压后使目录为 <root>/DanceTrack/{train,val,test}/dancetrackXXXX/{seqinfo.ini,gt,img1}"
        ),
    },
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeqInfo:
    """seqinfo.ini 解析结果（MOT 标准 [Sequence] 节）。"""
    name: str
    im_dir: str
    frame_rate: float
    seq_length: int
    im_width: int
    im_height: int
    im_ext: str


def read_seqinfo(seq_dir: Path) -> SeqInfo:
    """解析 <seq_dir>/seqinfo.ini；缺文件或缺字段抛 FileNotFoundError/ValueError。"""
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.is_file():
        raise FileNotFoundError(f"缺少 seqinfo.ini: {ini_path}")
    parser = configparser.ConfigParser()
    # seqinfo.ini 无编码声明，官方文件为 ASCII/UTF-8
    parser.read(ini_path, encoding="utf-8")
    if "Sequence" not in parser:
        raise ValueError(f"seqinfo.ini 缺少 [Sequence] 节: {ini_path}")
    sec = parser["Sequence"]

    def _req(key: str) -> str:
        val = sec.get(key)
        if val is None:
            raise ValueError(f"seqinfo.ini 缺少字段 {key}: {ini_path}")
        return val.strip()

    return SeqInfo(
        name=sec.get("name", seq_dir.name).strip(),
        im_dir=_req("imDir"),
        frame_rate=float(_req("frameRate")),
        seq_length=int(_req("seqLength")),
        im_width=int(_req("imWidth")),
        im_height=int(_req("imHeight")),
        im_ext=sec.get("imExt", ".jpg").strip(),
    )


def dataset_base(root: Path, dataset: str) -> Path:
    """数据集在 root 下的实际目录。"""
    return root / DATASETS[dataset]["relpath"]


def split_dir(root: Path, dataset: str, split: str) -> Path:
    return dataset_base(root, dataset) / split


def discover_sequences(root: Path, dataset: str, split: str) -> list[Path]:
    """列出 split 目录下所有含 seqinfo.ini 的序列目录（按名称排序）。"""
    sdir = split_dir(root, dataset, split)
    if not sdir.is_dir():
        return []
    return sorted(
        (p for p in sdir.iterdir() if p.is_dir() and (p / "seqinfo.ini").is_file()),
        key=lambda p: p.name,
    )


def locate_public_det(seq_dir: Path) -> Path | None:
    """定位公开检测文件 <seq>/det/det.txt（MOT17 各序列对应 FRCNN/DPM/SDP 之一）。"""
    det = seq_dir / "det" / "det.txt"
    return det if det.is_file() else None


# ---------------------------------------------------------------------------
# check：结构校验 + 下载指引
# ---------------------------------------------------------------------------

def check_dataset(root: Path, dataset: str) -> tuple[bool, list[str]]:
    """校验数据集结构。返回 (是否全部就绪, 报告行列表)。"""
    spec = DATASETS[dataset]
    lines: list[str] = []
    ok = True
    base = dataset_base(root, dataset)
    lines.append(f"[{dataset}] 根目录: {base}")
    if not base.is_dir():
        lines.append(f"  缺失: 数据集目录不存在 ({base})")
        lines.append("  下载指引:")
        lines.extend(f"    {ln}" for ln in spec["download"].splitlines())
        return False, lines

    for split in spec["splits"]:
        sdir = base / split
        has_gt = split in spec["gt_splits"]
        if not sdir.is_dir():
            # test split 缺失不视为阻塞（GT 不公开，仅里程碑用）
            level = "提示" if split == "test" else "缺失"
            lines.append(f"  [{split}] {level}: 目录不存在 ({sdir})")
            if split != "test":
                ok = False
            continue
        seqs = discover_sequences(root, dataset, split)
        if not seqs:
            lines.append(f"  [{split}] 缺失: 未找到任何含 seqinfo.ini 的序列目录 ({sdir})")
            ok = False
            continue
        n_gt = n_det = n_img = 0
        problems: list[str] = []
        for seq in seqs:
            try:
                info = read_seqinfo(seq)
            except (FileNotFoundError, ValueError) as exc:
                problems.append(f"{seq.name}: {exc}")
                continue
            if has_gt and not (seq / "gt" / "gt.txt").is_file():
                problems.append(f"{seq.name}: 缺少 gt/gt.txt")
            if spec["public_det"] and locate_public_det(seq) is None:
                problems.append(f"{seq.name}: 缺少 det/det.txt（公开检测）")
            if not (seq / info.im_dir).is_dir():
                n_img += 0
                problems.append(f"{seq.name}: 缺少图像目录 {info.im_dir}/")
            else:
                n_img += 1
            n_gt += 1 if (seq / "gt" / "gt.txt").is_file() else 0
            n_det += 1 if locate_public_det(seq) is not None else 0
        status = "OK" if not problems else "问题"
        lines.append(
            f"  [{split}] {status}: {len(seqs)} 序列, gt={n_gt}, det={n_det}, "
            f"img1 就绪={n_img}"
        )
        for p in problems[:20]:
            lines.append(f"      - {p}")
        if len(problems) > 20:
            lines.append(f"      ... 其余 {len(problems) - 20} 条省略")
        if problems:
            ok = False
    if not ok:
        lines.append("  下载指引:")
        lines.extend(f"    {ln}" for ln in spec["download"].splitlines())
    return ok, lines


# ---------------------------------------------------------------------------
# export-trackeval-gt：TrackEval GT 布局
# ---------------------------------------------------------------------------

def _generate_seqinfo(seq_dir: Path, out_path: Path) -> None:
    """按 img1 实际帧数生成 seqinfo.ini（DanceTrack 部分包缺 seqinfo 时的兜底）。"""
    img_dir = seq_dir / "img1"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"无法生成 seqinfo.ini：缺少图像目录 {img_dir}")
    exts = {p.suffix.lower() for p in img_dir.iterdir() if p.is_file()}
    im_ext = ".jpg" if ".jpg" in exts else (sorted(exts)[0] if exts else ".jpg")
    n_frames = sum(1 for p in img_dir.iterdir() if p.suffix.lower() == im_ext)
    if n_frames == 0:
        raise ValueError(f"图像目录为空: {img_dir}")
    # 尺寸无法从文件名得知时读首帧
    try:
        import cv2  # 延迟导入：仅生成 seqinfo 时需要
        first = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == im_ext)[0]
        img = cv2.imread(str(first))
        if img is None:
            raise ValueError(f"无法读取图像: {first}")
        h, w = img.shape[:2]
    except ImportError:
        raise RuntimeError("生成 seqinfo.ini 需要 opencv-python（pip install opencv-python）")
    out_path.write_text(
        "[Sequence]\n"
        f"name={seq_dir.name}\n"
        "imDir=img1\n"
        "frameRate=20\n"
        f"seqLength={n_frames}\n"
        f"imWidth={w}\n"
        f"imHeight={h}\n"
        f"imExt={im_ext}\n",
        encoding="utf-8",
    )


def export_trackeval_gt(
    root: Path,
    dataset: str,
    split: str,
    out_dir: Path,
    *,
    only_with_gt: bool = True,
) -> tuple[int, list[str]]:
    """把 split 下序列整理成 TrackEval MotChallenge2DBox 期望的 GT 布局：

        <out_dir>/<SEQ>/gt/gt.txt
        <out_dir>/<SEQ>/seqinfo.ini

    返回 (导出的序列数, 警告列表)。seqinfo.ini 缺失时按 img1 生成。
    """
    spec = DATASETS[dataset]
    if split not in spec["splits"]:
        raise ValueError(f"数据集 {dataset} 无 split={split}（可选: {spec['splits']}）")
    seqs = discover_sequences(root, dataset, split)
    if not seqs:
        raise FileNotFoundError(
            f"未找到序列: {split_dir(root, dataset, split)}\n"
            f"请先运行 check 子命令查看下载指引。"
        )
    warnings: list[str] = []
    n_export = 0
    for seq in seqs:
        gt_src = seq / "gt" / "gt.txt"
        if only_with_gt and not gt_src.is_file():
            warnings.append(f"{seq.name}: 无 gt/gt.txt，跳过")
            continue
        dst = out_dir / seq.name
        (dst / "gt").mkdir(parents=True, exist_ok=True)
        if gt_src.is_file():
            shutil.copy2(gt_src, dst / "gt" / "gt.txt")
        ini_src = seq / "seqinfo.ini"
        if ini_src.is_file():
            shutil.copy2(ini_src, dst / "seqinfo.ini")
        else:
            _generate_seqinfo(seq, dst / "seqinfo.ini")
            warnings.append(f"{seq.name}: seqinfo.ini 缺失，已按 img1 生成")
        n_export += 1
    return n_export, warnings


# ---------------------------------------------------------------------------
# index-public-det：公开检测统一访问层
# ---------------------------------------------------------------------------

def build_public_det_index(root: Path, dataset: str, split: str) -> dict:
    """输出 {seq: {det, split, seqinfo}} 的 JSON 可序列化索引。"""
    if not DATASETS[dataset]["public_det"]:
        raise ValueError(f"数据集 {dataset} 不提供公开检测（仅 MOT17/MOT20）")
    index: dict[str, dict] = {}
    for seq in discover_sequences(root, dataset, split):
        det = locate_public_det(seq)
        if det is None:
            continue
        info = read_seqinfo(seq)
        index[seq.name] = {
            "det": str(det),
            "split": split,
            "frames": info.seq_length,
            "fps": info.frame_rate,
            "size": [info.im_width, info.im_height],
        }
    return {
        "dataset": dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "sequences": index,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(p) -> None:
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS),
                   help="数据集（mot17 / mot20 / dancetrack）")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help=f"数据集根目录（默认 {DEFAULT_ROOT}）")


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="MOT 评测数据集准备与校验（MOT17 / MOT20 / DanceTrack）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="校验数据集目录结构，缺失时给出下载指引")
    _add_common(p_check)

    p_list = sub.add_parser("list-seqs", help="输出序列清单（seqinfo.ini 解析结果）")
    _add_common(p_list)
    p_list.add_argument("--split", default=None,
                        help="只列某个 split（默认列出全部本地 split）")

    p_idx = sub.add_parser("index-public-det",
                           help="定位公开检测 det/det.txt，输出统一访问层 JSON")
    _add_common(p_idx)
    p_idx.add_argument("--split", default="train", help="split（默认 train）")
    p_idx.add_argument("--out", type=Path, default=None,
                       help="输出 JSON 路径（默认 <root>/_det_cache/<dataset>_public_det_index.json）")

    p_exp = sub.add_parser("export-trackeval-gt",
                           help="生成 TrackEval 所需 GT 布局（gt/gt.txt + seqinfo.ini）")
    _add_common(p_exp)
    p_exp.add_argument("--split", required=True, help="split（如 train / val）")
    p_exp.add_argument("--out", type=Path, required=True, help="输出 GT 根目录")
    p_exp.add_argument("--include-without-gt", action="store_true",
                       help="无 GT 的序列也导出 seqinfo.ini（默认跳过）")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    if sys.platform == "win32":
        # Windows 中文系统默认 GBK 控制台：避免输出中文乱码
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    root: Path = args.root

    if args.command == "check":
        ok, lines = check_dataset(root, args.dataset)
        for ln in lines:
            print(ln)
        print("结论:", "就绪" if ok else "未就绪（见上方缺失项与下载指引）")
        return 0 if ok else 1

    if args.command == "list-seqs":
        splits = [args.split] if args.split else list(DATASETS[args.dataset]["splits"])
        for split in splits:
            seqs = discover_sequences(root, args.dataset, split)
            if not seqs:
                print(f"[{split}] 无序列（目录不存在或为空）")
                continue
            print(f"[{split}] {len(seqs)} 序列:")
            print(f"  {'name':<28} {'frames':>7} {'fps':>6} {'size':>13}  imDir")
            for seq in seqs:
                try:
                    info = read_seqinfo(seq)
                except (FileNotFoundError, ValueError) as exc:
                    print(f"  {seq.name:<28} <错误: {exc}>")
                    continue
                print(f"  {info.name:<28} {info.seq_length:>7} {info.frame_rate:>6.1f} "
                      f"{info.im_width:>5}x{info.im_height:<6}  {info.im_dir}")
        return 0

    if args.command == "index-public-det":
        out = args.out or (root / "_det_cache" / f"{args.dataset}_public_det_index.json")
        index = build_public_det_index(root, args.dataset, args.split)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {len(index['sequences'])} 条公开检测索引 -> {out}")
        return 0

    if args.command == "export-trackeval-gt":
        n, warnings = export_trackeval_gt(
            root, args.dataset, args.split, args.out,
            only_with_gt=not args.include_without_gt,
        )
        print(f"已导出 {n} 个序列的 TrackEval GT 布局 -> {args.out}")
        for w in warnings:
            print(f"  警告: {w}")
        return 0

    return 2  # pragma: no cover - argparse required=True 已拦截


if __name__ == "__main__":
    sys.exit(main())
