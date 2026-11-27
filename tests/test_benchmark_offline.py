#!/usr/bin/env python3
"""MOT 离线评测基建单元测试（合成小序列，不依赖真实数据集与 TrackEval 安装）。

覆盖：
  - prepare_mot_data: seqinfo 解析、结构校验（缺失→下载指引）、TrackEval GT 导出、公开检测索引；
  - run_detection_cache: 缓存路径/写盘格式（不触发 ultralytics）；
  - run_tracking_offline: det 解析（7 列私有 / 10 列公开）、坐标换算、stride 跳帧行为、
    MOT 结果输出格式、端到端 run() 与 summary.json；
  - evaluate_trackeval: 指标提取与 Markdown 渲染（纯函数，不依赖 trackeval 包）。

运行: venv/Scripts/python.exe -m pytest tests/test_benchmark_offline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import prepare_mot_data as prep
from benchmark import run_detection_cache as rdc
from benchmark import run_tracking_offline as rto
from benchmark import evaluate_trackeval as ev


# ---------------------------------------------------------------------------
# 合成数据集构造
# ---------------------------------------------------------------------------

SEQ_NAME = "SYNTH-01"
IM_W, IM_H, N_FRAMES = 640, 480, 6


def make_seq_dir(base: Path, *, with_gt: bool = True, with_det: bool = True,
                 with_seqinfo: bool = True) -> Path:
    """构造一个 MOT 布局序列目录（img1 用空目录占位，check 只验存在性）。"""
    seq = base / SEQ_NAME
    seq.mkdir(parents=True)
    if with_seqinfo:
        (seq / "seqinfo.ini").write_text(
            "[Sequence]\n"
            f"name={SEQ_NAME}\n"
            "imDir=img1\n"
            "frameRate=30\n"
            f"seqLength={N_FRAMES}\n"
            f"imWidth={IM_W}\n"
            f"imHeight={IM_H}\n"
            "imExt=.jpg\n",
            encoding="utf-8",
        )
    (seq / "img1").mkdir()
    if with_gt:
        (seq / "gt").mkdir()
        (seq / "gt" / "gt.txt").write_text(
            "1,1,10,20,30,40,1,1,1\n2,1,12,20,30,40,1,1,1\n", encoding="utf-8",
        )
    if with_det:
        (seq / "det").mkdir()
        # 10 列公开检测格式
        (seq / "det" / "det.txt").write_text(
            "1,-1,10.0,20.0,30.0,40.0,0.9000,-1,-1,-1\n"
            "2,-1,12.0,20.0,30.0,40.0,0.8500,-1,-1,-1\n",
            encoding="utf-8",
        )
    return seq


@pytest.fixture()
def mot17_root(tmp_path: Path) -> Path:
    """合成 MOT17 根目录（train 含 1 序列）。"""
    root = tmp_path / "root"
    make_seq_dir(root / "MOTChallenge" / "MOT17" / "train")
    return root


def write_private_det(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{f},-1,{l:.2f},{t:.2f},{w:.2f},{h:.2f},{c:.4f}" for f, l, t, w, h, c in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# prepare_mot_data
# ---------------------------------------------------------------------------

def test_read_seqinfo(tmp_path: Path):
    seq = make_seq_dir(tmp_path)
    info = prep.read_seqinfo(seq)
    assert info.name == SEQ_NAME
    assert info.seq_length == N_FRAMES
    assert (info.im_width, info.im_height) == (IM_W, IM_H)
    assert info.frame_rate == 30.0
    assert info.im_dir == "img1"


def test_read_seqinfo_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="seqinfo.ini"):
        prep.read_seqinfo(tmp_path / "nope")


def test_check_dataset_missing_gives_guidance(tmp_path: Path, capsys):
    ok, lines = prep.check_dataset(tmp_path, "mot17")
    assert not ok
    text = "\n".join(lines)
    assert "motchallenge.net" in text  # 明确下载指引而非崩溃
    assert "MOT17.zip" in text


def test_check_dataset_ok(mot17_root: Path):
    ok, lines = prep.check_dataset(mot17_root, "mot17")
    # test split 缺失只算提示，train 完整即就绪
    assert ok, "\n".join(lines)


def test_check_dataset_missing_gt(mot17_root: Path):
    gt = mot17_root / "MOTChallenge" / "MOT17" / "train" / SEQ_NAME / "gt" / "gt.txt"
    gt.unlink()
    ok, lines = prep.check_dataset(mot17_root, "mot17")
    assert not ok
    assert any("gt.txt" in ln for ln in lines)


def test_export_trackeval_gt(mot17_root: Path, tmp_path: Path):
    out = tmp_path / "gt_out"
    n, warnings = prep.export_trackeval_gt(mot17_root, "mot17", "train", out)
    assert n == 1 and not warnings
    assert (out / SEQ_NAME / "gt" / "gt.txt").is_file()
    assert (out / SEQ_NAME / "seqinfo.ini").is_file()


def test_build_public_det_index(mot17_root: Path):
    index = prep.build_public_det_index(mot17_root, "mot17", "train")
    assert SEQ_NAME in index["sequences"]
    entry = index["sequences"][SEQ_NAME]
    assert entry["det"].endswith("det.txt")
    assert entry["frames"] == N_FRAMES


def test_public_det_index_rejects_dancetrack(mot17_root: Path):
    with pytest.raises(ValueError, match="公开检测"):
        prep.build_public_det_index(mot17_root, "dancetrack", "val")


# ---------------------------------------------------------------------------
# run_detection_cache（不触发 ultralytics）
# ---------------------------------------------------------------------------

def test_cache_paths(tmp_path: Path):
    det, meta = rdc.cache_paths(tmp_path, "mot17", SEQ_NAME, "det_yolov8n_v1")
    assert det == tmp_path / "mot17" / SEQ_NAME / "det_yolov8n_v1.txt"
    assert meta.name == "det_yolov8n_v1.meta.json"


def test_write_det_cache_format(tmp_path: Path):
    out = tmp_path / "det.txt"
    rdc.write_det_cache([(1, 10.0, 20.0, 30.0, 40.0, 0.9)], out)
    line = out.read_text(encoding="utf-8").strip()
    parts = line.split(",")
    assert len(parts) == 7
    assert parts[0] == "1" and parts[1] == "-1"  # 私有检测约定 id=-1
    assert float(parts[6]) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# run_tracking_offline: 解析与坐标换算
# ---------------------------------------------------------------------------

def test_load_mot_detections_private_7col(tmp_path: Path):
    det = tmp_path / "det.txt"
    write_private_det(det, [(1, 10, 20, 30, 40, 0.9), (1, 50, 60, 30, 40, 0.8),
                            (3, 12, 20, 30, 40, 0.7)])
    by_frame = rto.load_mot_detections(det)
    assert set(by_frame) == {1, 3}
    assert len(by_frame[1]) == 2
    r = by_frame[1][0]
    assert (r.left, r.top, r.width, r.height) == (10, 20, 30, 40)


def test_load_mot_detections_public_10col(tmp_path: Path):
    det = tmp_path / "det.txt"
    det.write_text("1,-1,10,20,30,40,0.9,-1,-1,-1\n", encoding="utf-8")
    by_frame = rto.load_mot_detections(det)
    assert by_frame[1][0].conf == pytest.approx(0.9)


def test_load_mot_detections_skips_degenerate(tmp_path: Path):
    det = tmp_path / "det.txt"
    det.write_text("1,-1,10,20,0,40,0.9\n1,-1,10,20,30,40,0.8\n", encoding="utf-8")
    by_frame = rto.load_mot_detections(det)
    assert len(by_frame[1]) == 1  # 零宽退化框被跳过


def test_load_mot_detections_bad_line(tmp_path: Path):
    det = tmp_path / "det.txt"
    det.write_text("1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="7 列"):
        rto.load_mot_detections(det)


def test_coordinate_roundtrip():
    """像素→归一化（tracker 输入）→像素（结果输出）换算闭环。"""
    row = rto.DetRow(frame=1, left=64.0, top=48.0, width=64.0, height=96.0, conf=0.9)
    det = rto.det_to_tracker_input(row, IM_W, IM_H)
    assert det["x1"] == pytest.approx(0.1)
    assert det["y1"] == pytest.approx(0.1)
    assert det["x2"] == pytest.approx(0.2)
    assert det["y2"] == pytest.approx(0.3)
    assert det["label"] == "person"
    track = {"track_id": 7, "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.3, "confidence": 0.9}
    out = rto.track_to_result_row(5, track, IM_W, IM_H)
    assert (out.left, out.top) == (pytest.approx(64.0), pytest.approx(48.0))
    assert (out.width, out.height) == (pytest.approx(64.0), pytest.approx(96.0))


def test_track_to_result_row_clips():
    track = {"track_id": 1, "x1": -0.05, "y1": 0.9, "x2": 1.05, "y2": 1.1, "confidence": 0.5}
    out = rto.track_to_result_row(1, track, IM_W, IM_H)
    assert out.left == 0.0 and out.top == pytest.approx(0.9 * IM_H)
    assert out.left + out.width <= IM_W
    assert out.top + out.height <= IM_H


# ---------------------------------------------------------------------------
# run_tracking_offline: stride 跳帧行为
# ---------------------------------------------------------------------------

class _FakeTracker:
    """记录 update/predict_step 调用序列的假 tracker。"""

    def __init__(self):
        self.calls: list[str] = []

    def update(self, detections):
        self.calls.append("update")
        return [{"track_id": 1, "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2,
                 "confidence": 0.9, "label": "person"}]

    def predict_step(self):
        self.calls.append("predict_step")
        return [{"track_id": 1, "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2,
                 "confidence": 0.9, "label": "person", "predicted_only": True}]

    def reset(self):
        pass


def _seq_info() -> rto.SeqInfo:
    return rto.SeqInfo(name=SEQ_NAME, frame_rate=30.0, seq_length=N_FRAMES,
                       im_width=IM_W, im_height=IM_H)


def test_stride_call_pattern():
    tracker = _FakeTracker()
    rows = rto.run_sequence(tracker, {}, _seq_info(), stride=2)
    assert tracker.calls == ["update", "predict_step"] * (N_FRAMES // 2)
    # 每个输出帧都有行（跳帧预测也输出，与线上一致）
    assert sorted({r.frame for r in rows}) == list(range(1, N_FRAMES + 1))


def test_stride1_all_updates():
    tracker = _FakeTracker()
    rto.run_sequence(tracker, {}, _seq_info(), stride=1)
    assert tracker.calls == ["update"] * N_FRAMES


def test_stride_invalid():
    with pytest.raises(ValueError, match="stride"):
        rto.run_sequence(_FakeTracker(), {}, _seq_info(), stride=0)


def test_min_conf_filters_detections():
    tracker = _FakeTracker()
    dets = {1: [rto.DetRow(1, 0, 0, 10, 10, 0.9), rto.DetRow(1, 0, 0, 10, 10, 0.1)]}
    seen: list[int] = []

    class _Spy(_FakeTracker):
        def update(self, detections):
            seen.append(len(detections))
            return super().update(detections)

    rto.run_sequence(_Spy(), dets, _seq_info(), stride=1, min_conf=0.5)
    assert seen[0] == 1  # 低置信度检测被过滤


# ---------------------------------------------------------------------------
# run_tracking_offline: 真实 DetectionTracker 端到端 + 输出格式
# ---------------------------------------------------------------------------

def _moving_det_rows() -> list[tuple]:
    """单目标匀速右移的私有检测（像素坐标）。"""
    return [(f, 10.0 + 2.0 * (f - 1), 20.0, 30.0, 40.0, 0.9) for f in range(1, N_FRAMES + 1)]


def test_real_tracker_stride2_stable_id(tmp_path: Path):
    det = tmp_path / "det.txt"
    write_private_det(det, _moving_det_rows())
    by_frame = rto.load_mot_detections(det)
    tracker = rto.make_tracker(
        filter_type="ukf", backend="python", max_output_misses=0,
        matching_strategy="hungarian", iou_threshold=0.25, max_misses=4,
        min_hits_to_show=1,
    )
    rows = rto.run_sequence(tracker, by_frame, _seq_info(), stride=2)
    assert sorted({r.frame for r in rows}) == list(range(1, N_FRAMES + 1))
    ids = {r.track_id for r in rows}
    assert len(ids) == 1, f"单目标应保持稳定 ID: {ids}"


def test_write_mot_results_format(tmp_path: Path):
    rows = [rto.ResultRow(frame=3, track_id=7, left=1.5, top=2.5, width=30.0,
                          height=40.0, conf=0.87654)]
    out = tmp_path / "sub" / "SEQ.txt"
    rto.write_mot_results(out, rows)
    parts = out.read_text(encoding="utf-8").strip().split(",")
    assert len(parts) == 10  # MOT 结果 10 列（尾部 x,y,z=-1）
    assert parts[0] == "3" and parts[1] == "7"
    assert parts[7:] == ["-1", "-1", "-1"]
    assert float(parts[6]) == pytest.approx(0.8765, abs=1e-4)


def test_resolve_det_file_private_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="run_detection_cache"):
        rto.resolve_det_file(dataset="mot17", root=tmp_path, split="train",
                             seq=SEQ_NAME, det_source="private",
                             cache_root=tmp_path / "cache", det_version="v1")


def test_resolve_det_file_public_rejects_dancetrack(tmp_path: Path):
    with pytest.raises(ValueError, match="DanceTrack"):
        rto.resolve_det_file(dataset="dancetrack", root=tmp_path, split="val",
                             seq=SEQ_NAME, det_source="public-frcnn",
                             cache_root=tmp_path, det_version="v1")


def test_resolve_det_file_public_ok(mot17_root: Path):
    det = rto.resolve_det_file(dataset="mot17", root=mot17_root, split="train",
                               seq=SEQ_NAME, det_source="public-frcnn",
                               cache_root=mot17_root, det_version="v1")
    assert det.name == "det.txt" and det.parent.name == "det"


def test_run_end_to_end_private(mot17_root: Path, tmp_path: Path):
    """端到端：合成数据集 + 私有缓存 → 结果 txt + summary.json。"""
    cache_root = tmp_path / "cache"
    write_private_det(
        cache_root / "mot17" / SEQ_NAME / "det_yolov8n_v1.txt", _moving_det_rows(),
    )
    out_dir = tmp_path / "runs"
    summary = rto.run(
        root=mot17_root, dataset="mot17", split="train", seqs=["all"],
        det_source="private", cache_root=cache_root, det_version="det_yolov8n_v1",
        filter_type="ukf", backend="python", stride=2, max_output_misses=0,
        matching_strategy="hungarian", iou_threshold=0.25, max_misses=4,
        min_hits_to_show=1, min_conf=0.0, out_dir=out_dir,
        tracker_tag="test_tag", project_root=mot17_root,
    )
    result = out_dir / "test_tag" / f"{SEQ_NAME}.txt"
    assert result.is_file()
    lines = result.read_text(encoding="utf-8").strip().splitlines()
    frames = {int(ln.split(",")[0]) for ln in lines}
    assert frames == set(range(1, N_FRAMES + 1))
    # summary.json 落盘且含协议元信息（evaluate_trackeval 依赖）
    sjson = json.loads((out_dir / "test_tag" / "summary.json").read_text(encoding="utf-8"))
    assert sjson["det_protocol"] == "private"
    assert sjson["det_cache_version"] == "det_yolov8n_v1"
    assert sjson["stride"] == 2
    assert sjson["backend"] == "python"
    assert summary["sequences"][SEQ_NAME]["frames"] == N_FRAMES


# ---------------------------------------------------------------------------
# evaluate_trackeval: 指标提取与渲染（不依赖 trackeval 包）
# ---------------------------------------------------------------------------

def _fake_tracker_res() -> dict:
    """模拟真实 TrackEval 结构：{seq: {class: {metric: {field}}},
    "COMBINED_SEQ": {...}}；HOTA 系字段为 19 个 alpha 阈值的数组。"""
    import numpy as np

    def block(hota, assa, deta, idf1, mota, idsw):
        return {"pedestrian": {
            "HOTA": {"HOTA": np.full(19, hota), "AssA": np.full(19, assa),
                     "DetA": np.full(19, deta)},
            "Identity": {"IDF1": idf1},
            "CLEAR": {"MOTA": mota, "IDSW": idsw},
        }}
    return {
        "SEQ-01": block(0.55, 0.60, 0.50, 0.65, 0.70, 12.0),
        "SEQ-02": block(0.45, 0.50, 0.40, 0.55, 0.60, 20.0),
        "COMBINED_SEQ": block(0.50, 0.55, 0.45, 0.60, 0.65, 32.0),
    }


def test_extract_metrics():
    res = ev.extract_metrics(_fake_tracker_res())
    assert res["combined"]["HOTA"] == pytest.approx(0.50)
    assert res["combined"]["IDSW"] == pytest.approx(32.0)
    assert set(res["per_sequence"]) == {"SEQ-01", "SEQ-02"}
    assert res["per_sequence"]["SEQ-01"]["AssA"] == pytest.approx(0.60)


def test_render_markdown_contains_metadata():
    report = {
        "meta": {
            "tag": "m0_baseline", "date": "2026-11-27T00:00:00+00:00",
            "dataset": "mot17", "split": "train",
            "det_protocol": "private", "det_cache_version": "det_yolov8n_v1",
            "stride": 1, "filter_type": "ukf", "backend": "python",
            "revision": "abc123", "trackeval_version": "1.0", "gt_dir": "G:/x",
        },
        "trackers": {"tag_a": ev.extract_metrics(_fake_tracker_res())},
    }
    md = ev.render_markdown(report)
    assert "m0_baseline" in md
    assert "det_yolov8n_v1" in md
    assert "| tag_a | 0.5000 |" in md          # 总表
    assert "| SEQ-01 | 0.5500 |" in md          # 每序列明细
    assert "abc123" in md


def test_import_trackeval_bad_dir(tmp_path: Path):
    with pytest.raises(RuntimeError, match="trackeval"):
        ev.import_trackeval(tmp_path / "nonexistent")


def test_import_trackeval_guidance_when_missing():
    """未安装且未给目录时，错误信息须含安装指引。"""
    import importlib.util
    if importlib.util.find_spec("trackeval") is not None:
        pytest.skip("环境已安装 trackeval")
    with pytest.raises(RuntimeError, match="JonathonLuiten/TrackEval"):
        ev.import_trackeval(None)


# ---------------------------------------------------------------------------
# CLI --help 可用（argparse 仅在 main 构建，import 无副作用）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [prep, rdc, rto, ev])
def test_cli_help(module, capsys):
    with pytest.raises(SystemExit) as exc_info:
        module.main(["--help"])
    assert exc_info.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()

# ---------------------------------------------------------------------------
# 真实 TrackEval 端到端（可选：未安装 trackeval 时 skip，不作硬依赖）
# ---------------------------------------------------------------------------

def test_trackeval_end_to_end_synthetic(tmp_path: Path):
    """真实 TrackEval 端到端（子进程隔离）。

    trackeval 的导入与 evaluate_trackeval 的 NumPy 2.x 兼容 shim（np.float 等
    别名补回）都是进程级副作用；在主 pytest 进程内执行会使后续 Qt fixture
    teardown 崩溃（实测 access violation），故整个评测在干净子进程中运行
    （与 tests/_runtime_checks.py 同一隔离思路）。
    """
    import importlib.util
    import subprocess
    if importlib.util.find_spec("trackeval") is None:
        pytest.skip("未安装 trackeval")

    seq = "SYNTH-01"
    gt_dir = tmp_path / "gt"
    seq_gt = gt_dir / seq
    (seq_gt / "gt").mkdir(parents=True)
    (seq_gt / "gt" / "gt.txt").write_text(
        "1,1,10,20,30,40,1,1,1\n2,1,12,20,30,40,1,1,1\n"
        "3,1,14,20,30,40,1,1,1\n4,1,16,20,30,40,1,1,1\n",
        encoding="utf-8",
    )
    (seq_gt / "seqinfo.ini").write_text(
        "[Sequence]\nname=SYNTH-01\nimDir=img1\nframeRate=30\nseqLength=4\n"
        "imWidth=640\nimHeight=480\nimExt=.jpg\n",
        encoding="utf-8",
    )
    tdir = tmp_path / "runs" / "tag_a"
    tdir.mkdir(parents=True)
    (tdir / f"{seq}.txt").write_text(
        "1,1,10,20,30,40,0.9,-1,-1,-1\n2,1,12,20,30,40,0.9,-1,-1,-1\n"
        "3,1,14,20,30,40,0.9,-1,-1,-1\n4,1,16,20,30,40,0.9,-1,-1,-1\n",
        encoding="utf-8",
    )

    project_root = Path(__file__).resolve().parent.parent
    code = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(project_root)!r})
from benchmark import evaluate_trackeval as ev

rc = ev.main([
    "--dataset", "mot17", "--root", {str(tmp_path)!r}, "--split", "train",
    "--results-dir", {str(tmp_path / "runs")!r}, "--gt-dir", {str(gt_dir)!r},
    "--out-dir", {str(tmp_path / "results")!r}, "--tag", "e2e",
])
assert rc == 0, f"evaluate rc={{rc}}"
results = Path({str(tmp_path / "results")!r})
json_files = list(results.glob("*_e2e.json"))
assert len(json_files) == 1, json_files
data = json.loads(json_files[0].read_text(encoding="utf-8"))
combined = data["trackers"]["tag_a"]["combined"]
assert abs(combined["HOTA"] - 1.0) < 1e-9, combined
assert abs(combined["IDF1"] - 1.0) < 1e-9, combined
assert abs(combined["IDSW"] - 0.0) < 1e-9, combined
assert data["meta"]["dataset"] == "mot17"
md_files = list(results.glob("*_e2e.md"))
assert len(md_files) == 1 and "tag_a" in md_files[0].read_text(encoding="utf-8")
print("E2E_OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(project_root), timeout=300,
    )
    assert proc.returncode == 0 and "E2E_OK" in proc.stdout, (
        f"trackeval e2e subprocess failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
