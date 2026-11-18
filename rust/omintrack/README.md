# omintrack — DetectionTracker 的 Rust 核心

OminVisionFlowSight 跟踪模块的 Rust 实现（UKF 滤波族 / AUTO 自适应 / 匹配引擎 /
重检测与生命周期管理），通过 PyO3 暴露为 `omintrack.RustTracker`，由
`ai/tracker_backend.RustDetectionTracker` 适配为与 Python 版完全一致的公开 API。

## 启用方式

```python
from ai.tracker import DetectionTracker
tracker = DetectionTracker(backend="rust")   # 显式启用 Rust 后端
# 或设置环境变量 OMINTRACK_BACKEND=rust，所有默认构造点自动切换
```

默认后端为 `python`，行为与历史版本完全一致。

## 构建（Windows）

前置：Rust 工具链（本机使用 `x86_64-pc-windows-gnu`，无需 MSVC Build Tools）、
项目 venv 内已安装 maturin。

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
cd E:\Vision Bata
venv\Scripts\python.exe -m maturin develop --manifest-path rust\omintrack\Cargo.toml --release
```

国内网络建议配置镜像：rustup 安装用 `RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static`，
crates 用 `~/.cargo/config.toml` 配 rsproxy 源（本机已配置）。

## 验证

```powershell
venv\Scripts\python.exe -m pytest tests/test_tracker_rust_equivalence.py -q   # 18 个等价性测试
venv\Scripts\python.exe -m pytest tests/test_tracker_snapshot.py -q           # 14 个行为快照测试
venv\Scripts\python.exe benchmark/benchmark_tracker_rust.py                   # 性能基准
```

## 已知边界

- 数值精度：nalgebra 与 NumPy/LAPACK 的浮点路径不同，等价性测试容差 1e-6
  （实测最大偏差 7.2e-10）。track_id / 状态机 / 标志位精确一致。
- polygon IoU 按当前运行环境（无 shapely）的 `_fallback_bbox_iou` 语义实现；
  若 Python 侧安装了 shapely，两后端在 OBB 匹配上会出现语义分歧。
- cv2.compareHist(Bhattacharyya) 内部为 float32，Rust 按 float64 公式实现，
  与 cv2 交叉验证差 ~3e-9。
- 帧相关计算（HSV 直方图、ReID embedding）保留在 Python 适配层（cv2/模型后端），
  经 det_hist / det_emb 参数传入 Rust 核心。
