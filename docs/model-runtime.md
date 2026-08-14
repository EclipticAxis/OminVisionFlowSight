# Model Runtime Extraction Report (Milestone D9.2)

> **日期**：2026-08-04
> **范围**：从 `ai/inference.py` 提取模型生命周期到 `visioncore/runtime/model/`
> **状态**：已实现，30 项测试全通过 + 全量回归零破坏（33 个套件）

---

## 1. 分析：源代码定位

### 1.1 Legacy 代码（`ai/inference.py`）

| 方法 | 行号 | 功能 |
|---|---|---|
| `_initialize_backend` | L891-918 | 选择后端（PyTorch/ONNX），加载检测模型 |
| `_warmup` | L920-936 | 对 dummy 输入跑 2 次推理预热 |
| `_switch_model_if_needed` | L938-977 | 热切换模型（带回滚） |
| `_detect_device` | L979-988 | 检测 CUDA/CPU |
| `_load_detect_model` | L1012-1057 | 加载 PyTorch 检测模型 |
| `_load_onnx_detect_model` | L1060-1077 | 加载 ONNX 检测模型 |
| `_load_pose_model` | L990-1009 | 加载姿态模型 |
| `_select_inference_model` | L1091-1099 | 选择检测/姿态模型 |
| `_predict_model` | L1101-1109 | 运行推理 |

### 1.2 迁移的职责

```
ModelRuntime (ABC) ──── 单模型生命周期
  ├─ load(path, device)
  ├─ unload()
  ├─ warmup(imgsz)
  ├─ predict(frame, conf, imgsz) → raw output
  └─ health_check() → bool

ModelManager ──── 多模型协调
  ├─ detect_device() → "cpu"/"cuda"
  ├─ register_runtime(name, runtime)
  ├─ load(name, path, device)
  ├─ unload(name)
  ├─ warmup_all(imgsz)
  ├─ switch_model(name, new_path) → bool (rollback on failure)
  └─ health_check() → bool
```

---

## 2. 文件结构

```
visioncore/runtime/
├── __init__.py              # 包导出
└── model/
    ├── __init__.py          # 子包导出
    └── model_runtime.py     # ModelInfo, ModelRuntime, ModelManager, ModelManagerError

tests/test_model_runtime.py  # 30 项测试
docs/model-runtime.md        # 本文档
```

---

## 3. 接口

### 3.1 ModelRuntime（ABC）

```python
class ModelRuntime(ABC):
    def load(self, path=None, device="cpu"): ...
    def unload(self): ...
    def warmup(self, imgsz=640): ...
    def predict(self, frame, conf=0.25, imgsz=640) -> Any: ...
    def health_check(self) -> bool: ...
```

### 3.2 ModelManager

```python
class ModelManager:
    def __init__(self, device="cpu", default_imgsz=640): ...
    def detect_device() -> str: ...              # 静态方法
    def register_runtime(name, runtime): ...
    def get_runtime(name) -> ModelRuntime | None: ...
    def list_runtimes() -> list[str]: ...
    def load(name, path=None, device=None): ...
    def unload(name): ...
    def warmup_all(imgsz=None): ...
    def switch_model(name, new_path, device=None) -> bool: ...
    def health_check() -> bool: ...
```

### 3.3 Legacy 代码对应关系

| Legacy 代码 | Runtime 等价物 |
|---|---|
| `_initialize_backend` | `ModelManager.load()` |
| `_warmup` | `ModelManager.warmup_all()` |
| `_switch_model_if_needed` | `ModelManager.switch_model()` |
| `_detect_device` | `ModelManager.detect_device()` |
| `_load_detect_model` / `_load_onnx_detect_model` | 具体 `ModelRuntime` 子类（注入） |
| `_predict_model` | `ModelRuntime.predict()` |
| `_model` / `_detect_model` / `_pose_model` | `ModelManager` 命名槽位 |

### 3.4 使用示例

```python
from visioncore.runtime.model import ModelManager

manager = ModelManager(device="cuda")
manager.register_runtime("detect", my_pytorch_runtime)
manager.register_runtime("pose", my_onnx_runtime)

manager.load("detect", "yolov8n.pt")
manager.load("pose", "yolov8n-pose.onnx")
manager.warmup_all()

# 运行时查询
info = manager.get_info("detect")
print(info.loaded, info.device, info.path)

# 热切换
success = manager.switch_model("detect", "yolov8s.pt")
if not success:
    print("switch failed, old model restored")
```

---

## 4. 设计决策

### 4.1 为什么不包含检测逻辑？

规格要求 "禁止包含检测逻辑，Detector 只调用 Runtime"。`ModelRuntime.predict()` 返回**原始模型输出**（类型取决于后端）。NMS、box 转换、标签过滤等后处理留在 Detector Stage。

### 4.2 为什么 torch 是延迟导入？

`detect_device()` 内部 `import torch`（在 try 块中）——模块顶层不导入 torch，避免启动时加载 GPU 库。测试验证：模块级源码无 `import torch`。

### 4.3 switch_model 的回滚机制

热切换失败时：
1. 保存旧模型路径
2. 尝试 unload → load(new_path) → warmup
3. 如果任何步骤失败：unload → load(old_path)
4. 如果回滚也失败：标记 loaded=False

---

## 5. 测试覆盖（30 项）

| 分组 | 测试数 | 关键测试 |
|---|---|---|
| 模块表面 | 3 | ModelManagerError 继承、ModelInfo 默认值、ModelRuntime ABC |
| 构造 | 2 | 默认构造、自定义 device/imgsz |
| 设备检测 | 2 | 返回字符串、CPU fallback |
| 运行时注册 | 4 | 注册+获取、未知返回 None、多运行时、自动创建 ModelInfo |
| 生命周期 | 7 | load 调用运行时、未知抛异常、load 失败抛异常、unload、warmup_all、空 warmup、自定义 imgsz |
| 热切换 | 4 | 成功、失败回滚、未知运行时、warmup 状态 |
| 聚合健康 | 4 | 空管理器健康、全部健康、一个不健康、未加载不健康 |
| 无检测逻辑 | 1 | 源码无 NMS/box_iou/filter 函数 |
| 零依赖 | 2 | 顶层无 banned 导入、模块导入不触发 torch |

---

## 6. 零侵入声明

新增 `visioncore/runtime/` 包。**未修改任何既有文件**（`ai/inference.py` 未改）。全量回归：33 个测试套件全部通过。
