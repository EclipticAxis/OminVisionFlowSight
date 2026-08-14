# Debug Session: yolo-detection-overlay
- **Status**: [OPEN]
- **Issue**: YOLO 模型检测结果与界面标定/叠框表现异常，需要通过运行时证据确认是模型输出、信号链路还是绘制坐标映射问题。
- **Debug Server**: `http://127.0.0.1:7777/event`
- **Log File**: `.dbg/trae-debug-log-yolo-detection-overlay.ndjson`

## Reproduction Steps
1. 启动应用并接入任意摄像头。
2. 开启 AI 检测。
3. 观察检测框、关键点、标签文本与目标位置是否存在错位、抖动、重复或异常类别。

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | 推理输出使用原始归一化坐标，但绘制阶段使用了错误的目标矩形或缩放基准，导致框和关键点错位 | High | Med | Rejected |
| B | 当前加载的模型类型或类别/关键点结构与绘制逻辑不匹配，导致标定数据不合法 | High | Low | Rejected |
| C | 摄像头帧在捕获、推理、绘制三段链路中的尺寸或颜色空间不一致，造成结果映射偏差 | Med | Med | Rejected |
| D | `frame_ready` / `detection_ready` 信号重复连接，导致旧结果叠加或不同插槽结果串线 | Med | Low | Rejected |
| E | 模型路径或实际加载文件异常，导致推理使用了非预期模型或回退模型 | Med | Low | Confirmed |

## Log Evidence
- `pre-fix / E / ai/inference.py:_load_model`: `requested_model_path=F:\VisionBata\models\yolov8n-pose.pt`，但运行输出同时出现 `Local pose model not found... Auto-downloading from Ultralytics...`，确认本地路径解析错误。
- `post-fix / E / ai/inference.py:_load_model`: `requested_model_path=F:\VisionBata\yolov8n-pose.pt`，不再出现自动下载警告，说明已命中本地模型。
- `pre-fix + post-fix / B`: `task=pose`、`names_count=1`、`detections_count=4`，输出结构稳定，未见类别/关键点结构异常。
- `pre-fix + post-fix / C`: `frame_shape=[1080,810,3]`，`image_size=[810,1080]`，帧尺寸在推理到绘制前保持一致。
- `pre-fix + post-fix / A`: `paint rect={x:149,y:12,w:342,h:456}`，与 `frame_size=[810,1080]` 等比例缩放匹配，未见坐标映射错位。
- `pre-fix + post-fix / D`: 单次链路中仅收到 1 次 `detections delivered to cell`，未观察到结果串线或重复叠加证据。

## Verification Conclusion
- 根因已定位为模型路径解析错误，代码固定查找 `models/yolov8n-pose.pt`，但项目内实际模型文件位于项目根目录 `yolov8n-pose.pt`。
- 修复后改为基于项目根目录动态解析，并按 `models/文件 -> 根目录文件` 顺序查找 `yolov8n-pose.pt` 与 `yolov8n.pt`。
- `pre-fix` 与 `post-fix` 对比结果表明：模型已从错误路径回退下载，切换为正确的本地文件加载；推理输出和绘制映射在修复前后均保持稳定。
