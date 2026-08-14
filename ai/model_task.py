from enum import Enum, auto


class ModelTask(Enum):
    """模型任务类型枚举，用于统一路由推理与后处理逻辑。

    取代字符串硬编码（"detect" / "pose" / "obb"），避免后端在解析时出错。
    """

    DETECT = auto()
    POSE = auto()
    OBB = auto()
