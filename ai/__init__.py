def __getattr__(name: str):
    # PEP 562 惰性导入：避免 `import ai` 时立即加载 PyQt6 / torch / ultralytics 等重型依赖，
    # 防止这些模块作为副作用被连带加载（docs 准则：包内不得 import ai.*）。
    if name == "InferWorker":
        from ai.inference import InferWorker

        return InferWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")