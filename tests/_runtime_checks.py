"""子进程版"运行时 import 副作用"检查。

直接在 pytest 进程内断言 ``sys.modules`` 会受套件内其他用例的 import 顺序
影响：排在前面的用例一旦 import 了 torch / cv2 等，后续所有
``test_*_loaded_at_runtime`` 就会误报。子进程提供干净解释器，语义也更
准确 —— 检查的是"导入目标模块本身会不会把违禁模块带进来"。

各 ``test_*_loaded_at_runtime`` 用例通过
``from tests._runtime_checks import assert_import_clean`` 使用
（各测试文件头部已把仓库根目录加入 ``sys.path``，直接以脚本方式运行
单个测试文件时同样可用）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def assert_import_clean(modules, banned, *, timeout: float = 120.0) -> None:
    """在独立子进程中导入 ``modules``，断言 ``banned`` 均未进入 ``sys.modules``。

    Args:
        modules: 要导入的模块名（字符串或字符串列表）。
        banned: 禁止被副作用加载的模块名（字符串或字符串列表）。
        timeout: 子进程超时秒数。
    """
    if isinstance(modules, str):
        modules = [modules]
    if isinstance(banned, str):
        banned = [banned]
    import_lines = "\n".join(f"import {m}  # noqa: F401" for m in modules)
    code = (
        "import sys\n"
        f"{import_lines}\n"
        f"banned = {list(banned)!r}\n"
        "loaded = [b for b in banned if b in sys.modules]\n"
        "if loaded:\n"
        "    print('banned module(s) loaded as side effect:', loaded)\n"
        "    sys.exit(1)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"import side-effect check failed for {list(modules)}:\n"
        f"stdout: {proc.stdout.strip()}\n"
        f"stderr: {proc.stderr.strip()}"
    )
