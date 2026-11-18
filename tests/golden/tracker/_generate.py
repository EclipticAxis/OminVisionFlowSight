"""Regenerate golden snapshots for tests/test_tracker_snapshot.py.

Usage:
    venv\Scripts\python.exe tests\golden\tracker\_generate.py

用当前 ai/tracker.py 的实现行为重写 tests/golden/tracker/*.json。
 golden 内容 = 场景函数的 JSON 序列化结果（浮点已 round 到 9 位小数），
 与 pytest 断言路径使用完全相同的序列化管线，保证可重复。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TESTS_DIR))

import test_tracker_snapshot as m  # noqa: E402


def main() -> None:
    m.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name in sorted(m.SCENARIOS):
        data = m.SCENARIOS[name]()
        out = m.GOLDEN_DIR / f"{name}.json"
        out.write_text(
            json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
