"""Markdown 围栏配对检查契约测试（设计见 docs/markdown-fence-check.md）：

1. 守护用例：git 跟踪的全部 .md 文件必须围栏平衡（与 CI markdown-fences 同源）；
2. 判定规则单测：CommonMark §4.5 闭合三规则 + EOF 未闭合即违规；
3. 回归用例：PR #17 引入的四反引号开启 + 三反引号闭合事故必须被捕获。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.check_markdown_fences import (
    check_files,
    find_unclosed_fences,
    list_tracked_markdown_files,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_all_tracked_markdown_files_have_balanced_fences():
    """守护：仓库全部跟踪 .md 文件围栏平衡（CI markdown-fences 的本地同源检查）。"""
    paths = list_tracked_markdown_files(REPO_ROOT)
    assert paths, "git ls-files 未返回任何 .md 文件，检查器失去守护对象"
    issues = check_files(paths)
    assert not issues, "\n".join(issue.format() for issue in issues)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("plain text, no fences\n", id="no-fence"),
        pytest.param("```\ncode\n```\n", id="bare-pair"),
        pytest.param("```python\nx = 1\n```\n", id="info-string-pair"),
        pytest.param("```\ncode\n````\n", id="longer-closer-ok"),
        pytest.param("~~~\ncode\n~~~\n", id="tilde-pair"),
        pytest.param("~~~ info with ` backtick\ncode\n~~~~\n", id="tilde-info-allows-backtick"),
        pytest.param(
            "````markdown\n```python\ninner\n```\n````\n",
            id="nested-fence-example-ok",
        ),
        pytest.param("text\n    ```\nindented code, not a fence\n", id="indented-not-a-fence"),
        pytest.param("``` info `with` backtick\nnot an opener\n", id="backtick-in-info-not-a-fence"),
    ],
)
def test_balanced_cases(text: str):
    assert find_unclosed_fences(text) == []


@pytest.mark.parametrize(
    "text, expected_line",
    [
        pytest.param("```python\nx = 1\n", 1, id="unclosed-at-eof"),
        pytest.param("intro\n\n````bash\ncode\n```\n", 3, id="four-open-three-close"),
        pytest.param("```\ncode\n```python\n", 1, id="closer-with-info-string-fails"),
        pytest.param("```\ncode\n~~~\n", 1, id="tilde-cannot-close-backtick"),
        pytest.param("````bash\ncode\n```\n\n````bash\nmore\n```\n", 1, id="pr17-readme-regression"),
    ],
)
def test_unbalanced_cases(text: str, expected_line: int):
    issues = find_unclosed_fences(text, path="case.md")
    assert len(issues) == 1
    assert issues[0].line == expected_line
    assert issues[0].path == "case.md"


def test_pr17_regression_reports_failed_closer_candidates():
    """四反引号开启行必须列出三反引号闭合失败原因，便于定位笔误。"""
    text = "````bash\ngit clone x\n```\n\n````bash\npip install\n```\n"
    (issue,) = find_unclosed_fences(text)
    assert issue.line == 1
    assert issue.fence == "````"
    assert any("长度 3 < 开启长度 4" in c for c in issue.candidates)
    assert any("不得携带 info string" in c for c in issue.candidates)


def test_tilde_opener_reports_backtick_closer_candidate():
    issues = find_unclosed_fences("~~~\ncode\n```\n")
    assert len(issues) == 1
    assert any("字符" in c for c in issues[0].candidates)