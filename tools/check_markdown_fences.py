"""Markdown 代码围栏配对检查（CommonMark §4.5）。

设计契约见 docs/markdown-fence-check.md。判定规则：

1. 闭合围栏字符类型须与开启围栏一致（` 对 `，~ 对 ~）；
2. 闭合围栏长度必须 ≥ 开启围栏长度；
3. 闭合围栏不得携带 info string（围栏序列之后仅允许空白）；
4. 文件结束时仍开启的围栏视为违规——CommonMark 允许 EOF 隐式闭合，
   但本仓库文档中它几乎必然是笔误（PR #17 四反引号事故即此类：
   ````bash 开启后，``` 闭合行因规则 2 失效，其后 60% 文档渲染为代码块）。

v1 限制：不解析 blockquote / list 嵌套前缀（"> ```" 这类行不参与围栏判定），
也不检查围栏内部内容。仅做"每个开启围栏必须按规则闭合"的底线守护。

用法：
    python tools/check_markdown_fences.py            # 检查 git 跟踪的全部 .md
    python tools/check_markdown_fences.py a.md b.md  # 检查指定文件
退出码：0 = 全部平衡；1 = 存在未闭合围栏或文件读取失败。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# 围栏行：至多 3 空格缩进 + 3 个以上 ` 或 ~ + 其余内容（info string）
_FENCE_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})[ \t]*(.*)$")


@dataclass
class FenceIssue:
    """一个未闭合的开启围栏及其疑似闭合失败行。"""

    path: str
    line: int  # 开启行号（1-based）
    fence: str  # 开启围栏序列（如 "````"）
    candidates: list[str] = field(default_factory=list)

    def format(self) -> str:
        lines = [
            f"{self.path}:{self.line}: 围栏 {self.fence!r} 开启后直到文件结束未闭合",
        ]
        for cand in self.candidates:
            lines.append(f"    疑似闭合失败 {cand}")
        return "\n".join(lines)


def find_unclosed_fences(text: str, path: str = "<memory>") -> list[FenceIssue]:
    """按 CommonMark §4.5 扫描文本，返回未闭合围栏列表（应为空）。"""
    open_fence: FenceIssue | None = None
    open_char = ""
    open_len = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        m = _FENCE_RE.match(raw)
        if not m:
            continue
        _indent, seq, info = m.groups()
        # CommonMark §4.5：反引号围栏的 info string 不得含反引号，
        # 否则该行根本不是围栏行，按普通文本处理。
        if seq[0] == "`" and "`" in info:
            continue
        if open_fence is None:
            open_fence = FenceIssue(path=path, line=lineno, fence=seq)
            open_char, open_len = seq[0], len(seq)
            continue
        # 已处于开启围栏内部：该行若能闭合则闭合，否则只是内容，
        # 但记录"看起来像闭合却失败"的行，帮助定位笔误。
        if seq[0] == open_char and len(seq) >= open_len and not info.strip():
            open_fence = None
            continue
        reasons = []
        if seq[0] != open_char:
            reasons.append(f"字符 {seq[0]!r} 与开启字符 {open_char!r} 不同")
        if len(seq) < open_len:
            reasons.append(f"长度 {len(seq)} < 开启长度 {open_len}")
        if info.strip():
            reasons.append("闭合围栏不得携带 info string")
        open_fence.candidates.append(f"line {lineno}: {'; '.join(reasons)}")
    return [open_fence] if open_fence is not None else []


def list_tracked_markdown_files(root: Path) -> list[Path]:
    """返回 git 跟踪的全部 .md 文件（尊重 .gitignore，不含 venv/ 等）。"""
    # -z + core.quotepath=false：中文等非 ASCII 路径不做八进制转义/引号包裹
    out = subprocess.run(
        ["git", "-c", "core.quotepath=false", "ls-files", "-z", "--", "*.md"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return [root / line for line in out.stdout.split("\0") if line]


def check_files(paths: list[Path]) -> list[FenceIssue]:
    issues: list[FenceIssue] = []
    for p in paths:
        text = p.read_text(encoding="utf-8-sig")
        issues.extend(find_unclosed_fences(text, path=str(p)))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Markdown 代码围栏配对检查（CommonMark §4.5）")
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="待检查文件；缺省时检查 git 跟踪的全部 .md 文件",
    )
    args = parser.parse_args(argv)
    # Windows 中文区域控制台默认 GBK，统一 UTF-8 输出避免乱码
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    root = Path(__file__).resolve().parent.parent
    paths = args.files if args.files else list_tracked_markdown_files(root)
    try:
        issues = check_files(paths)
    except OSError as exc:
        print(f"读取失败: {exc}", file=sys.stderr)
        return 1
    if issues:
        for issue in issues:
            print(issue.format())
        print(f"\n共 {len(issues)} 处未闭合围栏（共检查 {len(paths)} 个文件）", file=sys.stderr)
        return 1
    print(f"OK: {len(paths)} 个 Markdown 文件围栏全部平衡")
    return 0


if __name__ == "__main__":
    sys.exit(main())