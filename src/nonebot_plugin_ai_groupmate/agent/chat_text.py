"""Make unsolicited Markdown emphasis readable in plain chat messages."""

import re

_PROTECTED = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`|“[^”\n]*”|‘[^’\n]*’|「[^」\n]*」|『[^』\n]*』|\"[^\"\n]*\"|(?<![a-zA-Z0-9])'[^'\n]*'(?![a-zA-Z0-9])|https?://[^\s]+")
_BOLD = re.compile(r"(?<!\\)\*\*(?=\S)([^\n]*?\S)(?<!\\)\*\*")
_HEADING = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+")
_MARKDOWN_REQUEST = re.compile(
    r"(?:(?:用|以|按|采用|转成|转换成|输出为|输出成|生成|提供|保留).{0,8}(?:markdown|(?<![a-z])md(?![a-z]))"
    r"|(?:markdown|(?<![a-z])md(?![a-z])).{0,8}(?:输出|发送|发给|给我|源码|原文))",
    re.IGNORECASE,
)
_MARKDOWN_NEGATION = re.compile(r"(?:不要|别|不用|禁止|去掉|去除|删除|取消).{0,8}(?:markdown|(?<![a-z])md(?![a-z]))", re.IGNORECASE)
_VERBATIM_REQUEST = re.compile(r"原样|逐字|照抄|原封不动|一字不差|代码|程序|脚本|正则表达式|命令行|\b(?:sql|json|yaml)\b", re.IGNORECASE)


def wants_original_format(request_text: str) -> bool:
    if _VERBATIM_REQUEST.search(request_text):
        return True
    return bool(_MARKDOWN_REQUEST.search(request_text) and not _MARKDOWN_NEGATION.search(request_text))


def plain_chat_text(text: str, *, request_text: str = "") -> str:
    """Remove heading and bold markers without altering quotes, code or URLs."""
    if wants_original_format(request_text):
        return text
    protected = [(match.start(), match.end()) for match in _PROTECTED.finditer(text)]

    def is_protected(index: int) -> bool:
        return any(start <= index < end for start, end in protected)

    changes: list[tuple[int, int, str]] = []
    for match in _BOLD.finditer(text):
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        expression = bool(re.fullmatch(r"[a-zA-Z0-9_\])]", before) and re.fullmatch(r"[a-zA-Z0-9_\[(]", after))
        if not expression and not is_protected(match.start()) and not is_protected(match.end() - 1):
            changes.append((match.start(), match.end(), match.group(1)))
    for match in _HEADING.finditer(text):
        if not is_protected(match.start()):
            changes.append((match.start(), match.end(), ""))
    for start, end, replacement in sorted(changes, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text
