"""Portable lexical checks before touching paths supplied by imported metadata."""
from pathlib import Path, PureWindowsPath
import re


def relative_parts(value: str) -> tuple[str, ...]:
    if (not isinstance(value, str) or not value or "\\" in value
            or PureWindowsPath(value).drive or PureWindowsPath(value).root
            or value.startswith(("/", "~"))):
        raise ValueError("path must be a relative POSIX path")
    parts = tuple(value.split("/"))
    for part in parts:
        if (part in ("", ".", "..") or ":" in part or part.endswith((".", " "))
                or any(ord(c) < 32 or ord(c) == 127 for c in part)
                or re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part)):
            raise ValueError("path contains an unsafe segment")
    return parts


def contained_path(root: Path, relative: str) -> Path:
    parts = relative_parts(relative)  # Must precede resolve/stat (especially on Windows).
    base = Path(root).resolve()
    target = base
    for part in parts:
        target = target / part
        if target.is_symlink():
            raise ValueError("paths must not follow symbolic links")
    if not target.resolve().is_relative_to(base):
        raise ValueError("path escapes its root")
    return target


def filename(value: str) -> str:
    if len(relative_parts(value)) != 1:
        raise ValueError("expected a filename without directories")
    return value
