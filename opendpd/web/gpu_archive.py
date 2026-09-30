"""Bounded, regular-file-only transfer; never extract an archive with extractall."""
from __future__ import annotations

import io
import os
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath
from opendpd.safe_paths import relative_parts

MAX_BYTES = 256 * 1024 * 1024
MAX_FILE = 64 * 1024 * 1024


def read_regular(root: Path, relative: str, offset=0, limit=MAX_FILE) -> bytes:
    """Read an active container's output without following even raced parent links."""
    parts = PurePosixPath(relative).parts
    if not parts or any(part in {".", ".."} for part in parts) or relative.startswith("/"):
        raise ValueError("invalid output path")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE:
                raise ValueError("invalid GPU output file")
            os.lseek(descriptor, offset, os.SEEK_SET)
            chunks = bytearray()
            while len(chunks) < limit:
                chunk = os.read(descriptor, min(1024 * 1024, limit - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
            return bytes(chunks)
        finally:
            os.close(descriptor)
    except FileNotFoundError:
        return b""
    finally:
        os.close(directory)


def unpack(data: bytes, root: Path, *, input_run_id: str | None = None) -> None:
    if len(data) > MAX_BYTES:
        raise ValueError("GPU transfer exceeds storage limit")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) > 4096 or sum(i.file_size for i in members) > MAX_BYTES:
            raise ValueError("GPU archive exceeds storage limit")
        seen = set()
        for item in members:
            relative_parts(item.filename)
            path = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if (not path.parts or path.as_posix() != item.filename or path.is_absolute() or any(p in {"", ".", ".."} or p.startswith(".gpu-") for p in path.parts)
                    or "\\" in item.filename or item.filename in seen or item.is_dir()
                    or stat.S_IFMT(mode) not in {0, stat.S_IFREG} or item.file_size > MAX_FILE):
                raise ValueError("invalid GPU archive member")
            if input_run_id is not None:
                # The VM supplies data to installed code, never importable modules.
                parts = path.parts
                allowed = (item.filename == "workspace.json" or
                           (len(parts) >= 3 and parts[0] in {"datasets", "runs"}
                            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", parts[1])
                            and path.suffix.lower() in {".json", ".jsonl", ".csv", ".npy", ".npz", ".pt", ".log", ".txt", ".png", ".svg", ".md", ".html"}))
                if not allowed:
                    raise ValueError("invalid GPU input layout")
            seen.add(item.filename)
            target = root.joinpath(*path.parts)
            # Reject links already on disk as well as links in the archive.
            if any(p.is_symlink() for p in [target, *target.parents]):
                raise ValueError("GPU transfer cannot follow links")
        if input_run_id is not None and not {"workspace.json", f"runs/{input_run_id}/config.resolved.json", f"runs/{input_run_id}/run.json"} <= seen:
            raise ValueError("GPU input is missing job metadata")
        if archive.testzip() is not None:
            raise ValueError("GPU archive checksum failed")
        for item in members:
            target = root / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(".gpu-transfer")
            with open(temporary, "wb") as output:
                output.write(archive.read(item))
            os.replace(temporary, target)


def pack(root: Path, paths=None, *, subtree: str | None = None) -> bytes:
    """Read through directory descriptors, including the root and all parents.

    Container output must be collected only after the container has been removed.
    No path-based stat/write can race a directory replacement into host files.
    """
    output = io.BytesIO()
    total = count = entries_seen = 0
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if subtree is not None:
            for part in relative_parts(subtree):
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            def add(parent, name, relative):
                nonlocal total, count
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_FILE:
                        raise ValueError("invalid GPU output file")
                    count += 1
                    if count > 4096:
                        raise ValueError("GPU workspace exceeds transfer limit")
                    data = bytearray()
                    while True:
                        chunk = os.read(fd, min(1024 * 1024, MAX_FILE - len(data) + 1))
                        if not chunk:
                            break
                        data.extend(chunk)
                        total += len(chunk)
                        if len(data) > MAX_FILE or total > MAX_BYTES:
                            raise ValueError("GPU workspace exceeds transfer limit")
                    archive.writestr(relative, data)
                finally:
                    os.close(fd)

            def walk(parent, prefix="", depth=0):
                nonlocal entries_seen
                if depth > 32:
                    raise ValueError("GPU directory nesting exceeds limit")
                with os.scandir(parent) as entries:
                    entries = list(entries)
                entries_seen += len(entries)
                if entries_seen > 4096:
                    raise ValueError("GPU workspace exceeds transfer limit")
                for entry in entries:
                    if entry.name.startswith(".gpu-"):
                        continue
                    relative_parts(entry.name)
                    relative = prefix + entry.name
                    if entry.is_dir(follow_symlinks=False):
                        child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                        try:
                            walk(child, relative + "/", depth + 1)
                        finally:
                            os.close(child)
                    else:
                        add(parent, entry.name, relative)

            if paths is None:
                walk(directory)
            else:
                for path in paths:
                    relative = path.relative_to(root / subtree if subtree else root).as_posix()
                    parts = relative_parts(relative)
                    if any(p.startswith(".gpu-") for p in parts):
                        continue
                    parent = os.dup(directory)
                    try:
                        for part in parts[:-1]:
                            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                            os.close(parent)
                            parent = child
                        info = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
                        if not stat.S_ISDIR(info.st_mode):
                            add(parent, parts[-1], relative)
                    finally:
                        os.close(parent)
    finally:
        os.close(directory)
    if output.tell() > MAX_BYTES:
        raise ValueError("GPU transfer exceeds storage limit")
    return output.getvalue()
