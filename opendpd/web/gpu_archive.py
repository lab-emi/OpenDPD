"""Bounded, regular-file-only transfer; never extract an archive with extractall."""
from __future__ import annotations

import io
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath

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


def unpack(data: bytes, root: Path) -> None:
    if len(data) > MAX_BYTES:
        raise ValueError("GPU transfer exceeds storage limit")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) > 4096 or sum(i.file_size for i in members) > MAX_BYTES:
            raise ValueError("GPU archive exceeds storage limit")
        seen = set()
        for item in members:
            path = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if (not path.parts or path.as_posix() != item.filename or path.is_absolute() or any(p in {"", ".", ".."} or p.startswith(".gpu-") for p in path.parts)
                    or "\\" in item.filename or item.filename in seen or item.is_dir()
                    or stat.S_IFMT(mode) not in {0, stat.S_IFREG} or item.file_size > MAX_FILE):
                raise ValueError("invalid GPU archive member")
            seen.add(item.filename)
            target = root.joinpath(*path.parts)
            # Reject links already on disk as well as links in the archive.
            if any(p.is_symlink() for p in [target, *target.parents]):
                raise ValueError("GPU transfer cannot follow links")
        if archive.testzip() is not None:
            raise ValueError("GPU archive checksum failed")
        for item in members:
            target = root / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(".gpu-transfer")
            with open(temporary, "wb") as output:
                output.write(archive.read(item))
            os.replace(temporary, target)


def pack(root: Path, paths) -> bytes:
    output = io.BytesIO()
    total = count = 0
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in paths:
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(p.startswith(".gpu-") for p in relative.parts):
                continue
            size = path.stat().st_size
            total += size
            count += 1
            if size > MAX_FILE or total > MAX_BYTES or count > 4096:
                raise ValueError("GPU workspace exceeds transfer limit")
            archive.write(path, relative.as_posix())
    if output.tell() > MAX_BYTES:
        raise ValueError("GPU transfer exceeds storage limit")
    return output.getvalue()
