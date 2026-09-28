"""Inspect NumPy headers and resource limits before allocating imported arrays."""
import math
import zipfile
from pathlib import Path
import numpy as np
from opendpd.safe_paths import filename

MAX_ARRAY_BYTES = 256 * 1024 * 1024
MAX_ARRAYS = 64


def _header(stream, size):
    version = np.lib.format.read_magic(stream)
    reader = {(1, 0): np.lib.format.read_array_header_1_0,
              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
    if reader is None:
        raise ValueError("unsupported NumPy header version")
    shape, _, dtype = reader(stream, max_header_size=16384)
    if dtype.kind not in "fiuc" or len(shape) > 4 or any(n < 0 for n in shape):
        raise ValueError("NumPy input must contain numeric arrays with at most four dimensions")
    size_bytes = math.prod(shape) * dtype.itemsize
    if size_bytes > MAX_ARRAY_BYTES or size_bytes + stream.tell() != size:
        raise ValueError("NumPy array exceeds the size limit or has an invalid shape")
    return {"dtype": str(dtype), "shape": list(shape)}


def inspect_numpy(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        with path.open("rb") as stream:
            return {"array": _header(stream, path.stat().st_size)}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not members or len(members) > MAX_ARRAYS or sum(i.file_size for i in members) > MAX_ARRAY_BYTES:
            raise ValueError("NumPy archive exceeds the 256 MiB expanded size limit")
        result = {}
        for item in members:
            filename(item.filename)
            key = item.filename[:-4]
            if not item.filename.endswith(".npy") or key in result:
                raise ValueError("NumPy archive must contain uniquely named .npy arrays")
            with archive.open(item) as stream:
                result[key] = _header(stream, item.file_size)
        return result
