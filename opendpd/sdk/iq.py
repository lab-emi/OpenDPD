"""Explicit MATLAB/NumPy I/Q conversion; no scaling or alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def as_iq(value, name="signal") -> np.ndarray:
    """Return finite float32 Nx2 I/Q from a complex vector or real Nx2 array.

    Complex row/column vectors are accepted. Real 1-D vectors are I-only;
    a real 2-D array must have two columns (including the one-sample case).
    """
    arr = np.asarray(value)
    if arr.dtype.kind not in "fc" or arr.size == 0:
        raise ValueError(f"{name}: provide a nonempty single/double I/Q array or complex vector")
    if np.iscomplexobj(arr):
        if arr.ndim != 1 and not (arr.ndim == 2 and 1 in arr.shape):
            raise ValueError(f"{name}: complex input must be a vector, got {arr.shape}")
        arr = arr.reshape(-1)
        arr = np.column_stack((arr.real, arr.imag))
    elif arr.ndim == 1:
        arr = np.column_stack((arr, np.zeros_like(arr)))
    elif arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name}: real I/Q must have shape (N, 2), got {arr.shape}")
    if not np.isfinite(arr).all() or np.max(np.abs(arr)) > np.finfo(np.float32).max:
        raise ValueError(f"{name}: samples must be finite and within the float32 range")
    return np.ascontiguousarray(arr, dtype=np.float32)


def read_mat(path, input_variable="x", output_variable="y"):
    """Read explicitly named numeric variables from a MATLAB v4/v6/v7 file."""
    from scipy.io import loadmat, whosmat

    path = Path(path).expanduser().resolve(strict=True)
    if input_variable == output_variable:
        raise ValueError("Input and output variables must be different")
    with path.open("rb") as stream:
        header = stream.read(128)
    if header.startswith((b"MATLAB 7.3 MAT-file", b"\x89HDF")):
        raise ValueError("MAT v7.3 is not supported yet; save numeric x/y using MATLAB save(..., '-v7')")
    try:
        variables = {name: kind for name, _, kind in whosmat(path)}
        for name in (input_variable, output_variable):
            if variables.get(name) not in ("single", "double"):
                raise ValueError(f"MAT variable '{name}' must exist and be a full single/double numeric array")
        values = loadmat(path, variable_names=[input_variable, output_variable])
    except NotImplementedError as err:
        raise ValueError("Unsupported MAT format; save numeric x/y using MATLAB save(..., '-v7')") from err
    arrays = []
    for name in (input_variable, output_variable):
        value = values[name]
        # MAT stores real vectors as 2-D arrays. Treat row/column vectors as
        # sample sequences, including a real two-sample row vector.
        if value.ndim == 2 and 1 in value.shape:
            value = value.reshape(-1)
        as_iq(value, name)  # validate without discarding source precision
        arrays.append(value)
    return path, arrays[0], arrays[1]
