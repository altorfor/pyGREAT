# pygreat/io/hdf5_utils.py
from __future__ import annotations
from typing import Any
import numpy as np

DIV_SLASH = "∕"  # U+2215 Unicode division slash

__all__ = [
    "nt_key",
    "mode_key",
    "ascii_name",
    "write_scalar",
    "write_1d",
    "write_group_attrs",
    "set_group_attrs",  # back-compat alias
]

# ---------------- keys & naming ----------------

def nt_key(nt: Any) -> str:
    """Group name for a timestep (8-digit, zero-padded)."""
    return f"{int(nt):08d}"

def mode_key(idx: Any) -> str:
    """Group name for a mode within a timestep."""
    return f"mode_{int(idx):04d}"

def ascii_name(name: str) -> str:
    """
    HDF5 dataset/group names cannot contain '/', so replace it by the
    Unicode division slash to keep labels human-readable.
    """
    return str(name).replace("/", DIV_SLASH)

# ---------------- writers ----------------

def write_scalar(grp, name: str, value: Any) -> None:
    """
    Write a scalar as a dataset. If '/' was replaced, keep the original label
    in the 'ascii_label' attribute.
    """
    dname = ascii_name(name)
    if dname in grp:
        del grp[dname]
    ds = grp.create_dataset(dname, data=np.asarray(value))
    if dname != name:
        ds.attrs["ascii_label"] = name

def write_1d(grp, name: str, arr, dtype=np.float64) -> None:
    """
    Write a 1-D array with gzip+shuffle and sensible chunking.
    """
    a = np.asarray(arr, dtype=dtype)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {a.shape}")
    dname = ascii_name(name)
    if dname in grp:
        del grp[dname]
    chunks = (min(a.size, 4096),) if a.size > 0 else None
    ds = grp.create_dataset(
        dname, data=a, dtype=dtype,
        compression="gzip", compression_opts=4,
        shuffle=True, chunks=chunks,
    )
    if dname != name:
        ds.attrs["ascii_label"] = name

def write_group_attrs(grp, **attrs) -> None:
    """Attach plain-Python scalars as HDF5 attributes on a group."""
    for k, v in attrs.items():
        grp.attrs[str(k)] = v

# Back-compat: some modules import this older name
def set_group_attrs(grp, **attrs) -> None:
    """Alias to write_group_attrs for compatibility."""
    write_group_attrs(grp, **attrs)
