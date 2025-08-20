# python/pygreat/ffi.py
from __future__ import annotations
import sys
from ctypes import CDLL
from pathlib import Path
import platform

def load_lib() -> CDLL:
    here = Path(__file__).resolve().parent
    libdir = (here.parent / "fortran" / "lib").resolve()
    if platform.system() == "Darwin":
        name = "libpygreat.dylib"
    elif platform.system() == "Linux":
        name = "libpygreat.so"
    else:
        raise OSError("Unsupported OS for libpygreat")
    libpath = libdir / name
    if not libpath.exists():
        raise FileNotFoundError(f"libpygreat not found at {libpath}")
    return CDLL(str(libpath))
