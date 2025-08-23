# pygreat/ffi.py
from __future__ import annotations
from ctypes import CDLL
from pathlib import Path
import platform


def _default_lib_path() -> Path:
    """
    Resolve the packaged GREAT shim:
      <repo>/fortran/lib/libpygreat.{dylib|so}
    """
    here = Path(__file__).resolve().parent
    libdir = (here.parent / "fortran" / "lib").resolve()
    name = "libpygreat.dylib" if platform.system() == "Darwin" else "libpygreat.so"
    return (libdir / name)


def load_lib() -> CDLL:
    """
    Load the GREAT shared library from the canonical repo location.
    (No environment-variable magic; explicit path for reproducibility.)
    """
    libpath = _default_lib_path()
    if not libpath.exists():
        raise FileNotFoundError(f"pyGREAT shared library not found at: {libpath}")
    return CDLL(str(libpath))
