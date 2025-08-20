# pygreat/core.py
from __future__ import annotations

import os
from ctypes import (
    CDLL, POINTER, byref, c_char_p, c_double, c_int
)
from pathlib import Path
import numpy as np


def _default_lib_path() -> Path:
    # lib lives at ../../fortran/lib/libpygreat.{dylib|so} relative to this file
    root = Path(__file__).resolve().parents[1]
    libname = {
        "darwin": "libpygreat.dylib",
        "linux": "libpygreat.so",
    }
    import sys
    fn = libname["darwin" if sys.platform == "darwin" else "linux"]
    return (root / "fortran" / "lib" / fn).resolve()


def _load_lib() -> CDLL:
    # Prefer explicit path in codebase; avoid env vars per your preference.
    libpath = _default_lib_path()
    if not libpath.exists():
        raise FileNotFoundError(f"pyGREAT shared library not found: {libpath}")
    return CDLL(str(libpath))


# -----------------------------------------------------------------------------

_lib = _load_lib()

# Fortran shim symbols (bind(C) names)
_lib.pg_capture_reset.argtypes = []
_lib.pg_capture_reset.restype  = None

_lib.pg_load_parameters.argtypes = [c_char_p, POINTER(c_int)]
_lib.pg_load_parameters.restype  = None

# pg_set_background(nt, time, m, r, rho, eps, p, cs2, phi, alpha, v1, ierr)
_lib.pg_set_background.argtypes = [
    c_int, c_double, c_int,
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_int),
]
_lib.pg_set_background.restype  = None

_lib.pg_analyze_current.argtypes = [POINTER(c_int)]
_lib.pg_analyze_current.restype  = None

_lib.pg_modes_count.argtypes = [POINTER(c_int)]
_lib.pg_modes_count.restype  = None

_lib.pg_mode_shape.argtypes = [c_int, POINTER(c_int)]
_lib.pg_mode_shape.restype  = None

# pg_mode_copy(k, r, nr, np_over_r, dp, dQ, dpsi, K_cap, Psi_cap, freq, nt, time, ierr)
_lib.pg_mode_copy.argtypes = [
    c_int,
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_double), POINTER(c_int), POINTER(c_double), POINTER(c_int),
]
_lib.pg_mode_copy.restype  = None

_lib.pg_background_shape.argtypes = [POINTER(c_int)]
_lib.pg_background_shape.restype  = None

# pg_background_copy(r, rho, eps, p, cs2, phi, alpha, h, q, e,
#                    gamma1, ggrav, n2, lamb2, Bstar, Qcap, inv_cs2, mass_v,
#                    nt, time, ierr)
_lib.pg_background_copy.argtypes = [
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_int), POINTER(c_double), POINTER(c_int),
]
_lib.pg_background_copy.restype  = None


# -----------------------------------------------------------------------------

def _as_f64_ptr(a: np.ndarray):
    """Ensure float64, contiguous (any order), return pointer."""
    if not isinstance(a, np.ndarray):
        a = np.asarray(a, dtype=np.float64)
    if a.dtype != np.float64:
        a = a.astype(np.float64)
    if not a.flags['C_CONTIGUOUS'] and not a.flags['F_CONTIGUOUS']:
        a = np.ascontiguousarray(a)
    return a, a.ctypes.data_as(POINTER(c_double))


class PyGreatSession:
    """
    Thin Python wrapper around the Fortran shim.

    Usage:
        sess = PyGreatSession()
        sess.load_parameters("parameters")
        sess.reset_capture()
        sess.set_background(nt, time, bg_dict)
        sess.analyze()
        nm = sess.get_modes_count()
        ...
    """

    def __init__(self) -> None:
        self._lib = _lib
        self._last_iR: int | None = None
        self._prev_modes_count: int = 0  # for delta-mode logic if needed

    # -------- Parameters --------
    def load_parameters(self, parfile: str) -> None:
        ierr = c_int(0)
        self._lib.pg_load_parameters(parfile.encode("utf-8"), byref(ierr))
        if ierr.value != 0:
            raise RuntimeError(f"Read_parameters() failed with ierr={ierr.value}")

    # -------- Capture buffer --------
    def reset_capture(self) -> None:
        self._lib.pg_capture_reset()
        self._prev_modes_count = 0

    # -------- Background --------
    def set_background(self, nt: int, time: float, bg: dict) -> None:
        """
        bg must contain: r, rho, eps, p, cs2 (or c_sound_squared), phi, alpha, v_1
        All arrays length m.
        """
        # Accept legacy/alternative names
        if "cs2" not in bg and "c_sound_squared" in bg:
            bg = dict(bg)  # avoid mutating caller
            bg["cs2"] = bg["c_sound_squared"]

        required = ["r", "rho", "eps", "p", "cs2", "phi", "alpha", "v_1"]
        missing = [k for k in required if k not in bg]
        if missing:
            raise KeyError(f"background missing keys: {missing}")

        r   = np.asarray(bg["r"],   dtype=np.float64, order="C")
        rho = np.asarray(bg["rho"], dtype=np.float64, order="C")
        eps = np.asarray(bg["eps"], dtype=np.float64, order="C")
        p   = np.asarray(bg["p"],   dtype=np.float64, order="C")
        cs2 = np.asarray(bg["cs2"], dtype=np.float64, order="C")
        phi = np.asarray(bg["phi"], dtype=np.float64, order="C")
        alp = np.asarray(bg["alpha"], dtype=np.float64, order="C")
        v1  = np.asarray(bg["v_1"], dtype=np.float64, order="C")

        m = int(r.shape[0])
        if not all(arr.shape[0] == m for arr in (rho, eps, p, cs2, phi, alp, v1)):
            raise ValueError("background arrays must have identical length")

        def _ptr(a):
            if not (a.flags['C_CONTIGUOUS'] or a.flags['F_CONTIGUOUS']):
                a = np.ascontiguousarray(a)
            return a.ctypes.data_as(POINTER(c_double))

        ierr = c_int(0)
        self._lib.pg_set_background(
            c_int(int(nt)), c_double(float(time)), c_int(m),
            _ptr(r), _ptr(rho), _ptr(eps), _ptr(p),
            _ptr(cs2), _ptr(phi), _ptr(alp), _ptr(v1),
            byref(ierr),
        )
        if ierr.value != 0:
            raise RuntimeError(f"pg_set_background failed with ierr={ierr.value}")
        self._last_iR = None  # unknown until analyze()


    # -------- Run analysis --------
    def analyze(self) -> None:
        ierr = c_int(0)
        self._lib.pg_analyze_current(byref(ierr))
        if ierr.value != 0:
            raise RuntimeError(f"pg_analyze_current failed with ierr={ierr.value}")

    # -------- Modes API --------
    def get_modes_count(self) -> int:
        n = c_int(0)
        self._lib.pg_modes_count(byref(n))
        return int(n.value)

    def get_mode_shape(self, k: int) -> int:
        iR = c_int(0)
        self._lib.pg_mode_shape(c_int(int(k)), byref(iR))
        return int(iR.value)

    def copy_mode(self, k: int) -> dict:
        iR = self.get_mode_shape(k)
        if iR <= 0:
            raise IndexError("invalid mode index or empty spectrum")

        # allocate numpy arrays
        r        = np.empty(iR, dtype=np.float64)
        nr       = np.empty(iR, dtype=np.float64)
        np_over_r= np.empty(iR, dtype=np.float64)
        dp       = np.empty(iR, dtype=np.float64)
        dQ       = np.empty(iR, dtype=np.float64)
        dpsi     = np.empty(iR, dtype=np.float64)
        K        = np.empty(iR, dtype=np.float64)
        Psi      = np.empty(iR, dtype=np.float64)

        pr   = r.ctypes.data_as(POINTER(c_double))
        pnr  = nr.ctypes.data_as(POINTER(c_double))
        pnp  = np_over_r.ctypes.data_as(POINTER(c_double))
        pdp  = dp.ctypes.data_as(POINTER(c_double))
        pdQ  = dQ.ctypes.data_as(POINTER(c_double))
        pdps = dpsi.ctypes.data_as(POINTER(c_double))
        pK   = K.ctypes.data_as(POINTER(c_double))
        pPsi = Psi.ctypes.data_as(POINTER(c_double))

        freq = c_double(0.0)
        nt   = c_int(0)
        time = c_double(0.0)
        ierr = c_int(0)

        self._lib.pg_mode_copy(
            c_int(int(k)),
            pr, pnr, pnp, pdp, pdQ, pdps, pK, pPsi,
            byref(freq), byref(nt), byref(time), byref(ierr),
        )
        if ierr.value != 0:
            raise RuntimeError(f"pg_mode_copy failed with ierr={ierr.value}")

        return {
            "iR": iR,
            "freq": float(freq.value),
            "nt": int(nt.value),
            "time": float(time.value),
            "r": r, "eta_r": nr, "eta_perp_over_r": np_over_r,
            "delta_P": dp, "delta_Q": dQ, "delta_psi": dpsi,
            "K": K, "Psi": Psi,
        }

    # -------- Background API --------
    def get_background_shape(self) -> int:
        iR = c_int(0)
        self._lib.pg_background_shape(byref(iR))
        return int(iR.value)

    def copy_background(self) -> dict:
        iR = self.get_background_shape()
        if iR <= 0:
            raise RuntimeError("background not yet analyzed or empty")

        def arr():
            a = np.empty(iR, dtype=np.float64)
            return a, a.ctypes.data_as(POINTER(c_double))

        r,pr         = arr()
        rho,prho     = arr()
        eps,peps     = arr()
        p,pp         = arr()
        cs2,pcs2     = arr()
        phi,pphi     = arr()
        alpha,palpha = arr()
        h,ph         = arr()
        q,pq         = arr()
        e,pe         = arr()
        gamma1,pg1   = arr()
        ggrav,pg     = arr()
        n2,pn2       = arr()
        lamb2,pl2    = arr()
        Bstar,pb     = arr()
        Qcap,pQ      = arr()
        inv_cs2,pinv = arr()
        mass_v,pmass = arr()

        nt   = c_int(0)
        time = c_double(0.0)
        ierr = c_int(0)

        self._lib.pg_background_copy(
            pr, prho, peps, pp, pcs2, pphi, palpha,
            ph, pq, pe, pg1, pg, pn2, pl2, pb, pQ, pinv, pmass,
            byref(nt), byref(time), byref(ierr),
        )
        if ierr.value != 0:
            raise RuntimeError(f"pg_background_copy failed with ierr={ierr.value}")

        return {
            "iR": iR, "nt": int(nt.value), "time": float(time.value),
            "r": r, "rho": rho, "eps": eps, "p": p, "cs2": cs2,
            "phi": phi, "alpha": alpha, "h": h, "q": q, "e": e,
            "gamma1": gamma1, "ggrav": ggrav, "n2": n2, "lamb2": lamb2,
            "Bstar": Bstar, "Qcap": Qcap, "inv_cs2": inv_cs2, "mass_v": mass_v,
        }
