# pygreat/core.py
from __future__ import annotations

from ctypes import (
    CDLL, POINTER, byref, c_char_p, c_double, c_int
)
from pathlib import Path
import numpy as np

from .ffi import load_lib


# ----------------------------------------------------------------------------- #
#                           Library + error handling                            #
# ----------------------------------------------------------------------------- #

_lib: CDLL = load_lib()

# Friendly messages for common ierr values (edit as needed to match Fortran codes)
_ERR_MAP = {
    # Loader / parameter stage
    101: "Invalid or unreadable parameter file.",
    # Background ingestion
    301: "Background arrays have inconsistent sizes or invalid values.",
    # Analysis stage
    401: "Eigenvalue search failed to converge.",
    901: "Analysis preconditions not satisfied (e.g., shock/outer boundary not found or "
         "background state invalid for this mode).",
}

def _err_msg(where: str, ierr: int) -> str:
    msg = _ERR_MAP.get(int(ierr), "Unspecified error (see GREAT logs).")
    return f"{where} failed with ierr={int(ierr)}: {msg}"


# ----------------------------------------------------------------------------- #
#                               Fortran bindings                                #
# ----------------------------------------------------------------------------- #

# Capture buffer
_lib.pg_capture_reset.argtypes = []
_lib.pg_capture_reset.restype  = None

# Parameters
_lib.pg_load_parameters.argtypes = [c_char_p, POINTER(c_int)]
_lib.pg_load_parameters.restype  = None

# Background set: pg_set_background(nt, time, m, r, rho, eps, p, cs2, phi, alpha, v1, ierr)
_lib.pg_set_background.argtypes = [
    c_int, c_double, c_int,
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_int),
]
_lib.pg_set_background.restype  = None

# Analyze
_lib.pg_analyze_current.argtypes = [POINTER(c_int)]
_lib.pg_analyze_current.restype  = None

# Modes meta
_lib.pg_modes_count.argtypes = [POINTER(c_int)]
_lib.pg_modes_count.restype  = None

_lib.pg_mode_shape.argtypes = [c_int, POINTER(c_int)]
_lib.pg_mode_shape.restype  = None

# Mode copy:
# pg_mode_copy(k, r, nr, np_over_r, dp, dQ, dpsi, K_cap, Psi_cap, freq, nt, time, ierr)
_lib.pg_mode_copy.argtypes = [
    c_int,
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    POINTER(c_double), POINTER(c_int), POINTER(c_double), POINTER(c_int),
]
_lib.pg_mode_copy.restype  = None

# Background result shape
_lib.pg_background_shape.argtypes = [POINTER(c_int)]
_lib.pg_background_shape.restype  = None

# Background copy:
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


# ----------------------------------------------------------------------------- #
#                                  Utilities                                    #
# ----------------------------------------------------------------------------- #

def _as_ptr_f64(a: np.ndarray):
    """Return (array_cast, pointer) as float64 contiguous."""
    if not isinstance(a, np.ndarray):
        a = np.asarray(a, dtype=np.float64)
    elif a.dtype != np.float64:
        a = a.astype(np.float64, copy=False)
    if not (a.flags["C_CONTIGUOUS"] or a.flags["F_CONTIGUOUS"]):
        a = np.ascontiguousarray(a)
    return a, a.ctypes.data_as(POINTER(c_double))


def _require_keys(d: dict, keys: tuple[str, ...]) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"background missing keys: {missing}")


def _assert_monotonic_increasing(r: np.ndarray) -> None:
    if r.size >= 2 and not np.all(np.diff(r) > 0):
        raise ValueError("background r must be strictly increasing (ascending).")


def _assert_finite(*arrays: np.ndarray) -> None:
    for a in arrays:
        if not np.all(np.isfinite(a)):
            raise ValueError("background contains NaN/Inf values.")


# ----------------------------------------------------------------------------- #
#                                 Main session                                  #
# ----------------------------------------------------------------------------- #

class PyGreatSession:
    """
    Thin Python wrapper around the Fortran GREAT shim.

    Typical usage:
        sess = PyGreatSession()
        sess.load_parameters("parameters")
        sess.reset_capture()
        sess.set_background(nt, time, bg_dict)  # bg_dict uses keys: r, rho, eps, p, cs2|c_sound_squared, phi, alpha, v_1
        sess.analyze()
        nm = sess.get_modes_count()
        mode1 = sess.copy_mode(1)
        bgout = sess.copy_background()
    """

    def __init__(self) -> None:
        self._lib = _lib
        self._last_iR: int | None = None
        self._prev_modes_count: int = 0

    # -------- Parameters --------
    def load_parameters(self, parfile: str) -> None:
        ierr = c_int(0)
        self._lib.pg_load_parameters(parfile.encode("utf-8"), byref(ierr))
        if ierr.value != 0:
            raise RuntimeError(_err_msg("pg_load_parameters", ierr.value))

    # -------- Capture buffer --------
    def reset_capture(self) -> None:
        self._lib.pg_capture_reset()
        self._prev_modes_count = 0

    # -------- Background --------
    def set_background(self, nt: int, time: float, bg: dict) -> None:
        """
        bg must contain (1D float64 arrays of equal length m):
            r, rho, eps, p, cs2 (or c_sound_squared), phi, alpha, v_1
        """
        # Accept legacy name for cs^2
        if "cs2" not in bg and "c_sound_squared" in bg:
            bg = dict(bg)  # avoid mutating caller
            bg["cs2"] = bg["c_sound_squared"]

        _require_keys(bg, ("r", "rho", "eps", "p", "cs2", "phi", "alpha", "v_1"))

        r   = np.asarray(bg["r"],   dtype=np.float64, order="C")
        rho = np.asarray(bg["rho"], dtype=np.float64, order="C")
        eps = np.asarray(bg["eps"], dtype=np.float64, order="C")
        p   = np.asarray(bg["p"],   dtype=np.float64, order="C")
        cs2 = np.asarray(bg["cs2"], dtype=np.float64, order="C")
        phi = np.asarray(bg["phi"], dtype=np.float64, order="C")
        alp = np.asarray(bg["alpha"], dtype=np.float64, order="C")
        v1  = np.asarray(bg["v_1"], dtype=np.float64, order="C")

        m = int(r.shape[0])
        if not all(arr.shape == (m,) for arr in (rho, eps, p, cs2, phi, alp, v1)):
            raise ValueError("background arrays must be 1D and have identical length")

        _assert_monotonic_increasing(r)
        _assert_finite(r, rho, eps, p, cs2, phi, alp, v1)

        # Pointers
        pr   = r.ctypes.data_as(POINTER(c_double))
        prho = rho.ctypes.data_as(POINTER(c_double))
        peps = eps.ctypes.data_as(POINTER(c_double))
        pp   = p.ctypes.data_as(POINTER(c_double))
        pcs2 = cs2.ctypes.data_as(POINTER(c_double))
        pphi = phi.ctypes.data_as(POINTER(c_double))
        palp = alp.ctypes.data_as(POINTER(c_double))
        pv1  = v1.ctypes.data_as(POINTER(c_double))

        ierr = c_int(0)
        self._lib.pg_set_background(
            c_int(int(nt)), c_double(float(time)), c_int(m),
            pr, prho, peps, pp, pcs2, pphi, palp, pv1,
            byref(ierr),
        )
        if ierr.value != 0:
            raise RuntimeError(_err_msg("pg_set_background", ierr.value))
        self._last_iR = None  # till analyze()

    # -------- Run analysis --------
    def analyze(self) -> None:
        ierr = c_int(0)
        self._lib.pg_analyze_current(byref(ierr))
        if ierr.value != 0:
            raise RuntimeError(_err_msg("pg_analyze_current", ierr.value))

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
        r         = np.empty(iR, dtype=np.float64)
        nr        = np.empty(iR, dtype=np.float64)
        np_over_r = np.empty(iR, dtype=np.float64)
        dp        = np.empty(iR, dtype=np.float64)
        dQ        = np.empty(iR, dtype=np.float64)
        dpsi      = np.empty(iR, dtype=np.float64)
        K         = np.empty(iR, dtype=np.float64)
        Psi       = np.empty(iR, dtype=np.float64)

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
            raise RuntimeError(_err_msg("pg_mode_copy", ierr.value))

        return {
            "iR": iR,
            "freq": float(freq.value),
            "nt": int(nt.value),
            "time": float(time.value),
            "r": r,
            "eta_r": nr,
            "eta_perp_over_r": np_over_r,
            "delta_P": dp,
            "delta_Q": dQ,
            "delta_psi": dpsi,
            "K": K,
            "Psi": Psi,
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

        def arr_ptr():
            a = np.empty(iR, dtype=np.float64)
            return a, a.ctypes.data_as(POINTER(c_double))

        r,         pr   = arr_ptr()
        rho,       prho = arr_ptr()
        eps,       peps = arr_ptr()
        p,         pp   = arr_ptr()
        cs2,       pcs2 = arr_ptr()
        phi,       pphi = arr_ptr()
        alpha,     palp = arr_ptr()
        h,         ph   = arr_ptr()
        q,         pq   = arr_ptr()
        e,         pe   = arr_ptr()
        gamma1,    pg1  = arr_ptr()
        ggrav,     pg   = arr_ptr()
        n2,        pn2  = arr_ptr()
        lamb2,     pl2  = arr_ptr()
        Bstar,     pb   = arr_ptr()
        Qcap,      pQ   = arr_ptr()
        inv_cs2,   pinv = arr_ptr()
        mass_v,    pmass= arr_ptr()

        nt   = c_int(0)
        time = c_double(0.0)
        ierr = c_int(0)

        self._lib.pg_background_copy(
            pr, prho, peps, pp, pcs2, pphi, palp,
            ph, pq, pe, pg1, pg, pn2, pl2, pb, pQ, pinv, pmass,
            byref(nt), byref(time), byref(ierr),
        )
        if ierr.value != 0:
            raise RuntimeError(_err_msg("pg_background_copy", ierr.value))

        return {
            "iR": iR,
            "nt": int(nt.value),
            "time": float(time.value),
            "r": r, "rho": rho, "eps": eps, "p": p, "cs2": cs2,
            "phi": phi, "alpha": alpha, "h": h, "q": q, "e": e,
            "gamma1": gamma1, "ggrav": ggrav, "n2": n2, "lamb2": lamb2,
            "Bstar": Bstar, "Qcap": Qcap, "inv_cs2": inv_cs2, "mass_v": mass_v,
        }
