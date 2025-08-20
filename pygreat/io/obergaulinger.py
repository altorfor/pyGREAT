# src/pygreat/io/obergaulinger.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..constants import C_LIGHT, RHO_GEOM, P_GEOM  # cgs constants

__all__ = [
    "ObergaulingerData",
    "read_coco2d_evol",
    "select_background",
]

# -------------------------
# Data container (raw file)
# -------------------------
@dataclass
class ObergaulingerData:
    num_t: int
    m: int
    time: np.ndarray
    r: np.ndarray
    rho: np.ndarray
    y_e: np.ndarray
    p: np.ndarray
    e: np.ndarray
    t: np.ndarray
    cs: np.ndarray
    gamma1: np.ndarray
    v1: np.ndarray
    g: np.ndarray
    nv: np.ndarray

    def to_bg(self, n: int) -> dict:
        """Map time slice n → GREAT background dict (cgs, Newtonian metric)."""
        if n < 0 or n >= self.num_t:
            raise IndexError(f"time index {n} out of range [0,{self.num_t-1}]")

        r = np.array(self.r, dtype=np.float64, order="F", copy=True)

        rho_raw = np.asarray(self.rho[n, :], dtype=np.float64, order="F")
        p_raw   = np.asarray(self.p[n,   :], dtype=np.float64, order="F")
        e_raw   = np.asarray(self.e[n,   :], dtype=np.float64, order="F")
        cs_raw  = np.asarray(self.cs[n,  :], dtype=np.float64, order="F")
        g_raw   = np.asarray(self.g[n,   :], dtype=np.float64, order="F")
        v1_raw  = np.asarray(self.v1[n,  :], dtype=np.float64, order="F")
        y_e     = np.asarray(self.y_e[n, :], dtype=np.float64, order="F")
        T       = np.asarray(self.t[n,   :], dtype=np.float64, order="F")
        gamma1  = np.asarray(self.gamma1[n,:], dtype=np.float64, order="F")

        # Unit conversions (Obergaulinger → GREAT expectations)
        rho = rho_raw * RHO_GEOM
        p   = p_raw   * P_GEOM
        eps = e_raw / (rho_raw * C_LIGHT**2) + (939.57 - 8.8) / 931.49 - 1.0
        cs2 = (cs_raw / C_LIGHT) ** 2

        return {
            "time": float(self.time[n]),
            "nt": int(n),
            "r": r,
            "rho": rho,
            "eps": eps,
            "p": p,
            "c_sound_squared": cs2,
            "phi": np.ones_like(r),
            "alpha": 1.0 + g_raw / (C_LIGHT ** 2),
            "U": g_raw,
            "gamma_one": gamma1,
            "v_1": v1_raw / C_LIGHT,
            "y_e": y_e,
            "t": T,
            "entropy": np.zeros_like(r),
            "calculate_n2": True,
        }
# -------------------------
# Fixed-format I/O helpers
# -------------------------
def _read_record4(f) -> bytes:
    """Read one Fortran unformatted sequential record with 4-byte big-endian markers."""
    h = f.read(4)
    if not h:
        raise EOFError("unexpected EOF in record header")
    if len(h) != 4:
        raise ValueError("truncated record header")
    n = int.from_bytes(h, "big", signed=True)
    payload = f.read(n)
    if len(payload) != n:
        raise ValueError("truncated record payload")
    t = f.read(4)
    if len(t) != 4 or int.from_bytes(t, "big", signed=True) != n:
        raise ValueError("record length mismatch")
    return payload

def _vec_f8_be(buf: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(buf, dtype=">f8")
    if a.size != n:
        raise ValueError(f"vector has {a.size} elements; expected {n}")
    return a.astype(np.float64, copy=True)

def _mat_f8_be_F(buf: bytes, rows: int, cols: int) -> np.ndarray:
    a = np.frombuffer(buf, dtype=">f8")
    if a.size != rows * cols:
        raise ValueError(f"matrix has {a.size} elements; expected {rows*cols}")
    return np.asfortranarray(a.reshape((rows, cols), order="F"))

def _maybe_drop_bad_t0(time, rho, y_e, p, e, t, cs, gamma1, v1, g, nv):
    """
    If the first time slice is invalid (NaNs in gamma1 row OR all zeros in key rows),
    drop it and return Fortran-contiguous arrays again.
    """
    bad_nan = not np.isfinite(gamma1[0, :]).all()
    bad_zero = (
        np.allclose(rho[0, :], 0.0) and
        np.allclose(p[0,   :], 0.0) and
        np.allclose(e[0,   :], 0.0) and
        np.allclose(cs[0,  :], 0.0)
    )
    drop = bad_nan or bad_zero
    if not drop:
        return False, (time, rho, y_e, p, e, t, cs, gamma1, v1, g, nv)
    # Drop t=0 and re-fortranize 2D matrices
    time = time[1:]
    rho    = np.asfortranarray(rho[1:, :])
    y_e    = np.asfortranarray(y_e[1:, :])
    p      = np.asfortranarray(p[1:, :])
    e      = np.asfortranarray(e[1:, :])
    t      = np.asfortranarray(t[1:, :])
    cs     = np.asfortranarray(cs[1:, :])
    gamma1 = np.asfortranarray(gamma1[1:, :])
    v1     = np.asfortranarray(v1[1:, :])
    g      = np.asfortranarray(g[1:, :])
    nv     = nv[1:]
    return True, (time, rho, y_e, p, e, t, cs, gamma1, v1, g, nv)

# -------------------------
# Reader (fixed layout)
# -------------------------
def read_coco2d_evol(path: str | Path) -> ObergaulingerData:
    """
    Read CoCo2d-evol.dat with fixed format:
      - 4-byte record markers (big-endian)
      - header (num_t, m) as big-endian int32
      - payloads as big-endian float64
      - 2D fields stored in Fortran order

    Record order:
      0: (num_t, m)      int32
      1: time(num_t)     f8
      2: r(m)            f8
      3: rho(num_t,m)    f8 (F)
      4: y_e(num_t,m)
      5: p(num_t,m)
      6: e(num_t,m)
      7: t(num_t,m)
      8: cs(num_t,m)
      9: gamma1(num_t,m)
     10: v1(num_t,m)
     11: g(num_t,m)
     12: nv(num_t)       f8
    """
    path = Path(path)
    with path.open("rb") as f:
        # Header
        hdr = _read_record4(f)
        if len(hdr) != 8:
            raise ValueError(f"header record is {len(hdr)} bytes; expected 8")
        num_t, m = np.frombuffer(hdr, dtype=">i4", count=2).astype(np.int64)
        num_t, m = int(num_t), int(m)

        # 1D vectors
        time = _vec_f8_be(_read_record4(f), num_t)
        r_raw = _vec_f8_be(_read_record4(f), m)

        # Normalize radius to cm (if file gives km)
        if float(r_raw[0]) > 1.0e4:
            r_cm = r_raw
        else:
            r_cm = (r_raw * 1.0e5).astype(np.float64, copy=False)

        # 2D matrices (Fortran order)
        rho    = _mat_f8_be_F(_read_record4(f), num_t, m)
        y_e    = _mat_f8_be_F(_read_record4(f), num_t, m)
        p      = _mat_f8_be_F(_read_record4(f), num_t, m)
        e      = _mat_f8_be_F(_read_record4(f), num_t, m)
        t      = _mat_f8_be_F(_read_record4(f), num_t, m)
        cs     = _mat_f8_be_F(_read_record4(f), num_t, m)
        gamma1 = _mat_f8_be_F(_read_record4(f), num_t, m)
        v1     = _mat_f8_be_F(_read_record4(f), num_t, m)
        g      = _mat_f8_be_F(_read_record4(f), num_t, m)

        nv     = _vec_f8_be(_read_record4(f), num_t)

        # sanitize: drop bad t=0 (zeros/NaNs) if present
        dropped, (time, rho, y_e, p, e, t, cs, gamma1, v1, g, nv) = _maybe_drop_bad_t0(
            time, rho, y_e, p, e, t, cs, gamma1, v1, g, nv
        )
        if dropped:
            num_t = len(time)
            if num_t <= 0:
                raise ValueError("after dropping invalid t=0, file has no time slices")

    return ObergaulingerData(
        num_t=num_t, m=m,
        time=time, r=r_cm,
        rho=rho, y_e=y_e, p=p, e=e, t=t, cs=cs,
        gamma1=gamma1, v1=v1, g=g, nv=nv,
    )

# -------------------------
# Per-snapshot mapping to GREAT inputs
# -------------------------
# add near the bottom of pygreat/io/obergaulinger.py

def select_background(data, n, **_ignored):
    """Back-compat shim—use data.to_bg(n) going forward."""
    return data.to_bg(n)
