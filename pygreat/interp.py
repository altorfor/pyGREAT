# pygreat/interp.py
from __future__ import annotations
"""
Interpolation utilities for pyGREAT (compute-only).

Design goals
------------
- Depend on NumPy; optionally use SciPy (PCHIP, Cubic, Akima) when available.
- Minimal, well-documented public surface—no I/O here.
- No “magic” renaming: use canonical pre-GREAT keys:
    r, rho, eps, p, c_sound_squared, phi, alpha, v_1
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# Optional SciPy interpolators
try:
    from scipy.interpolate import (
        PchipInterpolator,
        CubicSpline,
        Akima1DInterpolator,
    )  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


__all__ = [
    "SonicInfo",
    "sonic_idx_from_bg",
    "find_sonic_point",
    "uniform_refined_grid",
    "interior_uniform_grid",
    "interpolate_bg",
    "validate_bg",
]


# ------------------------------- helpers --------------------------------------

def _unique_increasing(x: np.ndarray) -> np.ndarray:
    """Return a strictly increasing 1D array (drops duplicates)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return x.copy()
    keep = np.concatenate(([True], np.diff(x) > 0))
    return x[keep]


class _LinearPlan:
    """
    Precompute indices and weights for linear/nearest/loglinear interpolation
    from x -> x_new. Assumes x is strictly increasing.
    """
    __slots__ = ("idx_l", "idx_r", "alpha")

    def __init__(self, x: np.ndarray, x_new: np.ndarray):
        x = np.asarray(x, dtype=np.float64).ravel()
        xn = np.asarray(x_new, dtype=np.float64).ravel()

        idx = np.searchsorted(x, xn, side="left")
        idx_r = np.clip(idx, 0, len(x) - 1)
        idx_l = np.clip(idx - 1, 0, len(x) - 1)

        x_l = x[idx_l]
        x_r = x[idx_r]
        denom = np.maximum(x_r - x_l, 1e-300)
        alpha = (xn - x_l) / denom

        self.idx_l = idx_l
        self.idx_r = idx_r
        self.alpha = alpha

    def apply_linear(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        return (1.0 - self.alpha) * y[self.idx_l] + self.alpha * y[self.idx_r]

    def apply_nearest(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        # choose nearer endpoint; tie -> left
        pick_left = (self.alpha <= 0.5) | (self.idx_r == 0)
        return np.where(pick_left, y[self.idx_l], y[self.idx_r])

    def apply_loglinear(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        y_pos = np.maximum(y, 1e-300)
        logy = np.log(y_pos)
        out = (1.0 - self.alpha) * logy[self.idx_l] + self.alpha * logy[self.idx_r]
        return np.exp(out)


def _safe_interp(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, kind: str = "linear") -> np.ndarray:
    """
    Interpolate y(x) onto x_new.

    kind ∈ {'linear','nearest','loglinear','pchip','cubic','akima'}.
    Falls back to 'linear' if SciPy-based methods are unavailable.
    Extrapolation is clamped to edge values.
    """
    x = _unique_increasing(x)
    y = np.asarray(y, dtype=np.float64).ravel()
    x_new = np.asarray(x_new, dtype=np.float64).ravel()

    if kind == "nearest":
        plan = _LinearPlan(x, x_new)
        return plan.apply_nearest(y)

    if kind == "loglinear":
        plan = _LinearPlan(x, x_new)
        return plan.apply_loglinear(y)

    if kind == "pchip" and _HAS_SCIPY:
        f = PchipInterpolator(x, y, extrapolate=False)
        out = f(x_new)
        out = np.where(~np.isfinite(out) & (x_new <= x[0]), y[0], out)
        out = np.where(~np.isfinite(out) & (x_new >= x[-1]), y[-1], out)
        return out

    if kind == "cubic" and _HAS_SCIPY:
        f = CubicSpline(x, y, extrapolate=False)
        out = f(x_new)
        out = np.where(~np.isfinite(out) & (x_new <= x[0]), y[0], out)
        out = np.where(~np.isfinite(out) & (x_new >= x[-1]), y[-1], out)
        return out

    if kind == "akima" and _HAS_SCIPY:
        f = Akima1DInterpolator(x, y)
        out = f(x_new)
        out = np.where(~np.isfinite(out) & (x_new <= x[0]), y[0], out)
        out = np.where(~np.isfinite(out) & (x_new >= x[-1]), y[-1], out)
        return out

    # default fast path: linear via plan
    plan = _LinearPlan(x, x_new)
    return plan.apply_linear(y)


# ----------------------------- sonic point -----------------------------------

@dataclass
class SonicInfo:
    """Sonic point summary on the ORIGINAL grid."""
    idx: int       # 0-based index
    r: float       # radius at sonic point
    method: str    # "bg_iR(1-based)" | "zero_cross_innermost" | "min_abs"


def sonic_idx_from_bg(bg: Dict[str, np.ndarray], r: np.ndarray) -> SonicInfo:
    """
    Prefer the sonic index provided by the background (e.g., iR, Fortran 1-based).
    Clamps to valid range and returns 0-based idx.
    """
    r = _unique_increasing(np.asarray(r, float))
    n = len(r)
    if n == 0:
        return SonicInfo(idx=0, r=0.0, method="bg_iR(1-based)")

    for key in ("iR", "ir", "IR"):
        if key in bg:
            v = int(np.asarray(bg[key]).item())
            j = max(0, min(n - 1, v - 1))
            return SonicInfo(idx=j, r=float(r[j]), method="bg_iR(1-based)")

    raise KeyError("No 'iR' in background; cannot use background-provided sonic index.")


def find_sonic_point(
    r: np.ndarray,
    v_1: np.ndarray,
    c_sound_squared: np.ndarray,
) -> SonicInfo:
    """
    Locate sonic point from v_1 and c_sound_squared on the ORIGINAL grid.
    We find zero crossings of f = c_s^2 - v_1^2; if multiple exist, choose the
    innermost (smallest r). If none, use argmin |f|.
    """
    r  = _unique_increasing(np.asarray(r, float))
    v1 = np.asarray(v_1, float).ravel()
    cs2 = np.asarray(c_sound_squared, float).ravel()
    n = len(r)
    if n == 0:
        return SonicInfo(idx=0, r=0.0, method="min_abs")

    f = cs2 - v1**2
    s = np.signbit(f)
    cross = np.where(s[:-1] != s[1:])[0]
    if cross.size > 0:
        j = int(cross[0] + (abs(f[cross[0]]) > abs(f[cross[0] + 1])))  # closer endpoint
        j = int(np.clip(j, 0, n - 1))
        return SonicInfo(idx=j, r=float(r[j]), method="zero_cross_innermost")

    j = int(np.argmin(np.abs(f)))
    return SonicInfo(idx=j, r=float(r[j]), method="min_abs")


# ------------------------------ grids ----------------------------------------

def uniform_refined_grid(r: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Subdivide each original cell into `factor` equal subcells (factor>=1).
    This means we insert (factor-1) interior points per interval.
    Returns a strictly increasing grid that includes original endpoints.
    """
    r = _unique_increasing(r)
    factor = max(1, int(factor))
    if factor == 1 or r.size < 2:
        return r.copy()

    pieces: List[np.ndarray] = [r[[0]]]
    for i in range(len(r) - 1):
        a, b = r[i], r[i + 1]
        if factor > 1:
            # factor subcells ⇒ (factor-1) interior points
            # use factor+1 total points inclusive, then drop endpoints
            interior = np.linspace(a, b, factor + 1, endpoint=True)[1:-1]
            if interior.size:
                pieces.append(interior)
        pieces.append(np.array([b]))
    out = np.concatenate(pieces)
    return _unique_increasing(out)

def interior_uniform_grid(
    r: np.ndarray,
    shock_idx: int,
    buffer_cells: int = 15,
    inner_ncells: int = 400,
) -> np.ndarray:
    """
    Uniform grid from r[0] up to r[j_end], untouched beyond.
      j_end = min(n-1, shock_idx + buffer_cells)
    'inner_ncells' is the TOTAL number of points in [r[0], r[j_end]] (incl. endpoints).
    """
    r = _unique_increasing(np.asarray(r, float))
    n = len(r)
    if n == 0:
        return r.copy()

    j_end = int(min(n - 1, max(0, int(shock_idx) + int(buffer_cells))))
    if j_end < 1:
        return r.copy()

    inner_ncells = max(2, int(inner_ncells))
    inner = np.linspace(r[0], r[j_end], inner_ncells, endpoint=True)
    tail = r[j_end + 1 :] if (j_end + 1) < n else np.empty(0, dtype=r.dtype)
    out = np.concatenate([inner, tail]) if tail.size else inner
    return _unique_increasing(out)


# --------------------------- interpolation core ------------------------------

# Canonical keys expected in a background dict (pyGREAT naming).
_BG_KEYS = ["r", "rho", "eps", "p", "c_sound_squared", "phi", "alpha", "v_1"]

# Additional/optional extras to pass through if present.
_EXTRA_KEYS = ["y_e", "t", "entropy", "U"]

def interpolate_bg(
    bg: Dict[str, np.ndarray],
    r_new: np.ndarray,
    kind: str = "linear",
) -> Dict[str, np.ndarray]:
    """
    Interpolate a background dict onto r_new (1D). Returns a new dict with the same
    canonical keys + 'iR' set to len(r_new). Scalars 'nt' and 'time' are preserved.

    Fast kinds ('linear','nearest','loglinear') reuse a precomputed plan; spline-like
    kinds use SciPy per-field if available, else fall back to linear.
    Raises KeyError if a required key is missing.
    """
    out: Dict[str, np.ndarray] = {}
    r = _unique_increasing(np.asarray(bg["r"], dtype=np.float64))
    r_new = _unique_increasing(np.asarray(r_new, dtype=np.float64))

    fast_kind = kind in {"linear", "nearest", "loglinear"}
    plan = _LinearPlan(r, r_new) if fast_kind else None

    def _apply(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if fast_kind:
            if kind == "linear":
                return plan.apply_linear(y)  # type: ignore[union-attr]
            if kind == "nearest":
                return plan.apply_nearest(y)  # type: ignore[union-attr]
            if kind == "loglinear":
                return plan.apply_loglinear(y)  # type: ignore[union-attr]
        return _safe_interp(r, y, r_new, kind=kind)

    # Core fields
    for k in _BG_KEYS:
        if k == "r":
            out["r"] = r_new
            continue
        if k not in bg:
            have = ", ".join(sorted(bg.keys()))
            raise KeyError(f"background missing key: '{k}'. Available: {have}")
        out[k] = _apply(bg[k])

    # Optional extras
    for k in _EXTRA_KEYS:
        if k in bg:
            out[k] = _apply(bg[k])

    out["nt"]   = int(bg.get("nt", 0))
    out["time"] = float(bg.get("time", 0.0))
    out["iR"]   = int(len(r_new))
    return out


def validate_bg(bg_old: Dict[str, np.ndarray], bg_new: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Quick validation metrics:
      - max relative diff at original radii: rho, p, c_sound_squared
      - positivity: rho, p, c_sound_squared, alpha
      - monotonicity of r_new
    """
    r0 = _unique_increasing(np.asarray(bg_old["r"], float))
    r1 = _unique_increasing(np.asarray(bg_new["r"], float))

    def back_diff(key: str) -> float:
        y0 = np.asarray(bg_old[key], float)
        y1 = np.interp(r0, r1, np.asarray(bg_new[key], float), left=np.nan, right=np.nan)
        mask = np.isfinite(y1) & (np.abs(y0) + 1e-300 > 0)
        if not np.any(mask):
            return np.nan
        return float(np.nanmax(np.abs((y1[mask] - y0[mask]) / (np.abs(y0[mask]) + 1e-300))))

    metrics = {
        "reldiff_rho": back_diff("rho"),
        "reldiff_p": back_diff("p"),
        "reldiff_c_sound_squared": back_diff("c_sound_squared"),
        "min_r_new": float(np.min(r1)) if r1.size else np.nan,
        "max_r_new": float(np.max(r1)) if r1.size else np.nan,
        "len_r_new": int(len(r1)),
        "r_monotonic": 1.0 if (r1.size < 2 or np.all(np.diff(r1) > 0)) else 0.0,
    }

    for key in ["rho", "p", "c_sound_squared", "alpha"]:
        if key in bg_new and np.any(np.asarray(bg_new[key]) < 0):
            metrics[f"neg_{key}"] = 1.0

    return metrics
