# pygreat/interp.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np

try:
    import h5py  # type: ignore
    _HAS_H5PY = True
except Exception:
    _HAS_H5PY = False

# Optional interpolation methods
try:
    from scipy.interpolate import PchipInterpolator  # type: ignore
    _HAS_PCHIP = True
except Exception:
    _HAS_PCHIP = False

try:
    from scipy.interpolate import CubicSpline, Akima1DInterpolator  # type: ignore
    _HAS_SPLINE = True
except Exception:
    _HAS_SPLINE = False


__all__ = [
    "SonicInfo",
    "sonic_idx_from_bg",
    "find_sonic_point",
    "uniform_refined_grid",
    "shock_focused_grid",
    "interpolate_bg",
    "validate_bg",
    "write_intermediate_h5",
]


# ------------------------------- utils --------------------------------------

def _unique_increasing(x: np.ndarray) -> np.ndarray:
    """Return a strictly increasing 1D copy of x (drop duplicates)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return x.copy()
    keep = np.concatenate(([True], np.diff(x) > 0))
    return x[keep]


def _interp_nearest(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(x, x_new, side="left")
    idx_right = np.clip(idx, 0, len(x) - 1)
    idx_left  = np.clip(idx - 1, 0, len(x) - 1)
    pick_left = (idx_right == 0) | (
        (idx_right < len(x)) & (np.abs(x_new - x[idx_left]) <= np.abs(x_new - x[idx_right]))
    )
    return np.where(pick_left, y[idx_left], y[idx_right])


def _safe_interp(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, kind: str = "linear") -> np.ndarray:
    """
    Interpolate y(x) onto x_new.
    kind: 'linear' (default), 'nearest', 'loglinear', 'pchip', 'cubic', 'akima'
    Falls back to 'linear' if a SciPy-dependent method isn't available.
    """
    x = _unique_increasing(x)
    y = np.asarray(y, dtype=np.float64).ravel()
    x_new = np.asarray(x_new, dtype=np.float64).ravel()

    if kind == "nearest":
        return _interp_nearest(x, y, x_new)

    if kind == "loglinear":
        y_pos = np.maximum(y, 1e-300)
        return np.exp(np.interp(x_new, x, np.log(y_pos), left=np.log(y_pos[0]), right=np.log(y_pos[-1])))

    if kind == "pchip" and _HAS_PCHIP:
        f = PchipInterpolator(x, y, extrapolate=False)
        out = f(x_new)
        if np.any(~np.isfinite(out)):
            out = np.where((~np.isfinite(out)) & (x_new <= x[0]), y[0], out)
            out = np.where((~np.isfinite(out)) & (x_new >= x[-1]), y[-1], out)
        return out

    if kind == "cubic" and _HAS_SPLINE:
        f = CubicSpline(x, y, extrapolate=False)
        out = f(x_new)
        out = np.where((~np.isfinite(out)) & (x_new <= x[0]), y[0], out)
        out = np.where((~np.isfinite(out)) & (x_new >= x[-1]), y[-1], out)
        return out

    if kind == "akima" and _HAS_SPLINE:
        f = Akima1DInterpolator(x, y)
        out = f(x_new)
        out = np.where((~np.isfinite(out)) & (x_new <= x[0]), y[0], out)
        out = np.where((~np.isfinite(out)) & (x_new >= x[-1]), y[-1], out)
        return out

    return np.interp(x_new, x, y, left=y[0], right=y[-1])


# ----------------------------- sonic point -----------------------------------

@dataclass
class SonicInfo:
    idx: int       # index on the ORIGINAL grid (0-based)
    r: float       # radius at sonic point
    method: str    # "bg_iR(1-based)" | "zero_cross" | "min_abs"


def sonic_idx_from_bg(bg: Dict[str, np.ndarray], r: np.ndarray) -> SonicInfo:
    """
    Prefer the sonic index provided by the background (e.g., iR).
    Assumes 'iR' is 1-based (Fortran); clamps to valid range.
    """
    r = _unique_increasing(np.asarray(r, float))
    n = len(r)
    if n == 0:
        return SonicInfo(idx=0, r=0.0, method="bg_iR(1-based)")

    if "iR" not in bg:
        raise KeyError("No 'iR' in background; cannot use background-provided sonic index.")

    v = int(np.asarray(bg["iR"]).item())
    idx0 = max(0, min(n - 1, v - 1))  # 1-based -> 0-based
    return SonicInfo(idx=idx0, r=float(r[idx0]), method="bg_iR(1-based)")


def find_sonic_point(
    r: np.ndarray,
    v1: np.ndarray,
    c_sound_squared: np.ndarray,
) -> SonicInfo:
    """
    Sonic point from v_1 and c_sound_squared.

    Policy (robust for SN backgrounds):
      1) Find zero-crossings of f = c_s^2 - v_1^2.
      2) Choose the *first* crossing when scanning outward in r (innermost crossing).
      3) If no crossing exists, pick argmin |f|.

    Notes:
      - This avoids picking far-out spurious crossings where both terms are ~0.
      - Returned idx is a grid-point index (0-based).
    """
    r  = _unique_increasing(np.asarray(r, float))
    v1 = np.asarray(v1, float).ravel()
    cs2 = np.asarray(c_sound_squared, float).ravel()
    n = len(r)
    if n == 0:
        return SonicInfo(idx=0, r=0.0, method="min_abs")

    f = cs2 - v1**2
    # robust: ignore NaNs and infs
    good = np.isfinite(f)
    if not np.all(good):
        # restrict to finite window, but keep indexing consistent
        f = np.where(good, f, np.nan)

    # zero-crossings: indices i where [i, i+1] straddles zero
    sgn = np.signbit(f)
    cross = np.where((sgn[:-1] != sgn[1:]) & np.isfinite(f[:-1]) & np.isfinite(f[1:]))[0]

    if cross.size > 0:
        i = int(cross[0])           # innermost crossing
        # pick the closer endpoint to zero (purely for reporting r)
        j = i if abs(f[i]) <= abs(f[i+1]) else i+1
        j = int(np.clip(j, 0, n-1))
        return SonicInfo(idx=j, r=float(r[j]), method="zero_cross_innermost")

    # fallback: closest to sonic (min |f|)
    j = int(np.nanargmin(np.abs(f)))
    return SonicInfo(idx=j, r=float(r[j]), method="min_abs")



# ------------------------------ grids ----------------------------------------

def uniform_refined_grid(r: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Subdivide each original cell into `factor` equal subcells (factor>=1).
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
            sub = np.linspace(a, b, factor + 1, endpoint=False)[1:]
            pieces.append(sub)
        pieces.append(np.array([b]))
    out = np.concatenate(pieces)
    return _unique_increasing(out)


def shock_focused_grid(  # generic "window around idx" refinement; public name kept
    r: np.ndarray,
    shock_idx: int,
    buffer_cells: int = 15,
    refine_factor: int = 4,
) -> np.ndarray:
    """
    Keep original resolution outside [idx - buffer, idx + buffer],
    and refine uniformly inside that window by `refine_factor`.
    """
    r = _unique_increasing(r)
    n = len(r)
    if n == 0:
        return r.copy()

    i0 = max(0, min(n - 1, shock_idx - int(buffer_cells)))
    i1 = max(0, min(n - 1, shock_idx + int(buffer_cells)))

    if i0 >= i1:
        return uniform_refined_grid(r, factor=max(1, int(refine_factor)))

    left  = r[: i0 + 1]
    mid   = r[i0 : i1 + 1]
    right = r[i1 :]

    mid_ref = uniform_refined_grid(mid, factor=max(1, int(refine_factor)))
    merged = np.concatenate([left, mid_ref[1:], right[1:]])
    return _unique_increasing(merged)


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
    Interpolate the full background dict onto r_new (1D). Returns a new dict
    with the same keys. Uses canonical 'c_sound_squared' (not 'cs2').
    IMPORTANT: does NOT set/overwrite 'iR' (that name is reserved for the sonic index).
    """
    out: Dict[str, np.ndarray] = {}
    r = _unique_increasing(np.asarray(bg["r"], dtype=np.float64))
    r_new = _unique_increasing(np.asarray(r_new, dtype=np.float64))

    for k in _BG_KEYS:
        if k == "r":
            out["r"] = r_new
            continue
        if k not in bg:
            have = ", ".join(sorted(bg.keys()))
            raise KeyError(f"background missing key: '{k}'. Available: {have}")
        out[k] = _safe_interp(r, np.asarray(bg[k], dtype=np.float64), r_new, kind=kind)

    for k in _EXTRA_KEYS:
        if k in bg:
            out[k] = _safe_interp(r, np.asarray(bg[k], dtype=np.float64), r_new, kind=kind)

    out["nt"]   = int(bg.get("nt", 0))
    out["time"] = float(bg.get("time", 0.0))
    # DO NOT set out["iR"] here; use find_sonic_point() / GREAT to define it when needed.
    return out


def validate_bg(bg_old: Dict[str, np.ndarray], bg_new: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Quick sanity metrics:
      - max relative diff at original radii for a few fields (rho, p, c_sound_squared)
      - positivity checks (rho, p, c_sound_squared, alpha)
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
    }

    for key in ["rho", "p", "c_sound_squared", "alpha"]:
        if key in bg_new and np.any(np.asarray(bg_new[key]) < 0):
            metrics[f"neg_{key}"] = 1.0

    metrics["r_monotonic"] = 1.0 if (r1.size < 2 or np.all(np.diff(r1) > 0)) else 0.0
    return metrics


# ----------------------------- output writer ---------------------------------

def write_intermediate_h5(
    path: Path,
    bg_orig: Dict[str, np.ndarray],
    bg_interp: Dict[str, np.ndarray],
    sonic: Optional[SonicInfo] = None,
    kind: str = "linear",
) -> None:
    """
    Write HDF5 with two groups: /original and /interpolated
    Stores arrays as datasets and scalars as attributes.
    If h5py is unavailable, writes two .npz files as a fallback.
    """
    if _HAS_H5PY:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:

            def _dump_group(name: str, bg: Dict[str, np.ndarray]):
                g = f.create_group(name)
                for k, v in bg.items():
                    if isinstance(v, (int, float)):
                        g.attrs[k] = v
                    else:
                        arr = np.asarray(v)
                        g.create_dataset(
                            k,
                            data=arr,
                            compression="gzip",
                            compression_opts=3,
                            shuffle=True,
                        )
                return g

            _dump_group("original", bg_orig)
            _dump_group("interpolated", bg_interp)

            f.attrs["kind"] = kind
            if sonic is not None:
                f.attrs["sonic_idx"] = int(sonic.idx)
                f.attrs["sonic_r"]   = float(sonic.r)
                f.attrs["sonic_method"] = sonic.method
                # Back-compat for any helper that still reads shock_*:
                f.attrs["shock_idx"] = int(sonic.idx)
                f.attrs["shock_r"]   = float(sonic.r)
                f.attrs["shock_metric"] = sonic.method
    else:
        base = Path(path).with_suffix("")
        np.savez_compressed(
            str(base) + "_original.npz",
            **{k: np.asarray(v) for k, v in bg_orig.items()},
        )
        np.savez_compressed(
            str(base) + "_interpolated.npz",
            **{k: np.asarray(v) for k, v in bg_interp.items()},
        )
