# pygreat/pygreat.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from .core import PyGreatSession
from .io.obergaulinger import read_coco2d_evol
from .io.h5writer import init_h5, append_one_step, inspect_h5_tree
from .params import parse_parameters

# --- interpolation helpers (import + safe local fallback for interior refine) ---
from .interp import (
    uniform_refined_grid,
    interpolate_bg,
    find_sonic_point,
)

try:
    # prefer the version in interp.py if present
    from .interp import interior_uniform_grid  # type: ignore
except Exception:
    # minimal inside-only refinement fallback
    def interior_uniform_grid(
        r: np.ndarray,
        shock_idx: int,
        buffer_cells: int = 15,
        inner_ncells: int = 400,
    ) -> np.ndarray:
        r = np.asarray(r, float).ravel()
        if r.size == 0:
            return r.copy()
        n = r.size
        j_end = int(min(n - 1, max(0, int(shock_idx) + int(buffer_cells))))
        if j_end < 1:
            return r.copy()
        inner_ncells = max(2, int(inner_ncells))
        inner = np.linspace(r[0], r[j_end], inner_ncells, endpoint=True)
        tail = r[j_end + 1 :] if (j_end + 1) < n else np.empty(0, dtype=r.dtype)
        return np.concatenate([inner, tail]) if tail.size else inner


# ---------------------------------------------------------------------------------
# Parameter parsing (keep it local so run_pygreat.py stays untouched)
# ---------------------------------------------------------------------------------

def _lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in d.items()}

def _require(key: str, P: Dict[str, Any]):
    if key not in P:
        raise KeyError(f"Parameter '{key}' missing in parfile")
    return P[key]

@dataclass
class InterpConfig:
    enabled: bool
    mode: str                 # "uniform" or "shock" (inside-only)
    kind: str                 # "linear", "pchip", "nearest", "loglinear", "cubic", "akima"
    factor: int               # for mode == "uniform"
    buffer: int               # cells AFTER iR (inside-only)
    n_inner: int              # total points from r[0]..r[iR+buffer] (incl endpoints)

def _to_bool(x: Any) -> bool:
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    return s in {".true.", "true", "t", "1", "yes", "on"}

def _interp_config(P: Dict[str, Any]) -> InterpConfig:
    Pl = _lower_keys(P)
    enabled = _to_bool(Pl.get("interpolation", False))
    mode = str(Pl.get("interp_mode", "uniform")).strip().lower()
    if mode not in {"uniform", "shock", "interior"}:
        mode = "uniform"
    if mode == "interior":
        mode = "shock"  # alias

    kind = str(Pl.get("interp_kind", "linear")).strip().lower()
    factor = int(Pl.get("interp_factor", 2))
    buffer = int(Pl.get("interp_buffer", 15))       # cells AFTER iR
    n_inner = int(Pl.get("interp_n_inner", 400))    # total points inside
    return InterpConfig(enabled=enabled, mode=mode, kind=kind, factor=factor, buffer=buffer, n_inner=n_inner)

# ---------------------------------------------------------------------------------
# Background sonic index helpers
# ---------------------------------------------------------------------------------

def _get_ir_from_bg(bg: Dict[str, Any], m: int) -> int:
    """
    Use only background 'iR' (Fortran 1-based) if present; return 0-based idx.
    """
    for k in ("iR", "ir", "IR"):
        if k in bg:
            val = int(np.asarray(bg[k]).item())
            return max(0, min(m - 1, val - 1))
    raise KeyError("Background does not provide 'iR' (sonic index).")

# ---------------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------------

def run_from_parfile(parfile: str | Path, *, inspect_tree: bool = False) -> None:
    """
    Simple pipeline:
      - parse GREAT parameter file
      - choose reader (input_mode)
      - optional interpolation (uniform or inside-only up to iR+buffer)
      - run GREAT
      - write background.h5, eigen.h5, freqs.h5
    """
    parfile = Path(parfile).resolve()
    if not parfile.exists():
        raise FileNotFoundError(parfile)

    P  = parse_parameters(parfile)
    Pl = _lower_keys(P)
    cfg = _interp_config(P)

    # I/O paths
    input_dir  = Path(str(_require("input_directory", Pl))).expanduser().resolve()
    output_dir = Path(str(_require("output_directory", Pl))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Outputs
    bg_h5  = output_dir / "background.h5"
    eig_h5 = output_dir / "eigen.h5"
    frq_h5 = output_dir / "freqs.h5"
    init_h5(bg_h5, eig_h5, frq_h5)

    # Time selection
    nt_ini  = int(Pl.get("nt_ini", 0))
    nt_last = int(Pl.get("nt_last", 10**9))
    nt_step = max(1, int(Pl.get("nt_step", 1)))

    # Reader (mode 3 = Obergaulinger)
    input_mode = int(Pl.get("input_mode", 3))
    if input_mode == 3:
        data = read_coco2d_evol(input_dir / "CoCo2d-evol.dat")
        to_bg = data.to_bg
        n_max = data.num_t - 1
    else:
        raise NotImplementedError(f"input_mode={input_mode} not implemented in this runner")

    nt_last = min(nt_last, n_max)

    # GREAT session
    sess = PyGreatSession()
    sess.load_parameters(str(parfile))
    sess.reset_capture()

    # optional scalars for freqs.h5
    pfrac = Pl.get("pfrac", None)
    gfrac = Pl.get("gfrac", None)

    # Loop
    for n in range(nt_ini, nt_last + 1, nt_step):
        bg_in = to_bg(n)  # expected to have keys: r, rho, eps, p, c_sound_squared, phi, alpha, v_1, and nt,time; iR optional

        # ---- Interpolation (if enabled) ----
        if cfg.enabled:
            r = np.asarray(bg_in["r"], float)

            if cfg.mode == "uniform":
                r_new = uniform_refined_grid(r, factor=max(1, int(cfg.factor)))

            else:  # inside-only: uniform from r[0]..r[iR+buffer]; keep tail unchanged
                try:
                    idx0 = _get_ir_from_bg(bg_in, m=len(r))
                except KeyError:
                    # Only compute if needed for interpolation
                    v1  = np.asarray(bg_in["v_1"], float)
                    cs2 = np.asarray(bg_in["c_sound_squared"], float)
                    si  = find_sonic_point(r, v1, cs2)
                    idx0 = int(si.idx)

                r_new = interior_uniform_grid(
                    r,
                    shock_idx=idx0,
                    buffer_cells=int(cfg.buffer),
                    inner_ncells=int(cfg.n_inner),
                )

            bg_use = interpolate_bg(bg_in, r_new, kind=str(cfg.kind))

        else:
            bg_use = bg_in

        # ---- Extras for HDF5 (ASCII-like names preserved by writer) ----
        extras = {}
        if "y_e" in bg_use:      extras["Y_e"] = np.asarray(bg_use["y_e"])
        if "t" in bg_use:        extras["T"]   = np.asarray(bg_use["t"])
        if "entropy" in bg_use:  extras["s"]   = np.asarray(bg_use["entropy"])
        if "U" in bg_use:        extras["U"]   = np.asarray(bg_use["U"])
        if "v_1" in bg_use:      extras["v_1"] = np.asarray(bg_use["v_1"])

        # ---- GREAT ----
        sess.set_background(int(bg_use["nt"]), float(bg_use["time"]), bg_use)
        sess.analyze()

        bg_out = sess.copy_background()

        append_one_step(bg_h5, eig_h5, frq_h5, sess, bg_out,
                        extras=extras, pfrac=pfrac, gfrac=gfrac)

        # Quick console summary of first 10 freqs at this nt
        nm_total = sess.get_modes_count()
        freqs_nt = []
        for k in range(1, nm_total + 1):
            md = sess.copy_mode(k)
            if int(md["nt"]) == int(bg_out["nt"]):
                freqs_nt.append(float(md["freq"]))
        freqs_nt.sort()
        print(
            f"nt={bg_out['nt']} time={bg_out['time']:.6f}s  freqs(Hz)[0:10]: "
            + ", ".join(f"{f:.3f}" for f in freqs_nt[:10])
        )

    if inspect_tree:
        inspect_h5_tree(bg_h5)
        inspect_h5_tree(eig_h5)
        inspect_h5_tree(frq_h5)
