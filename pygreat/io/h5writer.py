# pygreat/io/h5writer.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import h5py

# Pull the shared primitives from one place
from .hdf5_utils import (
    nt_key,
    mode_key,
    ascii_name,        # kept for completeness (used by write_* internally)
    write_scalar,
    write_1d,
    set_group_attrs,
)

__all__ = [
    "init_h5",
    "write_background_group",
    "write_eigen_for_nt",
    "append_one_step",
    "inspect_h5_tree",
]

# ---------- background field maps (ASCII-like labels → keys) ------------------

_BG_FIELDS = [
    ("r [cm]"      , "r"),
    ("rho [cm^-2]" , "rho"),
    ("eps"         , "eps"),
    ("p [cm^-1]"   , "p"),
    ("cs^2"        , "cs2"),
    ("phi"         , "phi"),
    ("alpha"       , "alpha"),
    ("h"           , "h"),
    ("Gamma_1"     , "gamma1"),
    ("G [cm^-1]"   , "ggrav"),
    ("N^2 [cm^-2]" , "n2"),
    ("L^2 [cm^-2]" , "lamb2"),
    ("B [cm^-1]"   , "Bstar"),
    ("Q"           , "Qcap"),
    ("1/cs^2"      , "inv_cs2"),
    ("M [cm]"      , "mass_v"),
    ("q [cm^-2]"   , "q"),
    ("e [cm^-2]"   , "e"),
]

# optional extras from the reader
_BG_EXTRAS = [
    ("v_1", "v_1"),
    ("Y_e", "Y_e"),
    ("T",   "T"),
    ("s",   "s"),
    ("U",   "U"),
]

# ---------- background writer -------------------------------------------------

def write_background_group(Fbg: h5py.File, bg_out: dict, extras: dict | None = None) -> None:
    """
    Write one nt group into background.h5 using ASCII-like labels.

    - Group path: /{nt:08d}
    - Scalars written both as attributes and scalar datasets:
        'time [s]', 'nt'
    - 1D arrays written with compression and chunking via shared helpers.
    """
    nt   = int(bg_out["nt"])
    time = float(bg_out["time"])
    iR   = int(bg_out["iR"])

    gbg = Fbg.require_group(nt_key(nt))

    # store attrs and duplicate as scalar datasets for ASCII parity
    set_group_attrs(gbg, nt=nt, time=time, iR=iR)
    write_scalar(gbg, "time [s]", time)
    write_scalar(gbg, "nt", nt)

    # core fields (truncate to iR)
    for label, key in _BG_FIELDS:
        if key in bg_out:
            arr = np.asarray(bg_out[key])[:iR]
            write_1d(gbg, label, arr)

    # optional extras (truncate to iR when present and 1D)
    if extras:
        for label, key in _BG_EXTRAS:
            if key in extras:
                a = np.asarray(extras[key])
                if a.ndim == 1 and a.size >= iR:
                    write_1d(gbg, label, a[:iR])

# ---------- eigen + freqs writers --------------------------------------------

def write_eigen_for_nt(
    Feig: h5py.File,
    Ffrq: h5py.File,
    sess,
    nt: int,
    time: float,
    pfrac: float | None = None,
    gfrac: float | None = None,
) -> None:
    """
    For a single time step:
      - eigen.h5: /{nt}/mode_0001, ... with ASCII-like dataset names
      - freqs.h5: /{nt}/freq [Hz] plus scalars {'time [s]','nt','ne','pfrac','gfrac'}
    """
    Ge = Feig.require_group(nt_key(nt))
    set_group_attrs(Ge, nt=nt, time=time)

    Gf = Ffrq.require_group(nt_key(nt))
    set_group_attrs(Gf, nt=nt, time=time)

    # collect modes for this nt
    nm = sess.get_modes_count()
    pairs = []
    for k in range(1, nm + 1):
        m = sess.copy_mode(k)
        if int(m["nt"]) == nt:
            pairs.append((k, float(m["freq"])))

    freqs = np.array([f for _, f in pairs], dtype=np.float64)

    # freqs.h5
    write_scalar(Gf, "time [s]", time)
    write_scalar(Gf, "nt", nt)
    write_scalar(Gf, "ne", int(len(freqs)))
    if pfrac is not None:
        write_scalar(Gf, "pfrac", float(pfrac))
    if gfrac is not None:
        write_scalar(Gf, "gfrac", float(gfrac))
    write_1d(Gf, "freq [Hz]", freqs)

    # eigen.h5 per-mode groups
    for idx, (k, f) in enumerate(pairs, start=1):
        m  = sess.copy_mode(k)
        iR = int(m["iR"])
        Gj = Ge.require_group(mode_key(idx))
        set_group_attrs(Gj, idx=idx, nt=nt, time=time, freq_Hz=float(f))

        write_scalar(Gj, "time [s]", time)
        write_scalar(Gj, "nt", nt)
        write_scalar(Gj, "freq [Hz]", float(f))

        # ASCII field names (with ∕ for slash handled by helpers)
        write_1d(Gj, "r [cm]",      np.asarray(m["r"])[:iR])
        write_1d(Gj, "eta_r",       np.asarray(m["eta_r"])[:iR])
        write_1d(Gj, "eta_perp/r",  np.asarray(m["eta_perp_over_r"])[:iR])
        write_1d(Gj, "delta P",     np.asarray(m["delta_P"])[:iR])
        write_1d(Gj, "delta Q",     np.asarray(m["delta_Q"])[:iR])
        write_1d(Gj, "delta psi",   np.asarray(m["delta_psi"])[:iR])
        write_1d(Gj, "K",           np.asarray(m["K"])[:iR])
        write_1d(Gj, "Psi",         np.asarray(m["Psi"])[:iR])

# ---------- file init / append / inspect -------------------------------------

def init_h5(bg_h5: Path, eig_h5: Path, frq_h5: Path) -> None:
    """Create/overwrite HDF5 files and set file-level schema tags."""
    for p in (bg_h5, eig_h5, frq_h5):
        p = Path(p)
        if p.exists():
            p.unlink()
    with h5py.File(bg_h5, "w") as F:
        F.attrs["schema"] = "pygreat.background.ascii_names.v1"
    with h5py.File(eig_h5, "w") as F:
        F.attrs["schema"] = "pygreat.eigen.ascii_names.v1"
    with h5py.File(frq_h5, "w") as F:
        F.attrs["schema"] = "pygreat.freqs.ascii_names.v1"

def append_one_step(
    bg_h5: Path,
    eig_h5: Path,
    frq_h5: Path,
    sess,
    bg_out: dict,
    extras: dict | None = None,
    pfrac: float | None = None,
    gfrac: float | None = None,
) -> None:
    """Append one processed time step to the three output files."""
    nt   = int(bg_out["nt"])
    time = float(bg_out["time"])
    with h5py.File(bg_h5, "a") as Fbg:
        write_background_group(Fbg, bg_out, extras=extras)
    with h5py.File(eig_h5, "a") as Fe, h5py.File(frq_h5, "a") as Ff:
        write_eigen_for_nt(Fe, Ff, sess, nt, time, pfrac=pfrac, gfrac=gfrac)

def inspect_h5_tree(path: Path, max_children: int = 8) -> None:
    """Quick, human-friendly tree printout for debugging."""
    print(f"\n[{Path(path).name}]")
    with h5py.File(path, "r") as F:
        print(" root keys:", sorted(F.keys()))
        for k in sorted(F.keys()):
            g = F[k]
            if isinstance(g, h5py.Group):
                print(f"  - {k}: group; attrs={dict(g.attrs)}; kids={len(g.keys())}")
                for j, ck in enumerate(sorted(g.keys())):
                    if j >= max_children:
                        print("     …")
                        break
                    node = g[ck]
                    kind = "group" if isinstance(node, h5py.Group) else "dataset"
                    extra = f", shape={node.shape}" if hasattr(node, "shape") else ""
                    print(f"     • {ck} ({kind}{extra})")
            else:
                print(f"  - {k}: dataset shape={g.shape}")
