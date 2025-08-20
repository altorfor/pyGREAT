# pygreat/io/h5writer.py
from __future__ import annotations
import h5py
import numpy as np
from pathlib import Path

# ---------- helpers ----------
def _nt_key(nt: int) -> str:
    return f"{int(nt):08d}"

def _mode_key(idx: int) -> str:
    return f"mode_{int(idx):04d}"

def _ascii_name(name: str) -> str:
    # HDF5 forbids "/" in names; use Unicode division slash "∕" (U+2215)
    return name.replace("/", "∕")

def _write_scalar(grp: h5py.Group, name: str, value):
    dname = _ascii_name(name)
    if dname in grp:
        del grp[dname]
    ds = grp.create_dataset(dname, data=np.asarray(value))
    if dname != name:  # store the exact ASCII label too
        ds.attrs["ascii_label"] = name

def _write_1d(grp: h5py.Group, name: str, arr, dtype=np.float64):
    a = np.asarray(arr, dtype=dtype)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {a.shape}")
    dname = _ascii_name(name)
    if dname in grp:
        del grp[dname]
    chunks = (min(a.size, 4096),) if a.size > 0 else None
    ds = grp.create_dataset(
        dname, data=a, dtype=dtype,
        compression="gzip", compression_opts=4,
        shuffle=True, chunks=chunks
    )
    if dname != name:
        ds.attrs["ascii_label"] = name

def _write_group_attrs(grp: h5py.Group, **attrs):
    for k, v in attrs.items():
        grp.attrs[str(k)] = v

# ---------- background ----------
_BG_FIELDS = [
    ("r [cm]"        , "r"),
    ("rho [cm^-2]"   , "rho"),
    ("eps"           , "eps"),
    ("p [cm^-1]"     , "p"),
    ("cs^2"          , "cs2"),
    ("phi"           , "phi"),
    ("alpha"         , "alpha"),
    ("h"             , "h"),
    ("Gamma_1"       , "gamma1"),
    ("G [cm^-1]"     , "ggrav"),
    ("N^2 [cm^-2]"   , "n2"),
    ("L^2 [cm^-2]"   , "lamb2"),
    ("B [cm^-1]"     , "Bstar"),
    ("Q"             , "Qcap"),
    ("1/cs^2"        , "inv_cs2"),
    ("M [cm]"        , "mass_v"),
    ("q [cm^-2]"     , "q"),
    ("e [cm^-2]"     , "e"),
]

# optional extras, if provided by the reader
_BG_EXTRAS = [
    ("v_1"           , "v_1"),
    ("Y_e"           , "Y_e"),
    ("T"             , "T"),
    ("s"             , "s"),
    ("U"             , "U"),
]

def write_background_group(Fbg: h5py.File, bg_out: dict, extras: dict | None = None):
    """
    Write one nt group in background.h5 using ASCII-like names.
    Also writes scalar datasets 'time [s]' and 'nt'.
    """
    nt   = int(bg_out["nt"])
    time = float(bg_out["time"])
    iR   = int(bg_out["iR"])
    gbg  = Fbg.require_group(_nt_key(nt))

    # store as both attrs and scalar datasets (for "exact header" feel)
    _write_group_attrs(gbg, nt=nt, time=time, iR=iR)
    _write_scalar(gbg, "time [s]", time)
    _write_scalar(gbg, "nt", nt)

    # core fields
    for label, key in _BG_FIELDS:
        if key in bg_out:
            _write_1d(gbg, label, bg_out[key][:iR])

    # optional extras
    if extras:
        for label, key in _BG_EXTRAS:
            if key in extras:
                a = np.asarray(extras[key])
                if a.ndim == 1 and a.size >= iR:
                    _write_1d(gbg, label, a[:iR])

# ---------- eigen + freqs ----------
def write_eigen_for_nt(Feig: h5py.File, Ffrq: h5py.File, sess, nt: int, time: float, pfrac: float | None = None, gfrac: float | None = None):
    """
    Write eigenfunctions and freqs for ONE nt, using ASCII-like names.
    - eigen.h5: /{nt}/mode_0001,... each with datasets:
        'r [cm]', 'eta_r', 'eta_perp/r', 'delta P', 'delta Q', 'delta psi', 'K', 'Psi'
        and scalar datasets 'time [s]', 'nt', 'freq [Hz]'
    - freqs.h5: /{nt}/{'freq [Hz]'} + scalars 'time [s]','nt','ne','pfrac','gfrac'
    """
    Ge = Feig.require_group(_nt_key(nt))
    _write_group_attrs(Ge, nt=nt, time=time)

    Gf = Ffrq.require_group(_nt_key(nt))
    _write_group_attrs(Gf, nt=nt, time=time)

    # collect modes for this nt
    nm = sess.get_modes_count()
    pairs = []
    for k in range(1, nm + 1):
        m = sess.copy_mode(k)
        if int(m["nt"]) == nt:
            pairs.append((k, float(m["freq"])))
    freqs = np.array([f for _, f in pairs], dtype=np.float64)

    # freqs.h5 datasets
    _write_scalar(Gf, "time [s]", time)
    _write_scalar(Gf, "nt", nt)
    _write_scalar(Gf, "ne", int(len(freqs)))
    if pfrac is not None: _write_scalar(Gf, "pfrac", float(pfrac))
    if gfrac is not None: _write_scalar(Gf, "gfrac", float(gfrac))
    _write_1d(Gf, "freq [Hz]", freqs)

    # eigen.h5 per-mode groups
    for idx, (k, f) in enumerate(pairs, start=1):
        m  = sess.copy_mode(k)
        iR = int(m["iR"])
        Gj = Ge.require_group(_mode_key(idx))
        _write_group_attrs(Gj, idx=idx, nt=nt, time=time, freq_Hz=float(f))
        _write_scalar(Gj, "time [s]", time)
        _write_scalar(Gj, "nt", nt)
        _write_scalar(Gj, "freq [Hz]", float(f))

        # field names as in ASCII (with ∕ for slash)
        _write_1d(Gj, "r [cm]",             np.asarray(m["r"])[:iR])
        _write_1d(Gj, "eta_r",              np.asarray(m["eta_r"])[:iR])
        _write_1d(Gj, "eta_perp/r",         np.asarray(m["eta_perp_over_r"])[:iR])  # gets 'eta_perp∕r' in HDF5
        _write_1d(Gj, "delta P",            np.asarray(m["delta_P"])[:iR])
        _write_1d(Gj, "delta Q",            np.asarray(m["delta_Q"])[:iR])
        _write_1d(Gj, "delta psi",          np.asarray(m["delta_psi"])[:iR])
        _write_1d(Gj, "K",                  np.asarray(m["K"])[:iR])
        _write_1d(Gj, "Psi",                np.asarray(m["Psi"])[:iR])

# ---------- file init / append ----------
def init_h5(bg_h5: Path, eig_h5: Path, frq_h5: Path):
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

def append_one_step(bg_h5: Path, eig_h5: Path, frq_h5: Path, sess, bg_out: dict, extras: dict | None = None, pfrac: float | None = None, gfrac: float | None = None):
    nt   = int(bg_out["nt"])
    time = float(bg_out["time"])
    with h5py.File(bg_h5, "a") as Fbg:
        write_background_group(Fbg, bg_out, extras=extras)
    with h5py.File(eig_h5, "a") as Fe, h5py.File(frq_h5, "a") as Ff:
        write_eigen_for_nt(Fe, Ff, sess, nt, time, pfrac=pfrac, gfrac=gfrac)

def inspect_h5_tree(path: Path, max_children=8):
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
