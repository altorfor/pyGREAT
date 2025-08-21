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

def _ensure_del(grp: h5py.Group, name: str) -> None:
    if name in grp:
        del grp[name]

def _write_scalar(grp: h5py.Group, name: str, value):
    dname = _ascii_name(name)
    _ensure_del(grp, dname)
    ds = grp.create_dataset(dname, data=np.asarray(value))
    if dname != name:  # store the exact ASCII label too
        ds.attrs["ascii_label"] = name
    return ds

def _write_1d(grp: h5py.Group, name: str, arr, dtype=np.float64):
    a = np.asarray(arr, dtype=dtype)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {a.shape}")
    dname = _ascii_name(name)
    _ensure_del(grp, dname)
    chunks = (min(a.size, 4096),) if a.size > 0 else None
    ds = grp.create_dataset(
        dname, data=a, dtype=dtype,
        compression="gzip", compression_opts=4,
        shuffle=True, chunks=chunks
    )
    if dname != name:
        ds.attrs["ascii_label"] = name
    return ds

def _link_alias(grp: h5py.Group, alias: str, target_ds: h5py.Dataset) -> None:
    """Create a hard link alias -> target_ds (no data copy)."""
    alias = _ascii_name(alias)
    _ensure_del(grp, alias)
    grp[alias] = target_ds  # hard link

def _write_group_attrs(grp: h5py.Group, **attrs):
    for k, v in attrs.items():
        grp.attrs[str(k)] = v

# ---------- canonical mapping ----------
# Each entry: (canonical_name, pretty_alias, src_key_in_bg_out, units_attr_or_None)
_BG_CANON = [
    ("r",                "r [cm]"       , "r"      , "cm"),
    ("rho",              "rho [cm^-2]"  , "rho"    , "cm^-2"),
    ("eps",              "eps"          , "eps"    , None),
    ("p",                "p [cm^-1]"    , "p"      , "cm^-1"),
    ("c_sound_squared",  "cs^2"         , "cs2"    , None),   # src uses 'cs2'
    ("phi",              "phi"          , "phi"    , None),
    ("alpha",            "alpha"        , "alpha"  , None),
    ("h",                "h"            , "h"      , None),
    ("gamma1",           "Gamma_1"      , "gamma1" , None),
    ("ggrav",            "G [cm^-1]"    , "ggrav"  , "cm^-1"),
    ("n2",               "N^2 [cm^-2]"  , "n2"     , "cm^-2"),
    ("lamb2",            "L^2 [cm^-2]"  , "lamb2"  , "cm^-2"),
    ("Bstar",            "B [cm^-1]"    , "Bstar"  , "cm^-1"),
    ("Qcap",             "Q"            , "Qcap"   , None),
    ("inv_cs2",          "1/cs^2"       , "inv_cs2", None),
    ("mass_v",           "M [cm]"       , "mass_v" , "cm"),
    ("q",                "q [cm^-2]"    , "q"      , "cm^-2"),
    ("e",                "e [cm^-2]"    , "e"      , "cm^-2"),
]

# optional extras, if provided by the reader
_BG_EXTRAS = [
    ("v_1",   "v_1", "v_1", None),
    ("Y_e",   "Y_e", "Y_e", None),
    ("T",     "T",   "T",   None),
    ("s",     "s",   "s",   None),
    ("U",     "U",   "U",   None),
]

# ---------- background ----------
def write_background_group(Fbg: h5py.File, bg_out: dict, extras: dict | None = None):
    """
    Write one nt group in background.h5 with CANONICAL names and pretty aliases.
    Scalars written as both attrs and scalar datasets (canonical + pretty alias).
    """
    nt   = int(bg_out["nt"])
    time = float(bg_out["time"])
    iR   = int(bg_out["iR"])
    gbg  = Fbg.require_group(_nt_key(nt))

    # attrs
    _write_group_attrs(gbg, nt=nt, time=time, iR=iR)

    # scalar datasets: canonical + alias
    ds_time = _write_scalar(gbg, "time", time)
    _link_alias(gbg, "time [s]", ds_time)
    _write_scalar(gbg, "nt", nt)

    # core fields
    for canon, alias, src, units in _BG_CANON:
        if src in bg_out:
            ds = _write_1d(gbg, canon, bg_out[src][:iR])
            if units:
                ds.attrs["units"] = units
            if alias != canon:
                _link_alias(gbg, alias, ds)

    # optional extras
    if extras:
        for canon, alias, src, units in _BG_EXTRAS:
            if src in extras:
                a = np.asarray(extras[src])
                if a.ndim == 1 and a.size >= iR:
                    ds = _write_1d(gbg, canon, a[:iR])
                    if units:
                        ds.attrs["units"] = units
                    if alias != canon:
                        _link_alias(gbg, alias, ds)

# ---------- eigen + freqs ----------
def _ensure_flat_freqs_groups(Ffrq: h5py.File):
    """Make sure /flat datasets exist and are resizable for easy pandas reads."""
    g = Ffrq.require_group("flat")
    # create resizable 1D datasets if absent
    for name, dtype in (("nt", np.int64), ("time", np.float64),
                        ("mode", np.int64), ("freq", np.float64)):
        if name not in g:
            g.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype,
                             compression="gzip", compression_opts=3, shuffle=True)
    return g

def _append_flat_freqs(gflat: h5py.Group, nt: int, time: float, freqs: np.ndarray):
    n_old = gflat["freq"].shape[0]
    n_add = int(len(freqs))
    if n_add == 0:
        return
    for name in ("nt","time","mode","freq"):
        ds = gflat[name]
        ds.resize((n_old + n_add,))
    gflat["nt"][n_old:]   = nt
    gflat["time"][n_old:] = time
    gflat["mode"][n_old:] = np.arange(1, n_add + 1, dtype=np.int64)
    gflat["freq"][n_old:] = np.asarray(freqs, dtype=np.float64)

def write_eigen_for_nt(Feig: h5py.File, Ffrq: h5py.File, sess, nt: int, time: float,
                       pfrac: float | None = None, gfrac: float | None = None):
    """
    - eigen.h5: /{nt}/mode_0001,... with canonical names + pretty aliases:
        canonical: r, eta_r, eta_perp_over_r, delta_P, delta_Q, delta_psi, K, Psi
        aliases:   'r [cm]', 'eta_perp/r', etc.
    - freqs.h5:
        * per-nt group: /{nt}/freq (alias 'freq [Hz]') + scalars
        * flat table at /flat: nt, time, mode, freq (append-only)
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

    # freqs.h5 datasets (canonical + alias)
    ds_time = _write_scalar(Gf, "time", time)
    _link_alias(Gf, "time [s]", ds_time)
    _write_scalar(Gf, "nt", nt)
    _write_scalar(Gf, "ne", int(len(freqs)))
    if pfrac is not None: _write_scalar(Gf, "pfrac", float(pfrac))
    if gfrac is not None: _write_scalar(Gf, "gfrac", float(gfrac))
    ds_f = _write_1d(Gf, "freq", freqs)
    _link_alias(Gf, "freq [Hz]", ds_f)

    # also append to flat table
    gflat = _ensure_flat_freqs_groups(Ffrq)
    _append_flat_freqs(gflat, nt, time, freqs)

    # eigen.h5 per-mode groups
    for idx, (k, f) in enumerate(pairs, start=1):
        m  = sess.copy_mode(k)
        iR = int(m["iR"])
        Gj = Ge.require_group(_mode_key(idx))
        _write_group_attrs(Gj, idx=idx, nt=nt, time=time, freq_Hz=float(f))

        # scalars canonical + alias
        ds_time = _write_scalar(Gj, "time", time)
        _link_alias(Gj, "time [s]", ds_time)
        _write_scalar(Gj, "nt", nt)
        ds_f = _write_scalar(Gj, "freq", float(f))
        _link_alias(Gj, "freq [Hz]", ds_f)

        # canonical arrays + pretty aliases
        ds_r   = _write_1d(Gj, "r",              np.asarray(m["r"])[:iR])
        _link_alias(Gj, "r [cm]", ds_r)
        _write_1d(Gj, "eta_r",           np.asarray(m["eta_r"])[:iR])
        ds_ep  = _write_1d(Gj, "eta_perp_over_r", np.asarray(m["eta_perp_over_r"])[:iR])
        _link_alias(Gj, "eta_perp∕r", ds_ep)
        _write_1d(Gj, "delta_P",         np.asarray(m["delta_P"])[:iR])
        _write_1d(Gj, "delta_Q",         np.asarray(m["delta_Q"])[:iR])
        _write_1d(Gj, "delta_psi",       np.asarray(m["delta_psi"])[:iR])
        _write_1d(Gj, "K",               np.asarray(m["K"])[:iR])
        _write_1d(Gj, "Psi",             np.asarray(m["Psi"])[:iR])

# ---------- file init / append ----------
def init_h5(bg_h5: Path, eig_h5: Path, frq_h5: Path):
    for p in (bg_h5, eig_h5, frq_h5):
        p = Path(p)
        if p.exists():
            p.unlink()
    with h5py.File(bg_h5, "w") as F:
        F.attrs["schema"] = "pygreat.background.v2"  # canonical + aliases
    with h5py.File(eig_h5, "w") as F:
        F.attrs["schema"] = "pygreat.eigen.v2"
    with h5py.File(frq_h5, "w") as F:
        F.attrs["schema"] = "pygreat.freqs.v2"

def append_one_step(bg_h5: Path, eig_h5: Path, frq_h5: Path, sess, bg_out: dict,
                    extras: dict | None = None, pfrac: float | None = None, gfrac: float | None = None):
    nt   = int(bg_out["nt"])
    time = float(bg_out["time"])
    with h5py.File(bg_h5, "a") as Fbg:
        write_background_group(Fbg, bg_out, extras=extras)
    with h5py.File(eig_h5, "a") as Fe, h5py.File(frq_h5, "a") as Ff:
        write_eigen_for_nt(Fe, Ff, sess, nt, time, pfrac=pfrac, gfrac=gfrac)

def inspect_h5_tree(path: Path, max_children=8):
    print(f"\n[{Path(path).name}]")
    with h5py.File(path, "r") as F:
        print(" root attrs:", dict(F.attrs))
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
