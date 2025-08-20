# pygreat/pygreat.py
from __future__ import annotations

from pathlib import Path
import numpy as np

from .core import PyGreatSession
from .io.obergaulinger import read_coco2d_evol
from .io.h5writer import init_h5, append_one_step, inspect_h5_tree


# --- local minimal parser (do NOT depend on other modules) ------------------
_TRUE  = {".true.", "true", "t", "1", "yes", "on"}
_FALSE = {".false.", "false", "f", "0", "no", "off"}

def _coerce_token(s: str):
    s = s.strip()
    if not s:
        return s
    sl = s.lower()
    if sl in _TRUE:  return True
    if sl in _FALSE: return False
    # Fortran 'd' exponents → 'e'
    if "d" in sl or "e" in sl:
        try:
            return float(sl.replace("d", "e"))
        except ValueError:
            pass
    # try int, else raw string (strip quotes)
    try:
        return int(s, 10)
    except ValueError:
        pass
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def _parse_parfile(parfile: Path) -> dict:
    out = {}
    for raw in parfile.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        k, v = (p.strip() for p in line.split("=", 1))
        out[k] = _coerce_token(v)
    return out


# --- main pipeline ----------------------------------------------------------
def run_from_parfile(parfile: str | Path, *, inspect_tree: bool = False) -> None:
    """
    Simple, clean pipeline:
      - read GREAT parameter file
      - choose reader (input_mode)
      - run GREAT for the requested time range
      - write background.h5, eigen.h5, freqs.h5 (ASCII-like dataset names)
    """
    parfile = Path(parfile).resolve()
    if not parfile.exists():
        raise FileNotFoundError(parfile)

    P = _parse_parfile(parfile)

    # Required I/O paths from the parameter file
    input_dir  = Path(str(P["input_directory"])).expanduser().resolve()
    output_dir = Path(str(P["output_directory"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output files
    bg_h5  = output_dir / "background.h5"
    eig_h5 = output_dir / "eigen.h5"
    frq_h5 = output_dir / "freqs.h5"
    init_h5(bg_h5, eig_h5, frq_h5)

    # Time range
    # (defaults: all available; we'll clamp below once we know num_t)
    nt_ini  = int(P.get("nt_ini", 0))
    nt_last = int(P.get("nt_last", 10**9))
    nt_step = int(P.get("nt_step", 1))

    # Choose and load the input reader
    input_mode = int(P.get("input_mode", 3))
    if input_mode == 3:
        # Obergaulinger CoCo2d-evol.dat in input_directory
        coco_path = input_dir / "CoCo2d-evol.dat"
        data = read_coco2d_evol(coco_path)
        to_bg = data.to_bg
        n_max = data.num_t - 1
    else:
        raise NotImplementedError(f"input_mode={input_mode} is not implemented in this simple runner")

    nt_last = min(nt_last, n_max)

    # Start GREAT
    sess = PyGreatSession()
    sess.load_parameters(str(parfile))
    sess.reset_capture()

    # For freqs.h5 (optional scalars)
    pfrac = P.get("pfrac", None)
    gfrac = P.get("gfrac", None)

    # Loop over time indices
    prev_modes = 0
    for n in range(nt_ini, nt_last + 1, nt_step):
        bg_in = to_bg(n)

        # Map optional extras for background writer (ASCII names):
        # 'Y_e' (from 'y_e'), 'T' (from 't'), 's' (from 'entropy'), 'U' (from 'U'), and 'v_1'
        extras = {}
        if "y_e" in bg_in:      extras["Y_e"] = np.asarray(bg_in["y_e"])
        if "t" in bg_in:        extras["T"]   = np.asarray(bg_in["t"])
        if "entropy" in bg_in:  extras["s"]   = np.asarray(bg_in["entropy"])
        if "U" in bg_in:        extras["U"]   = np.asarray(bg_in["U"])
        if "v_1" in bg_in:      extras["v_1"] = np.asarray(bg_in["v_1"])

        # Feed background to GREAT and analyze
        sess.set_background(int(bg_in["nt"]), float(bg_in["time"]), bg_in)
        sess.analyze()

        # Copy processed background (has iR + derived quantities)
        bg_out = sess.copy_background()

        # Append one step to all three HDF5 files
        append_one_step(bg_h5, eig_h5, frq_h5, sess, bg_out, extras=extras, pfrac=pfrac, gfrac=gfrac)

        # Console: show first 10 frequencies for this nt (for quick sanity)
        nm_total = sess.get_modes_count()
        freqs_nt = []
        for k in range(1, nm_total + 1):
            md = sess.copy_mode(k)
            if int(md["nt"]) == int(bg_out["nt"]):
                freqs_nt.append(float(md["freq"]))
        freqs_nt.sort()
        print(f"nt={bg_out['nt']} time={bg_out['time']:.6f}s  freqs(Hz)[0:10]: " +
              ", ".join(f"{f:.3f}" for f in freqs_nt[:10]))

    if inspect_tree:
        # quick tree summary of outputs
        inspect_h5_tree(bg_h5)
        inspect_h5_tree(eig_h5)
        inspect_h5_tree(frq_h5)
