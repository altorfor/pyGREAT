#!/usr/bin/env python3
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import h5py

# --- make `pygreat` importable when running from repo root ---
THIS = Path(__file__).resolve()
sys.path.append(str(THIS.parents[1]))

from pygreat import PyGreatSession
from pygreat.io.obergaulinger import read_coco2d_evol


# ---------------------------
# tiny parser for GREAT .par
# ---------------------------
def parse_parfile(path: Path) -> dict:
    d = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # strip trailing inline comments
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        k, v = [x.strip() for x in line.split("=", 1)]
        if not k:
            continue
        # booleans
        if v.lower() in (".true.", "true"):
            d[k] = True
            continue
        if v.lower() in (".false.", "false"):
            d[k] = False
            continue
        # ints
        try:
            d[k] = int(v)
            continue
        except ValueError:
            pass
        # floats
        try:
            d[k] = float(v)
            continue
        except ValueError:
            pass
        # strings (strip quotes)
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        d[k] = v
    return d


# ---------------------------
# HDF5 helpers
# ---------------------------
def write_1d(group: h5py.Group, name: str, arr: np.ndarray):
    """Create/overwrite a 1D dataset with gzip and shuffle."""
    if name in group:
        del group[name]
    dset = group.create_dataset(name, data=np.asarray(arr, dtype=np.float64),
                                compression="gzip", shuffle=True)
    return dset


def main():
    ap = argparse.ArgumentParser(description="Run GREAT from parfile and write background/eigen/freqs to HDF5.")
    ap.add_argument("--parfile", type=str, default="parameters", help="GREAT-style parameter file")
    args = ap.parse_args()

    parfile = Path(args.parfile).expanduser().resolve()
    if not parfile.exists():
        raise FileNotFoundError(f"parfile not found: {parfile}")

    # Parse params (Python-side for I/O decisions) and also load into GREAT
    P = parse_parfile(parfile)

    # Resolve I/O paths
    input_dir = Path(P.get("input_directory", ".")).expanduser().resolve()
    output_dir = Path(P.get("output_directory", "./output")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Input file: prefer explicit 'input_file', otherwise assume CoCo2d-evol.dat in input_directory
    input_file = P.get("input_file", None)
    if input_file:
        input_path = Path(input_file).expanduser()
        if not input_path.is_absolute():
            input_path = input_dir / input_path
    else:
        input_path = input_dir / "CoCo2d-evol.dat"
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input data not found: {input_path}")

    # Time range
    nt_ini = int(P.get("nt_ini", 0))
    nt_last = int(P.get("nt_last", 10**9))
    nt_step = int(P.get("nt_step", 1))

    # Load the data file
    data = read_coco2d_evol(input_path)

    # Clamp the time range to available slices
    nt_ini = max(0, nt_ini)
    nt_last = min(data.num_t - 1, nt_last)
    if nt_last < nt_ini:
        print("Nothing to do (empty time range).")
        return

    # Prepare GREAT session
    sess = PyGreatSession()
    rc = sess.load_parameters(str(parfile))
    if rc != 0:
        raise RuntimeError(f"GREAT Read_parameters() failed with ierr={rc}")

    sess.reset_capture()  # clear any old modes in the in-memory store

    # HDF5 outputs (open ONCE)
    bg_h5  = output_dir / "background.h5"
    eig_h5 = output_dir / "eigen.h5"
    fr_h5  = output_dir / "freqs.h5"

    with h5py.File(bg_h5, "w") as f_bg, \
         h5py.File(eig_h5, "w") as f_eig, \
         h5py.File(fr_h5, "w") as f_fr:

        gbg  = f_bg.require_group("background")
        geig = f_eig.require_group("eigen")
        gfr  = f_fr.require_group("freqs")

        # Loop over times
        for n in range(nt_ini, nt_last + 1, nt_step):
            # Build background dict (cgs -> GREAT-ready) and send to GREAT
            bg = data.to_bg(n)  # returns dict with r, rho, eps, p, cs2, phi, alpha, v_1, etc.

            rc = sess.set_background(int(bg["nt"]), float(bg["time"]), bg)
            if rc != 0:
                print(f"[WARN] set_background failed at nt={n} (ierr={rc}); skipping")
                continue

            rc = sess.analyze_current()
            if rc != 0:
                print(f"[WARN] analyze_current failed at nt={n} (ierr={rc}); skipping")
                continue

            # Pull background (post-processed by GREAT; includes iR and derived fields)
            bg_out = sess.get_background()
            iR = int(bg_out.get("iR", 0))
            if iR <= 0:
                print(f"[WARN] nt={n} produced iR={iR}; skipping writes")
                continue

            # ---- background: /background/000000nn/...
            gbg_nt = gbg.require_group(f"{bg_out['nt']:08d}")
            gbg_nt.attrs["time"] = float(bg_out["time"])
            gbg_nt.attrs["nt"]   = int(bg_out["nt"])
            gbg_nt.attrs["iR"]   = iR

            write_1d(gbg_nt, "r",       bg_out["r"][:iR])
            write_1d(gbg_nt, "rho",     bg_out["rho"][:iR])
            write_1d(gbg_nt, "eps",     bg_out["eps"][:iR])
            write_1d(gbg_nt, "p",       bg_out["p"][:iR])
            write_1d(gbg_nt, "cs2",     bg_out["cs2"][:iR])
            write_1d(gbg_nt, "phi",     bg_out["phi"][:iR])
            write_1d(gbg_nt, "alpha",   bg_out["alpha"][:iR])
            write_1d(gbg_nt, "h",       bg_out["h"][:iR])
            write_1d(gbg_nt, "q",       bg_out["q"][:iR])
            write_1d(gbg_nt, "e",       bg_out["e"][:iR])
            write_1d(gbg_nt, "gamma1",  bg_out["gamma1"][:iR])
            write_1d(gbg_nt, "g",       bg_out["ggrav"][:iR])
            write_1d(gbg_nt, "n2",      bg_out["n2"][:iR])
            write_1d(gbg_nt, "l2",      bg_out["lamb2"][:iR])
            write_1d(gbg_nt, "B",       bg_out["Bstar"][:iR])
            write_1d(gbg_nt, "Q",       bg_out["Qcap"][:iR])
            write_1d(gbg_nt, "inv_cs2", bg_out["inv_cs2"][:iR])
            write_1d(gbg_nt, "M",       bg_out["mass_v"][:iR])

            # ---- eigenmodes: /eigen/000000nn/000m/
            modes = sess.get_modes()
            geig_nt = geig.require_group(f"{bg_out['nt']:08d}")
            geig_nt.attrs["time"] = float(bg_out["time"])
            geig_nt.attrs["nt"]   = int(bg_out["nt"])

            for idx, m in enumerate(modes, start=1):
                gk = geig_nt.require_group(f"{idx:04d}")
                gk.attrs["freq_Hz"] = float(m["freq"])
                gk.attrs["iR"]      = int(m["iR"])
                write_1d(gk, "r",                 m["r"])
                write_1d(gk, "eta_r",             m["nr"])
                write_1d(gk, "eta_perp_over_r",   m["np_over_r"])
                write_1d(gk, "delta_P",           m["delta_p"])
                write_1d(gk, "delta_Q",           m["delta_Q"])
                write_1d(gk, "delta_psi",         m["delta_phi"])
                write_1d(gk, "K",                 m["k_cap"])
                write_1d(gk, "Psi",               m["phi_cap"])

            # ---- freqs summary: /freqs/000000nn/freq_Hz
            gfr_nt = gfr.require_group(f"{bg_out['nt']:08d}")
            gfr_nt.attrs["time"] = float(bg_out["time"])
            gfr_nt.attrs["nt"]   = int(bg_out["nt"])
            freqs = np.array([m["freq"] for m in modes], dtype=np.float64)
            if "freq_Hz" in gfr_nt:
                del gfr_nt["freq_Hz"]
            gfr_nt.create_dataset("freq_Hz", data=freqs, compression="gzip", shuffle=True)

            # Show first 10 freqs for quick sanity check
            if freqs.size > 0:
                to_show = ", ".join(f"{f:.3f}" for f in freqs[:10])
                print(f"nt={bg_out['nt']} time={bg_out['time']:.6f}s  freqs(Hz)[0:10]: {to_show}")
            else:
                print(f"nt={bg_out['nt']} time={bg_out['time']:.6f}s  (no modes detected)")

    print(f"\nWrote:\n  {bg_h5}\n  {eig_h5}\n  {fr_h5}")


if __name__ == "__main__":
    main()
