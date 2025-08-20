#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Make 'pygreat' importable when running directly from repo
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pygreat.core import PyGreatSession
from pygreat.io.obergaulinger import read_coco2d_evol
from pygreat.io.h5writer import init_h5, append_one_step, inspect_h5_tree

def parse_parfile(path: Path) -> dict:
    params = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"): continue
        if "#" in line: line = line.split("#", 1)[0].strip()
        if "=" not in line: continue
        k, v = [s.strip() for s in line.split("=", 1)]
        vl = v.lower()
        if vl in (".true.", "true"):   params[k] = True
        elif vl in (".false.", "false"): params[k] = False
        else:
            try: params[k] = int(v)
            except ValueError:
                try: params[k] = float(v.replace("d","e"))
                except ValueError: params[k] = v
    return params

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parfile", required=True, type=str)
    args = ap.parse_args()

    parfile = Path(args.parfile).resolve()
    params  = parse_parfile(parfile)

    input_dir  = Path(str(params["input_directory"])).expanduser().resolve()
    output_dir = Path(str(params["output_directory"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bg_h5  = output_dir / "background.h5"
    eig_h5 = output_dir / "eigen.h5"
    frq_h5 = output_dir / "freqs.h5"
    init_h5(bg_h5, eig_h5, frq_h5)

    # data reader
    coco_path = input_dir / "CoCo2d-evol.dat"
    data = read_coco2d_evol(coco_path)

    nt_ini  = int(params.get("nt_ini", 0))
    nt_last = min(int(params.get("nt_last", data.num_t - 1)), data.num_t - 1)
    nt_step = int(params.get("nt_step", 1))

    # GREAT session
    sess = PyGreatSession()
    sess.load_parameters(str(parfile))
    sess.reset_capture()

    pfrac = float(params.get("pfrac", 1.0))
    gfrac = float(params.get("gfrac", 1.0))

    prev_modes = 0
    for n in range(nt_ini, nt_last + 1, nt_step):
        bg_in = data.to_bg(n)  # includes extra fields (v_1, y_e, t, entropy, U)

        # run GREAT
        # Feed background and run analysis
        sess.set_background(int(bg_in["nt"]), float(bg_in["time"]), bg_in)
        sess.analyze()

        # Copy background back (up to iR) and define nt/t **before** printing
        bg_out = sess.copy_background()
        nt = int(bg_out["nt"])
        iR = int(bg_out["iR"])
        t  = float(bg_out["time"])

        # Collect new modes for this nt
        nm_total = sess.get_modes_count()
        new_modes = []
        for k in range(prev_modes + 1, nm_total + 1):
            md = sess.copy_mode(k)
            if md["nt"] == nt:
                new_modes.append(md)
        prev_modes = nm_total

        # Build list of freqs and print preview
        freqs = [md["freq"] for md in new_modes]
        print(
            f"nt={nt} time={t:.6f}s  freqs(Hz)[0:10]: "
            + ", ".join(f"{f:.3f}" for f in freqs[:10])
        )

        # Write HDF5
        append_one_step(bg_h5, eig_h5, frq_h5, sess, bg_out)

    # quick inspection
    inspect_h5_tree(bg_h5,  max_children=6)
    inspect_h5_tree(eig_h5, max_children=6)
    inspect_h5_tree(frq_h5, max_children=6)

if __name__ == "__main__":
    main()
