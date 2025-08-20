# scripts/run_obergaulinger.py
import sys
from pathlib import Path

# add <repo>/python to sys.path so "import pygreat" works
THIS = Path(__file__).resolve()
PKG_ROOT = THIS.parents[1] / "python"
sys.path.insert(0, str(PKG_ROOT))  # ← key line

from pygreat import PyGreatSession
from pygreat.io.base import from_generator
from pygreat.io.obergaulinger import read_coco2d_evol

def obergaulinger_reader(input_file: str, nt_ini=0, nt_last=None, nt_step=1):
    data = read_coco2d_evol(input_file)
    if nt_last is None or nt_last >= data.num_t:
        nt_last = data.num_t - 1
    for n in range(nt_ini, nt_last + 1, nt_step):
        bg_full = data.to_bg(n)  # your tested mapping
        yield int(bg_full["nt"]), float(bg_full["time"]), bg_full  # (nt, time, bg)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parfile", required=True)
    ap.add_argument("--input",   required=True)
    ap.add_argument("--nt-ini",  type=int, default=0)
    ap.add_argument("--nt-last", type=int, default=None)
    ap.add_argument("--nt-step", type=int, default=1)
    args = ap.parse_args()

    sess = PyGreatSession()
    sess.load_parameters(args.parfile)

    reader = from_generator(obergaulinger_reader(
        args.input, nt_ini=args.nt_ini, nt_last=args.nt_last, nt_step=args.nt_step
    ))
    sess.analyze_timeseries(reader)

    modes = sess.fetch_modes()
    print(f"[pyGREAT] modes captured: {len(modes)}")
    if modes:
        m0 = modes[0]
        print(f"  first: nt={m0.nt} t={m0.time:.6g}s f={m0.freq:.6g}Hz iR={m0.r.size}")

    bg = sess.fetch_background()
    if bg:
        print(f"[pyGREAT] last background: nt={bg.nt} t={bg.time:.6g}s iR={bg.r.size}")

if __name__ == "__main__":
    main()
