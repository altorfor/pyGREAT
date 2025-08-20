# scripts/readers/obergaulinger_stream.py
from pathlib import Path
from pygreat.io.obergaulinger import read_coco2d_evol
from pygreat.io.base import from_generator

def obergaulinger_reader(input_file: str, nt_ini=0, nt_last=None, nt_step=1):
    data = read_coco2d_evol(Path(input_file))
    if nt_last is None:
        nt_last = data.num_t - 1
    for n in range(nt_ini, min(nt_last, data.num_t - 1) + 1, nt_step):
        bg_full = data.to_bg(n)  # your function (Fortran-style names)
        # Yield nt, time, bg-dict
        yield int(bg_full["nt"]), float(bg_full["time"]), bg_full

def make_reader(input_file: str, **kws):
    return from_generator(obergaulinger_reader(input_file, **kws))
