# pygreat/io/base.py
from __future__ import annotations
from typing import Protocol, Iterable, Iterator, Tuple, Dict, Any
import numpy as np

# Canonical background schema expected by the shim (names kept short & stable)
#   r, rho, eps, p, cs2, phi, alpha, v1  -> all float64, shape (m,)
BG = Dict[str, np.ndarray]

class BackgroundReader(Protocol):
    """Any reader that yields (nt, time, canonical_bg) tuples."""
    def __iter__(self) -> Iterator[Tuple[int, float, BG]]: ...

def canonize_bg(bg: Dict[str, Any]) -> BG:
    """
    Map various field names from upstream readers into canonical keys for the shim.
    Accepts variants:
      cs2         <- "cs2" or "c_sound_squared"
      v1          <- "v1"  or "v_1"
    Always returns contiguous float64 arrays.
    """
    def arr(x): return np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    # required keys in some form
    r    = arr(bg["r"])
    rho  = arr(bg["rho"])
    eps  = arr(bg["eps"])
    p    = arr(bg["p"])
    # tolerate both names for cs^2 and v1
    cs2  = arr(bg.get("cs2", bg.get("c_sound_squared")))
    v1   = arr(bg.get("v1",  bg.get("v_1")))
    phi  = arr(bg["phi"])
    alpha= arr(bg["alpha"])
    # basic sanity
    m = len(r)
    for k, v in dict(r=r, rho=rho, eps=eps, p=p, cs2=cs2, phi=phi, alpha=alpha, v1=v1).items():
        if v.shape != (m,):
            raise ValueError(f"canonize_bg: '{k}' shape {v.shape} != ({m},)")
    return dict(r=r, rho=rho, eps=eps, p=p, cs2=cs2, phi=phi, alpha=alpha, v1=v1)

def from_generator(gen: Iterable[Tuple[int, float, Dict[str, Any]]]) -> BackgroundReader:
    """Wrap a simple generator as a BackgroundReader with canonical mapping."""
    class _GenReader:
        def __iter__(self):
            for nt, time, bg in gen:
                yield nt, float(time), canonize_bg(bg)
    return _GenReader()
