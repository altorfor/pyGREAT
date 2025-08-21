# python/pygreat/params.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

_TRUE = {".true.", "true", "t", "1", "yes", "on"}
_FALSE = {".false.", "false", "f", "0", "no", "off"}

def _coerce(s: str) -> Any:
    s = s.strip()
    if not s:
        return s
    s_l = s.lower()
    if s_l in _TRUE:  return True
    if s_l in _FALSE: return False
    # Fortran 'd' exponents -> 'e'
    if any(c in s_l for c in ("e", "d")):
        try:
            return float(s_l.replace("d", "e"))
        except ValueError:
            pass
    # int?
    try:
        return int(s, 10)
    except ValueError:
        pass
    # string (strip quotes)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def parse_parameters(parfile: str | Path) -> Dict[str, Any]:
    p = {}
    for raw in Path(parfile).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line: continue
        if "=" not in line: continue
        k, v = line.split("=", 1)
        p[k.strip()] = _coerce(v.strip())
    return p
# --- interpolation config ----------------------------------------------------
from dataclasses import dataclass

@dataclass
class InterpConfig:
    enabled: bool           # interpolation = .true./.false.
    mode: str               # "shock" | "uniform"
    kind: str               # "linear" | "pchip" | "nearest" | "loglinear" | "cubic" | "akima"
    factor: int             # for uniform
    buffer: int             # cells on each side of iR (shock/sonic window)
    refine: int             # refine factor inside the window

_ALLOWED_MODES  = {"shock", "uniform"}
_ALLOWED_KINDS  = {"linear", "pchip", "nearest", "loglinear", "cubic", "akima"}

def _as_bool(x) -> bool:
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    return s in _TRUE

def get_interp_config(P: Dict[str, Any]) -> InterpConfig:
    """Build an InterpConfig from a parsed parfile dict. Keys are case-insensitive."""
    Pl = {str(k).lower(): v for k, v in P.items()}

    enabled = _as_bool(Pl.get("interpolation", False))

    # Defaults chosen to avoid extra deps and to match your sonic-window use
    mode  = str(Pl.get("interp_mode", "shock" if enabled else "uniform")).strip().lower()
    if mode not in _ALLOWED_MODES:
        mode = "shock" if enabled else "uniform"

    kind  = str(Pl.get("interp_kind", "linear")).strip().lower()
    if kind not in _ALLOWED_KINDS:
        kind = "linear"

    factor = int(Pl.get("interp_factor", 2))
    buffer = int(Pl.get("interp_buffer", 15))
    refine = int(Pl.get("interp_refine", 4))

    # clamp sanely
    factor = max(1, factor)
    buffer = max(0, buffer)
    refine = max(1, refine)

    return InterpConfig(enabled=enabled, mode=mode, kind=kind, factor=factor, buffer=buffer, refine=refine)
