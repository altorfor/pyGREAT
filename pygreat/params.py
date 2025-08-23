# python/pygreat/params.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Basic parameter parsing
# -----------------------------------------------------------------------------

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
    p: Dict[str, Any] = {}
    for raw in Path(parfile).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        p[k.strip()] = _coerce(v.strip())
    return p

# -----------------------------------------------------------------------------
# Interpolation config (used by pipeline)
# -----------------------------------------------------------------------------

@dataclass
class InterpConfig:
    enabled: bool            # interpolation = .true./.false.
    mode: str                # "uniform" | "interior"  (accept "shock" as alias of "interior")
    kind: str                # "linear" | "pchip" | "nearest" | "loglinear" | "cubic" | "akima"
    factor: int              # only for mode="uniform"
    buffer: int              # cells AFTER iR, for mode="interior"
    n_inner: Optional[int]   # total pts in [r[0], r[iR+buffer]] (incl endpoints), for mode="interior"

_ALLOWED_MODES = {"uniform", "interior", "shock"}  # "shock" -> alias for "interior"
_ALLOWED_KINDS = {"linear", "pchip", "nearest", "loglinear", "cubic", "akima"}

def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in _TRUE

def get_interp_config(P: Dict[str, Any]) -> InterpConfig:
    """
    Build an InterpConfig from a parsed parfile dict. Keys are case-insensitive.

    Recognized keys:
      interpolation        (.true./.false.)
      interp_mode          ("uniform" | "interior" | "shock")
      interp_kind          ("linear","pchip","nearest","loglinear","cubic","akima")
      interp_factor        (int, only for mode="uniform")
      interp_buffer        (int, only for mode="interior")
      interp_n_inner       (int or empty, only for mode="interior")
    """
    Pl = {str(k).lower(): v for k, v in P.items()}

    enabled = _as_bool(Pl.get("interpolation", False))

    mode = str(Pl.get("interp_mode", "uniform")).strip().lower()
    if mode not in _ALLOWED_MODES:
        mode = "uniform"
    if mode == "shock":            # backward-compat: "shock" means inside-only refinement
        mode = "interior"

    kind = str(Pl.get("interp_kind", "linear")).strip().lower()
    if kind not in _ALLOWED_KINDS:
        kind = "linear"

    factor = max(1, int(Pl.get("interp_factor", 2)))      # for uniform
    buffer = max(0, int(Pl.get("interp_buffer", 15)))     # for interior

    # n_inner can be absent → None (pipeline will fall back to a default)
    raw_n_inner = Pl.get("interp_n_inner", None)
    if raw_n_inner is None or raw_n_inner == "":
        n_inner: Optional[int] = None
    else:
        n_inner = max(2, int(raw_n_inner))

    return InterpConfig(
        enabled=enabled,
        mode=mode,
        kind=kind,
        factor=factor,
        buffer=buffer,
        n_inner=n_inner,
    )
