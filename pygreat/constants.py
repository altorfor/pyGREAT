# pygreat/constants.py
"""
Shared physical constants and GREAT unit-conversion factors (cgs).
Mirrors GREAT's Fortran values so Python ↔ Fortran stay consistent.
"""

__all__ = [
    "C_LIGHT", "G_GRAV", "PI", "M_SUN",
    "RHO_GEOM", "P_GEOM",
]

# Core constants (cgs)
C_LIGHT: float = 2.99792458e10   # cm/s
G_GRAV:  float = 6.673e-8        # cm^3 g^-1 s^-2   (matches GREAT)
PI:      float = 3.141592653589793
M_SUN:   float = 1.989e33        # g

# GREAT geometric-unit conversion factors (match module_background.f90):
#   p_geom_factor  = G / c^4
#   rho_geom_factor= G / c^2
P_GEOM:  float = G_GRAV / (C_LIGHT**4)  # converts pressure to code units
RHO_GEOM:float = G_GRAV / (C_LIGHT**2)  # converts density  to code units
