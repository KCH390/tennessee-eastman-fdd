"""
Antoine vapor-pressure parameters for the quaternary system, used by
column.py for bubble-point / VLE calculations. Same species order and
indices as physics/thermo.py, physics/reaction.py, physics/vle_params.py.

All four are expressed internally in a single common form so column.py
never has to think about units per species:

    log10(P / kPa) = A - B / (T_degC + C)

Two species (CH3OH, CH3COOCH3) were already in exactly this form, sourced
from Dykyj et al. (1984) as tabulated in Graczova, Dobcsanyi & Steltenpohl
(2017), Table 1 -- the same paper used to validate the NRTL azeotrope
prediction in test_nrtl.py.

The other two (CH3COOH, H2O) were sourced in the more common mmHg/degC
Antoine form and converted here (A_kPa = A_mmHg - log10(1/0.133322) =
A_mmHg - 0.87506; B, C unchanged, since only the pressure axis is being
rescaled, not the temperature-dependent term):
  - H2O: A=8.07131, B=1730.63, C=233.426 (mmHg, degC; the standard
    textbook Antoine set for water, valid 1-100 degC).
  - CH3COOH: A=7.38782, B=1533.313, C=222.309 (mmHg, degC; NIST
    Chemistry WebBook, as tabulated in an undergraduate thermodynamics
    textbook's physical-property appendix).

All four were checked against their real normal boiling points (64.7,
117.9, 56.9, 100.0 degC for CH3OH/CH3COOH/CH3COOCH3/H2O respectively).
Three land within ~0.1 kPa of 101.3 kPa there; CH3OH is off by about 4%
(97.0 vs. 101.3 kPa) -- expected regression scatter, since that fit
covers a very broad range (5-224 degC) with a single equation rather than
being re-fit locally per boiling point. Still close enough to validate
the sourcing/unit conversion above is self-consistent, not swapped or
mis-transcribed.
"""

import numpy as np

SPECIES = ["CH3OH", "CH3COOH", "CH3COOCH3", "H2O"]

# (A, B, C) in the log10(P/kPa) = A - B/(T_degC + C) form -- see docstring.
_ANTOINE_KPA = {
    "CH3OH": (7.6278, 1905.90, 273.15),
    "CH3COOH": (7.38782 - 0.87506, 1533.313, 222.309),
    "CH3COOCH3": (6.19052, 1157.622, 219.724),
    "H2O": (8.07131 - 0.87506, 1730.63, 233.426),
}

NORMAL_BOILING_POINT_C = {
    "CH3OH": 64.7,
    "CH3COOH": 117.9,
    "CH3COOCH3": 56.9,
    "H2O": 100.0,
}


def psat_kpa(species: str, T_K: float) -> float:
    """Saturation pressure, kPa, at temperature T_K (Kelvin)."""
    A, B, C = _ANTOINE_KPA[species]
    T_C = T_K - 273.15
    return 10 ** (A - B / (T_C + C))


def psat_kpa_vector(T_K: float) -> np.ndarray:
    """Saturation pressures for all 4 species (SPECIES order), kPa."""
    return np.array([psat_kpa(s, T_K) for s in SPECIES])
