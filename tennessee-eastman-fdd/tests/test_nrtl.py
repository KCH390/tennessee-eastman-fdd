"""
Tests for physics/nrtl.py and physics/vle_params.py.

No cantera/scipy dependency issues here -- this module is pure numpy, so
these tests run directly (no mocking needed), unlike test_physics_cstr.py.

Run directly with:  python tests/test_nrtl.py
(no pytest install required -- see _test_support.py. Also runs fine under
`pytest tests/` if you do have pytest installed.)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _test_support
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # for physics/

from _test_support import approx, run_tests  # noqa: E402
from physics import nrtl, vle_params  # noqa: E402

IDX = {s: i for i, s in enumerate(vle_params.SPECIES)}


def test_pure_component_limit_gives_unit_activity():
    # x_i -> 1 (species alone) must give gamma_i -> 1 for every species,
    # regardless of the parameter matrix -- a structural property of NRTL,
    # not specific to this system's parameters.
    for i in range(vle_params.N):
        x = np.zeros(vle_params.N)
        x[i] = 1.0
        gamma = nrtl.activity_coefficients(x, vle_params.DG, vle_params.ALPHA, T=350.0)
        assert gamma[i] == approx(1.0, abs=1e-8)


def test_fully_ideal_system_gives_unit_activity_everywhere():
    dg_zero = np.zeros((vle_params.N, vle_params.N))
    alpha_zero = np.zeros((vle_params.N, vle_params.N))
    x = np.array([0.2, 0.3, 0.1, 0.4])
    gamma = nrtl.activity_coefficients(x, dg_zero, alpha_zero, T=350.0)
    assert np.allclose(gamma, 1.0)


def test_methanol_water_shows_known_positive_deviation():
    # Methanol + water is a well-known positive-deviation (gamma > 1)
    # system -- not azeotropic at atmospheric pressure, but non-ideal.
    # Check at an intermediate composition using only these two species
    # (acetic acid, methyl acetate absent -> x=0 for those).
    x = np.zeros(vle_params.N)
    x[IDX["CH3OH"]] = 0.4
    x[IDX["H2O"]] = 0.6
    gamma = nrtl.activity_coefficients(x, vle_params.DG, vle_params.ALPHA, T=337.0)
    assert 1.0 < gamma[IDX["CH3OH"]] < 3.0
    assert 1.0 < gamma[IDX["H2O"]] < 3.0


# --- Antoine data for the azeotrope check below, from the SAME source as
# the methyl-acetate/methanol NRTL regression context (Orchilles et al.
# 2007 data, as tabulated in Graczova et al. 2017 Table 1). log10(P/kPa) =
# A - B/(C + t), t in degC. Kept local to this test (not in vle_params.py)
# since it's only used here to sanity-check the NRTL fit, not part of the
# CSTR/column's actual property set.
_ANTOINE = {
    "CH3COOCH3": (6.19052, 1157.622, 219.724),
    "CH3OH": (7.6278, 1905.90, 273.15),
}


def _psat_kpa(species: str, T_K: float) -> float:
    A, B, C = _ANTOINE[species]
    t_C = T_K - 273.15
    return 10 ** (A - B / (C + t_C))


def test_methyl_acetate_methanol_azeotrope_near_literature_composition():
    # Literature (Orchilles et al. 2007, as reported in Graczova et al.
    # 2017): MA-methanol azeotrope at ~101.3 kPa is x_MA ~= 0.67, boiling
    # ~56.5-56.9 degC. We didn't use that paper's own NRTL fit (see
    # vle_params.py docstring for why -- two sources disagree on this
    # pair's parameters), so this is a check that our DECHEMA-sourced
    # parameters land in a physically reasonable neighborhood, not an
    # exact reproduction.
    T = 329.6  # ~56.5 degC, approx literature azeotrope temperature
    P_target = 101.3  # kPa

    best_x, best_gap = None, np.inf
    for x_ma in np.linspace(0.05, 0.95, 181):
        x = np.zeros(vle_params.N)
        x[IDX["CH3COOCH3"]] = x_ma
        x[IDX["CH3OH"]] = 1.0 - x_ma
        gamma = nrtl.activity_coefficients(x, vle_params.DG, vle_params.ALPHA, T=T)

        p_ma = x_ma * gamma[IDX["CH3COOCH3"]] * _psat_kpa("CH3COOCH3", T)
        p_m = (1 - x_ma) * gamma[IDX["CH3OH"]] * _psat_kpa("CH3OH", T)
        y_ma = p_ma / (p_ma + p_m)

        # Azeotrope: vapor composition equals liquid composition.
        gap = abs(y_ma - x_ma)
        if gap < best_gap:
            best_gap, best_x = gap, x_ma

    assert best_gap < 0.05  # a clean crossing was found
    assert 0.45 < best_x < 0.85  # in the right neighborhood of the literature ~0.67


def test_unsourced_pairs_are_flagged_and_currently_ideal():
    for i, j in vle_params.UNSOURCED_PAIRS:
        assert vle_params.DG[i, j] == 0.0
        assert vle_params.DG[j, i] == 0.0


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
