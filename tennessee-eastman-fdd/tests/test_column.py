"""
Tests for physics/column.py and physics/properties.py.

Pure numpy/scipy -- no cantera dependency (CMO means this column doesn't
need Cp/Hvap at all), so no mocking needed here.

Run directly with:  python tests/test_column.py
(no pytest install required -- see _test_support.py. Also runs fine under
`pytest tests/` if you do have pytest installed.)

Runtime note: this file is the slow one in the suite -- expect roughly
1-2 minutes total. Each derivative evaluation does N+1 bubble-point
root-finds, and steady_state() needs a stiff implicit integrator; see the
performance note in physics/column.py's steady_state() docstring.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _test_support
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # for physics/

from _test_support import approx, raises, run_tests  # noqa: E402
from physics import column, properties, vle_params  # noqa: E402

IDX = {s: i for i, s in enumerate(vle_params.SPECIES)}


def _default_params(**overrides):
    # N=5 (not e.g. 15) specifically to keep this test suite fast: each
    # derivative evaluation does N+1 bubble-point root-finds, and
    # steady_state() needs a stiff (BDF) implicit integrator whose
    # finite-difference Jacobian cost scales with state count -- see the
    # performance note in column.py's steady_state() docstring. A
    # realistic column (15-20+ trays) works the same way, just slower;
    # test_realistic_column_size_still_converges below checks one N=15
    # case so that path isn't completely untested, just not run repeatedly.
    z_F = np.array([8.955, 5.329, 3.405, 3.405])
    z_F = z_F / z_F.sum()
    defaults = dict(
        N=5, feed_stage=3, F=10.0, z_F=z_F, q=1.0,
        reflux_ratio=2.0, V_boilup=15.0,
        P_kPa=101.3, M_tray=5.0, M_condenser=10.0, M_reboiler=20.0,
    )
    defaults.update(overrides)
    return column.ColumnParams(**defaults)


# --------------------------------------------------------- Antoine/bubble --
def test_pure_component_bubble_points_match_known_boiling_points():
    for i, s in enumerate(properties.SPECIES):
        x = np.zeros(4)
        x[i] = 1.0
        T, y = column.bubble_point(x, P_kPa=101.3)
        expected = properties.NORMAL_BOILING_POINT_C[s] + 273.15
        assert T == approx(expected, abs=2.0)  # within 2 K
        assert y[i] == approx(1.0, abs=1e-6)


def test_bubble_point_rejects_out_of_bracket_composition():
    # Not really reachable with a normalized x, but confirms the guard
    # rail raises instead of silently returning garbage.
    with raises(RuntimeError):
        column.bubble_point(np.array([0.25, 0.25, 0.25, 0.25]), P_kPa=1e6)


# ------------------------------------------------------------------ flows --
def test_cmo_flows_satisfy_overall_mass_balance():
    params = _default_params()
    L, V, D, W = params.flows()
    assert D + W == approx(params.F, rel=1e-10)


def test_total_moles_invariant_holds():
    # Sum of derivatives across components, per stage, should be exactly
    # zero given CMO-consistent flows -- this is what makes steady_state's
    # forward-integration approach valid (see column.py docstring).
    params = _default_params()
    col = column.DynamicColumn(params)
    y0 = col.default_initial_state()
    d = col.derivatives(0.0, y0).reshape(params.N + 2, 4).sum(axis=1)
    assert np.abs(d).max() < 1e-5  # matches bubble_point's xtol=1e-4 (see column.py)


# ----------------------------------------------------------- steady state --
def test_steady_state_mole_fractions_sum_to_one():
    col = column.DynamicColumn(_default_params())
    x_cond, x_trays, x_reb = col.steady_state()
    assert x_cond.sum() == approx(1.0, abs=1e-6)
    assert x_reb.sum() == approx(1.0, abs=1e-6)
    assert np.allclose(x_trays.sum(axis=1), 1.0, atol=1e-6)


def test_steady_state_component_balance_closes():
    params = _default_params()
    col = column.DynamicColumn(params)
    _, _, D_flow, W_flow = params.flows()  # note: flows() returns (L,V,D,W); grab D,W by name below
    L, V, D, W = params.flows()
    x_cond, x_trays, x_reb = col.steady_state()

    comp_in = params.F * params.z_F
    comp_out = D * x_cond + W * x_reb
    assert np.allclose(comp_in, comp_out, atol=1e-6)


def test_column_separates_volatile_from_heavy_components():
    # The two heavy/less-volatile species (acetic acid, water) should end
    # up concentrated in the bottoms; the two light/volatile species
    # (methanol, methyl acetate) concentrated in the distillate.
    col = column.DynamicColumn(_default_params())
    x_cond, _, x_reb = col.steady_state()

    light_in_distillate = x_cond[IDX["CH3OH"]] + x_cond[IDX["CH3COOCH3"]]
    light_in_bottoms = x_reb[IDX["CH3OH"]] + x_reb[IDX["CH3COOCH3"]]
    assert light_in_distillate > light_in_bottoms

    heavy_in_bottoms = x_reb[IDX["CH3COOH"]] + x_reb[IDX["H2O"]]
    heavy_in_distillate = x_cond[IDX["CH3COOH"]] + x_cond[IDX["H2O"]]
    assert heavy_in_bottoms > heavy_in_distillate


def test_distillate_approaches_but_does_not_cross_the_azeotrope():
    # A minimum-boiling azeotrope is a distillation pinch: no amount of
    # reflux/trays can push the distillate's methyl-acetate mole fraction
    # past it. Correct behavior is asymptotic approach from below (the
    # feed here sits on the methanol-rich side of the azeotrope), not
    # landing exactly on it at any one arbitrary operating point.
    #
    # NOTE: an earlier version of this test (and an earlier claim made
    # in conversation about a specific RR=2, N=15 run) asserted the
    # distillate composition WAS the azeotrope at typical operating
    # conditions. That was wrong -- it eyeballed x_MeOAc=0.32 against the
    # azeotrope's ~0.65-0.67 and mistook a coincidental resemblance to a
    # DIFFERENT number (x_MeOH, the complement) for a match. The actual
    # behavior, checked properly here: x_MeOAc climbs from ~0.32 (RR=2,
    # N=5) to ~0.62 (RR=5, N=10) to ~0.65 (RR=15, N=20) as reflux/trays
    # increase -- exactly the expected pinch behavior, and a stronger
    # check than the single-point version it replaced.
    results = []
    for rr, n, fs in [(2.0, 5, 3), (5.0, 10, 5), (15.0, 20, 10)]:
        params = _default_params(N=n, feed_stage=fs, reflux_ratio=rr)
        col = column.DynamicColumn(params)
        x_cond, _, _ = col.steady_state(t_max=800)
        results.append(x_cond[IDX["CH3COOCH3"]])

    assert results[0] < results[1] < results[2]  # monotonic approach
    assert all(r < 0.68 for r in results)  # never crosses the ~0.65-0.67 azeotrope ceiling
    assert results[2] == approx(0.65, abs=0.05)  # closest run lands near the literature value


def test_realistic_column_size_mass_balance_closes():
    # Reuses the N=20 run from the azeotrope-approach test's parametrized
    # loop conceptually, but run standalone so this test's pass/fail is
    # legible on its own (mass balance specifically), not folded into the
    # azeotrope test's assertions.
    params = _default_params(N=15, feed_stage=8, reflux_ratio=2.0)
    col = column.DynamicColumn(params)
    L, V, D, W = params.flows()
    x_cond, x_trays, x_reb = col.steady_state(t_max=1500)
    comp_in = params.F * params.z_F
    comp_out = D * x_cond + W * x_reb
    assert np.allclose(comp_in, comp_out, atol=1e-4)


# ------------------------------------------------- the actual swap-point --
def test_column_output_changes_if_vle_params_are_edited():
    """
    The whole point of keeping vle_params.py as the single source of NRTL
    truth: editing it should be enough to change the column's behavior,
    with zero changes to column.py. This test edits (and restores) a
    currently-ideal pair to confirm that wiring actually works, so a
    future refactor that accidentally hardcodes something in column.py
    would break this test.
    """
    params = _default_params()
    col = column.DynamicColumn(params)
    x_cond_before, _, _ = col.steady_state()

    i, j = IDX["CH3COOH"], IDX["H2O"]
    original_dg_ij = vle_params.DG[i, j]
    original_dg_ji = vle_params.DG[j, i]
    original_alpha = vle_params.ALPHA[i, j]
    try:
        # A stand-in "experimental" parameter set for acetic acid-water
        # (deliberately not claimed as real data -- just needs to be
        # non-ideal enough to change the result measurably).
        vle_params.DG[i, j] = -800.0
        vle_params.DG[j, i] = 1600.0
        vle_params.ALPHA[i, j] = vle_params.ALPHA[j, i] = 0.3

        col_after = column.DynamicColumn(_default_params())  # fresh instance, same params
        x_cond_after, _, _ = col_after.steady_state()

        assert not np.allclose(x_cond_before, x_cond_after, atol=1e-4)
    finally:
        vle_params.DG[i, j] = original_dg_ij
        vle_params.DG[j, i] = original_dg_ji
        vle_params.ALPHA[i, j] = vle_params.ALPHA[j, i] = original_alpha


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
