"""
Tests for physics/reaction.py and physics/cstr.py.

Cantera isn't available in every environment this repo might be tested in
(it's a heavier, less universally-installed dependency than the rest of the
stack), so these tests mock cantera.Solution with a stand-in that returns
the same constant per-species Cp physics/thermo.py would build from its
YAML definition. This exercises all of the CSTR's actual ODE/steady-state
math -- everything except Cantera's own YAML parsing -- without requiring
Cantera to be installed to run this file. If Cantera *is* installed,
test_thermo_yaml_loads below additionally checks the real YAML parses.

Run directly with:  python tests/test_physics_cstr.py
(no pytest install required -- see _test_support.py. Also runs fine under
`pytest tests/` if you do have pytest installed.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _test_support
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # for physics/

import types

import numpy as np

from _test_support import approx, importorskip, run_tests, skip  # noqa: E402

# --- mock cantera before importing physics.* (see module docstring) -------
_FAKE_CP_ORDER = [81.1, 123.3, 143.5, 75.3]  # J/(mol K); matches physics/thermo.py


class _FakeSolution:
    standard_cp_R = np.array(_FAKE_CP_ORDER) / 8.314462618


def _install_fake_cantera_if_needed():
    """
    Only install the fake if real cantera ISN'T importable. An earlier
    version of this always installed the fake unconditionally -- which
    meant that even on a machine with real cantera installed, it would
    silently get clobbered in sys.modules before physics.thermo ever
    touched it, and test_thermo_yaml_loads_with_real_cantera below would
    then always skip (since it would find the fake already sitting in
    sys.modules, not real cantera) instead of actually exercising the real
    YAML parsing it's meant to check. Fixed: try the real import first.
    """
    try:
        import cantera  # noqa: F401
        return  # real cantera is available -- don't touch sys.modules at all
    except ImportError:
        pass

    fake_ct = types.ModuleType("cantera")
    fake_ct.Solution = lambda **kwargs: _FakeSolution()
    fake_ct._is_fake = True
    sys.modules["cantera"] = fake_ct


_install_fake_cantera_if_needed()

from physics import cstr, reaction, thermo  # noqa: E402


# --------------------------------------------------------------- reaction --
def test_rate_constants_match_published_arrhenius_fit():
    # Mekala & Goli (2014) Fig. 13 regression, checked against their
    # Table 2 tabulated (kf, kb) at each temperature. The Arrhenius fit is

    # a regression (R^2 ~ 0.96), so we allow ~25% scatter, not exact match.
    table_2 = {
        305.15: (0.001304, 0.000261),
        313.15: (0.003589, 0.000717),
        323.15: (0.006346, 0.001268),
        333.15: (0.010673, 0.002133),
    }
    for T, (kf_table, kb_table) in table_2.items():
        kf = reaction.forward_rate_constant(T)
        kb = reaction.backward_rate_constant(T)
        assert kf == approx(kf_table, rel=0.3)
        assert kb == approx(kb_table, rel=0.3)


def test_equilibrium_constant_in_literature_range():
    # Source paper reports Ke (concentration-based) for this reaction
    # falls in the 3.9-9.0 range across studies; their own fitted value was
    # 4.95. Not strongly temperature-dependent given the similar Ea for
    # forward/backward.
    for T in [298.15, 313.15, 333.15, 353.15]:
        Ke = reaction.equilibrium_constant(T)
        assert 3.0 < Ke < 10.0


def test_rate_is_zero_at_equilibrium_composition():
    # By construction: pick C such that kf*C0*C1 == kb*C2*C3.
    T = 320.0
    kf = reaction.forward_rate_constant(T)
    kb = reaction.backward_rate_constant(T)
    C = np.array([2.0, 2.0, kf * 2.0 * 2.0 / kb, 1.0])
    # solve for C3 s.t. kb*C2*C3 == kf*C0*C1 given C2 above and C3=1 chosen,
    # so instead just check the constructed point directly satisfies rate=0
    C2 = 1.0
    C3 = kf * C[0] * C[1] / (kb * C2)
    C_eq = np.array([C[0], C[1], C2, C3])
    assert reaction.rate(C_eq, T) == approx(0.0, abs=1e-12)


# ------------------------------------------------------------------- CSTR --
def _default_reactor(UA=2000.0, T_jacket=298.15, Q_acid=1.0, Q_meoh=1.0, V=100.0):
    params = cstr.CSTRParams(
        V=V, Q_acid=Q_acid, Q_meoh=Q_meoh,
        T_acid_in=298.15, T_meoh_in=298.15,
        UA=UA, T_jacket=T_jacket,
    )
    return cstr.CSTR(params)


def test_steady_state_satisfies_derivatives_zero():
    reactor = _default_reactor()
    C_ss, T_ss = reactor.steady_state()
    residual = reactor.derivatives(0.0, np.concatenate([C_ss, [T_ss]]))
    assert np.allclose(residual, 0.0, atol=1e-6)


def test_steady_state_mass_balance_closes():
    # Total moles/min in must equal total moles/min out: this reaction
    # conserves total moles (1 mol reactants -> 1 mol products), so overall
    # molar throughput in and out must match exactly regardless of
    # conversion.
    reactor = _default_reactor()
    p = reactor.p
    C_ss, _ = reactor.steady_state()

    mol_in = p.Q_acid * np.sum(p.C_acid_in) + p.Q_meoh * np.sum(p.C_meoh_in)
    mol_out = (p.Q_acid + p.Q_meoh) * np.sum(C_ss)
    assert mol_out == approx(mol_in, rel=1e-6)


def test_steady_state_conversion_between_zero_and_one():
    reactor = _default_reactor()
    C_ss, _ = reactor.steady_state()
    conv = reactor.conversion(C_ss)
    assert 0.0 < conv < 1.0


def test_more_residence_time_increases_conversion():
    # Bigger volume at fixed flow -> longer residence time -> should push
    # conversion higher (toward, but not past, equilibrium).
    small = _default_reactor(V=20.0)
    large = _default_reactor(V=500.0)
    conv_small = small.conversion(small.steady_state()[0])
    conv_large = large.conversion(large.steady_state()[0])
    assert conv_large > conv_small


def test_cooling_failure_raises_steady_state_temperature():
    normal = _default_reactor(UA=2000.0)
    failed = _default_reactor(UA=20.0)  # ~two orders of magnitude worse heat transfer
    _, T_normal = normal.steady_state()
    _, T_failed = failed.steady_state()
    assert T_failed > T_normal


def test_transient_simulation_reaches_new_steady_state_after_cooling_failure():
    normal = _default_reactor(UA=2000.0)
    C0, T0 = normal.steady_state()

    failed = _default_reactor(UA=20.0)
    sol = failed.simulate(y0=np.concatenate([C0, [T0]]), t_span=(0, 400))
    assert sol.success

    T_final_transient = sol.y[4, -1]
    _, T_expected_ss = failed.steady_state()
    assert T_final_transient == approx(T_expected_ss, rel=0.02)


def test_feed_ratio_disturbance_shifts_conversion():
    # A feed-ratio disturbance (Phase 8 fault) should measurably change
    # steady-state conversion relative to a balanced 1:1 feed -- holding
    # TOTAL feed flow (and therefore residence time) constant, so what's
    # being isolated is the equilibrium-shift effect of excess methanol,
    # not a confounding residence-time change. (An earlier version of this
    # test added extra methanol flow on top of a fixed acid flow instead,
    # which shortens residence time enough to swamp the equilibrium effect
    # and actually *lowers* conversion -- a real, physically correct model
    # behavior, just not what that test meant to check.)
    balanced = _default_reactor(Q_acid=1.0, Q_meoh=1.0)
    unbalanced = _default_reactor(Q_acid=0.8, Q_meoh=1.2)  # same total flow, methanol excess
    conv_balanced = balanced.conversion(balanced.steady_state()[0])
    conv_unbalanced = unbalanced.conversion(unbalanced.steady_state()[0])
    assert conv_unbalanced > conv_balanced  # excess methanol should push AcOH conversion up


# ------------------------------------------------------------ real cantera -
def test_thermo_yaml_loads_with_real_cantera():
    importorskip("cantera", reason="Cantera not installed in this environment")
    # If real cantera IS available (and _install_fake_cantera_if_needed
    # above correctly left it alone), double check the actual YAML we
    # ship parses and gives the expected constant Cp values, not just
    # the mock's hardcoded array.
    import importlib
    real_cantera = importlib.import_module("cantera")
    if getattr(real_cantera, "_is_fake", False):
        skip("only the mock cantera is installed here")

    sol = thermo.make_solution()
    cp = thermo.species_cp(sol)
    assert np.allclose(cp, _FAKE_CP_ORDER, rtol=1e-3)


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
