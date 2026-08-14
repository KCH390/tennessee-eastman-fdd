"""
Runnable Track 2 demo: CSTR steady state + cooling-failure fault, and the
methyl acetate/methanol azeotrope prediction from the NRTL module.

    python demo_track2.py

Requires Cantera for the real run (physics/thermo.py). If Cantera isn't
installed, this script still runs everything except the Cantera-backed Cp
lookup, using the same tiny stand-in physics/thermo.py's own docstring
describes -- see USE_FAKE_CANTERA below.

Produces two PNGs in this directory:
    track2_cooling_failure.png  -- CSTR temperature response to a cooling
                                    failure fault (Phase 8 scenario)
    track2_azeotrope.png        -- x-y diagram for methyl acetate/methanol
                                    showing the predicted azeotrope
"""

import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics import cstr, nrtl, vle_params

# --- Cantera availability check -------------------------------------------
try:
    import cantera  # noqa: F401
    USE_FAKE_CANTERA = False
except ImportError:
    USE_FAKE_CANTERA = True
    _fake_ct = types.ModuleType("cantera")

    class _FakeSolution:
        # Same constant Cp values physics/thermo.py's real YAML would build.
        standard_cp_R = np.array([81.1, 123.3, 143.5, 75.3]) / 8.314462618

    _fake_ct.Solution = lambda **kwargs: _FakeSolution()
    sys.modules["cantera"] = _fake_ct
    print("NOTE: Cantera not installed here -- using the same mocked Cp "
          "stand-in the test suite uses so this demo can still run end to "
          "end. Install cantera and re-run for the real YAML-backed values.")



IDX = {s: i for i, s in enumerate(vle_params.SPECIES)}


def run_cstr_demo():
    print("\n=== CSTR: steady state and cooling-failure fault ===")
    normal_params = cstr.CSTRParams(
        V=100.0, Q_acid=1.0, Q_meoh=1.0,
        T_acid_in=298.15, T_meoh_in=298.15,
        UA=2000.0, T_jacket=298.15,
    )
    normal = cstr.CSTR(normal_params)
    C_ss, T_ss = normal.steady_state()
    conv = normal.conversion(C_ss)
    print(f"Normal operation steady state: T = {T_ss:.2f} K, "
          f"acetic acid conversion = {conv:.1%}")
    print(f"  concentrations (mol/L) [MeOH, AcOH, MeOAc, H2O] = "
          f"{np.round(C_ss, 3)}")

    # Cooling-failure fault: same reactor, jacket heat transfer degrades
    # sharply (UA drops ~100x) starting from the normal steady state.
    failed_params = cstr.CSTRParams(
        V=100.0, Q_acid=1.0, Q_meoh=1.0,
        T_acid_in=298.15, T_meoh_in=298.15,
        UA=20.0, T_jacket=298.15,
    )
    failed = cstr.CSTR(failed_params)
    t_eval = np.linspace(0, 400, 200)
    sol = failed.simulate(y0=np.concatenate([C_ss, [T_ss]]), t_span=(0, 400), t_eval=t_eval)
    _, T_new_ss = failed.steady_state()
    print(f"After cooling failure (UA: 2000 -> 20 J/min/K): "
          f"new steady state T = {T_new_ss:.2f} K "
          f"(+{T_new_ss - T_ss:.2f} K)")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sol.t, sol.y[4], color="crimson", lw=2)
    ax.axhline(T_ss, color="gray", ls="--", lw=1, label=f"pre-fault steady state ({T_ss:.1f} K)")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("reactor temperature (K)")
    ax.set_title("CSTR temperature response to a cooling-jacket failure")
    ax.legend()
    fig.tight_layout()
    fig.savefig("track2_cooling_failure.png", dpi=150)
    print("Saved track2_cooling_failure.png")


_ANTOINE = {
    "CH3COOCH3": (6.19052, 1157.622, 219.724),
    "CH3OH": (7.6278, 1905.90, 273.15),
}


def _psat_kpa(species, T_K):
    A, B, C = _ANTOINE[species]
    return 10 ** (A - B / (C + T_K - 273.15))


def run_azeotrope_demo():
    print("\n=== NRTL: methyl acetate / methanol x-y diagram and azeotrope ===")
    T = 329.6  # K, ~56.5 degC, approx atmospheric-pressure boiling region
    x_ma_range = np.linspace(0.001, 0.999, 200)
    y_ma_range = np.zeros_like(x_ma_range)

    for i, x_ma in enumerate(x_ma_range):
        x = np.zeros(vle_params.N)
        x[IDX["CH3COOCH3"]] = x_ma
        x[IDX["CH3OH"]] = 1.0 - x_ma
        gamma = nrtl.activity_coefficients(x, vle_params.DG, vle_params.ALPHA, T=T)
        p_ma = x_ma * gamma[IDX["CH3COOCH3"]] * _psat_kpa("CH3COOCH3", T)
        p_m = (1 - x_ma) * gamma[IDX["CH3OH"]] * _psat_kpa("CH3OH", T)
        y_ma_range[i] = p_ma / (p_ma + p_m)

    gap = np.abs(y_ma_range - x_ma_range)
    az_idx = np.argmin(gap)
    print(f"Predicted azeotrope: x_MeOAc = y_MeOAc = {x_ma_range[az_idx]:.3f} "
          f"(literature, Orchilles et al. 2007: ~0.67)")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(x_ma_range, y_ma_range, color="steelblue", lw=2, label="NRTL prediction")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="y = x")
    ax.scatter([x_ma_range[az_idx]], [y_ma_range[az_idx]], color="crimson", zorder=5,
               label=f"predicted azeotrope (x={x_ma_range[az_idx]:.2f})")
    ax.set_xlabel("x, methyl acetate (liquid mole fraction)")
    ax.set_ylabel("y, methyl acetate (vapor mole fraction)")
    ax.set_title("Methyl acetate / methanol x-y diagram (NRTL, ~56.5 degC)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig("track2_azeotrope.png", dpi=150)
    print("Saved track2_azeotrope.png")


if __name__ == "__main__":
    run_cstr_demo()
    run_azeotrope_demo()
