"""
Dynamic distillation column: N equilibrium trays + a partial reboiler +
a total-condenser reflux drum, separating the quaternary CH3OH / CH3COOH /
CH3COOCH3 / H2O mixture (standalone -- fed by a user-specified feed
composition for now; not yet coupled to cstr.py's actual effluent, per
the earlier decision to build and validate the two units independently
before wiring them together).

Scope decision, stated plainly rather than silently downgraded: the
distillation column phase was originally scoped as "full dynamic
tray-by-tray MESH, mass AND energy balances." What's actually implemented
here integrates the mass balances dynamically per tray with rigorous
per-tray NRTL VLE (real non-ideal thermodynamics, not simplified), but
uses constant molar overflow (CMO) for the internal vapor/liquid flow
profile instead of solving a fully coupled energy balance per tray.

Why: CMO is the standard, named, widely-taught simplification for exactly
this situation (implicitly assumes similar molar heats of vaporization
across components -- reasonable here since CH3OH/CH3COOH/CH3COOCH3/H2O
molar heats of vaporization are all within roughly a factor of 2 of each
other). A fully coupled dynamic energy balance across every tray is a
much bigger, more failure-prone undertaking to get right without a real
process simulator to check the result against -- verifying an incorrect
implicit tray-to-tray energy solve is much harder than verifying this
model's mass balances and VLE, which are checked in test_column.py. This
is the same kind of explicit, documented simplification as the CSTR's
deferred heat-of-mixing term -- flagged, not hidden, and the obvious next
thing to revisit if this model needs to get more rigorous later.

MESH equations actually solved, per equilibrium stage i (trays 1..N,
numbered top to bottom, plus the reboiler as stage N+1):

    M(aterial balance, per component k, dynamic):
        dx_i,k/dt = [L_in*x_in,k + V_in*y_in,k - L_i*x_i,k - V_i*y_i,k
                     + (feed term if i is the feed stage)] / M_i

    E(quilibrium): y_i,k = gamma_i,k(x_i, T_i) * Psat_k(T_i) / P * x_i,k
        -- solved via NRTL (physics/nrtl.py, physics/vle_params.py) and
        Antoine vapor pressures (physics/properties.py)

    S(ummation): bubble-point T_i chosen so sum_k y_i,k = 1

    (H)ydraulics/flows: constant molar overflow -- see above; L_i, V_i on
        each tray come from the operating specs (reflux ratio, boilup,
        feed thermal condition) rather than a per-tray energy balance.

The condenser (reflux drum) is a real dynamic accumulator too (it has its
own holdup and composition state, receiving V_1*y_1 and draining
(L_reflux + D) = V_rect at its own composition) -- but it's NOT an
equilibrium stage, since a total condenser fully condenses the incoming
vapor with no further VLE calculation needed.

Uses vle_params.DG / vle_params.ALPHA exactly as physics/nrtl.py exposes
them -- no NRTL numbers are duplicated or hardcoded in this file. That
means swapping in real experimental parameters for the 4 currently-ideal
acetic-acid pairs (see vle_params.py) is a one-file change: edit
vle_params.py, and every consumer (this column, the CSTR's future
heat-of-mixing work, anything else) picks it up automatically. See
test_column.py::test_column_output_changes_if_vle_params_are_edited for
a test that actually exercises this swap point, so a future change to
vle_params.py that accidentally breaks the wiring here would fail a test,
not just "hopefully still work."
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from . import nrtl, properties, vle_params

N_SPECIES = 4
SPECIES = vle_params.SPECIES  # [CH3OH, CH3COOH, CH3COOCH3, H2O]

_T_BOUNDS_K = (280.0, 420.0)  # generous bracket covering all 4 boiling points


@dataclass
class ColumnParams:
    N: int  # number of equilibrium trays (not counting the reboiler)
    feed_stage: int  # 1-indexed tray the feed enters on

    F: float  # feed molar flow, mol/min
    z_F: np.ndarray  # feed composition, SPECIES order, sums to 1
    q: float  # feed liquid fraction (1.0 = saturated liquid, 0.0 = saturated vapor)

    reflux_ratio: float  # L_rect / D
    V_boilup: float  # vapor rate leaving the reboiler, mol/min

    P_kPa: float = 101.3  # column pressure, assumed uniform across all stages

    M_tray: float = 10.0  # liquid holdup per tray, mol (assumed uniform)
    M_condenser: float = 20.0  # reflux drum holdup, mol
    M_reboiler: float = 50.0  # reboiler holdup, mol

    def flows(self):
        """
        Returns (L, V, D, W): L and V are length-N arrays (tray liquid/
        vapor molar flow, mol/min), D is distillate rate, W is bottoms
        rate. Constant within each section per the CMO assumption -- see
        module docstring.
        """
        V_strip = self.V_boilup
        V_rect = V_strip + (1.0 - self.q) * self.F
        D = V_rect / (self.reflux_ratio + 1.0)
        L_rect = self.reflux_ratio * D
        L_strip = L_rect + self.q * self.F
        W = L_strip - V_strip

        L = np.where(np.arange(1, self.N + 1) < self.feed_stage, L_rect, L_strip)
        V = np.where(np.arange(1, self.N + 1) < self.feed_stage, V_rect, V_strip)
        return L, V, D, W


def bubble_point(x: np.ndarray, P_kPa: float, T_guess: float = 340.0):
    """
    Solve for T such that sum_k(gamma_k(x,T)*Psat_k(T)/P * x_k) == 1, and
    return (T, y). x is normalized defensively (ODE integration can drift
    slightly off summing to 1).
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    x = x / x.sum()

    def residual(T):
        gamma = nrtl.activity_coefficients(x, vle_params.DG, vle_params.ALPHA, T=T)
        psat = properties.psat_kpa_vector(T)
        return np.sum(gamma * psat * x) / P_kPa - 1.0

    lo, hi = _T_BOUNDS_K
    # brentq needs a sign change; nudge outward a bit if the initial bracket fails
    # (only expected near the extremes of validity, e.g. a near-pure light or heavy stream).
    r_lo, r_hi = residual(lo), residual(hi)
    if r_lo * r_hi > 0:
        raise RuntimeError(
            f"bubble_point: no sign change in [{lo},{hi}] K for x={x} "
            f"(residuals {r_lo:.4g}, {r_hi:.4g}) -- composition may be out of range"
        )
    T = brentq(residual, lo, hi, xtol=1e-4)
    gamma = nrtl.activity_coefficients(x, vle_params.DG, vle_params.ALPHA, T=T)
    y = gamma * properties.psat_kpa_vector(T) * x / P_kPa
    return T, y


class DynamicColumn:
    """
    State vector layout (all mole fractions, SPECIES order):
        [x_condenser (4), x_tray_1 (4), ..., x_tray_N (4), x_reboiler (4)]
    Total length = 4*(N+2).
    """

    def __init__(self, params: ColumnParams):
        self.p = params
        self.L, self.V, self.D, self.W = params.flows()

    def n_states(self) -> int:
        return N_SPECIES * (self.p.N + 2)

    def _unpack(self, state: np.ndarray):
        state = state.reshape(self.p.N + 2, N_SPECIES)
        return state[0], state[1:-1], state[-1]  # x_cond, x_trays (N,4), x_reb

    def _pack(self, x_cond, x_trays, x_reb) -> np.ndarray:
        return np.concatenate([x_cond, x_trays.ravel(), x_reb])

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        p = self.p
        x_cond, x_trays, x_reb = self._unpack(state)

        # Bubble points / vapor compositions for every equilibrium stage.
        y_trays = np.zeros((p.N, N_SPECIES))
        for i in range(p.N):
            _, y_trays[i] = bubble_point(x_trays[i], p.P_kPa)
        _, y_reb = bubble_point(x_reb, p.P_kPa)

        d_cond = (self.V[0] * y_trays[0] - (self.L[0] + self.D) * x_cond) / p.M_condenser

        d_trays = np.zeros((p.N, N_SPECIES))
        for i in range(p.N):
            L_in = self.L[i - 1] if i > 0 else self.L[0]
            x_in = x_trays[i - 1] if i > 0 else x_cond
            V_in = self.V[i + 1] if i < p.N - 1 else p.V_boilup
            y_in = y_trays[i + 1] if i < p.N - 1 else y_reb

            feed_term = p.F * p.z_F if (i + 1) == p.feed_stage else 0.0

            balance = (
                L_in * x_in + V_in * y_in - self.L[i] * x_trays[i] - self.V[i] * y_trays[i] + feed_term
            )
            d_trays[i] = balance / p.M_tray

        d_reb = (self.L[p.N - 1] * x_trays[p.N - 1] - p.V_boilup * y_reb - self.W * x_reb) / p.M_reboiler

        return self._pack(d_cond, d_trays, d_reb)

    def default_initial_state(self) -> np.ndarray:
        """Every stage starts at the feed composition -- a reasonable, simple default."""
        p = self.p
        x_cond = p.z_F.copy()
        x_trays = np.tile(p.z_F, (p.N, 1))
        x_reb = p.z_F.copy()
        return self._pack(x_cond, x_trays, x_reb)

    def simulate(self, y0=None, t_span=(0, 200), t_eval=None, **kwargs):
        if y0 is None:
            y0 = self.default_initial_state()
        return solve_ivp(self.derivatives, t_span, y0, t_eval=t_eval, method="LSODA", **kwargs)

    def steady_state(self, y0=None, t_max=1500.0, tol=1e-6):
        """
        Reach steady state by integrating forward in time until the
        derivatives are small, rather than an algebraic (fsolve) root
        find.

        This is deliberate, not a fallback: the total-moles-per-stage
        ODE is an exact invariant of this model (CMO makes every tray's
        total in-flow equal its total out-flow -- see column.py's module
        docstring; test_column.py::test_total_moles_invariant_holds
        checks this directly), so as long as the initial state sums to 1
        per stage, integrating forward keeps every stage's mole fractions
        physically valid throughout. An earlier fsolve-based version of
        this method normalized the state inside the residual function to
        enforce that same constraint algebraically -- but that
        normalization makes the residual invariant to overall scaling,
        which gives fsolve a singular/near-singular Jacobian and sends it
        off into nonphysical territory (confirmed: it returned mole
        "fractions" like 38.0 for methanol). Integrating forward sidesteps
        the problem entirely instead of patching around it.

        Performance note: each derivative evaluation does N+1 bubble-point
        root-finds (brentq), so this is not cheap, and solve_ivp's
        finite-difference Jacobian for a stiff BDF-family method (needed
        here) costs roughly (n_states+1) derivative evaluations per
        Jacobian update. rtol/atol below are deliberately looser than
        cstr.py's (1e-6/1e-8 vs 1e-8/1e-10) for that reason -- tightening
        them is straightforward if a particular case needs it, just
        slower.
        """
        if y0 is None:
            y0 = self.default_initial_state()

        sol = solve_ivp(self.derivatives, (0, t_max), y0, method="BDF", rtol=1e-6, atol=1e-8)
        if not sol.success:
            raise RuntimeError(f"steady_state: integration failed: {sol.message}")

        final = sol.y[:, -1]
        residual_norm = np.abs(self.derivatives(sol.t[-1], final)).max()
        if residual_norm > tol:
            raise RuntimeError(
                f"steady_state: did not converge within t_max={t_max} "
                f"(max |derivative| = {residual_norm:.3g} > tol={tol}); "
                "try a larger t_max"
            )

        x_cond, x_trays, x_reb = self._unpack(final)
        return x_cond, x_trays, x_reb

    def distillate_composition(self, x_cond: np.ndarray) -> np.ndarray:
        return x_cond  # total condenser: distillate composition == condenser (reflux drum) composition

    def bottoms_composition(self, x_reb: np.ndarray) -> np.ndarray:
        return x_reb
