"""
Non-isothermal CSTR for methyl acetate esterification.

    CH3OH + CH3COOH <=> CH3COOCH3 + H2O

Two separate feed streams -- a methanol feed and an acetic acid feed -- enter
and mix within a single well-mixed liquid volume, rather than assuming a
pre-mixed feed. This is deliberate (see design discussion): it's the mixing
phenomenon most directly relevant to this reactor, and it's what makes a
feed-ratio disturbance (Phase 8 fault scenario) physically meaningful later
-- "feed disturbance" can mean literally unbalancing the two stream flow
rates, not just perturbing an already-combined feed concentration.

Mass balance (per species i, well-mixed constant-volume liquid CSTR):

    dC_i/dt = (Q_A*CA_in_i + Q_M*CM_in_i - (Q_A+Q_M)*C_i) / V + nu_i * r(C, T)

Energy balance:

    Because species Cp is constant-in-T in our thermo model (physics/thermo.py
    uses constant-cp species thermo), the accumulation term reduces exactly
    (not just approximately) to (V * sum_i C_i*cp_i) * dT/dt -- no need to
    separately track d(cp)/dt. Sensible heat is computed per species (each
    inlet species carried in at its own cp, weighted by its own molar flow),
    which is more direct than assuming a single lumped "stream Cp":

    V*sum(C_i*cp_i)*dT/dt =
          sum_i cp_i * [Q_A*CA_in_i*(T_A_in - T) + Q_M*CM_in_i*(T_M_in - T)]
        + (-DELTA_H_RXN) * r(C,T) * V
        - UA*(T - T_jacket)

Cooling jacket term UA*(T - T_jacket) is the hook for a "cooling failure"
fault scenario (Phase 8): drop UA toward 0, or step T_jacket up, mid-run and
watch the temperature response -- see tests/test_physics_cstr.py.

Known simplifications, stated rather than hidden:
  - Heat of mixing (excess enthalpy of the non-ideal liquid mixture) is NOT
    included here -- the energy balance above only has heat of reaction and
    sensible heat. Modeling it properly needs an activity-coefficient model
    (NRTL) for this quaternary system, which is being built for the
    distillation column (where it matters more, since methanol/methyl
    acetate form an azeotrope). Once that NRTL module exists it can be
    wired into this energy balance too; deferred for now rather than
    guessing at un-cited excess-enthalpy parameters.
  - Constant liquid volume/density (no volume-of-mixing or volume change on
    reaction) -- standard simplifying assumption for a liquid-phase CSTR.
  - Catalyst (H2SO4) concentration is implicit in the rate law (see
    reaction.py) rather than tracked as a state.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from . import reaction, thermo

N_SPECIES = 4  # CH3OH, CH3COOH, CH3COOCH3, H2O -- see thermo.SPECIES


@dataclass
class CSTRParams:
    V: float  # working liquid volume, L

    Q_acid: float  # acetic-acid feed stream volumetric flow, L/min
    Q_meoh: float  # methanol feed stream volumetric flow, L/min

    T_acid_in: float  # acetic-acid feed temperature, K
    T_meoh_in: float  # methanol feed temperature, K

    UA: float  # cooling jacket UA, J/(min K)
    T_jacket: float  # cooling jacket temperature, K

    # Feed stream compositions default to (near-)pure liquids; override for
    # e.g. aqueous acetic acid or a methanol/water feed.
    C_acid_in: np.ndarray = None
    C_meoh_in: np.ndarray = None

    def __post_init__(self):
        if self.C_acid_in is None:
            self.C_acid_in = np.array(
                [0.0, thermo.pure_liquid_molar_concentration("CH3COOH"), 0.0, 0.0]
            )
        if self.C_meoh_in is None:
            self.C_meoh_in = np.array(
                [thermo.pure_liquid_molar_concentration("CH3OH"), 0.0, 0.0, 0.0]
            )


class CSTR:
    """
    Usage:
        params = CSTRParams(V=100, Q_acid=1.0, Q_meoh=1.0, T_acid_in=298.15,
                             T_meoh_in=298.15, UA=500.0, T_jacket=298.15)
        reactor = CSTR(params)
        C_ss, T_ss = reactor.steady_state()
        t, y = reactor.simulate(y0=[*C_ss, T_ss], t_span=(0, 60))
    """

    def __init__(self, params: CSTRParams):
        self.p = params
        self._solution = thermo.make_solution()
        self._cp = thermo.species_cp(self._solution)  # J/(mol K), constant

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        C = y[:N_SPECIES]
        T = y[N_SPECIES]
        p = self.p

        r = reaction.rate(C, T)
        Q_total = p.Q_acid + p.Q_meoh

        dCdt = (
            p.Q_acid * p.C_acid_in + p.Q_meoh * p.C_meoh_in - Q_total * C
        ) / p.V + reaction.NU * r

        sensible_in = np.sum(
            self._cp
            * (
                p.Q_acid * p.C_acid_in * (p.T_acid_in - T)
                + p.Q_meoh * p.C_meoh_in * (p.T_meoh_in - T)
            )
        )
        reaction_heat = -reaction.DELTA_H_RXN * r * p.V
        cooling = p.UA * (T - p.T_jacket)

        heat_capacitance = p.V * np.sum(C * self._cp)  # J/K, > 0 as long as C>0
        dTdt = (sensible_in + reaction_heat - cooling) / heat_capacitance

        return np.concatenate([dCdt, [dTdt]])

    def simulate(self, y0, t_span, t_eval=None, **solve_ivp_kwargs):
        """
        Integrate the CSTR ODEs. t in minutes (matches reaction.py's rate
        constant units). y0 = [C_MeOH, C_AcOH, C_MeOAc, C_H2O, T].
        """
        sol = solve_ivp(
            self.derivatives,
            t_span,
            y0,
            t_eval=t_eval,
            method="LSODA",  # handles the stiffness a fast reaction + slow thermal response can cause
            **solve_ivp_kwargs,
        )
        return sol

    def steady_state(self, y0_guess=None):
        """
        Solve dydt = 0 for the steady-state concentrations and temperature,
        starting from y0_guess (defaults to feed conditions, unreacted).
        Returns (C_ss, T_ss).
        """
        p = self.p
        if y0_guess is None:
            Q_total = p.Q_acid + p.Q_meoh
            C_guess = (p.Q_acid * p.C_acid_in + p.Q_meoh * p.C_meoh_in) / Q_total
            T_guess = (
                p.Q_acid * p.T_acid_in + p.Q_meoh * p.T_meoh_in
            ) / Q_total
            y0_guess = np.concatenate([C_guess, [T_guess]])

        def residual(y):
            return self.derivatives(0.0, y)

        y_ss = fsolve(residual, y0_guess)
        return y_ss[:N_SPECIES], y_ss[N_SPECIES]

    def conversion(self, C: np.ndarray, limiting_index: int = 1) -> float:
        """
        Fractional conversion of the limiting reactant (default: acetic
        acid, index 1) relative to its feed concentration at the current
        operating point (mass-balance based, not composition-based).
        """
        p = self.p
        Q_total = p.Q_acid + p.Q_meoh
        C_in_limiting = (
            p.Q_acid * p.C_acid_in[limiting_index] + p.Q_meoh * p.C_meoh_in[limiting_index]
        ) / Q_total
        if C_in_limiting <= 0:
            return 0.0
        return 1.0 - C[limiting_index] / C_in_limiting
