"""
Reaction kinetics for methyl acetate esterification, from real published data.

    CH3OH + CH3COOH <=> CH3COOCH3 + H2O
    (methanol)  (acetic acid)  (methyl acetate)  (water)

This is the reaction behind Eastman Chemical's landmark methyl acetate
reactive-distillation process (V. H. Agreda, L. R. Partin, W. H. Heise,
"High-Purity Methyl Acetate via Reactive Distillation," Chem. Eng. Prog.
86(2), 40-46, 1990) -- the same Eastman as the Tennessee Eastman Process
benchmark this whole repo is built around.

Rate law and Arrhenius parameters
----------------------------------
Second-order reversible rate law, homogeneously catalyzed by H2SO4 (2 wt%
relative to the reaction mixture), 1:1 initial mole ratio, well-mixed batch
reactor (agitation speed was shown to have no effect on conversion in the
source study, confirming the reaction is kinetically controlled rather than
mass-transfer limited -- reasonable grounds to treat it as intrinsic
kinetics in a CSTR too):

    r = kf * [CH3OH] * [CH3COOH]  -  kb * [CH3COOCH3] * [H2O]      (mol/(L*min))

    kf(T) = exp(18.207 - 7544.7 / T)      L/(mol*min)
    kb(T) = exp(16.577 - 7538.3 / T)      L/(mol*min)

    (equivalently Ea_f = 62.7 kJ/mol, Ea_b = 62.7 kJ/mol, consistent with
    the source paper's independently reported activation energies of
    62,721 and 62,670 J/mol for forward/backward)

Source: Mekala, M.; Goli, V. R. "Comparative kinetics of esterification of
methanol-acetic acid in the presence of liquid and solid catalysts." Asia-
Pac. J. Chem. Eng. 2014. DOI: 10.1002/apj.1798. (Table 2 gives kf, kb at
four temperatures for the H2SO4-catalyzed case; the Arrhenius fit above is
their Fig. 13 regression, ln(k) vs 1/T.)

Caveat carried over from that source: this rate law was fit at one specific
catalyst loading (2 wt% H2SO4) and 1:1 mole ratio. It's used here as-is,
i.e. catalyst concentration is treated as fixed/implicit rather than as a
simulated state -- reasonable for a CSTR operating at a designed catalyst
dosing, but worth remembering if the model is ever pushed to conditions far
from those the paper tested.

An alternative, more rigorous activity-based (not simple concentration-based)
kinetic model for this same reaction exists in Popken, T.; Gotze, L.;
Gmehling, J. "Reaction Kinetics and Chemical Equilibrium of Homogeneously
and Heterogeneously Catalyzed Acetic Acid Esterification with Methanol and
Methyl Acetate Hydrolysis." Ind. Eng. Chem. Res. 2000, 39, 2601-2611. That
model requires activity coefficients (UNIFAC/ASOG) even for the homogeneous
case and its exact fitted coefficients aren't reproduced in any source
available here -- the concentration-based Mekala & Goli model was chosen
instead specifically because its parameters are fully and exactly
reported, so nothing here is fabricated or approximated from an
inaccessible source.

Reaction enthalpy
------------------
Song et al. reported the enthalpy of methyl acetate synthesis as
approximately -6.5 kJ/mol (cited via Altiokka et al., "Esterification of
acetic acid with methanol over a cation..." reviewing W. Song et al.,
Ind. Eng. Chem. Res. 1998, 37, 1917-1928). Mildly exothermic, consistent
with the small reaction enthalpies (order -3 to -10 kJ/mol) reported across
this whole family of esterification reactions (e.g. ethyl acetate
synthesis measured at -3.6 +/- 0.2 kJ/mol by direct calorimetry).
"""

import numpy as np

# Species order matches physics/thermo.py: [MeOH, AcOH, MeOAc, H2O]
NU = np.array([-1.0, -1.0, 1.0, 1.0])  # stoichiometric coefficients

R_GAS = 8.314462618  # J/(mol K)

# Arrhenius parameters, Mekala & Goli (2014) -- see module docstring.
_LN_A_F, _EA_F_OVER_R = 18.207, 7544.7  # forward: kf = exp(_LN_A_F - _EA_F_OVER_R / T)
_LN_A_B, _EA_B_OVER_R = 16.577, 7538.3  # backward: kb = exp(_LN_A_B - _EA_B_OVER_R / T)

# Song et al. (1998), as cited above. J/mol, per mole of methyl acetate formed.
DELTA_H_RXN = -6500.0


def forward_rate_constant(T: float) -> float:
    """kf(T) in L/(mol*min). T in Kelvin."""
    return np.exp(_LN_A_F - _EA_F_OVER_R / T)


def backward_rate_constant(T: float) -> float:
    """kb(T) in L/(mol*min). T in Kelvin."""
    return np.exp(_LN_A_B - _EA_B_OVER_R / T)


def equilibrium_constant(T: float) -> float:
    """Ke(T) = kf/kb -- concentration-based equilibrium constant."""
    return forward_rate_constant(T) / backward_rate_constant(T)


def rate(concentrations: np.ndarray, T: float) -> float:
    """
    Net rate of reaction, mol/(L*min), positive in the forward
    (ester-forming) direction.

    concentrations: array-like [C_MeOH, C_AcOH, C_MeOAc, C_H2O], mol/L.
    """
    c_meoh, c_acoh, c_meoac, c_h2o = concentrations
    kf = forward_rate_constant(T)
    kb = backward_rate_constant(T)
    return kf * c_meoh * c_acoh - kb * c_meoac * c_h2o
