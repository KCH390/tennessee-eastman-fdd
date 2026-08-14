"""
Cantera species thermodynamics for the methyl acetate esterification system.

CH3OH + CH3COOH <=> CH3COOCH3 + H2O   (methanol + acetic acid <=> methyl acetate + water)

Scope note (see design discussion in project chat / PR description): Cantera's
built-in Reactor/ReactorNet classes and most of its phase models are built for
gas-phase/combustion use cases, and there's no first-class Cantera support for
non-ideal liquid VLE (NRTL/UNIQUAC-type activity coefficients). So Cantera's
job here is narrow and deliberate: provide the per-species standard-state
molar heat capacity (cp) for the four liquid species, via a minimal
`ideal-condensed` phase with `constant-cp` species thermo. The CSTR's mass
and energy balance ODEs are integrated separately (see cstr.py, scipy
solve_ivp) rather than through Cantera's Reactor/ReactorNet, and the reaction
rate law is evaluated directly from the published kinetics (see reaction.py)
rather than through Cantera's reaction-rate machinery, since that machinery
is built around elementary-reaction Arrhenius forms and doesn't add anything
here.

Species order used throughout Track 2 (CSTR, and later the column):
    0: CH3OH      methanol
    1: CH3COOH    acetic acid
    2: CH3COOCH3  methyl acetate
    3: H2O        water

Data sources for the constants below:
  - Liquid heat capacities (cp0, J/mol/K, ~298 K): representative literature
    values (NIST WebBook / Perry's Chemical Engineers' Handbook order of
    magnitude for these four common liquids). Treated as constant over the
    reactor's operating temperature range -- a stated simplification, not a
    temperature-dependent correlation (e.g. DIPPR). If the column phase later
    needs tighter accuracy this is the first place to revisit.
  - Liquid molar volumes: back-calculated from standard liquid densities near
    room temperature and each species' molecular weight. Only used to satisfy
    the `ideal-condensed` phase's equation-of-state requirement (Cantera
    needs *some* volumetric info for a condensed phase); not used elsewhere
    in the CSTR model.
  - Reference enthalpy/entropy (h0, s0) are set to 0 for all four species.
    This is deliberate: we are NOT using Cantera to compute the heat of
    reaction from a difference of formation enthalpies (that would require
    verified liquid-phase heat-of-formation data for methyl acetate, which
    we don't have to hand). Instead, the heat of reaction is taken directly
    from a cited literature value in reaction.py. Cantera is only asked for
    Cp here, so h0/s0 are irrelevant to every calculation this module is
    used for -- leaving them at 0 avoids implying a precision we don't have.
"""

import cantera as ct

SPECIES = ["CH3OH", "CH3COOH", "CH3COOCH3", "H2O"]
SPECIES_LABELS = {
    "CH3OH": "methanol",
    "CH3COOH": "acetic acid",
    "CH3COOCH3": "methyl acetate",
    "H2O": "water",
}

# Molecular weights, g/mol (standard values).
MOLECULAR_WEIGHTS = {
    "CH3OH": 32.04,
    "CH3COOH": 60.05,
    "CH3COOCH3": 74.08,
    "H2O": 18.02,
}

# Liquid densities near 298 K, g/cm^3 (standard reference values), used only
# to derive a molar volume for the ideal-condensed phase's equation of state.
_DENSITIES_G_PER_CM3 = {
    "CH3OH": 0.792,
    "CH3COOH": 1.049,
    "CH3COOCH3": 0.932,
    "H2O": 0.997,
}

# Liquid heat capacities near 298 K, J/(mol K) -- see module docstring.
_CP_J_PER_MOL_K = {
    "CH3OH": 81.1,
    "CH3COOH": 123.3,
    "CH3COOCH3": 143.5,
    "H2O": 75.3,
}


def _molar_volume_cm3(species: str) -> float:
    return MOLECULAR_WEIGHTS[species] / _DENSITIES_G_PER_CM3[species]


def pure_liquid_molar_concentration(species: str) -> float:
    """
    Molar concentration (mol/L) of the pure liquid species at ~298 K, from
    the same density values used for the Cantera equation-of-state above.
    Used by cstr.py to characterize the feed streams (e.g. "the acetic acid
    feed is ~glacial acetic acid, so its inlet concentration is essentially
    the pure-liquid concentration").
    """
    density_g_per_l = _DENSITIES_G_PER_CM3[species] * 1000.0
    return density_g_per_l / MOLECULAR_WEIGHTS[species]


def _species_yaml_block(species: str) -> str:
    return f"""
- name: {species}
  composition: {{}}
  thermo:
    model: constant-cp
    T0: 298.15 K
    h0: 0.0 J/mol
    s0: 0.0 J/mol/K
    cp0: {_CP_J_PER_MOL_K[species]} J/mol/K
  equation-of-state:
    model: constant-volume
    molar-volume: {_molar_volume_cm3(species):.4f} cm^3/mol
"""


def _build_yaml() -> str:
    species_blocks = "".join(_species_yaml_block(s) for s in SPECIES)
    species_names = ", ".join(SPECIES)
    return f"""
phases:
- name: liquid_mixture
  thermo: ideal-condensed
  species: [{species_names}]
  state: {{T: 298.15 K, P: 1 atm}}

species:{species_blocks}
"""


def make_solution() -> "ct.Solution":
    """
    Build the Cantera `Solution` for the four-species liquid mixture.

    NOTE: `composition: {}` is used for each species (no elemental formula)
    since we never invoke element-balance/equilibrium features of Cantera
    here -- only per-species Cp. If a later phase of this project wants
    Cantera's equilibrium solver too, these need real elemental compositions
    (e.g. CH3OH -> {C:1, H:4, O:1}) filled in first.
    """
    return ct.Solution(yaml=_build_yaml())


def species_cp(solution: "ct.Solution" = None):
    """
    Per-species molar heat capacity, J/(mol K), in SPECIES order.

    Constant-cp species thermo means this doesn't actually depend on the
    solution's current T -- included as a function (rather than just
    exporting the _CP_J_PER_MOL_K dict) so callers go through the Cantera
    object, consistent with "Cantera supplies the thermo" from the design.
    """
    sol = solution or make_solution()
    r = 8.314462618  # J/(mol K)
    # standard_cp_R is cp/R for each species at the solution's current T, P.
    return sol.standard_cp_R * r
