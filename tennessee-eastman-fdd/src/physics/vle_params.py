"""
NRTL binary interaction parameters for the quaternary system:
    0: CH3OH      methanol
    1: CH3COOH    acetic acid
    2: CH3COOCH3  methyl acetate
    3: H2O        water

(Same species order as physics/thermo.py and physics/reaction.py.)

Status: 2 of the 6 binary pairs are backed by real, cited regressions.
The other 4 are NOT YET SOURCED and are left as ideal (dg=0, i.e. tau=0,
gamma=1) rather than filled with invented numbers -- see
UNSOURCED_PAIRS below and the "what's missing" note at the bottom of
this docstring.

Sourced pairs
-------------
CH3OH-H2O (methanol-water), indices (0,3):
    dg[0,3] = -1062.2 J/mol   (b12 = -253.88 cal/mol)
    dg[3,0] =  3536.4 J/mol   (b21 =  845.21 cal/mol)
    alpha   =  0.2994
    Source: Table 12.5 / 13.10 in Smith, Van Ness & Abbott, "Introduction
    to Chemical Engineering Thermodynamics" -- values there are stated as
    "those recommended by Gmehling et al., Vapor-Liquid Equilibrium Data
    Collection, Chemistry Data Series, vol. I, parts 1a, 1b, 2c and 2e,
    DECHEMA, Frankfurt/Main, 1981-1988." (i.e. the standard DECHEMA VLE
    compilation, not a value we derived ourselves).

CH3COOCH3-CH3OH (methyl acetate-methanol), indices (2,0):
    dg[2,0] =  -130.5 J/mol   (b12 =  -31.19 cal/mol)
    dg[0,2] =  3402.6 J/mol   (b21 =  813.18 cal/mol)
    alpha   =  0.2965
    Source: same DECHEMA-recommended table as above.

    This is the single most important pair in the whole system: it's the
    one that forms the azeotrope (~67 mol% methyl acetate at atmospheric
    pressure) that makes distillation alone insufficient to separate
    unreacted methanol from product -- the whole reason this is a
    "distillation isn't enough either" system worth modeling. A second,
    independent source (Graczova, Dobcsanyi & Steltenpohl, 2017,
    "Separation of Methyl Acetate-Methanol Azeotropic Mixture Using
    [Emim][triflate]," Chem. Eng. Trans. 61, 1183-1188, Table 2 --
    regressed from Orchilles et al. 2007 isobaric VLE data) gives a
    materially different parameter set for this same pair (dg12=1498.5,
    dg21=1550.1 J/mol, alpha=0.300). Both are legitimate regressions
    against real data, just fit to different data sets/temperature
    ranges -- they are NOT reconciled here. The DECHEMA-table values
    were chosen for this module for consistency with the also-DECHEMA-
    sourced methanol-water pair; the Graczova et al. values are the
    better choice if this module is ever extended toward the reduced-
    pressure/ionic-liquid conditions that paper actually studied.
    test_nrtl.py checks the DECHEMA-parameter azeotrope prediction lands
    in a physically reasonable range, not against one specific literature
    x value, given this discrepancy between sources.

NOT YET SOURCED
----------------
    CH3OH-CH3COOH    (0,1)  methanol-acetic acid
    CH3COOCH3-CH3COOH (2,1) methyl acetate-acetic acid
    H2O-CH3COOH       (3,1) water-acetic acid
    H2O-CH3COOCH3     (3,2) water-methyl acetate

The real target for these is Blazej, Kroupa & Dohnal (2006), "Isothermal
vapour-liquid equilibrium with chemical reaction in the quaternary water +
methanol + acetic acid + methyl acetate system, and in five binary
subsystems," Fluid Phase Equilib. -- confirmed (via its abstract) to
report NRTL parameters for exactly this quaternary system, including the
reacting pairs. Its full parameter table sits behind a paywall this
project doesn't have access to, so rather than approximate or invent
those 4 pairs, they're left at dg=0 (ideal solution, gamma=1) and marked
in UNSOURCED_PAIRS below. Any VLE/distillation result that leans heavily
on acetic acid's non-ideality (e.g. exact bubble points in acetic-acid-
rich regions) should be treated as provisional until these are replaced
with real data -- the two sourced pairs above are the ones that matter
most for the methanol/methyl acetate azeotrope specifically, which is
usable now.
"""

import numpy as np

SPECIES = ["CH3OH", "CH3COOH", "CH3COOCH3", "H2O"]
N = len(SPECIES)

_CAL_TO_J = 4.184

# dg[i, j] in J/mol. Only entries listed below are non-zero.
DG = np.zeros((N, N))
ALPHA = np.zeros((N, N))

# --- CH3OH (0) - H2O (3): DECHEMA/Gmehling-recommended -------------------
DG[0, 3] = -253.88 * _CAL_TO_J
DG[3, 0] = 845.21 * _CAL_TO_J
ALPHA[0, 3] = ALPHA[3, 0] = 0.2994

# --- CH3COOCH3 (2) - CH3OH (0): DECHEMA/Gmehling-recommended --------------
DG[2, 0] = -31.19 * _CAL_TO_J
DG[0, 2] = 813.18 * _CAL_TO_J
ALPHA[2, 0] = ALPHA[0, 2] = 0.2965

# Pairs still needing real literature parameters (see docstring above).
# Left as ideal (dg=0) -- NOT a claim that these pairs are actually ideal.
UNSOURCED_PAIRS = [(0, 1), (2, 1), (3, 1), (3, 2)]
