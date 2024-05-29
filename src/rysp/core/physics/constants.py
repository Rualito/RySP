import numpy as np
# ! NIST book of constants
# ! scipy + cross-reference with NIST
import scipy.constants as scst

c = scst.c
hbar = scst.hbar
eps0 = scst.epsilon_0
e0 = scst.elementary_charge
Eh = scst.physical_constants['Hartree energy'][0]
a0 = scst.physical_constants['atomic unit of length'][0]
me = scst.physical_constants['electron mass'][0]
amu = scst.physical_constants['unified atomic mass unit'][0]
alpha = scst.physical_constants['inverse fine-structure constant'][0]
kb = scst.physical_constants['Boltzmann constant'][0]
mub = scst.physical_constants['Bohr magneton'][0]  # Bohr magneton
# electron spin factor
gs = abs(scst.physical_constants['electron g factor'][0])
m_elec = me  # mass of strontium atom
