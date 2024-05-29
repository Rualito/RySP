# -*- coding: utf-8 -*-

import arc  # Import ARC (Alkali Rydberg Calculator)
import numpy as np
import rysp.core.physics.constants as cst
import rysp.core.physics.units as units
import csv
import os

from functools import cache

mub = 9.2740100783*10**(-24)  # Bohr magneton
gs = 2.002318  # electron spin factor
atom = arc.Strontium88()
gl = 1-1/(atom.mass/cst.me)  # electron orbital factor

# Load in all the energies (in cm^(-1)) and RDME (in a.u.)
rdme_energy_file = open(os.path.dirname(__file__) +
                        '/atom_data/Sr88_dipole_matrix_elements_energies.csv')
rdme_energy_table = csv.reader(rdme_energy_file)
header = []
header = next(rdme_energy_table)
rdme_energies = []
for row in rdme_energy_table:
    rdme_energies.append(row[0].split(';'))

rdme_energy_rydberg_file = open(os.path.dirname(
    __file__)+'/atom_data/Sr88_dipole_matrix_elements_energies_rydberg.csv')
rdme_energy_rydberg_table = csv.reader(rdme_energy_rydberg_file)
header = []
header = next(rdme_energy_rydberg_table)
for row in rdme_energy_rydberg_table:
    rdme_energies.append(row[0].split(';'))

rdme_energies = np.array(rdme_energies)


def Saturation_Intensity(decayrate, energy_diff):
    """
    Calculates the saturation intensity as in Eq.(61) of https://steck.us/alkalidata/rubidium85numbers.pdf

    Parameters
    ----------
    decayrate : float
        decayrate Gamma of the transition in Hz.
    energy_diff : float
        energy difference of the transition in Hz.

    Returns
    -------
    float
        saturation intensity in W/m^2.

    """
    return cst.hbar*energy_diff**3*decayrate/(12*np.pi*cst.c**2)


def rabi_frequency_calc(decayrate, I, energy_diff):
    """
    Calculates the Rabi Frequency as in Eq.(49) of https://steck.us/alkalidata/rubidium85numbers.pdf

    Parameters
    ----------
    decayrate : float
        decayrate Gamma of the transition in Hz.
    I : float
        intensity of the laser in W/m^2.
    energy_diff : float
        energy difference of the transition in Hz.

    Returns
    -------
    float
        Rabi frequency in Hz.

    """
    Isat = Saturation_Intensity(decayrate, energy_diff)
    return np.sqrt(decayrate**2*I/(2*Isat))


def total_off_resonance_scattering_rate_calc(transition, frequency, I):
    """
    Calculates the off resonance scattering rate as in Eq.(48) of https://steck.us/alkalidata/rubidium85numbers.pdf

    Parameters
    ----------
    decayrate : float
        decayrate Gamma of the transition in Hz.
    detuning : float
        detuning from the transition in Hz.
    I : float
        intensity of the laser in W/m^2.
    energy_diff : float
        energy difference of the transition in Hz.

    Returns
    -------
    float
        off resonance scattering rate of the transition in Hz.

    """
    n1, l1, j1, m1, s1, n2, l2, j2, m2, s2 = transition
    from_state = n1, l1, j1, m1, s1
    to_state = n2, l2, j2, m2, s2

    index_trans_1 = np.where((rdme_energies[:, [2, 4, 5, 6, 9, 11, 12, 13]] == [str(
        n1), str(l1), str(j1), str(s1), str(n2), str(l2), str(j2), str(s2)]).all(axis=1))

    index_trans_2 = np.where((rdme_energies[:, [9, 11, 12, 13, 2, 4, 5, 6]] == [str(
        n1), str(l1), str(j1), str(s1), str(n2), str(l2), str(j2), str(s2)]).all(axis=1))

    # indexes which correspond to the transition (may be empty - clock transition)
    index_trans = np.hstack([index_trans_1, index_trans_2])

    # it is fine if the transition isn't found, which is the case for the clock transition
    # if len(index_trans[0]) == 0:
    #     print(index_trans)
    #     raise ValueError(
    #         f"total_off_resonance_scattering_rate_calc: no transition found for given values, make sure transition is in the right order. {transition=}")

    from_state_index = [2, 4, 5, 6]
    to_state_index = [9, 11, 12, 13]
    # indexes from
    indexes1 = np.where((rdme_energies[:, from_state_index] == [
                        str(n1), str(l1), str(j1), str(s1)]).all(axis=1))

    states1 = []
    # 'indexes from' which do not correspond to the transition
    for ind in np.setdiff1d(indexes1, index_trans):
        state = rdme_energies[ind, to_state_index]
        n, l, j, s = [int(st) for st in state]
        energydiff, decayrate = get_transition_parameters(
            (*from_state, n, l, j, 0, s))

        if (energydiff is None) or (decayrate is None):
            continue
        detuning = frequency-energydiff

        scat_rate = off_resonance_scattering_rate_calc(
            decayrate, detuning, I, energydiff)

        states1 += [[(n, l, j, s), scat_rate]]

    # indexes from (transposed)
    indexes2 = np.where((rdme_energies[:, to_state_index] == [
                        str(n1), str(l1), str(j1), str(s1)]).all(axis=1))

    # 'indexes from (transposed)' which do not correspond to the transition
    for ind in np.setdiff1d(indexes2, index_trans):
        state = rdme_energies[ind, from_state_index]
        n, l, j, s = [int(st) for st in state]
        energydiff, decayrate = get_transition_parameters(
            (n, l, j, 0, s, *to_state))

        if (energydiff is None) or (decayrate is None):
            continue

        detuning = frequency-energydiff

        scat_rate = off_resonance_scattering_rate_calc(
            decayrate, detuning, I, energydiff)

        states1 += [[(n, l, j, s), scat_rate]]

    # indexes to (transposed)
    indexes3 = np.where((rdme_energies[:, from_state_index] == [
                        str(n2), str(l2), str(j2), str(s2)]).all(axis=1))

    states2 = []
    # 'indexes to (transposed)' which do not correspond to the transition
    for ind in np.setdiff1d(indexes3, index_trans):
        state = rdme_energies[ind, to_state_index]
        n, l, j, s = [int(st) for st in state]
        energydiff, decayrate = get_transition_parameters(
            (*from_state, n, l, j, 0, s))

        if (energydiff is None) or (decayrate is None):
            continue

        detuning = frequency-energydiff

        scat_rate = off_resonance_scattering_rate_calc(
            decayrate, detuning, I, energydiff)

        states2 += [[(n, l, j, s), scat_rate]]

    # indexes to
    indexes4 = np.where((rdme_energies[:, to_state_index] == [
                        str(n2), str(l2), str(j2), str(s2)]).all(axis=1))

    # 'indexes to' which do not correspond to the transition
    for ind in np.setdiff1d(indexes4, index_trans):
        state = rdme_energies[ind, from_state_index]
        n, l, j, s = [int(st) for st in state]
        energydiff, decayrate = get_transition_parameters(
            (n, l, j, 0, s, *to_state))

        if (energydiff is None) or (decayrate is None):
            continue

        detuning = frequency-energydiff

        scat_rate = off_resonance_scattering_rate_calc(
            decayrate, detuning, I, energydiff)

        states2 += [[(n, l, j, s), scat_rate]]

    total_rate_1 = 0
    total_rate_2 = 0

    # summing over all of the scattering rates
    for k in range(len(states1)):
        total_rate_1 += states1[k][1]
    for k in range(len(states2)):
        total_rate_2 += states2[k][1]

    return total_rate_1, total_rate_2, states1, states2


def off_resonance_scattering_rate_calc(decayrate, detuning, I, energy_diff):
    """
    Calculates the off resonance scattering rate as in Eq.(48) of https://steck.us/alkalidata/rubidium85numbers.pdf

    Parameters
    ----------
    decayrate : float
        decayrate Gamma of the transition in Hz.
    detuning : float
        detuning from the transition in Hz.
    I : float
        intensity of the laser in W/m^2.
    energy_diff : float
        energy difference of the transition in Hz.

    Returns
    -------
    float
        off resonance scattering rate of the transition in Hz.

    """

    Isat = Saturation_Intensity(decayrate, energy_diff)
    s0 = I/Isat
    return s0*decayrate/2*1/(1+s0+(2*detuning/decayrate)**2)


def rabi_frequency_calc_clock(I, B):
    """
    Returns the Rabi frequency on the clock transition

    Parameters
    ----------
    I : float
        intensity of the laser in W/m^2.

    Returns
    -------
    float
        Rabi frequency in Hz.

    """
    # muc = np.sqrt(2/3)*(gl - cst.gs)*cst.mub
    # energydiff3P03P1 = 2*np.pi*5.606*10**12  # Hz
    # Gamma3P11S0 = 2*np.pi*7.476*10**3  # Hz
    # energydiff3P01S0 = units.cmmin1_to_freq(14317.507)
    # Gamma3P01S0 = Gamma3P11S0*muc**2*B**2/cst.hbar**2/energydiff3P03P1**2  # Hz
    energydiff3P01S0, Gamma3P01S0 = get_transition_parameters(
        (5, 0, 0, 0, 0, 5, 1, 0, 0, 1), B)
    Isat = Saturation_Intensity(Gamma3P01S0, energydiff3P01S0)
    return np.sqrt(Gamma3P01S0**2*I/(2*Isat))


@cache
def get_energy_diff(transition):
    n1, l1, j1, m1, s1, n2, l2, j2, m2, s2 = transition
    if [n1, l1, j1, s1, n2, l2, j2, s2] == [5, 0, 0, 0, 5, 1, 0, 1]:
        return units.cmmin1_to_freq(14317.507)
    else:

        indexes = np.where((rdme_energies[:, [2, 4, 5, 6, 9, 11, 12, 13]] == [str(
            n1), str(l1), str(j1), str(s1), str(n2), str(l2), str(j2), str(s2)]).all(axis=1))
        if len(indexes[0]) == 0:
            raise ValueError(
                f"No transition found for given values, make sure transition is in the right order. {transition=}")
        energycmmin1 = np.float64(rdme_energies[indexes, 15])
        energydiff = units.cmmin1_to_freq(energycmmin1)
        return energydiff


def rabi_frequency_calc_trans(transition, I, B=0):
    """
    Calculates the Rabi Frequency as in Eq.(49) of https://steck.us/alkalidata/rubidium85numbers.pdf

    Parameters
    ----------
    transition : [n1,l1,j1,s1,n2,l2,j2,s2]
    I : float
        intensity of the laser in W/m^2.
    B : float
        magnetic field strength in T (only for clock transition)

    Returns
    -------
    float
        Rabi frequency in Hz.

    """
    energydiff, decayrate = get_transition_parameters(transition, B)
    Isat = Saturation_Intensity(decayrate, energydiff)
    return np.sqrt(decayrate**2*I/(2*Isat))


def decay_rate(transition, B=0):
    return get_transition_parameters(transition, B)[1]


def energy_diff(transition):
    return get_transition_parameters(transition)[0]


def decay_rate_state(state, B=0):
    n1, l1, j1, m1, s1 = state

    if state == [5, 1, 0, 0, 1]:
        return 0  # hardcoded 0 decay rate for the clock state

    from_state_index = [2, 4, 5, 6]
    to_state_index = [9, 11, 12, 13]

    # indexes that include the state
    indexes1 = np.where((rdme_energies[:, from_state_index] == [
        str(n1), str(l1), str(j1), str(s1)]).all(axis=1))
    indexes2 = np.where((rdme_energies[:, to_state_index] == [
        str(n1), str(l1), str(j1), str(s1)]).all(axis=1))
    total_rate = 0
    for ind in indexes1[0]:
        state_from = state
        n, l, j, s = np.int32(rdme_energies[ind, to_state_index])
        state_to = n, l, j, 0, s
        total_rate += decay_rate((*state_from, *state_to))

    for ind in indexes2[0]:
        state_to = state
        n, l, j, s = np.int32(rdme_energies[ind, from_state_index])
        state_from = n, l, j, 0, s
        total_rate += decay_rate((*state_from, *state_to))

    return total_rate


@cache
def get_transition_parameters(transition: tuple, B=0) -> tuple[float | None, float | None]:
    """
    get_transition_parameters Given a transition tuple, obtains the energy difference and the decay rate associated with the transition. The 3P0 to 1S0 transition is hardcoded. 


    Parameters
    ----------
    transition : tuple
        With the format (n1, l1, j1, m1, s1, n2, l2, j2, m2, s2). Note that m1, m2 are not used, and are considered only for convinience of use. 

    Returns
    -------
    tuple[float, float]
        Tuple (energy, decay_rate) for the given transition, in units of (Hz, Hz)

    Raises
    ------
    ValueError
        Is thrown if no transition is found on the aggregated Sr88_dipole_matrix_elements[...].csv file
    """
    n1, l1, j1, m1, s1, n2, l2, j2, m2, s2 = transition
    if [n1, l1, j1, s1, n2, l2, j2, s2] == [5, 0, 0, 0, 5, 1, 0, 1]:
        # For I=10 W/m^2 =1 mW/cm^2 and B=1T we get 200 as in https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.96.083001

        muc = np.sqrt(2/3)*(gl - cst.gs)*cst.mub
        energydiff3P03P1 = 2*np.pi*5.606*10**12  # Hz
        Gamma3P11S0 = 2*np.pi*7.476*10**3  # Hz
        Gamma3P01S0 = Gamma3P11S0*muc**2*B**2/cst.hbar**2/energydiff3P03P1**2

        energydiff3P01S0 = units.cmmin1_to_freq(14317.507)
        return energydiff3P01S0, Gamma3P01S0
    indexes = np.where((rdme_energies[:, [2, 4, 5, 6, 9, 11, 12, 13]] == [str(
        n1), str(l1), str(j1), str(s1), str(n2), str(l2), str(j2), str(s2)]).all(axis=1))
    if len(indexes[0]) == 0:
        return None, None
        # raise ValueError(
        #     f"No transition found for given values, make sure transition is in the right order. {transition=}")

    energycmmin1 = np.float64(rdme_energies[indexes, 15])
    rdme = np.float64(rdme_energies[indexes, 14])
    energydiff = units.cmmin1_to_freq(energycmmin1)
    decayrate = units.rdme_to_rate(rdme, j2, energydiff, 0)
    # print(rdme, energydiff, indexes)
    return energydiff, decayrate
