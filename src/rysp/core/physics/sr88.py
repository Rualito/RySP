# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:09:42 2022

@author: s165827
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:27:23 2022

@author: s165827
"""


# Load in all the energies (in cm^(-1)) and RDME (in a.u.)
import functools
import csv
import numpy as np
import rysp.core.physics.constants as cst
import os
from arc.wigner import Wigner6j
import arc  # Import ARC (Alkali Rydberg Calculator)
from rysp.core.physics import units
rdme_energy_file = open(
    os.path.dirname(__file__)+'/atom_data/Sr88_dipole_matrix_elements_energies.csv')
rdme_energy_table = csv.reader(rdme_energy_file)
header = []
header = next(rdme_energy_table)
rdme_energies = []
for row in rdme_energy_table:
    rdme_energies.append(row[0].split(';'))
rdme_energies = np.array(rdme_energies)
# Loading in the 5sns 3S1 electron wave functions and polarization factors for n=50-70
# as generated in ElectronPolarizationFactorCalc.py
ElectronWaveFuncMNorm = np.load(os.path.dirname(
    __file__)+"/electron_wf/ElectronWaveFuncMNorm.npy")
ElectronPolFactors = np.load(os.path.dirname(
    __file__)+"/electron_wf/ElectronPolFactors.npy")


def pol_scal(j0, w, energies, pols, js):
    """
    Calculates the scalar polarizability based on Eq.2 of DOI:10.1103/PhysRevA.94.012505

    Parameters
    ----------
    basis : np.array
        Array containing the list of strings describing the coupled states as
        [num1,sym1,num2,sym2,l,j,s]
    j0 : Float
        Total angular momentum of state to calculate
    w : Float
        Wavelength of the laser
    energies : Array
        Array of interacting state and energies as [num1,sym1,num2,sym2,l,j,s,energy]
    pols : Array
        Array of interacting state and reduced density matrix elements as [num1,sym1,num2,sym2,l,j,s,DME]

    Returns
    -------
    Float
        Scalar polarizability of state.
    """
    pol_scal = 0
    for k in range(np.size(energies)):
        dme = np.float64(pols[0][k])
        dE = np.float64(energies[0][k])
        deltaE = units.wavenumber_to_frequency(dE)
        pol_scal += units.hartree_to_freq(dme**2)*10**9*deltaE/(deltaE**2-w**2)
    return pol_scal*2/(3*(2*j0+1))


def pol_tens(j0, w, energies, pols, js):
    """
    Calculates the tensor polarizability based on Eq.2 of DOI:10.1103/PhysRevA.94.012505

    Parameters
    ----------
    basis : np.array
        Array containing the list of strings describing the coupled states as
        [num1,sym1,num2,sym2,l,j,s]
    j0 : Float
        Total angular momentum of state to calculate
    w : Float
        Wavelength of the laser
    energies : Array
        Array of interacting state and energies as [num1,sym1,num2,sym2,l,j,s,energy]
    pols : Array
        Array of interacting state and matrix elements as [num1,sym1,num2,sym2,l,j,s,DME]

    Returns
    -------
    Float
        Scalar polarizability of state.
    """
    pol_tens = 0
    C = (5*j0*(2*j0-1)/(6*(j0+1)*(2*j0+1)*(2*j0+3)))**(1/2)
    for k in range(np.size(energies)):
        dme = np.float64(pols[0][k])
        dE = np.float64(energies[0][k])
        deltaE = units.wavenumber_to_frequency(dE)
        j = float(js[0][k])
        try:
            pol_tens += (-1)**(j0+j)*Wigner6j(j0, 1, j, 1, j0, 2) * \
                units.hartree_to_freq(dme**2)*10**9*deltaE/(deltaE**2-w**2)
        except ValueError as e:
            if str(e) != '6j-Symbol is not triangular!':
                raise
    return pol_tens*4*C


def pol_core_other(n, m, l, j, s, w):
    """
    These are fits of the other contributions e.g. a.c., core or valence contributions
    based on data from DOI: 10.1103/PhysRevA.87.012509. See also mathematica notebook
    florian schreck

    Parameters
    ----------
    level : string
        string describing the level.
    w : float
        laser wavelength.

    Returns
    -------
    float

    """
    w = units.freq_to_wavelength(w)*10**9
    # 5s5s1S0
    if [n, m, l, j, s] == [5, 0, 0, 0, 0]:
        return 5.3+7.3-0.1/4.8*(w-515.2)
    # 5s5p3P0
    elif [n, m, l, j, s] == [5, 0, 1, 0, 1]:
        return 19.97+2173.54/(w-322.644)
    # 5s5p3P1
    elif [n, m, l, j, s] == [5, 0, 1, 1, 1]:
        return 5.8+17.9259+10231/(w-306.295)
    # 5s5p3P2
    elif [n, m, l, j, s] == [5, 0, 1, 2, 1]:
        return 5.6 + 41.2
    # 4s4s2S12
    elif [n, m, l, j, s] == [4, 0.5, 0, 0.5, 0.5]:
        return 5.4
    else:
        return 0


def pol_total(pol_scal, pol_tens, n0, m0, l0, j0, s0, w):
    """
    Calculates the total polarizability of a state based on DOI:10.1103/PhysRevA.94.012505

    Parameters
    ----------
    pol_scal : float
        scalar polarizability as in pol_scal()
    pol_tens : float
        tensor polarizability as in pol_tens()
    m0 : int
        magnetic quantum number of the state
    j0 : float
        total angular momentum of the state
    level : string
        basis level as string.
    w : float
        wavelength of the laser.

    Returns
    -------
    float
        total polarizability of the level given the laser.

    """

    if (m0 == 0 and j0 == 0) or (m0 == 0.5 and j0 == 0.5):
        return pol_scal+pol_core_other(n0, m0, l0, j0, s0, w)
    else:
        print(f'{j0=}, {m0=}')
        return pol_scal+pol_tens*(3*m0**2-j0*(j0+1))/(j0*(2*j0-1))+pol_core_other(n0, m0, l0, j0, s0, w)


def calc_electron_polarizability(wavelength):
    """
    calculates the electron polarizability (See Eq.9 of DOI:1308.0573)

    Parameters
    ----------
    wavelength : float
        wavelength of laser in m.

    Returns
    -------
    float
        electron polarizability in a.u..

    """
    return units.cm2Vmin1_to_AU(cst.e0**2/(cst.me*(units.wavelength_to_freq(wavelength))**2))


def calc_polarizability(wavelength, n, m, l, j=0., s=0., laser_extend=True, mode="normal"):
    """
    Calculates the polarizability of the state in a.u. given a laser wavelength

    Parameters
    ----------
    wavelength : float
        wavelength of laser in [m].
    n : int, optional
        primal quantum number. The default is 5.
    m : float, optional
        magnetic quantum number. The default is 0.
    l : float, optional
        azimuthal quantum number. The default is 0.
    j : float, optional
        total angular momentum. The default is 0.
    s : float, optional
        spin number. The default is 0.
    laser_extend : bool, optional
        take into account the finite extend of the laser, if false electron_pol_factor=1    
    mode : "normal", "ionic"
        determines whether a normal calculation is done or the ion+electron approximation
    Returns
    -------
    pol_total_state : float
        polarizability of the state in [a.u.].

    """

    w = units.wavelength_to_freq(wavelength)
    # 5s^2 1S0
    if mode == "normal" and n < 10:
        indexes = np.where((rdme_energies[:, [2, 4, 5, 6]] == [
                           str(n), str(l), str(j), str(s)]).all(axis=1))
        rdme = rdme_energies[indexes, 14]
        energies = rdme_energies[indexes, 15]
        js = rdme_energies[indexes, 12]
        pol_scal_state = pol_scal(j, w, energies, rdme, js)
        pol_tens_state = pol_tens(j, w, energies, rdme, js)
        pol_total_state = pol_total(
            pol_scal_state, pol_tens_state, n, m, l, j, s, w)
    else:
        if 10 < n < 20:
            print("n is quite low to be a Rydberg state")
        # 5sns 3S1 Rydberg state as ion + valence electron decomposition
        if l == 0 and j == 1 and s == 1:
            # selects index of n
            indexnumber = np.where(ElectronPolFactors[:, 0] == n)[0]
            if len(indexnumber) == 0:
                print("polarization for state [n,m,l,j,s]= " +
                      str([n, m, l, j, s])+" not defined for Sr88")
                pol_total_state = 0
            # ion contribution
            pol_total_state = np.real(calc_polarizability(
                wavelength, 4, 0.5, 0, 0.5, 0.5, True))
            # electron contribution with or without laser extend
            if laser_extend:
                pol_total_state += ElectronPolFactors[indexnumber,
                                                      1]*calc_electron_polarizability(wavelength)
            else:
                pol_total_state += -1*calc_electron_polarizability(wavelength)

        else:
            print("polarization for state [n,m,l,j,s]= " +
                  str([n, m, l, j, s])+" not defined for Sr88")
            pol_total_state = 0
    return pol_total_state


@functools.cache
def calc_C6_coefficient(n1, l1, j1, m1, n2, l2, j2, m2, s):
    calc = arc.PairStateInteractions(
        arc.Strontium88(),
        int(n1), int(l1), int(j1),
        int(n2), int(l2), int(j2),
        int(m1), int(m2),
        s=int(s)
    )
    # int(...) added by Raul
    # For some reason it will fail otherwise
    # Likely has to do with numpy.int deprecation + ARC
    theta = 0  # relative orientation of the two atoms [0,pi]
    phi = 0  # relative orientation of the two atoms [0,2pi]
    deltaN = 5  # levels above and below to consider
    deltaE = 30e9  # maximum energy difference to consider in Hz
    c6, eigenvectors = calc.getC6perturbatively(
        theta, phi, deltaN,
        deltaE, degeneratePerturbation=True)
    # getC6perturbatively returns the C6 coefficients
    # expressed in units of GHz mum^6.
    return c6[0]*10**9*(10**(-6))**6


def load_pol_from_file(state, config):
    # Currently loads in units of MHz cm^2 V^{-2}

    with open(config, 'rU') as infile:
        # read the file as a dictionary for each row ({header : value})
        reader = csv.DictReader(infile)
        data = {}
        for row in reader:
            for header, value in row.items():
                try:
                    data[header].append(value)
                except KeyError:
                    data[header] = [value]
    # extract the variables you want
    n = data['n']
    print(n)
    pol = data['alpha']
    return float(pol[n.index(str(state))])


def stray_field_shift(efield, pol): 
    return -0.5*2*np.pi*pol*efield**2


# atom = arc.Strontium88()



if __name__ == '__main__':
    # Identifying the magic wavelength at 813.4 nm and quantifying anti trapping
    wave_low = 813e-9
    wave_high = 813.1e-9
    N = 100

    # Calculation of Sr88 polarizabilities using calc_polarizability
    pols = np.zeros([23, N])
    Laser_Extend = True
    for k in range(N):
        wavelength = wave_low+(wave_high-wave_low)/N*k
        pols[0, k] = calc_polarizability(
            wavelength, 5, 0, 0, 0, 0, Laser_Extend)
        pols[1, k] = calc_polarizability(
            wavelength, 5, 0, 1, 0, 1, Laser_Extend)
        # pols[22,k]=calc_polarizability(wavelength,4,0.5,0,0.5,0.5,Laser_Extend)
        # for l in range(20):
        #     pols[2+l,k]=calc_polarizability(wavelength,50+l,0,0,1,1,Laser_Extend,mode="ionic")

    # plotting the polarizabilities
    # commenting out because it needs visualization package
    # plt.plot(wave_low+(wave_high-wave_low)/N*np.arange(N),np.transpose(pols))
    # plt.ylim([280,282])
    # plt.ylabel("Polarizability (a.u.)")
    # plt.xlabel("Wavelength (m)")
    # plt.legend(["1S0","3P0","Ion","Rydberg State n=50-70"])
