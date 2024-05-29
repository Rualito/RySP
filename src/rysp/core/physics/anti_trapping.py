# -*- coding: utf-8 -*-

import math
import rysp.core.physics.constants as cst
from math import factorial as fact
import numpy as np
from scipy.special import poch
import fractions
import rysp.core.physics.atom_wf_integrals as mat_elem
# import RSP_old.Hamiltonian.calc_matrix_elements as mat_elem
# import matplotlib.pyplot as plt
# import cProfile
import scipy


def doublefactorial(n):
    if n <= 0:
        return 1
    else:
        return n * doublefactorial(n-2)


def gammaf(z):
    """
    returns the gamma function at integer z (defined for precission)

    Parameters
    ----------
    z : int
        input of gamma function

    Returns
    -------
    float
        output of gamma function

    """
    return fact(z-1)


def gammafhalf(z):
    return doublefactorial(2*z-1)/2**z*np.sqrt(np.pi)


def pochf(z, m):
    """
    returns the pochhammer symbol of z at m

    Parameters
    ----------
    z : float
        input of the pochhammer symbol.
    m : int
        order of the pochhammer symbol.

    Returns
    -------
    fraction
        output of the pochammer function.

    """
    return fractions.Fraction(gammaf(z+m), gammaf(z))


def IntHermSumTerm(n, k, j, ass):
    """
    One part of the hermitian integral sum

    Parameters
    ----------
    n : int
    k : int
    j : int
    ass : float

    Returns
    -------
    fraction
        one term of the sum

    """
    return 2**j*fact(n+k-2*j)/(fact(j)*fact(n-j)*fact(k-j)*fact(int((n+k)/2-j)))*((1-(-ass+1))/(-ass+1))**(((n + k)/2 - j))


def IntHerm(n, k, ass):
    """
    Calculates a hermitian exponential intergal

    Parameters
    ----------
    n : int
    k : int
    ass : float

    Returns
    -------
    float

    """
    # Integral form as in \https://math.stackexchange.com/questions/592624/integrals-involving-\hermite-polynomials
    if np.mod(n-k, 2) == 0:
        prefac = 1/math.sqrt(2**n*fact(n))*(1/np.pi)**(1/4)*1/math.sqrt(
            2**k*fact(k))*(1/np.pi)**(1/4)*math.sqrt(np.pi/(-ass+1))*fact(n)*fact(k)
        sumpart = sum([2**j*fact(n+k-2*j)/(fact(j)*fact(n-j)*fact(k-j)*fact(int((n+k)/2-j)))
                      * ((1-(-ass+1))/(-ass+1))**(((n + k)/2 - j)) for j in range(0, min(n, k)+1)])
        return prefac*sumpart
    else:
        return 0


def matrix_element(m, n, U, w):
    """
    matrix element for 1d Gaussian potential Hamiltonian for boundstates

    Parameters
    ----------
    m : int
        order of first eigenstate of HO
    n : int
        order of second eigenstate of HO
    U : float
        trap depth in Joule
    w : float
        trap frequency in Hz

    Returns
    -------
    float
        matrix element between m and n for Gaussian potential.

    """
    prefac = -1/2
    if m == n:
        HOterm = -1/2*(2*n+1)
    elif m == n-2:
        HOterm = 1/2*np.sqrt(n*(n-1))
    elif m == n+2:
        HOterm = 1/2*np.sqrt((n+1)*(n+2))
    else:
        HOterm = 0
    return prefac*HOterm+U/(cst.hbar*w)*IntHerm(m, n, 1/2*cst.hbar*w/U)

# U = -50*10**(-6)*cst.kb
# w = 2*np.pi*25000
# def matrix_element_2(n1, n2, l1, l2):
#     """
#     matrix element for 2d Gaussian potential

#     Parameters
#     ----------
#     n1 : int
#         order of first eigenstate of HO
#     n2 : int
#         order of second eigenstate of HO
#     l1 : int
#         order of angular momentum first eigenstate of HO
#     l2 : int
#         order of angular momentum second eigenstate of HO

#     Returns
#     -------
#     float
#         matrix element between (n1,l1) and (n2,l2) in 2D Gaussian potential.

#     """
#     if l1 != l2 or abs(l1) > n1 or abs(l2) > n2 or (l1 + n1) % 2 == 1 or (l2 + n2) % 2 == 1:
#         return 0
#     else:
#         alpha_r2 = int(abs(l1) + 2)
#         alpha_V = int(abs(l1) + 1)
#         pr2 = fractions.Fraction(1, 1)
#         pV = fractions.Fraction.from_float(
#             (1 - cst.hbar*w/(2*U))).limit_denominator(400000000)
#         p1 = int((n1 - abs(l1))/2)
#         p2 = int((n2 - abs(l2))/2)
#         if n1 == n2:
#             n1n2term = (n1 + 1)
#         else:
#             n1n2term = 0

#         prefac1 = -1/4*math.sqrt(2*fact(p1)/fact(p1 + abs(l1)))*math.sqrt(2*fact(p2)/fact(p2 + abs(l2))) \
#             * ((math.gamma(alpha_r2) * poch(abs(l1) + 1, p1)
#                 * poch(abs(l2) + 1, p2)) / (pr2**alpha_r2 * (fact(p1) * fact(p2))))

#         sum12 = sum([fractions.Fraction(((-1)**j * int(gammaf(1 + p1))), int(gammaf(1 - j + p1)))
#                      * fractions.Fraction(int(pochf(alpha_r2, j)), int(pochf(abs(l1) + 1, j))
#                                           * fact(j)) * (1/pr2)**j * sum([fractions.Fraction(((-1)**k * int(gammaf(1 + p2))) / int(gammaf(1 - k + p2)))
#                                                                          * fractions.Fraction(int(pochf(alpha_r2 + j, k)), int(pochf(abs(l2) + 1, k)) * fact(k))
#                                                                          * (1/pr2)**k for k in range(p2 + 1)]) for j in range(p1 + 1)])

#         prefac2 = U/(cst.hbar*w)*1/2*math.sqrt(2*fact(p1) / (fact(p1 + abs(l1))))\
#             * math.sqrt(2*fact(p2) / (fact(p2 + abs(l2)))) * ((math.gamma(alpha_V)
#                                                                * poch(abs(l1) + 1, p1) * poch(abs(l2) + 1, p2)) / (pV**alpha_V * (fact(p1) * fact(p2))))

#         sum22 = sum([fractions.Fraction(((-1)**j * gammaf(1 + p1)), gammaf(1 - j + p1))
#                      * fractions.Fraction(pochf(alpha_V, j), pochf(abs(l1) + 1, j) * fact(j))
#                      * (1/pV)**j * sum([fractions.Fraction(((-1)**k * gammaf(1 + p2)) / gammaf(1 - k + p2))
#                                         * fractions.Fraction(pochf(alpha_V + j, k), pochf(abs(l2) + 1, k) * fact(k))
#                                         * (1/pV)**k for k in range(p2+1)]) for j in range(p1+1)])

#         return n1n2term+prefac1*sum12+prefac2*sum22


def Hgauss(NumApproxStates, U, w):
    """
    Used to create the Gaussian matrix based on HO eigenstates

    Parameters
    ----------
    NumApproxStates : int
        Number of states considered in building the matrix
    U : float
        trap depth in Joule
    w : float
        trap frequency in Hz

    Returns
    -------
    H : np.matrix of NumApproxStatesxNumApproxStates
        matrix filled with matrix elements of HO eigenstates with 
        the Gaussian potential

    """
    H = np.zeros([NumApproxStates, NumApproxStates])
    for k in range(NumApproxStates):
        for j in range(NumApproxStates):
            H[k, j] = matrix_element(k, j, U, w)
    return H


def Boundstates_calc(H):
    """
    Calculates the boundstates of the matrix H

    Parameters
    ----------
    H : np.matrix
        matrix filled with matrix elements of HO eigenstates with 
        the Gaussian potential

    Returns
    -------
    boundstates : numpy matrix
        contains the boundstates as decomposed in HO eigenstates
    energies : numpy array
        has the energies belonging to the boundstate
    vals : numpy array
        contains all energies belonging to the eigenstaets
    vecs : numpy matrix
        contains all eigenstates as decomposed in HO eigenstates

    """
    vals, vecs = scipy.linalg.eig(H)
    energies = vals[vals < 0]
    states = vecs[:, np.where(vals < 0)[0]]
    sorted_indexes = np.argsort(energies)
    boundstates = states[:, sorted_indexes]
    energies = energies[sorted_indexes]
    return boundstates, energies, vals, vecs


def psi_IHO_anal(t, n):
    """
    returns the polynomial and exponential coefficients of the analytic IHO solution
    at time t and order n. Terms calculated analytically using Mathematica script

    Parameters
    ----------
    t : float
        dimensionless time parameter.
    n : int
        excited state number of psi0.

    Returns
    -------
    polcoeff : np array
        list of polynomial coefficients
    expcoeff : float
        exponential coefficient

    """
    if n == 0:
        polcoeff = [1/np.pi**(1/4)/np.sqrt(np.cosh(t)+1j*np.sinh(t))]
        expcoeff = 1/2*1j*1/np.tanh(t)-1j*1 / \
            np.sinh(t)/2/(np.cosh(t)+1j*np.sinh(t))
    elif n == 1:
        polcoeff = [0, (1-1j)*1/np.sinh(t)/(np.pi**(1/4) *
                                            (-1j+1/np.tanh(t))*np.sqrt(1j*np.cosh(t)+np.sinh(t)))]
        expcoeff = -(1j+1/np.tanh(t))/(2*(-1j+1/np.tanh(t)))
    elif n == 2:
        polcoeff = [(1/2+1j/2)*np.cosh(2*t)*np.sqrt(1j*np.cosh(t)+np.sinh(t))/(np.pi**(1/4)*(np.cosh(t)-1j*np.sinh(t))**3), 0,
                    -(1+1j)*np.sqrt(1j*np.cosh(t)+np.sinh(t))/(np.pi**(1/4)*(np.cosh(t)-1j*np.sinh(t))**3)]
        expcoeff = -(1j+1/np.tanh(t))/(2*(-1j+1/np.tanh(t)))
    elif n == 3:
        polcoeff = [0, (-1)**(3/4)*np.sqrt(3)*np.cosh(2*t)/(np.pi**(1/4)*(np.cosh(t)-1j*np.sinh(t))**3*np.sqrt(1j*np.cosh(t)+np.sinh(t))), 0,
                    -(2*(-1)**(3/4))/(np.sqrt(3)*np.pi**(1/4)*(np.cosh(t)-1j*np.sinh(t))**3*np.sqrt(1j*np.cosh(t)+np.sinh(t)))]
        expcoeff = -(1j+1/np.tanh(t))/(2*(-1j+1/np.tanh(t)))
    elif n == 4:
        polcoeff = [-(-1)**(3/4)*np.sqrt(3)*(1+np.cosh(4*t))/(4*np.sqrt(2)*np.pi**(1/4)*(1j*np.cosh(t)+np.sinh(t))**(9/2)), 0,
                    (-1)**(3/4)*np.sqrt(6)*np.cosh(2*t) /
                    (np.pi**(1/4)*(1j*np.cosh(t)+np.sinh(t))**(9/2)), 0,
                    -(-1)**(3/4)*np.sqrt(2/3)/(np.pi**(1/4)*(1j*np.cosh(t)+np.sinh(t))**(9/2))]
        expcoeff = -(1j+1/np.tanh(t))/(2*(-1j+1/np.tanh(t)))
    elif n == 5:
        polcoeff = [0, np.sqrt(15)*(-1/np.pi)**(1/4)*(1+np.cosh(4*t))/(4*(1j*np.cosh(t)+np.sinh(t))**(11/2)), 0,
                    2*np.sqrt(5/3)*(-1/np.pi)**(1/4)*np.cosh(2*t) /
                    ((1j*np.cosh(t)+np.sinh(t))**(11/2)), 0,
                    2*(-1/np.pi)**(1/4)/(np.sqrt(15)*(1j*np.cosh(t)+np.sinh(t))**(11/2))]
        expcoeff = -(1j+1/np.tanh(t))/(2*(-1j+1/np.tanh(t)))
    else:
        raise ValueError(f"n={n} is too high for analytical calculation.")
    return polcoeff, expcoeff


def psi_Free_anal(t, n):
    """
    returns the polynomial and exponential coefficients of the analytic Free solution
    at time t and order n. Terms calculated analytically using Mathematica script

    Parameters
    ----------
    t : float
        dimensionless time parameter.
    n : int
        excited state number of psi0.

    Returns
    -------
    polcoeff : np array
        list of polynomial coefficients
    expcoeff : float
        exponential coefficient

    """
    if n == 0:
        polcoeff = [-(-1)**(3/4)/((np.pi)**(1/4)*np.sqrt(-1j+t))]
        expcoeff = 1j/(2*(-1j+t))
    elif n == 1:
        polcoeff = [0, -(1+1j)/(np.pi**(1/4)*(-1j+t)**(3/2))]
        expcoeff = 1j/(2*(-1j+t))
    elif n == 2:
        polcoeff = [(1/2-1j/2)*(1+t**2)/(np.pi**(1/4)*(-1j+t)**(5/2)), 0,
                    -(1-1j)/(np.pi**(1/4)*(-1j+t)**(5/2))]
        expcoeff = 1j/(2*(-1j+t))
    elif n == 3:
        polcoeff = [0, -(1+1j)*np.sqrt(3)*(1+t**2)/(np.sqrt(2)*np.pi**(1/4)*(-1j+t)**(7/2)), 0,
                    (1+1j)*np.sqrt(2/3)/(np.pi**(1/4)*(-1j+t)**(7/2))]
        expcoeff = 1j/(2*(-1j+t))
    elif n == 4:
        polcoeff = [(-0.144338+0.144338*1j)*3*(1+t**2)**2/(np.pi**(1/4)*(-1j+t)**(9/2)), 0,
                    -12*(1+t**2)*(-0.144338+0.144338*1j) /
                    (np.pi**(1/4)*(-1j+t)**(9/2)), 0,
                    4*(-0.144338+0.144338*1j)/(np.pi**(1/4)*(-1j+t)**(9/2))]
        expcoeff = 1j/(2*(-1j+t))
    elif n == 5:
        polcoeff = [0, -np.sqrt(15)*(-1/np.pi)**(1/4)*(1+t**2)**2/(2*(-1j+t)**(11/2)), 0,
                    2*np.sqrt(5/3)*(-1/np.pi)**(1/4) *
                    (1+t**2)/((-1j+t)**(11/2)), 0,
                    -2*(-1/np.pi)**(1/4)/(np.sqrt(15)*(-1j+t)**(11/2))]
        expcoeff = 1j/(2*(-1j+t))
    else:
        raise ValueError(f"n={n} is too high for analytical calculation.")
    return polcoeff, expcoeff


def construct_func(polcoeff, expcoeff, x):
    """
    Creates the function from the polynomial and exponential coefficients at x

    Parameters
    ----------
    polcoeff : np array
        list of polynomial coefficients
    expcoeff : float
        exponential coefficient
    x : float
        dimensionless space parameter

    Returns
    -------
    float
        value of the function at x

    """
    return np.sum(np.array([polcoeff[n]*x**n*np.exp(x**2*expcoeff) for n in range(len(polcoeff))]), axis=0)


def Gauss(U, w, x):
    """
    Outputs the gauss potential at x

    Parameters
    ----------
    U : float
        trap depth in Joule
    w : float
        trap frequency in Hz
    x : float
        dimensionless space parameter

    Returns
    -------
    float
        Gauss potential at x

    """
    return U/(cst.hbar*w)*(-1+np.exp(1/2*x**2*cst.hbar*w/U))


def Gen_Boundstate(x, vec, w, m):
    """
    Generates the boundstate at x from its HO eigenstate vector

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    vec : numpy array of floats
        array indicating a bound state decomposed as 
        eigenstates of the harmonic oscillator potential.
    w : float
        trap frequency in Hz.
    m_atom : float
        mass of atom in kg.

    Returns
    -------
    float
        value of the boundstate at x

    """
    return np.transpose(np.sum(np.sum(np.array([np.array([np.array([vec[k, n]*mat_elem.Harmonic_Oscillator_1D_wavefunction(1, n, x, m) for n in range(vec.shape[1])])]) for k in range(vec.shape[0])]), axis=2), axis=1))


def ExpIntegral1D(polcoeff, expcoeff, xpower):
    """
    Calculates the integral of a polynomial and exponential integral

    Parameters
    ----------
    polcoeff : float
        coefficient of the polynomial.
    expcoeff : float
        coefficient of the exponent
    xpower : int
        power of the polynomial.

    Returns
    -------
    float
        value of the integral.

    """
    if np.mod(xpower, 2) == 0:
        expthing = (-expcoeff)**(-1/2-xpower/2)
        q = expthing*polcoeff*gammafhalf(int(xpower/2))
        return q
    else:
        return 0


def calc_overlap_coeffs(polcoeffs_ev, expcoeff_ev, polcoeffs_bound, expcoeff_bound):
    """
    Calculates the overlap between a bound state and evolved state
    based on their polynomial and exponential coefficients

    Parameters
    ----------
    polcoeffs_ev : np.array of floats
        floats of the polynomial coefficient of the evolved state
        as [c,x,x**2,x**3,...]
    expcoeff_ev : float
        exponential coefficient of the evolved state
    polcoeffs_bound : floats of the polynomial coefficient of the bound state
            as [c,x,x**2,x**3,...]
    expcoeff_bound : float
        exponential coefficient of the bound state

    Returns
    -------
    float
        overlap between the bound state and evolved state.

    """
    total_expcoeff = expcoeff_ev+expcoeff_bound
    total_polcoeffs = np.convolve(polcoeffs_ev, polcoeffs_bound)
    return np.sum([ExpIntegral1D(total_polcoeffs[k], total_expcoeff, k) for k in range(len(total_polcoeffs))])


def calc_overlap_coeffs2(polcoeffs_ev, expcoeff_ev, polcoeffs_bound, expcoeff_bound):
    """
    Calculates the overlap between a bound state and evolved state
    based on their polynomial and exponential coefficients

    Parameters
    ----------
    polcoeffs_ev : np.array of floats
        floats of the polynomial coefficient of the evolved state
        as [c,x,x**2,x**3,...]
    expcoeff_ev : float
        exponential coefficient of the evolved state
    polcoeffs_bound : floats of the polynomial coefficient of the bound state
            as [c,x,x**2,x**3,...]
    expcoeff_bound : float
        exponential coefficient of the bound state

    Returns
    -------
    float
        overlap between the bound state and evolved state.

    """
    total_expcoeff = expcoeff_ev+expcoeff_bound
    total_polcoeffs = np.convolve(polcoeffs_ev, polcoeffs_bound)

    totsum = 0
    for k in range(len(total_polcoeffs)):
        xpower = k
        expcoeff = total_expcoeff
        polcoeff = total_polcoeffs[k]
        totsum += 1/2*(1+(-1)**xpower)*(-expcoeff)**(-1/2 -
                                                     xpower/2)*polcoeff*gammafhalf(int(xpower/2))
    return totsum


def harmcoeff_to_polcoeff(boundstate):
    """
    outputs the polynomial coefficient as function of the Harmonic oscillator coefficients

    Parameters
    ----------
    boundstate : np array of floats
        contains the boundstate as decomposed in HO eigenstates

    Returns
    -------
    total_coeff : np array of floats
        contains the polynomial coefficients of the boundstate 

    """
    total_coeff = np.zeros(len(boundstate)+1)
    for k in range(len(boundstate)):
        total_coeff[0:(k+1)] += boundstate[k] * \
            mat_elem.Harmonic_Oscillator_1D_wavefunction_coeff(k)[0]
    return total_coeff


def calc_overlap(state, boundstate):
    """
    calculates the overlap between an evolved state and a boundstate

    Parameters
    ----------
    state : [string,float,int]
        contains the mode of evolution "Free" or "IHO"
        the time t as float
        the excited state number n as int
    boundstate : np array of floats
        contains the boundstate as decomposed in HO eigenstates

    Returns
    -------
    float
        overlap between evolved state and bound state.

    """
    if state[0] == "Free":
        polcoeffs_ev, expcoeff_ev = psi_Free_anal(state[1], state[2])
    elif state[0] == "IHO":
        polcoeffs_ev, expcoeff_ev = psi_IHO_anal(state[1], state[2])
    else:
        raise ValueError(
            f"Not a specified state as 'Free' or 'IHO', instead got {state[0]}.")
    polcoeffs_bound = boundstate
    expcoeff_ev = np.real(expcoeff_ev)
    expcoeff_bound = -0.5
    return calc_overlap_coeffs(polcoeffs_ev, expcoeff_ev, polcoeffs_bound, expcoeff_bound)


def calc_total_overlap(state, boundstates):
    """
    Calculates the total overlap between an evolved state and all bound states

    Parameters
    ----------
    state : [string,float,int]
        contains the mode of evolution "Free" or "IHO"
        the time t as float
        the excited state number n as int
    boundstate : np matrix of floats
        contains the boundstates as decomposed in HO eigenstates

    Returns
    -------
    overlap: float
        overlap between evolved state and bound states.

    """
    overlap = 0
    for k in range(len(boundstates)):
        overlap += np.abs(calc_overlap(state, boundstates[k]))**2
    return overlap


def calc_survival(mode, U, w, n, times, Boundstates):
    """
    Calculates the total overlap between an evolved state and all bound states
    at various times

    Parameters
    ----------
    mode : string
        the mode of evolution "Free" or "IHO"
    U : float
        trap depth in Joule
    w : float
        trap frequency in Hz
    n : int
        excited state number of the evolved state psi0.
    times : np array of floats
        times at which to evaluate the overlap

    Returns
    -------
    overlaps : np array of floats
        total overlap between evolved state and bound states at various times

    """
    overlaps = np.zeros(len(times))
    for k in range(len(times)):
        if times[k] == 0:
            overlaps[k] = 1
        else:
            overlaps[k] = calc_total_overlap(
                [mode, times[k]*w, n], Boundstates)
            print([times[k], overlaps[k]])
    return overlaps


# def plot_bound_states(U,w,Boundstates,energies):
#     """
#     plots the boundstates together with the Gaussian potential

#     Parameters
#     ----------
#     U : float
#         trap depth in Joule.
#     w : float
#         trap frequency in Hz.
#     Boundstates : numpy matrix of floats
#         matrix where each row indicates a bound state decomposed as
#         eigenstates of the harmonic oscillator potential.
#     energies : np.array of floats
#         list of energies for the boundstates

#     Returns
#     -------
#     None.

#     """

#     xs=np.linspace(-20,20,100000) #x parameters for plot
#     plot_per=5 # plot per so many bound states (otherwise crowded)
#     plt.plot(xs,Gauss(U,w,xs)) #plots the Gaussian potential
#     indexes=np.arange(0,states.shape[0],plot_per)
#     plt.plot(xs,Gen_Boundstate(xs,Boundstates[indexes],1,1)+energies[indexes]-U/(cst.hbar*w))
#     plt.show()

# def plot_evolution(t,n):
#     """
#     Plots the evolution of the dimensionless Free and IHO potentials

#     Parameters
#     ----------
#     t : float
#         dimensionless time parameter.
#     n : int
#         excited state number of psi0.

#     Returns
#     -------
#     None.

#     """
#     xs=np.linspace(-20,20,100000)
#     polcoeff_0, expcoeff_0= psi_Free_anal(0,n)
#     psi_0=construct_func(polcoeff_0,expcoeff_0,xs)
#     polcoeff_IHO, expcoeff_IHO= psi_IHO_anal(t, n)
#     psi_IHO=construct_func(polcoeff_IHO, expcoeff_IHO, xs)
#     polcoeff_Free, expcoeff_Free= psi_Free_anal(t, n)
#     psi_Free=construct_func(polcoeff_Free, expcoeff_Free, xs)
#     plt.plot(xs,np.abs(psi_0))
#     plt.plot(xs,np.abs(psi_IHO))
#     plt.plot(xs,np.abs(psi_Free))
#     plt.xlabel(r"x [($\hbar/m \omega )^{1/2}$]")
#     plt.text(10,0.5,"t="+str(t))
#     plt.show()

# def plot_survival(T_end,n,U,w,m_atom,Boundstates):
#     """
#     Plots the recapture rates of the atom

#     Parameters
#     ----------
#     T_end : float
#         end time in units of 1/w.
#     n : int
#         excited state number of psi0.
#     U : float
#         trap depth in Joule.
#     w : float
#         trap frequency in Hz.
#     m_atom : float
#         mass of atom in electron masses.
#     Boundstates : numpy matrix of floats
#         matrix where each row indicates a bound state decomposed as
#         eigenstates of the harmonic oscillator potential.

#     Returns
#     -------
#     None.

#     """
#     N=200
#     m_atom=m_atom*cst.m_elec
#     X=np.sqrt(cst.hbar/(m_atom*w)) #length scale
#     T=1/w #time scale
#     ts=np.linspace(T_end/N*T,T_end*T/N*(N+1),N)
#     survIHO=np.zeros(N)
#     survFree=np.zeros(N)
#     for k in range(N):
#         survIHO[k]=calc_total_overlap(["IHO",T_end/N*(k+1),n], Boundstates)
#         survFree[k]=calc_total_overlap(["Free",T_end/N*(k+1),n], Boundstates)
#     plt.plot(ts,survIHO)
#     plt.plot(ts,survFree)
#     plt.xlabel("t [s]")
#     plt.ylabel("Recapture Prob.")
#     plt.legend(["IHO","Free"])
#     plt.show()

# def high_level_function():
#     calc_total_overlap(["IHO",1,0], Boundstates_pol)

# if __name__ == '__main__':
#     U = -50*10**(-6)*cst.kb
#     w = 2*np.pi*25000
#     NumApproxStates=45
#     m_atom=88*1800*cst.m_elec
#     NumApproxStates=20
#     H=Hgauss(NumApproxStates,U,w)
#     Boundstates,energies,vals,vecs=Boundstates_calc(H)
#     Boundstates_pol=np.zeros([Boundstates.shape[1],Boundstates.shape[0]+1])
#     for k in range(Boundstates.shape[1]):
#         Boundstates_pol[k]=harmcoeff_to_polcoeff(Boundstates[:,k])
#     print(calc_total_overlap(["IHO",1,0], Boundstates_pol))
#     plot_evolution(3,0)
#     plot_survival(7/3*4,0,U,w,m_atom,Boundstates_pol)
#     cProfile.run("high_level_function()", 'results2.prof')
