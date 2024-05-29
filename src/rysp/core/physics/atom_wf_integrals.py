
"""
Created on Fri Mar 11 15:16:38 2022

@author: s165827
"""
import warnings
import functools
from qutip.qip.gates import expand_operator
import scipy.integrate as integrate
import scipy.special as special
import math
import numpy as np
import qutip as qt
result = integrate.quad(lambda x: special.jv(2.5, x), 0, 4.5)


h = 6.626*10**(-34)
hbar = 6.626*10**(-34)/(2*np.pi)
me = 9.109*10**(-31)
kb = 1.38*10**(-23)

# HARMONIC OSCILLATOR WAVEFUNCTIONS


def Harmonic_Oscillator_3D_wavefunction(w, n, r, m):
    Xwave = Harmonic_Oscillator_1D_wavefunction(w[0], n[0], r[0], m)
    Ywave = Harmonic_Oscillator_1D_wavefunction(w[1], n[1], r[1], m)
    Zwave = Harmonic_Oscillator_1D_wavefunction(w[2], n[2], r[2], m)
    return Xwave*Ywave*Zwave


def Harmonic_Oscillator_1D_wavefunction(w, n, x, m):
    x0 = 1/(m*w)**(1/2)
    return 1/(np.pi)**(1/4)*1/np.sqrt(2**n*math.factorial(n)*x0) * special.hermite(n)(x/x0)*np.exp(-1/2*(x/x0)**2)

## EXPI INTEGRALS##


def Harmonic_Oscillator_1D_wavefunction_coeff(n):
    prefac = 1/(np.pi)**(1/4)*1/np.sqrt(2**n*math.factorial(n))
    powercoeff = prefac*special.hermite(n).c
    expcoeff = -1/2
    return np.flip(powercoeff), expcoeff


@functools.cache
def expi_integrate_1D(n1, n2, k, w1, w2, m, sign, mode="analytic"):
    """
    Calculates the 1D matrix elements for the photon recoil exp matrix

    Parameters
    ----------
    n1 : float
        describes the first motional state in HO states
    n2 : float
        describes the second motional state in HO states
    k : float
        wave vector element of the laser beam
    w1 : float
        oscillation frequency of the first state
    w2 : float
        oscillation frequency of the second state
    m : float
        mass of the atom.
    sign : +1 or -1
        describes whether we need exp(-) or exp(+).
    mode : calculation mode
        whether calculations should be done numeric or analytic.
        The default is "numeric".

    Returns
    -------
    float
        1D matrix element of the exp matrix between state 1 and 2.

    """
    if mode == "analytic":
        if w1 == w2:
            result = expi_integrate_1D_analytic(n1, n2, k, w1, m)
        else:
            raise ValueError(
                f"No analytic expression for w1=/=w2 ({w1=}, {w2=})")
    elif mode == "numeric":
        result = expi_integrate_1D_numeric(n1, n2, k, w1, w2, m)
    else:
        raise ValueError(
            f"Unknown mode of calculation. has to be either 'analytic' or 'numeric'. received {mode}")
    if sign == -1:
        result = np.conj(result)

    if np.isnan(result):
        warnings.warn(
            f"expi_integrate_1D: result is nan. Inputs: \n{n1=},{n2=},{k=},{w1=},{w2=},{m=},{sign=}\n")
    return result


def expi_integrate_1D_numeric(n1, n2, k, w1, w2, m):
    """
    Calculates the 1D matrix elements for the photon recoil exp matrix

    Parameters
    ----------
    n1 : float
        describes the first motional state in HO states
    n2 : float
        describes the second motional state in HO states
    k : float
        wave vector element of the laser beam
    w1 : float
        oscillation frequency of the first state
    w2 : float
        oscillation frequency of the second state
    m : float
        mass of the atom.
    sign : +1 or -1
        describes whether we need exp(-) or exp(+).

    Returns
    -------
    float
        matrix element of the exp matrix between state 1 and 2.

    """
    w1 = np.real(w1)
    w2 = np.real(w2)
    # change constants because of me and h=1
    x1 = np.sqrt(hbar/me)*np.real(1/(m*w1)**(1/2))
    x2 = np.sqrt(hbar/me)*np.real(1/(m*w2)**(1/2))
    n1 = int(n1)
    n2 = int(n2)

    prefac = 1/(np.pi)**(1/2)*1/np.sqrt(2**(n1+n2) *
                                        math.factorial(n1)*math.factorial(n2))/np.sqrt(x1*x2)

    integral_real = integrate.quad(lambda x: np.real(special.hermite(n1)(
        x/x1)*np.exp(1j*k*x)*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -100*max(x1, x2), 100*max(x1, x2))

    integral_real2 = integrate.quad(lambda x: np.real(special.hermite(n1)(
        x/x1)*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -100*max(x1, x2), 100*max(x1, x2))

    integral_imag = integrate.quad(lambda x: np.imag(special.hermite(n1)(
        x/x1)*np.exp(1j*k*x)*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -100*max(x1, x2), 100*max(x1, x2))

    return prefac*(integral_real[0]+1j*integral_imag[0])


def expi_integrate_1D_analytic(n1, n2, k, w, m):
    """
    Analytically calculates the 1D matrix elements for the photon recoil exp matrix

    Parameters
    ----------
    n1 : float
        describes the first motional state in HO states
    n2 : float
        describes the second motional state in HO states
    k : float
        wave vector element of the laser beam
    w : float
        oscillation frequency of the first and second state
    m : float
        mass of the atom.

    Returns
    -------
    float
        matrix element of the exp matrix between state 1 and 2.

    """
    # See https://functions.wolfram.com/Polynomials/HermiteH/21/ShowAll.html
    if k == 0:
        return int(n1 == n2)
    else:
        if n2 > n1:
            temp = int(n1)
            n1 = int(n2)
            n2 = int(temp)
        lam = 1j*k*np.sqrt(hbar/(2*m*me*w))
        return np.sqrt(math.factorial(n2)/math.factorial(n1))*lam**(n1-n2)*np.exp(lam**2/2)*special.genlaguerre(n2, n1-n2)(-lam**2)


def expi_integrate_3D(n1, n2, k, w1, w2, m, sign, mode="analytic"):
    """
    Calculates the 3D matrix elements for the photon recoil exp matrix

    Parameters
    ----------
    n1 : np.array 3x1
        describes the first motional state in HO states
    n2 : np.array 3x1
        describes the second motional state in HO states
    k : np.array 3x1
        wave vector of the laser beam
    w1 : np.array 3x1
        oscillation frequencies of the first state
    w2 : np.array 3x1
        oscillation frequencies of the second state
    m : float
        mass of the atom.
    sign : +1 or -1
        describes whether we need exp(-) or exp(+).
    mode : calculation mode
        whether calculations should be done numeric or analytic.
        The default is "numeric".

    Returns
    -------
    float
        matrix element of the exp matrix between state 1 and 2.

    """
    # Can calculate individually by seperability
    xdir = expi_integrate_1D(n1[0], n2[0], k[0], w1[0], w2[0], m, sign, mode)
    ydir = expi_integrate_1D(n1[1], n2[1], k[1], w1[1], w2[1], m, sign, mode)
    zdir = expi_integrate_1D(n1[2], n2[2], k[2], w1[2], w2[2], m, sign, mode)
    return xdir*ydir*zdir

## TRAP INTEGRALS ##


def overlap_integrate_1D(n1, n2, w1, w2, m, mode):
    """
    Calculates the overlap matrix element

    Parameters
    ----------
    n1 : int
        motional state of HO 1
    n2 : int
        motional state of HO 2
    w1 : float
        frequency of HO 1 in Hz
    w2 : float
        frequency of HO 2 in Hz.
    m : float
        mass of atom in electron masses.
    mode : string, optional
        mode "numeric" or "analytic. The default is "numeric".

    Returns
    -------
    float
        the 1D overlap matrix element between the two states.

    """

    if mode == "analytic":
        if w1 == w2:
            result = overlap_integrate_1d_analytic(n1, n2, w1, m)
        else:
            raise ValueError(
                f"No analytic expression for w1=/=w2 ({w1=}, {w2=})")
    elif mode == "numeric":
        result = overlap_integrate_1D_numeric(n1, n2, w1, w2, m)
    else:
        raise ValueError(f"Unknown mode of operation. {mode=}")
    return result


def trap_integrate_1D(n1, n2, w0, w1, w2, m, mode):
    """
    Calculates the HO oscillator matrix element

    Parameters
    ----------
    n1 : int
        motional state of HO 1
    n2 : int
        motional state of HO 2
    w0 : float
        trap frequency in Hz.
    w1 : float
        frequency of HO 1 in Hz
    w2 : float
        frequency of HO 2 in Hz.
    m : float
        mass of atom in electron masses.
    mode : string, optional
        mode "numeric" or "analytic. The default is "numeric".

    Returns
    -------
    float
        the 1D trap Hamiltonian matrix element between the two states.

    """
    if mode == "analytic":
        if w1 == w2:
            result = trap_integrate_1d_analytic(n1, n2, w0, w1, m)
        else:
            raise ValueError(
                f"No analytic expression for w1=/=w2 ({w1=}, {w2=})")
    elif mode == "numeric":
        result = trap_integrate_1D_numeric(n1, n2, w0, w1, w2, m)
    else:
        raise ValueError(f"Unknown mode of operation. {mode=}")
    return result


def trap_integrate_1d_analytic(n1, n2, w0, w1, m):
    """
    Calculates analytically the HO oscillator matrix element

    Parameters
    ----------
    n1 : int
        motional state of HO 1
    n2 : int
        motional state of HO 2
    w0 : float
        trap frequency in Hz.
    w1 : float
        frequency of HO 1 in Hz
    w2 : float
        frequency of HO 2 in Hz.
    m : float
        mass of atom in electron masses.

    Returns
    -------
    float
        the 1D trap Hamiltonian matrix element between the two states.

    """
    if n1 == n2:
        return 1/2*w1*(n1+1/2)*(1+w0**2/w1**2)
    elif n1 == n2-2:
        return 1/4*(w0**2-w1**2)/w1*np.sqrt(n2*(n2-1))
    elif n1 == n2+2:
        return 1/4*(w0**2-w1**2)/w1*np.sqrt(n1*(n1-1))
    else:
        return 0


def overlap_integrate_1d_analytic(n1, n2, w1, m):
    """
    Calculates analytically the overlap matrix element

    Parameters
    ----------
    n1 : int
        motional state of HO 1
    n2 : int
        motional state of HO 2
    w1 : float
        frequency of HO 1 in Hz
    w2 : float
        frequency of HO 2 in Hz.
    m : float
        mass of atom in electron masses.
    mode : string, optional
        mode "numeric" or "analytic. The default is "numeric".

    Returns
    -------
    float
        the 1D overlap matrix element between the two states.

    """
    if n1 == n2:
        return 1
    else:
        return 0


def trap_integrate_1D_numeric(n1, n2, w0, w1, w2, m):
    """
    Calculates numerically the HO oscillator matrix element

    Parameters
    ----------
    n1 : int
        motional state of HO 1
    n2 : int
        motional state of HO 2
    w0 : float
        trap frequency in Hz.
    w1 : float
        frequency of HO 1 in Hz
    w2 : float
        frequency of HO 2 in Hz.
    m : float
        mass of atom in electron masses.

    Returns
    -------
    float
        the 1D trap Hamiltonian matrix element between the two states.

    """

    return (-1/4*w1*d2x2_integrate_1d_numeric(n1, n2, w0, w1, w2, m)[0]+1/2*(w0**2)/w1*x2_integrate_1d_numeric(n1, n2, w0, w1, w2, m)[0])


def overlap_integrate_1D_numeric(n1, n2, w1, w2, m):
    """
    Calculates numerically the overlap matrix element

    Parameters
    ----------
    n1 : int
        motional state of HO 1
    n2 : int
        motional state of HO 2
    w1 : float
        frequency of HO 1 in Hz
    w2 : float
        frequency of HO 2 in Hz.
    m : float
        mass of atom in electron masses.

    Returns
    -------
    float
        the 1D trap overlap matrix element between the two states.

    """

    # length scales
    x1 = np.sqrt(hbar/me)*np.real(1/(m*w1)**(1/2))
    x2 = np.sqrt(hbar/me)*np.real(1/(m*w2)**(1/2))
    n1 = int(n1)
    n2 = int(n2)
    prefac = 1/(np.pi)**(1/2)*1/np.sqrt(2**(n1+n2)*math.factorial(n1)
                                        * math.factorial(n2))/np.sqrt(x1)/np.sqrt(x2)
    integral_real = integrate.quad(lambda x: np.real(special.hermite(n1)(
        x/x1)*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -100*min(x1, x2), 100*min(x1, x2))
    integral_imag = integrate.quad(lambda x: np.imag(special.hermite(n1)(
        x/x1)*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -100*min(x1, x2), 100*min(x1, x2))
    total = prefac*(integral_real[0]+1j*integral_imag[0]
                    ), prefac*(integral_real[1]+1j*integral_imag[1])
    return total[0]


def x2_integrate_1d_numeric(n1, n2, w0, w1, w2, m):
    """
    Calculates numerically the x^2 matrix element

    Parameters
    ----------
    n1 : int
        motional state of HO 1
    n2 : int
        motional state of HO 2
    w0 : float
        trap frequency in Hz.
    w1 : float
        frequency of HO 1 in Hz
    w2 : float
        frequency of HO 2 in Hz.
    m : float
        mass of atom in electron masses.

    Returns
    -------
    float
        the 1D x^2 Hamiltonian matrix element between the two states.

    """

    # length scales
    x1 = np.sqrt(hbar/me)*np.real(1/(m*w1)**(1/2))
    x2 = np.sqrt(hbar/me)*np.real(1/(m*w2)**(1/2))
    n1 = int(n1)
    n2 = int(n2)
    prefac = 1/(np.pi)**(1/2)*1/np.sqrt(2**(n1+n2)*math.factorial(n1)
                                        * math.factorial(n2))/np.sqrt(x1)/np.sqrt(x2)
    integral_real = integrate.quad(lambda x: np.real(special.hermite(n1)(
        x/x1)*(x/x1)*(x/x2)*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -100*min(x1, x2), 100*min(x1, x2))
    integral_imag = integrate.quad(lambda x: np.imag(special.hermite(n1)(
        x/x1)*x**2*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -100*min(x1, x2), 100*min(x1, x2))
    return prefac*(integral_real[0]+1j*integral_imag[0]), prefac*(integral_real[1]+1j*integral_imag[1])


def d2x2_integrate_1d_numeric(n1, n2, w0, w1, w2, m):
    """
    Calculates numerically the d^2/dx^2 matrix element

    Parameters
    ----------
    n1 : int
        motional state of HO 1
    n2 : int
        motional state of HO 2
    w0 : float
        trap frequency in Hz.
    w1 : float
        frequency of HO 1 in Hz
    w2 : float
        frequency of HO 2 in Hz.
    m : float
        mass of atom in electron masses.

    Returns
    -------
    float
        the 1D d^2/dx^2 Hamiltonian matrix element between the two states.

    """

    # length scales
    x1 = np.sqrt(hbar/me)*np.real(1/(m*w1)**(1/2))
    x2 = np.sqrt(hbar/me)*np.real(1/(m*w2)**(1/2))
    n1 = int(n1)
    n2 = int(n2)

    # for other values integral is zero, use recurrent expression for second derivative Hermite polynomial
    if n1 == n2:
        prefac = 1/(np.pi)**(1/2)*1/np.sqrt(2**(n1+n2)*math.factorial(n1)
                                            * math.factorial(n2))/np.sqrt(x1)/np.sqrt(x2)
        integral_real = integrate.quad(lambda x: np.real(special.hermite(n1)(
            x/x1)*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -100*min(x1, x2), 100*min(x1, x2))
        integral_imag = integrate.quad(lambda x: np.imag(special.hermite(n1)(
            x/x1)*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -100*min(x1, x2), 100*min(x1, x2))
        prefac = prefac*(-(2*n1+1))
        return prefac*(integral_real[0]+1j*integral_imag[0]), prefac*(integral_real[1]+1j*integral_imag[1])
    elif n2 == n1+2:
        n1a = n1+2
        prefac = 1/(np.pi)**(1/2)*1/np.sqrt(2**(n1a+n2) *
                                            math.factorial(n1a)*math.factorial(n2))/np.sqrt(x1)/np.sqrt(x2)
        integral_real = integrate.quad(lambda x: np.real(special.hermite(n1a)(
            x/x1)*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -100*min(x1, x2), 100*min(x1, x2))
        integral_imag = integrate.quad(lambda x: np.imag(special.hermite(n1a)(
            x/x1)*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -100*min(x1, x2), 100*min(x1, x2))
        prefac = prefac*(np.sqrt((n1+1)*(n1+2)))
        return prefac*(integral_real[0]+1j*integral_imag[0]), prefac*(integral_real[1]+1j*integral_imag[1])
    elif n2 == n1-2:
        n1a = n1-2
        prefac = 1/(np.pi)**(1/2)*1/np.sqrt(2**(n1a+n2) *
                                            math.factorial(n1a)*math.factorial(n2))/np.sqrt(x1)/np.sqrt(x2)
        integral_real = integrate.quad(lambda x: np.real(special.hermite(n1a)(
            x/x1)*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -100*min(x1, x2), 100*min(x1, x2))
        integral_imag = integrate.quad(lambda x: np.imag(special.hermite(n1a)(
            x/x1)*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -100*min(x1, x2), 100*min(x1, x2))
        prefac = prefac*(np.sqrt((n1*(n1-1))))
        return prefac*(integral_real[0]+1j*integral_imag[0]), prefac*(integral_real[1]+1j*integral_imag[1])
    else:
        return [0, 0]


def trap_integrate_3D(n1, n2, w0, w1, w2, m, mode="analytic"):
    """
    Calculates the trap matrix elements between states n1 and n2 in 3D

    Parameters
    ----------
    n1 : array of ints [nx1,ny1,nz1]
        first considered motional state
    n2 : array of ints [nx2,ny2,nz2]
        second considered motional state
    w0 : array of floats [wx,wy,wz]
        trap frequencies of the laser optical dipole trap in Hz.
    w1 : array of floats [wx,wy,wz]
        HO frequencies [wx,wy,wz] of state 1 in Hz.
    w2 : array of floats [wx,wy,wz]
        HO frequencies [wx,wy,wz] of state 2 in Hz.
    m : float
        mass of the atom in electron masses.
    mode : string, optional
        mode "numeric" or "analytic. The default is "numeric".

    Returns
    -------
    float
        the trap Hamiltonian matrix elements between the two states.

    """

    # We can decompose x,y,z by seperability of Schrodinger equation
    overlapx = overlap_integrate_1D(n1[0], n2[0], w1[0], w2[0], m, mode)
    overlapy = overlap_integrate_1D(n1[1], n2[1], w1[1], w2[1], m, mode)
    overlapz = overlap_integrate_1D(n1[2], n2[2], w1[2], w2[2], m, mode)

    xdir = trap_integrate_1D(n1[0], n2[0], w0[0], w1[0], w2[0], m, mode)
    xdir = xdir*overlapy*overlapz

    ydir = trap_integrate_1D(n1[1], n2[1], w0[1], w1[1], w2[1], m, mode)
    ydir = ydir*overlapx*overlapz

    zdir = trap_integrate_1D(n1[2], n2[2], w0[2], w1[2], w2[2], m, mode)
    zdir = zdir*overlapx*overlapy

    return xdir+ydir+zdir


def x_integrate_1d(n1, n2, w1, w2, m, mode="analytic"):
    if mode == "analytic":
        if w1 == w2:
            return x_integrate_1d_analytic(n1, n2, w1, m)
        else:
            raise ValueError(
                f"No analytic expression for w1=/=w2 ({w1=}, {w2=})")
    elif mode == "numeric":
        return x_integrate_1d_numeric(n1, n2, w1, w2, m)[0]
    else:
        raise ValueError(f"Unknown mode of calculation: '{mode}'")


def x_integrate_1d_analytic(n1, n2, w, m):
    if n1 == n2+1:
        return 1/np.sqrt(2*w*m)*np.sqrt(n1)
    elif n1 == n2-1:
        return 1/np.sqrt(2*w*m)*np.sqrt(n2)
    else:
        return 0


def xyz_integrate_1d(n1, n2, w1, w2, m, direction, mode="analytic"):
    if direction == "x":
        if n1[1] == n2[1] and n1[2] == n2[2]:
            return x_integrate_1d(n1[0], n2[0], w1[0], w2[0], m, mode)
        else:
            return 0
    if direction == "y":
        if n1[0] == n2[0] and n1[2] == n2[2]:
            return x_integrate_1d(n1[1], n2[1], w1[1], w2[1], m, mode)
        else:
            return 0
    if direction == "z":
        if n1[0] == n2[0] and n1[1] == n2[1]:
            return x_integrate_1d(n1[2], n2[2], w1[2], w2[2], m, mode)
        else:
            return 0
    raise ValueError(f"Invalid direction {direction}")


def VdW_integrate_3d(n1, n2, n3, n4, w1, w2, w3, w4, m, R, C6, mode="analytic"):
    total_int = 0
    Rabs = np.linalg.norm(R)
    x0 = np.sqrt(hbar/me) * \
        np.sqrt(1/(m*min(min(w1), min(w2), min(w3), min(w4))))
    if Rabs <= (10*x0):
        print("Calculation inaccurate due to seperation not being big enough")
    directions = ["x", "y", "z"]
    if (n1 == n3).all() and (n2 == n4).all():
        total_int += -C6/Rabs**6
    for k in range(3):
        if not R[k] == 0:
            total_int += 6*C6/Rabs**8*R[k]*(xyz_integrate_1d(n1, n3, w1, w3, m, directions[k],
                                            mode)-xyz_integrate_1d(n2, n4, w2, w4, m, directions[k], mode))
    return total_int


def x_integrate_1d_numeric(n1, n2, w1, w2, m):
    x1 = np.sqrt(hbar/me)*np.real(1/(m*w1)**(1/2))
    x2 = np.sqrt(hbar/me)*np.real(1/(m*w2)**(1/2))
    w1 = np.real(w1)
    w2 = np.real(w2)
    x1 = np.real(x1)
    x2 = np.real(x2)
    n1 = int(n1)
    n2 = int(n2)
    prefac = 1/(np.pi)**(1/2)*1/np.sqrt(2**(n1+n2)*math.factorial(n1)
                                        * math.factorial(n2))/np.sqrt(x1)/np.sqrt(x2)
    integral_real = integrate.quad(lambda x: np.real(special.hermite(n1)(
        x/x1)*x*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -10*min(x1, x2), 10*min(x1, x2))
    integral_imag = integrate.quad(lambda x: np.imag(special.hermite(n1)(
        x/x1)*x*special.hermite(n2)(x/x2)*np.exp(-1/2*(x/x1)**2-1/2*(x/x2)**2)), -10*min(x1, x2), 10*min(x1, x2))
    return prefac*(integral_real[0]+1j*integral_imag[0]), prefac*(integral_real[1]+1j*integral_imag[1])


def VdW_integrate_3D_basis(basis1, basis2, w1, w2, mass, R, C6, num_spin_states_1, num_spin_states_2, interacting_state):
    num_states_1 = basis1.shape[0]
    num_states_2 = basis2.shape[0]
    Hamiltonian = np.zeros(
        [num_states_1*num_states_2, num_states_1*num_states_2], dtype=complex)
    for k in range(num_states_1):
        for l in range(num_states_2):
            for m in range(num_states_1):
                for n in range(num_states_2):
                    Hamiltonian[k*num_states_2+l, m*num_states_2+n] = VdW_integrate_3d(
                        basis1[k, :], basis2[l, :], basis1[m, :], basis2[n, :], w1, w2, w1, w2, mass, R, C6)
    Hamiltonian = qt.Qobj(
        Hamiltonian, dims=[[num_states_1, num_states_2], [num_states_1, num_states_2]])
    spinHam = np.zeros([num_spin_states_1*num_spin_states_2,
                       num_spin_states_1*num_spin_states_2])
    spinHam[(num_spin_states_1+1)*interacting_state,
            (num_spin_states_1+1)*interacting_state] = 1
    spinHam = qt.Qobj(spinHam, dims=[[num_spin_states_1, num_spin_states_2], [
                      num_spin_states_1, num_spin_states_2]])
    spinHam = qt.Qobj(expand_operator(spinHam, 4, [0, 2], dims=[
        num_spin_states_1, num_states_1,
        num_spin_states_2, num_states_2]))
    Hamiltonian = qt.Qobj(expand_operator(Hamiltonian, 4, [1, 3], dims=[
        num_spin_states_1, num_states_1,
        num_spin_states_2, num_states_2]))
    return qt.Qobj(spinHam*Hamiltonian, dims=[[num_spin_states_1, num_states_1,
                                               num_spin_states_2, num_states_2],
                                              [num_spin_states_1, num_states_1,
                                               num_spin_states_2, num_states_2]])

## INTEGRATE WRT BASIS ##


def collapse_operators_3D_basis(basis, w, mass, T, rate):
    num_ops = 0
    for k in range(basis.shape[0]):
        for l in range(basis.shape[0]):
            if np.count_nonzero(basis[k, :]-basis[l, :]) == 1:
                num_ops += 1
    collapse_operators = np.ndarray([num_ops], dtype=object)
    Hamiltonian_spin = qt.qeye(w.shape[0])
    temp_freq = kb*T/hbar
    counter = 0
    for k in range(basis.shape[0]):
        for l in range(basis.shape[0]):
            if np.count_nonzero(basis[k, :]-basis[l, :]) == 1:
                direction = np.where(basis[k, :]-basis[l, :])
                energydifference = (
                    w[0, direction]*(basis[l, direction]-basis[k, direction]))[0, 0]
                energydifference = (
                    (basis[l, direction]-basis[k, direction]))[0, 0]
                collapse_operator = np.zeros([basis.shape[0], basis.shape[0]])
                collapse_operator[k, l] = rate * \
                    np.exp(energydifference/(2*temp_freq))
                collapse_operator[k, l] = rate*np.exp(energydifference/(2))
                collapse_operator = qt.Qobj(collapse_operator)
                collapse_operators[counter] = qt.tensor(
                    Hamiltonian_spin, collapse_operator)
                counter += 1
    return collapse_operators


def light_atom_integrate_3D_basis(transition, motional_basis, w, wave_vec, mass, lambdicke_regime, ignore_motional=False):
    """
    Creates the exp|a><b| exp|b><a| and detuning |b><b| operators

    Parameters
    ----------
    transition : np.array 3x1 [a,b,#levels]
        array describing which level the transition occurs between with the last 
        entry being the total number of spin states
    motional_basis : np.array 3xN
        describes the motional basis of the atom
    w : np.array
        trap frequencies of the motional states considered.
    wave_vec : np.array 3x1
        wave vector of the laser
    mass : float
        mass of the atom.
    lambdicke_regime : boolean
        indicates whether the lambdicke regime should be considered
        by setting the motional part to the identity

    Returns
    -------
    Hamiltonian_ab : qutip operator
        The exp|a><b| (plus) part of the transition
    Hamiltonian_ba : qutip operator
        The exp|b><a| (min) part of the transition
    Hamiltonian_bb : qutip operator
        The |b><b| (detuning part of the transition

    """

    Hamiltonian_spin_plus = np.zeros([transition[-1], transition[-1]])
    Hamiltonian_spin_plus[transition[0], transition[1]] = 0.5
    Hamiltonian_spin_plus = qt.Qobj(Hamiltonian_spin_plus)

    if lambdicke_regime:
        Hamiltonian_motional_plus = np.eye(motional_basis.shape[0])
    else:
        Hamiltonian_motional_plus = expi_integrate_3D_basis(
            motional_basis, w, mass, wave_vec, -1)
    Hamiltonian_motional_plus = qt.Qobj(Hamiltonian_motional_plus)

    Hamiltonian_spin_min = np.zeros([transition[-1], transition[-1]])
    # factor half in front of Omega/2 in atom-light interaction
    Hamiltonian_spin_min[transition[1], transition[0]] = 0.5
    Hamiltonian_spin_min = qt.Qobj(Hamiltonian_spin_min)

    if lambdicke_regime:
        Hamiltonian_motional_min = np.eye(motional_basis.shape[0])
    else:
        Hamiltonian_motional_min = expi_integrate_3D_basis(
            motional_basis, w, mass, wave_vec, 1)
    Hamiltonian_motional_min = qt.Qobj(Hamiltonian_motional_min)

    Hamiltonian_spin_det = np.zeros([transition[-1], transition[-1]])
    Hamiltonian_spin_det[transition[1], transition[1]] = 1
    Hamiltonian_spin_det = qt.Qobj(Hamiltonian_spin_det)

    Hamiltonian_motional_det = np.eye(motional_basis.shape[0])
    Hamiltonian_motional_det = qt.Qobj(Hamiltonian_motional_det)

    if not (ignore_motional and motional_basis.shape[0] == 1):
        Hamiltonian_ab = qt.tensor(
            Hamiltonian_spin_plus, Hamiltonian_motional_plus)
        Hamiltonian_ba = qt.tensor(
            Hamiltonian_spin_min, Hamiltonian_motional_min)
        Hamiltonian_bb = qt.tensor(
            Hamiltonian_spin_det, Hamiltonian_motional_det)
    else:
        Hamiltonian_ab = Hamiltonian_spin_plus
        Hamiltonian_ba = Hamiltonian_spin_min
        Hamiltonian_bb = Hamiltonian_spin_det
    return Hamiltonian_ab, Hamiltonian_ba, Hamiltonian_bb


def expi_integrate_3D_basis(motional_basis, w, mass, wave_vec, sign):
    """
    Creates the exponential part of a transition Hamiltonian

    Parameters
    ----------
    motional_basis : np.array 3xN
        describes the motional basis of the atom
    w : np.array
        trap frequencies of the motional states considered.
    wave_vec : np.array 3x1
        wave vector of the laser
    mass : float
        mass of the atom.
    sign : +1 or -1
        describes whether we need exp(-) or exp(+).

    Returns
    -------
    Hamiltonian : qutip operator
        the exp operator for motional states in the transition.

    """
    num_states = motional_basis.shape[0]
    Hamiltonian = np.zeros([num_states, num_states], dtype=complex)
    for k in range(num_states):
        for l in range(num_states):
            Hamiltonian[k, l] = expi_integrate_3D(
                motional_basis[k, :], motional_basis[l, :], wave_vec, w, w, mass, sign)
    return Hamiltonian


def trap_integrate_3D_basis(basis, w, mass):
    """
    Calculates for 1 atom the trap Hamiltonian

    Parameters
    ----------
    basis : np.array
        motional basis of the atom as [[nx,ny,nz],...].
    w : np.array
        trap frequencies per spin state.
    mass : float.
        mass of the atom in electron masses.

    Returns
    -------
    Qutup operator
        1 atom trap Hamiltonian in Hz.

    """
    num_states = basis.shape[0]*w.shape[0]
    Hamiltonian = np.zeros([num_states, num_states], dtype=complex)
    for k in range(basis.shape[0]):  # loop over motional states
        for l in range(basis.shape[0]):
            for t in range(w.shape[0]):  # loop over spin states
                Hamiltonian[k+t*basis.shape[0], l+t*basis.shape[0]] = trap_integrate_3D(
                    basis[k, :], basis[l, :], w[t, :], w[0, :], w[0, :], mass)
    return qt.Qobj(Hamiltonian, dims=[[w.shape[0], basis.shape[0]], [w.shape[0], basis.shape[0]]])
