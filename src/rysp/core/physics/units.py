#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:28:05 2022

@author: robert
"""

import numpy as np
import rysp.core.physics.constants as cst
# Physical constants


# magnetic field in Gauss to magnetic field in Tesla
def gauss_to_tesla(B):
    return B*10**(-4)

# magnetic field in Tesla to magnetic field in Gauss


def tesla_to_gauss(B):
    return B*10**4

# from Einstein coefficient to Radial density matrix element
# See Eq. 27 from DOI: 10.1119/1.12937


def rate_to_rdme(Aki, J, En1, En2):
    return np.sqrt(3*np.pi*cst.eps0*cst.hbar*cst.c**3/(cst.a0**2*cst.e0**2*np.abs(En1 - En2)**3)*(2*J + 1)*Aki)


def rdme_to_rate(rdme, J, En1, En2):
    return np.abs(En1 - En2)**3*cst.e0**2/(3*np.pi*cst.eps0*cst.hbar*cst.c**3*(2*J + 1))*(cst.a0*rdme)**2

# Polarizability in SI units (Cm^2/V) to Polarizability in atomic units (a.u.)


def cm2Vmin1_to_AU(pol):
    return (cst.e0**2*cst.a0**2/cst.Eh)**(-1)*pol

# Polarizability in SI units (Cm^2/V) to Polarizability in atomic units (a.u.)


def AU_to_cm2Vmin1(pol):
    return (cst.e0**2*cst.a0**2/cst.Eh)*pol

# Wavenumber in cm^(-1) to Wavelength in m


def cmmin1_to_wavelength(k):
    return 1/(10**2*k)

# Wavenumber in cm^(-1) to Frequency in Hz


def cmmin1_to_freq(k):
    return 2*np.pi*cst.c*(10**2*k)

# Frequency in Hz to Wavenumber in cm^(-1)


def freq_to_cmmin1(w):
    return w/(2*np.pi*cst.c*(10**2))

# Wavelength in m to Wavenumber in cm^(-1)


def wavelength_to_cmmin1(w):
    return 1/(10**2*w)

# Wavenumber in cm^(-1) to Energy in J


def cmmin1_to_joule(k):
    return cst.hbar*2*np.pi*cst.c*(10**2)*k

# Wavenumber in cm^(-1) to Energy in Hartree


def cmmin1_to_hartree(k):
    return 2*np.pi*(10**2)*cst.alpha*cst.a0*k

# Wavelength in m to Energy in Joule


def wavelength_to_joule(w):
    return cst.hbar*2*np.pi*cst.c/w

# intensity in Watt/m^2 to intensity in miliWatt/cm^2


def wm2_to_mwcm2(I):
    return I*10**3*10**(-4)

# energy in Joule to Wavelength in m


def joule_to_wavelength(E):
    return cst.hbar*2*np.pi*cst.c/E

# Energy in Joule to Frequency in Hz


def joule_to_frequency(E):
    return E/cst.hbar

# frequency in Hz to energy in Joule


def frequency_to_joule(f):
    return f*cst.hbar

# wavelength in m to frequency in Hz


def wavelength_to_freq(w):
    return 2*np.pi*cst.c/w

# frequency in Hz to wavelength in m


def freq_to_wavelength(w):
    return 2*np.pi*cst.c/w

# energy in electronvolt to energy in joule


def ev_to_joule(E):
    return E*cst.e0

# energy in ev to frequency in Hz


def ev_to_frequency(E):
    return E*cst.e0/cst.hbar

# energy in electronvolt to wavelength in m


def ev_to_wavelength(E):
    return cst.hbar*2*np.pi*cst.c/(E*cst.e0)

# wavenumber in cm^(-1) to wavelength in m


def wavenumber_to_wavelength(k):
    return 10**(-2)/k

# wavenumber in cm^(-1) to frequency in Hz


def wavenumber_to_frequency(k):
    return 2*np.pi*cst.c*k*10**(2)

# wavelength in m to energy in hartree


def wavelength_to_hartree(wl):
    return cst.alpha*cst.a0*2*np.pi/wl

# energy in hartree to wavelength in m


def hartree_to_wavelength(E):
    return cst.alpha*cst.a0*2*np.pi/E

# energy in hartree to frequency in Hz


def hartree_to_freq(E):
    return cst.c*E/(cst.alpha*cst.a0*10**9)

# frequency in Hz to energy in Hartree


def freq_to_hartree(E):
    return E*(cst.alpha*cst.a0*10**9)/cst.c
