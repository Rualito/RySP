# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 09:47:01 2023

@author: s165827
"""

import numpy as np
import qutip as qt
import rysp.core.physics.anti_trapping as anti_trap

def integrated_loss_time(psi, Hamiltonians_loss, T):
    """
    returns the time spent in each state as prescribed by the loss Hamiltonians

    Parameters
    ----------
    psi : list of Qobj
        The state of the quantum system as various times
    Hamiltonians_loss : list of Qobj
        Diagonal Hamiltonians indicating the loss at a certain state.
        inner products with this can measure how much of the state is in 
        the state prescribed by the Hamiltonian
    T : float
        End time of the evolution.

    Returns
    -------
    int_times : numpy array of floats
        Indicates the effective time spent in the loss states per time interval
    int_times_cum : numpy array of floats
        Indicates the cumulative effective time spent in the loss states per time interval

    """
    N = len(psi)
    num_loss = len(Hamiltonians_loss)
    int_times = np.zeros([num_loss, N])
    labels = []
    int_l = 0
    for l in Hamiltonians_loss:
        lossham = Hamiltonians_loss[l].Hamiltonian
        labels += [Hamiltonians_loss[l].label]
        for k in range(N):
            int_times[int_l, k] = qt.expect(lossham, psi[k])
        int_l += 1
    int_times = int_times*T/N
    int_times_cum = np.cumsum(int_times, 1)
    return int_times, int_times_cum, labels


def loss_rates(psi, Hamiltonian, T):
    """
    Calculates the loss rates due to exponential losses for the given 
    loss states

    Parameters
    ----------
    psi : list of Qobj
        The state of the quantum system as various times
    Hamiltonian: EvolutionHamiltonian Class operator
        Hamiltonian object that has the loss Hamiltonians in it
    T : float
        total evolution time T.

    Returns
    -------
    exp_losses : np array of floats
        the actual losses experienced by the loss terms at every time interval
    losses_cum : np array of floats
        has all the total rates multiplied by the total time spent in the loss states
        at the time intervals

    """
    int_times, int_times_cum, labels = integrated_loss_time(
        psi, Hamiltonian.Hamiltonians_loss, T)
    N = len(psi)
    num_loss = len(Hamiltonian.Hamiltonians_loss)
    losses = np.zeros([num_loss, N])
    for k in range(N):
        int_l = 0
        for l in Hamiltonian.Hamiltonians_loss:
            if Hamiltonian.Hamiltonians_loss[l].Control_Func == None:
                rate = 0
            else:
                rate = Hamiltonian.Hamiltonians_loss[l].Control_Func.z(
                    T/N*k, [])
            losses[int_l, k] = int_times[int_l, k]*rate
            int_l += 1
    losses_cum = np.cumsum(losses, 1)
    exp_losses = np.exp(-losses_cum)
    return exp_losses, losses_cum, labels


def loss_rates_anti_trapping(psi, Hamiltonian, T):
    """
    Calculates the loss rates due to non-exponential losses from anti-trapping
    for the given loss states


    Parameters
    ----------
    psi : list of Qobj
        The state of the quantum system as various times
    Hamiltonian: EvolutionHamiltonian Class operator
        Hamiltonian object that has the anti-trapping loss Hamiltonians in it
    T : float
        total evolution time T.

    Returns
    -------
    psi :list of Qobj
        The state of the quantum system as various times after anti-trapping losses
    rates :  np array of floats
        has all the total losses for total time spent in the loss states
        at the time intervals

    """
    int_times, int_times_cum, loss_labels = integrated_loss_time(
        psi, Hamiltonian.Hamiltonians_anti_trapping_loss, T)
    N = len(psi)
    num_loss = len(Hamiltonian.Hamiltonians_anti_trapping_loss)
    losses = np.zeros([num_loss, N])
    NumIntervals = 100
    rates = np.ones([num_loss, NumIntervals])
    labels = []
    int_l = 0
    for l in Hamiltonian.Hamiltonians_anti_trapping_loss:
        U, w, onoff = Hamiltonian.Hamiltonians_anti_trapping_loss[l].Control_Func
        if onoff == "on":
            mode = "IHO"
        elif onoff == "off":
            mode = "Free"
        else:
            raise ValueError("Specify onoff as on or off")
        # For now we only consider n=0
        NumApproxStates = 45
        H = anti_trap.Hgauss(NumApproxStates, U, w)
        Boundstates, energies, vals, vecs = anti_trap.Boundstates_calc(H)
        Boundstates_pol = np.zeros(
            [Boundstates.shape[1], Boundstates.shape[0]+1])
        interval_times = np.linspace(min(int_times_cum[int_l, :]), max(
            int_times_cum[int_l, :]), NumIntervals)
        for k in range(Boundstates.shape[1]):
            Boundstates_pol[k] = anti_trap.harmcoeff_to_polcoeff(
                Boundstates[:, k])
        ratespertime = anti_trap.calc_survival(
            mode, U, w, n=0, times=interval_times, Boundstates=Boundstates_pol)
        rates[int_l, :] = anti_trap.calc_survival(
            mode, U, w, n=0, times=interval_times, Boundstates=Boundstates_pol)
        time_count = 0
        for k in range(N):
            while int_times_cum[int_l, k] > interval_times[time_count]:
                time_count += 1
            psi[k] = psi[k]*rates[int_l, time_count]
        labels += [Hamiltonian.Hamiltonians_anti_trapping_loss[l].label]
        int_l += 1
    return psi, rates, labels


def loss_psi(psi, Hamiltonian, T):
    """
    post-processes the exponential and anti-trapping losses for psi

    Parameters
    ----------
    psi : list of Qobj
        The state of the quantum system as various times
    Hamiltonian: EvolutionHamiltonian Class operator
        Hamiltonian object that has the loss Hamiltonians in it
    T : float
        total evolution time T.

    Returns
    -------
    psi :list of Qobj
        The state of the quantum system as various times after losses

    """
    exp_losses, losses_cum, exp_labels = loss_rates(psi, Hamiltonian, T)
    N = len(psi)
    num_loss = len(Hamiltonian.Hamiltonians_loss)
    for k in range(N):
        for l in range(num_loss):
            psi[k] = psi[k]*exp_losses[l, k]
    psi, anti_trapping_losses, anti_trapping_labels = loss_rates_anti_trapping(
        psi, Hamiltonian, T)
    return psi
