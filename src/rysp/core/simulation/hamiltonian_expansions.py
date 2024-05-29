# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:33:02 2023

@author: s165827
"""

import numpy as np
import qutip as qt
from copy import deepcopy
from qutip.qip.gates import expand_operator

def Individual_to_Full(Identities,qubit_num,target_Hamiltonian):
    """
    Changes a 1 qubit hamiltonian into a N qubit Hamiltonian

    Parameters
    ----------
    Identities : np.array qutip operators
        Identities for all individual qubits with the right number of states
    qubit_num : int
        The qubit to which target_Hamiltonian should be applied.
    target_Hamiltonian : qutip Operator
        The 1 qubit operator that needs to be applied

    Returns
    -------
    full_Hamiltonian : qutip Operator
        operator that acts on the full system but actively only on the qubit_num qubit

    """
    
    #Extracting the right dimensions
    dimens=[]
    for k in range(Identities.shape[0]):
        dimens.append(Identities[k].dims[0][0])
        dimens.append(Identities[k].dims[0][1])
    #Use the qutip operator expand_operator to create the right Hamiltonian
    #Note that because we have a spin and motional space we apply it to two parts of the system
    full_Hamiltonian=expand_operator(target_Hamiltonian, len(dimens),[2*qubit_num,2*qubit_num+1],dims=dimens)
    return full_Hamiltonian

def Pairwise_to_Full(Identities,qubit_nums,target_Hamiltonian):
    """
    Changes a 2 qubit hamiltonian into a N qubit Hamiltonian

    Parameters
    ----------
    Identities : np.array qutip operators
        Identities for all individual qubits with the right number of states
    qubit_nums : [int,int]
        The qubits to which target_Hamiltonian should be applied.
    target_Hamiltonian : qutip Operator
        The 1 qubit operator that needs to be applied

    Returns
    -------
    full_Hamiltonian : qutip Operator
        operator that acts on the full system but actively only on the qubit_num qubit

    """
    
    #Extracting the right dimensions
    dimens=[]
    for k in range(Identities.shape[0]):
        dimens.append(Identities[k].dims[0][0])
        dimens.append(Identities[k].dims[0][1])
    #Use the qutip operator expand_operator to create the right Hamiltonian
    #Note that because we have a spin and motional space we apply it to four parts of the system
    full_Hamiltonian=expand_operator(target_Hamiltonian, len(dimens),[2*qubit_nums[0],2*qubit_nums[0]+1,2*qubit_nums[1],2*qubit_nums[1]+1],dims=dimens)
    return full_Hamiltonian
