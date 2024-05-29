# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:17:50 2023

@author: s165827
"""

import numpy as np
import qutip as qt
import copy

from qutip import Qobj

from typing import Callable, Any


class HamiltonianPart:

    """
    Class for parts of total evolution Hamiltonians
    """

    def __init__(self, Hamiltonian: Qobj,
                 func: Callable[[float, Any], float] | tuple[np.ndarray, float] | float | None,
                 label: str,
                 htype: str):
        self.Hamiltonian = Hamiltonian
        self.Control_Func = func
        self.label = label
        self.htype = htype


class EvolutionHamiltonian:
    """
    Class for evolution Hamiltonians
    """

    def __init__(self, dims):
        """
        Intialization function for evolution Hamiltonian
        """
        self.Hamiltonians = {}
        self.Hamiltonians_loss = {}
        self.Hamiltonians_anti_trapping_loss = {}
        self.cops = {}
        self.dims = dims

    def add_list(self,
                 Hamiltonians: list[Qobj],
                 htypes: list[str],
                 labels: list[str],
                 Control_Funcs: list[Callable[[
                     float, Any], float] | tuple[np.ndarray, float] | float | None] | None = None,
                 operator_types: list[str] = ['hamiltonian']):
        """
        Add Hamiltonians to the total evolution,
        Adding multiple Hamiltonians with one control function or one label
        will copy the function and label

        Parameters
        ----------
        Hamiltonian : array of qutip operators
            All the Hamiltonians that need to be added
        Control_Func : control function class
            All the control functions to multiply the Hamiltonian
        label : array of strings
            strings describing the Hamiltonians
        htype : array of strings
            describes whether the added Hamiltonian is "drift" or "control"
        operator_type: string - 'hamiltonian', 'loss', 'cops'

        Returns
        -------
        None

        """
        def check_length(obj, l, is_label=False):
            if len(obj) == 1:
                return obj[0] if not is_label else f"{obj[0]}_{l}"

            if l < len(obj):
                return obj[l]

            raise ValueError(
                f"Evolution_Hamiltonian: (htypes, Control_Funcs or labels) need to be added as length 1 or same length as Hamiltonians\n{Hamiltonians=}, {htypes=}, {labels=}, {Control_Funcs=}, {operator_types=}")

        for l in range(len(Hamiltonians)):
            cf = check_length(Control_Funcs, l,
                              False) if Control_Funcs is not None else None
            lb = check_length(labels, l, is_label=True)
            ht = check_length(htypes, l)
            ot = check_length(operator_types, l)
            self.add(Hamiltonian=Hamiltonians[l],
                     Control_Func_obj=cf,  # type: ignore
                     label=lb,
                     htype=ht,
                     operator_type=ot)

    def add(self,
            Hamiltonian: Qobj,
            htype: str,
            label: str,
            Control_Func_obj:  Callable[[
                float, Any], float] | tuple[np.ndarray, float] | float | None = None,
            operator_type: str = 'hamiltonian'):
        """ Add an Hamiltonian to the total evolution

        Parameters
        ----------
        Hamiltonian : qutip operator
            The Hamiltonians to be added
        Control_Func : control function class
            The control function to multiply the Hamiltonian
        label : string
            strings describing the Hamiltonian
        htype : string
            describes whether the added Hamiltonian is "drift" or "control"
        operator_type: string - 'hamiltonian', 'loss', 'cops'

        Returns
        -------
        None

        """

        Ham_Control_Func = None
        # Define Ham_Control_Func
        if (htype == "control") and (Control_Func_obj is not None) and not callable(Control_Func_obj):
            Ham_Control_Func = copy.copy(Control_Func_obj)
        else:
            Ham_Control_Func = Control_Func_obj

        if Hamiltonian.dims != self.dims:
            raise ValueError(
                f"Evolution_Hamiltonian: The above Hamiltonian with label {label}, does not have the right dimensions")
        Ham_Hamiltonian = Hamiltonian

        Ham = HamiltonianPart(Hamiltonian, Ham_Control_Func, label, htype)

        if operator_type == 'hamiltonian':
            self.Hamiltonians[label] = Ham
        elif operator_type == 'cops':
            self.cops[label] = Ham
        elif operator_type == 'loss':
            self.Hamiltonians_loss[label] = Ham

    def add_cops(self, cops, Control_Func, label, htype):
        """
        Add Hamiltonian(s) to the total evolution,
        Adding multiple Hamiltonians with one control function or one label
        will copy the function and label

        Parameters
        ----------
        Hamiltonian : array of qutip operators
            All the Hamiltonians that need to be added
        Control_Func : control function class
            All the control functions to multiply the Hamiltonian
        label : array of strings
            strings describing the Hamiltonians
        htype : array of strings
            describes whether the added Hamiltonian is "drift" or "control"

        Returns
        -------
        None

        """
        for l in range(len(cops)):
            # Define Ham_htype
            if len(htype) == 1:
                Ham_htype = htype[0]
            elif len(htype) == len(cops):
                Ham_htype = htype[l]
            else:
                print(
                    "htypes need to be added as length 1 or same length as Hamiltonians")
                return

            # Define Ham_Control_Func
            if Ham_htype == "control":
                if Control_Func == None:
                    Ham_Control_Func = None
                else:
                    if len(Control_Func) == 1:
                        Ham_Control_Func = Control_Func[0]
                    elif len(Control_Func) == len(cops):
                        Ham_Control_Func = Control_Func[l]
                    else:
                        print(
                            "control funcs need to be added as length 1 or same length as Hamiltonians")
                        return
            else:
                Ham_Control_Func = None

            # Define Ham_label
            if len(label) == 1:
                Ham_label = label[0]+"_"+str(l)
            elif len(label) == len(cops):
                Ham_label = label[l]
            else:
                print(
                    "labels need to be added as length 1 or same length as Hamiltonians")
                return

            # Define Ham_Hamiltonian
            if cops[l].dims == self.dims:
                Ham_Hamiltonian = cops[l]
            else:
                raise ValueError(
                    f"The above Hamiltonian with label {Ham_label}, does not have the right dimensions; {cops[l]=}")

            Ham = HamiltonianPart(
                Ham_Hamiltonian, Ham_Control_Func, Ham_label, Ham_htype)
            self.cops[Ham_label] = Ham

    def add_anti_trapping_loss_list(self,
                                    Hamiltonians: list[Qobj],
                                    paramss: tuple[list[float], list[float], list[str]],
                                    labels: list[str]):
        """
        Add Hamiltonian(s) to the total evolution,
        Adding multiple Hamiltonians with one control function or one label
        will copy the function and label

        Parameters
        ----------
        Hamiltonian : array of qutip operators
            All the Hamiltonians that need to be added
        Params=[U,w,onoff]
            U : float
                trapping potential in J
            w : float
                trapping frequency in Hz
            onoff : string
                string can be "on" or "off" indicating the status of the traps
        label : array of strings
            strings describing the Hamiltonians
        htype : array of strings
            describes whether the added Hamiltonian is "drift" or "control"

        Returns
        -------
        None

        """
        U_arr, w_arr, onoff_arr = paramss

        def check_length(obj, l):
            if len(obj) == 1:
                return obj[0]
            else:
                try:
                    return obj[l]
                except IndexError:  # Catching index error and raising appropriate exception
                    raise ValueError(
                        "Evolution_Hamiltonian: (htypes, Control_Funcs or labels) need to be added as length 1 or same length as Hamiltonians")

        for k in range(len(Hamiltonians)):
            ham = Hamiltonians[k]
            Up = check_length(U_arr, k)
            wp = check_length(w_arr, k)
            onoffp = check_length(onoff_arr, k)
            label = check_length(labels, k)

            self.add_anti_trapping_loss(ham, (Up, wp, onoffp), label)

    def add_anti_trapping_loss(self,
                               Hamiltonian: Qobj,
                               params: tuple[float, float, str],
                               label: str):
        """
        Add Hamiltonian(s) to the total evolution,
        Adding multiple Hamiltonians with one control function or one label
        will copy the function and label

        Parameters
        ----------
        Hamiltonian : array of qutip operators
            All the Hamiltonians that need to be added
        Params=[U,w,onoff]
            U : float
                trapping potential in J
            w : float
                trapping frequency in Hz
            onoff : string
                string can be "on" or "off" indicating the status of the traps
        label : array of strings
            strings describing the Hamiltonians
        htype : array of strings
            describes whether the added Hamiltonian is "drift" or "control"

        Returns
        -------
        None
        """
        if Hamiltonian.dims != self.dims:
            raise ValueError(
                f"Evolution_Hamiltonian: The above Hamiltonian with label {label}, does not have the right dimensions")
        Ham = HamiltonianPart(Hamiltonian, params, label,
                              htype='drift')  # type: ignore
        self.Hamiltonians_anti_trapping_loss[label] = Ham

    def create_total_Hamiltonian(self) -> tuple[list[Qobj | tuple[Qobj, Callable[[float], float]]], list[Qobj], list[str], list[str]]:
        """
        Creates the total evolution needed as input for propagator calculation
        functions

        Returns
        -------
        H : [full_drift_Hamiltonian,[control_Hamiltonian,control_Func],...]
            list of Hamiltonian operators (possibly time dependent) over which the evolution is run
        cops: Collapse operators
        H_labels
        cops_labels

        """
        H = []
        cops = []
        H_labels = []
        cops_labels = []

        # Create drift Hamiltonian as sums of all the drift parts
        drift_H = qt.Qobj(dims=self.dims)
        for k in self.Hamiltonians:

            # Adding all drift Hamiltonians
            if self.Hamiltonians[k].htype == "drift":
                # print(f"{self.Hamiltonians[k].Control_Func=}")
                if self.Hamiltonians[k].Control_Func is not None:
                    drift_H = drift_H + \
                        self.Hamiltonians[k].Hamiltonian * \
                        self.Hamiltonians[k].Control_Func
                else:
                    drift_H = drift_H + self.Hamiltonians[k].Hamiltonian

        H.append(drift_H)
        H_labels.append("drift")

        evol_time = 0
        evol_list_size = 0

        # Create the control Hamiltonian as list of Hamiltonian parts and control functions
        for k in self.Hamiltonians:
            if self.Hamiltonians[k].htype == "control":
                if self.Hamiltonians[k].Control_Func is not None:
                    # print( type(self.Hamiltonians[k].Control_Func))
                    # if isinstance(self.Hamiltonians[k].Control_Func, Control_Func):
                    #     H += [[self.Hamiltonians[k].Hamiltonian,
                    #            self.Hamiltonians[k].Control_Func.z]]
                    #     H_labels.append(k)

                    if callable(self.Hamiltonians[k].Control_Func):
                        H += [[self.Hamiltonians[k].Hamiltonian,
                               self.Hamiltonians[k].Control_Func]]
                        H_labels.append(k)

                    elif isinstance(self.Hamiltonians[k].Control_Func, tuple):

                        if evol_time == 0:
                            evol_time = self.Hamiltonians[k].Control_Func[1]
                        elif self.Hamiltonians[k].Control_Func[1] != evol_time:
                            raise ValueError(
                                f"Evolution_Hamiltonian.create_total_Hamiltonian: array functions of Hamiltonians are not compatible. Different evolution times were provided: {evol_time}!={self.Hamiltonians[k].Control_Func[1]}")
                        if evol_list_size == 0:
                            evol_list_size = len(
                                self.Hamiltonians[k].Control_Func[0])
                        elif len(self.Hamiltonians[k].Control_Func[0]) != evol_list_size:
                            raise ValueError(
                                f"Evolution_Hamiltonian.create_total_Hamiltonian: array functions of Hamiltonians are not compatible. They have different list sizes: {evol_list_size}!={len(self.Hamiltonians[k].Control_Func[0])}")

                        # print( f"tuple control func:  {self.Hamiltonians[k].Hamiltonian}")
                        H += [[self.Hamiltonians[k].Hamiltonian,
                               self.Hamiltonians[k].Control_Func[0]]]
                        H_labels.append(k)

                    elif isinstance(self.Hamiltonians[k].Control_Func, float):
                        H += [self.Hamiltonians[k].Hamiltonian *
                              self.Hamiltonians[k].Control_Func]
                        H_labels.append(k)

        for k in self.cops:
            copop = self.cops[k].Hamiltonian
            cops += [qt.liouvillian(0*copop, c_ops=copop)]
            cops_labels.append(k)

        # print({lab: len(h[1]) if type(h)==list else h for lab, h in zip(H_labels, H)})
        if len(H) == 1:
            H = H[0]
        return H, cops, H_labels, cops_labels
