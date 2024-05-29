# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:54:01 2022

@author: s165827
"""

import numpy as np

import rysp.core.physics.constants as cst
import rysp.core.physics.sr88 as sr88
from qutip import qeye, tensor, basis  # TODO: replace these imports with QTerm
import rysp.core.physics.units as units
import copy


class AtomInTrap:
    """
    Class for atoms (qubits) in an optical dipole trap
    """

    def __init__(self,
                 atom: dict,
                 trap_site: dict,
                 state: np.ndarray | str,
                 spin_basis: list[np.ndarray] | list[list[int]],
                 spin_basis_labels: list[str],
                 motional_basis: list[list[int]] | int | None = None,
                 motional_basis_labels: list[str] = ['000'],
                 Temp: float = 0,
                 interact_states: list | np.ndarray = ['r'],
                 motional_energy: int = 0):
        """
        Intialization function for the atom in optical dipole trap

        Parameters
        ----------
        atom : dict {'name': ..., 'mass': ...}
            contains the ARC supplied data of the atom
        trap_site : trap site dict -> trap {laser wavelength, laser intensity, laser w0, zR}
            Describes the trap in which the atom is located
        state : np.array()/string
            Describes the state in which the atom is in
        spin_basis : np.array() [[n,l,j,m,s],...]
            Describes the basis of spin states considered for the atom
        spin_basis_labels: list of strings
            Describes the spin basis as strings
        motional_basis : np.array() [[x,y,z],...]/int
            Describes the basis of motional states considered for the atom
            If int, then consider the everything up to n=int, e.g.
            int=2 then [0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1]
                        [1,0,1],[2,0,0],[0,2,0],[0,0,2]
        motional_basis_labels: list of strings
            Describes the motional basis as strings
        Temp : float
            Temperature of the atom in K.
        interact_states : np.array
            Describes which spin states interact

        """
        self.element_name = atom['name']  # atom.elementName
        # self.atom=atom
        self.trap_site: dict = trap_site
        self.m = atom['mass']  # atom.mass
        self.Temp = Temp
        self.interact_states = []
        self.motional_energy = motional_energy
        self.pos = np.array([0, 0, 0])
        if type(interact_states[0]) == str:
            for i, st in enumerate(spin_basis_labels):
                if st in interact_states:
                    self.interact_states.append(i)
        elif type(interact_states[0]) == int:
            for i, st in enumerate(spin_basis_labels):
                if i in interact_states:
                    self.interact_states.append(i)
        else:
            raise ValueError("Interact states: wrong specification")

        self.spin_basis = np.array(spin_basis, dtype=np.int16)
        self.num_spin_state = self.spin_basis.shape[0]
        self.spin_basis_labels = spin_basis_labels
        self.motional_basis_labels = motional_basis_labels

        # check if motional_basis is specified as array or integer
        if type(motional_basis) == int:
            self.create_motional_basis(motional_basis)
        else:
            self.motional_basis = np.array(motional_basis, dtype=np.int16)
        if motional_basis is None:
            self.create_motional_basis(motional_energy)

        self.num_motional_state = self.motional_basis.shape[0]
        self.num_total_state = self.num_spin_state*self.num_motional_state
        self.create_total_basis()

        self.spin_sys_identity = qeye(dimensions=len(self.spin_basis))
        self.motional_sys_identity = qeye(dimensions=len(self.motional_basis))
        if len(self.motional_basis_labels) > 1:
            self.basis_identity = tensor(
                self.spin_sys_identity, self.motional_sys_identity)
        else:
            self.basis_identity = self.spin_sys_identity

        if type(state) == str and state == "ground_state":
            self.state = np.zeros([self.num_total_state])
            self.state[0] = 1
        else:
            self.state = state

        self.polarizability = np.zeros([self.num_spin_state])
        self.trap_frequencies = np.zeros(
            [self.num_spin_state, 3], dtype=complex)
        self.c6_coefficients = np.zeros(
            [self.num_spin_state, self.num_spin_state])

        self.calc_C6_coefficients()
        self.calc_polarizability()
        self.calc_trap_frequencies()

    def __deepcopy__(self, memo):

        not_there = []
        existing = memo.get(self, not_there)
        if existing is not not_there:  # avoid copy recursion
            return existing
        # Only supports Sr88 at the moment
        trapsite = copy.deepcopy(self.trap_site, memo)
        state = copy.deepcopy(self.state, memo)
        temp = copy.deepcopy(self.Temp, memo)
        spin_basis = copy.deepcopy(self.spin_basis, memo)
        spin_basis_labels = copy.deepcopy(self.spin_basis_labels, memo)
        motional_basis = copy.deepcopy(self.motional_basis, memo)
        motional_basis_labels = copy.deepcopy(self.motional_basis_labels, memo)
        interact_states = copy.deepcopy(self.interact_states, memo)
        motional_energy = copy.deepcopy(self.motional_energy, memo)

        atomit = AtomInTrap({'name': self.element_name, 'mass': self.m}, trapsite, state,
                            spin_basis, spin_basis_labels,
                            motional_basis, motional_basis_labels,
                            temp, interact_states, motional_energy)
        atomit.c6_coef_dict = copy.deepcopy(self.c6_coef_dict, memo)
        atomit.c6_coefficients = copy.deepcopy(self.c6_coefficients, memo)
        atomit.polarizability = copy.deepcopy(self.polarizability, memo)
        atomit.pos = copy.deepcopy(self.pos, memo)
        return atomit

    def change_state(self, new_state):
        """
        updates the state of thet atom

        Parameters
        ----------
        new_state : np.array
            wavefunction/density matrix of the state

        """
        self.state = new_state

    def update_pos(self, pos):
        self.pos = pos

    def calc_loss(self):
        """
        calculates the loss of the atom from the norm of the state

        Returns
        -------
        float
            norm of the wavefunction of the atom.

        """
        return np.linalg.norm(self.state)

    def calc_polarizability(self):
        """
        calculates the polarizability of the atom per spin state in a.u.

        """
        for k in range(self.num_spin_state):
            n, l, j, m, s = self.spin_basis[k, :]
            if self.element_name == "Sr88":
                self.polarizability[k] = \
                    sr88.calc_polarizability(self.trap_site['laser wavelength'],
                                             n, m, l, j, s,
                                             True)
            else:
                raise ValueError(f"Element {self.element_name} not supported")
            # elif self.element_name == "Rb85":
            #     self.polarizability[k] = atom_pol.Rb85.calc_polarizability(self.trap_site.laser.wavelength,n,m,l,j,s,True)

    def update_polarizability(self, pols):
        """
        updates the polarizability of the atom per spin state in a.u.

        Parameters
        ----------
        pols : np.array
            polarizability of the atom in a.u.

        """
        self.polarizability = pols

    def create_motional_basis(self, energy_level):
        """
        In case motional basis specified as int gives the motional basis as the
        first int excited states

        Parameters
        ----------
        energy_level : int
            excited states up to energy int to consider

        """
        if energy_level == 0:
            self.motional_basis = np.array([[0, 0, 0]], dtype=np.int16)

        elif energy_level == 1:
            self.motional_basis = np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int16)
        elif energy_level == 2:
            self.motional_basis = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [
                                           2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int16)
        else:
            raise ValueError("Atom_In_Trap: Motional energy level is too high")
            # print("energy level too high")
        # Autogenerate motional basis labels
        self.motional_basis_labels = [
            ''.join([str(x) for x in ri]) for ri in self.motional_basis]

    def create_total_basis(self):
        """
        generates the total basis as
        np.array = [[n_spin,x,y,z],...]
        """
        col1 = np.reshape(np.array(np.repeat(
            np.arange(self.num_spin_state), self.motional_basis.shape[0])), (-1, 1))
        col2 = np.array(np.tile(self.motional_basis, [self.num_spin_state, 1]))
        self.total_basis = np.hstack([col1, col2])

    def get_atom_basis_identity(self):
        '''
        Gets the indentity operator for this atom.
        '''
        return self.basis_identity

    def get_atom_basis_state(self, spin_state: str, motional_state: str = '_id'):
        '''
        Gets the basis state for this atom. if state='_id', then replaces the particular subspace with the identity 
        spin_state: label of state in level system - spin
        motional_state: label of state in level system - motional

        raises ValueError if spin_state or motional_state are not defined on the initialization of atom level system
        '''
        spin_sys = self.spin_sys_identity
        motional_sys = self.motional_sys_identity
        if spin_state != '_id':
            spin_sys = basis(dimensions=len(self.spin_basis),
                             n=self.spin_basis_labels.index(spin_state))

        if len(self.motional_basis_labels) > 1:
            # ignores motional subsystem if just 1 state is given
            # since there are no motional degress of freedom
            if motional_state != '_id':
                motional_sys = basis(dimensions=len(self.motional_basis),
                                     n=self.motional_basis_labels.index(motional_state))

            return tensor(spin_sys, motional_sys)
        else:
            return spin_sys

    def ket(self, spin, motional: str = '_id'):
        return self.get_atom_basis_state(spin, motional)

    def calc_trap_frequencies(self):
        """
        Calculates the trap frequencies in Hz of the atom in the dipole trap
        according to Eq. (2.16) and (2.17) in https://thesis.library.caltech.edu/14061/7/Thesis_Ivaylo_Madjarov.pdf

        gives trap frequencies per spin state in x,y,z
        """
        for k in range(self.num_spin_state):
            # trap depth in Joule
            trap_depth = units.AU_to_cm2Vmin1(
                self.polarizability[k])*(self.trap_site['laser intensity'](0, 0)/(2*cst.eps0*cst.c))
            self.trap_depth = trap_depth
            self.trap_frequencies[k, 0] = (1j)**((1-np.sign(self.trap_depth))/2)*np.sqrt(
                4*np.abs(self.trap_depth)/(self.m*self.trap_site['laser w0']**2))
            self.trap_frequencies[k, 1] = (1j)**((1-np.sign(self.trap_depth))/2)*np.sqrt(
                4*np.abs(self.trap_depth)/(self.m*self.trap_site['laser w0']**2))
            self.trap_frequencies[k, 2] = (1j)**((1-np.sign(self.trap_depth))/2)*np.sqrt(
                2*np.abs(self.trap_depth)/(self.m*self.trap_site['zR']**2))

    def update_trap_frequencies(self, freqs):
        """
        updates the trap frequencies of the atom per spin state in Hz

        Parameters
        ----------
        freqs : np.array [[x,y,z],...]
            trap frequencies of the atom per spin state in Hz

        """
        self.trap_frequencies = freqs

    def calc_C6_coefficients(self):
        """
        Calculates the C6 coefficients of the states prescibed by
        interacting states
        """
        self.c6_coef_dict = {}
        # self.spin_basis = np.int64(self.spin_basis)
        for k in range(self.num_spin_state):
            for l in range(k, self.num_spin_state):
                if (k in self.interact_states) and (l in self.interact_states):
                    if self.element_name == "Sr88":
                        try:
                            self.c6_coefficients[k, l] = sr88.calc_C6_coefficient(self.spin_basis[k, 0], self.spin_basis[k, 1], self.spin_basis[k, 2], self.spin_basis[
                                                                                  k, 3], self.spin_basis[l, 0], self.spin_basis[l, 1], self.spin_basis[l, 2], self.spin_basis[l, 3], self.spin_basis[l, 4])
                        except Exception as e:
                            print(self.spin_basis[k, :])
                            print(self.spin_basis[l, :])
                            raise e

                        interact_id = tuple(
                            [self.spin_basis_labels[k], self.spin_basis_labels[l]])
                        self.c6_coef_dict[interact_id] = self.c6_coefficients[k, l]
                    else:
                        raise ValueError(
                            f"Element {self.element_name} not supported")
                        # self.c6_coefficients[k,l] =  sr88.calc_C6_coefficientRb85(self.spin_basis[k,0],self.spin_basis[k,1],self.spin_basis[k,2],self.spin_basis[k,3],self.spin_basis[l,0],self.spin_basis[l,1],self.spin_basis[l,2],self.spin_basis[l,3],self.spin_basis[l,4])

                        # interact_id = tuple([self.spin_basis_labels[k], self.spin_basis_labels[l]])

                        # self.c6_coef_dict[interact_id] = self.c6_coefficients[k,l]
                        # # * Rubidium not yet implemented

    def update_C6_coefficients(self, coeffs):
        """
        updates the C6 coefficients of the atom per spin state in Hz m^6 

        Parameters
        ----------
        freqs : np.array 
            C6 coefficients of the atom per spin state in Hz m^6
        """

        self.c6_coefficients = coeffs

    # def get_C6_from_labels(self, state1, state2):

    #     idx1 = self.spin_basis_labels.index(state1)
    #     idx2 = self.spin_basis_labels.index(state2)
