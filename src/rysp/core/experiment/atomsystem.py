
from rysp.core.experiment.atom import AtomInTrap
import numpy as np
from rysp.core.simulation.qterm import tensor
import copy


class AtomSystem:
    # TODO: plot atom lattice (utils)
    def __init__(self, setup_dict: dict | None = None) -> None:
        self.atomsetup: dict = setup_dict if setup_dict is not None else {}
        self.atom_labels: list[str] = []  # defining atom order
        if len(self.atomsetup) > 0:
            self.atom_labels = [*self.atomsetup.keys()]

        self.full_basis = {}
        self.kets = {}
        self.atom_indexes = {}
        self.idx = 0

    def __deepcopy__(self, memo):
        not_there = []
        existing = memo.get(self, not_there)
        if existing is not not_there:  # avoid copy recursion
            return existing

        atomsys = AtomSystem(copy.deepcopy(self.atomsetup, memo))
        atomsys.full_basis = copy.deepcopy(self.full_basis)
        atomsys.atom_indexes = copy.deepcopy(self.atom_indexes)
        atomsys.kets = copy.deepcopy(self.kets)
        atomsys.idx = self.idx
        return atomsys

    @classmethod
    def from_positions(cls, atom: AtomInTrap, positions: dict):
        asys = cls()
        asys.add_atoms(atom, positions)
        return asys

    def add_atoms(self, atom: AtomInTrap, positions: dict):
        self.lattice_key = None

        for lbl, pos in positions.items():
            self.add_atom(lbl, atom, pos)

    def copy_lattice(self, atoms: list[str] | None = None):
        if atoms is None:
            return copy.deepcopy(self)
        else:
            lattice = AtomSystem({})
            for label, atomit in self.atomsetup.items():
                new_atomit = copy.deepcopy(atomit)

                lattice.add_atom(label, new_atomit, new_atomit.pos)
            return lattice

    def add_atom(self,
                 label: str,
                 atom: AtomInTrap,
                 position: np.ndarray):
        """
        add_atom Adds a new atom onto the system, 
        with a given (unique) label as identification. 

        Parameters
        ----------
        label : str
            Unique label for the atom
        atom : AtomInTrap
            Atom object loaded from the experimental setup
        position : np.ndarray[float]
            Atom position in SI units (m)
        """
        self.atomsetup[label] = copy.deepcopy(atom)
        self.atomsetup[label].update_pos(position)

        if label in self.atom_labels:
            print(f"Atom with label {label} already exists, replacing...")
        else:
            self.atom_labels.append(label)
            dims = len(self.atomsetup[label].basis_identity.dims[0])
            self.atom_indexes[label] = [
                i for i in range(self.idx, self.idx+dims)]
            self.idx += dims

        if len(self.full_basis) > 0:
            print("Basis was already initialized. Resetting...")
            self.full_basis = {}
            self.kets = {}

    def move_atom(self, label, new_position: np.ndarray):
        '''
        Moves the labeled atom in the array, 
        returns the travel distance
        '''
        old_position = self.atomsetup[label].pos
        self.atomsetup[label].update_pos(new_position)
        return np.linalg.norm(old_position-new_position)

    def remap_atom_positions(self, position_mapping={}):
        '''
        Maps the labeled atoms in 'position_mapping' to the corresponding positions
        position_mapping: 
            key=atom label, 
            value=new position
        returns a list of the travelled positions 
        '''

        distances = []
        for k, v in position_mapping.items():
            distances.append(self.move_atom(k, v))
        return distances

    def get_atom_positions(self, atom_labels):
        poss = {}
        for lbl in atom_labels:
            poss[lbl] = copy.deepcopy(self.atomsetup[lbl].pos)
        return poss

    def switch_atom(self, label, new_atom: AtomInTrap):
        self.atomsetup[label] = new_atom

    def ket_str(self, state_str):
        # TODO: sanitize inputs
        state = []
        for i, s in enumerate(state_str):
            atom_label = self.atom_labels[i]

            state.append(self.atomsetup[atom_label].ket(s))
        return tensor(state)

    def get_interaction_hamiltonian_info(self, ryd='r', tidyup=1e-6):
        '''
        Gets the rydberg interaction terms for the current atomic configuration
        returns {(a1, a2, r1, r2): Hterm} for all near a1, a2

        Only computes VdW terms (r1=r2)
        '''
        # 'standard' C6: 1e-25 SI
        # 'standard' R: 3e-6 SI
        # 'standard' C6/R**6: ~1e8 SI

        vdw_ham_info = {}

        for i, atom1 in enumerate(self.atom_labels[:-1]):
            c6 = self.atomsetup[atom1].c6_coef_dict[(ryd, ryd)]
            for atom2 in self.atom_labels[i+1:]:
                # Interaction strength: c6/R**6
                interaction_strength = c6 / \
                    np.linalg.norm(
                        self.atomsetup[atom1].pos-self.atomsetup[atom2].pos)**6
                if np.abs(interaction_strength) > tidyup * 1e8:
                    vdw_ham_info[(atom1, atom2, ryd, ryd)
                                 ] = interaction_strength
        return vdw_ham_info

    def _check_lattice_compatibility(self, lattice: 'AtomSystem', targets: list[str], lattice_match_tolerance: float) -> bool:
        '''
        Checks if the geometry of target atoms in atomsystem matches with the geometry of a given `lattice`. The geometry is assumed the same if all the relative distances between the atoms are the same. It is assumed that the `targets` atom list has the same ordering as the atoms in `lattice` 
        '''
        # TODO: Lattice compatibility algorithm needs validation

        def relative_error(a, b):
            return np.abs(2 * (a - b) / (a + b))

        # system atoms 0, lattice atoms 0 ()
        for i, (sa_0, la_0) in enumerate(zip(targets, lattice.atom_labels)):
            for j, (sa_1, la_1) in enumerate(zip(targets, lattice.atom_labels)):
                if j <= i:
                    continue  # skips iteration for atoms before the current label

                sys_atom0_pos = self.atomsetup[sa_0].pos
                sys_atom1_pos = self.atomsetup[sa_1].pos

                sys_relative_dist = np.linalg.norm(sys_atom0_pos-sys_atom1_pos)
                latt_atom0_pos = lattice.atomsetup[la_0].pos
                latt_atom1_pos = lattice.atomsetup[la_1].pos
                latt_relative_dist = np.linalg.norm(
                    latt_atom0_pos-latt_atom1_pos)

                if not (relative_error(sys_relative_dist, latt_relative_dist) < lattice_match_tolerance):
                    print(
                        f"Lattices do not match: {(sa_0, la_0)=}, {(sa_1, la_1)=}, {sys_atom0_pos=}, {sys_atom1_pos=}, {latt_atom0_pos=}, {latt_atom1_pos=}")
                    return False
        return True

    def __getitem__(self, args: tuple[str, str] | tuple[str, str, str]):
        if args not in self.full_basis:
            self._precompute_basis([args])
            if len(args) == 2:
                self._precompute_basis([args], ket=True)

        return self.full_basis[args]

    def ket(self, args):
        if args not in self.full_basis:
            self._precompute_basis([args])
            self._precompute_basis([args], ket=True)
        return self.kets[args]

    def _precompute_basis(self, terms: list[tuple[str, str] | tuple[str, str, str]], ket=False):
        '''
        Precalculate hamiltonian terms.
        Calculates the corresponding kets, for terms[args], 
        Where args:
            with 2 arguments: detuning (attached to delta)
                args[0] : atom label
                args[1] : atom state |s>_0
                > creates |s><s|
            with 3 arguments: atomic transition (attached to Omega)
                args[0] : atom label 0
                args[1] : atom state |s0>_0 (spin, motional)
                args[2] : atom state |s1>_0 (spin, motional)
                > creates |s1><s0|
        '''
        for args in terms:
            if len(args) == 2:  # (atom, state)
                # args[0] : atom label 0
                # args[1] : atom state
                atom = self.atomsetup[args[0]]
                if type(args[1]) == str:
                    state = (args[1], '_id')
                elif type(args[1]) == tuple:
                    state = (args[1][0], args[1][1])
                else:
                    raise ValueError(
                        f"BasisControl: State argument poorly defined: {args}")

                ops = []
                for atomlbl in self.atom_labels:
                    if atomlbl == args[0]:
                        ss = atom.ket(state[0], state[1])
                        if ket:
                            ops.append(ss)
                        else:
                            ops.append(ss * ss.dag())
                    else:
                        ops.append(
                            self.atomsetup[atomlbl].basis_identity)
                if ket:
                    self.kets[args] = tensor(ops)
                else:
                    self.full_basis[args] = tensor(ops)

            elif len(args) == 3:  # (atom, state_in, state_out)
                # raise ValueError("Lets avoid this for now")
                atom1 = self.atomsetup[args[0]]

                state1 = AtomSystem._validate_state_arg(args[1])
                state2 = AtomSystem._validate_state_arg(args[2])

                ops = []
                for atomlbl in self.atom_labels:
                    if atomlbl == args[0]:
                        ss1 = atom1.ket(state1[0], state1[1])
                        ss2 = atom1.ket(state2[0], state2[1])
                        ops.append(ss2 * ss1.dag())
                    else:
                        ops.append(
                            self.atomsetup[atomlbl].basis_identity)
                self.full_basis[args] = tensor(ops)
            else:
                raise IndexError("BasisControl: cannot get basis item")

    @staticmethod
    def _validate_state_arg(arg):
        if type(arg) == str:
            state = (arg, '_id')
        elif type(arg) == tuple:
            state = (arg[0], arg[1])
        else:
            raise ValueError(
                f"BasisControl: State argument poorly defined: {arg}")
        return state

    def get_permutation_list(self, targets: list[str]):
        '''
        Returns the permutation list used for Qobj.permute
        Assumes form Qobj = tensor( U_tgt, U_other )
        [Algorithm taken from qutip.qip.operations.gates.expand_operator]
        '''

        target_idx = []
        for tgt in targets:
            target_idx.extend(self.atom_indexes[tgt])

        remain_idx = []
        for atom in self.atom_labels:
            if atom not in targets:
                remain_idx.extend(self.atom_indexes[atom])
        N = len(target_idx) + len(remain_idx)

        new_order = [0]*N
        for i, t in enumerate(target_idx):
            new_order[t] = i
        remain_qb = [*range(len(target_idx), N)]
        for i, ind in enumerate(remain_idx):
            new_order[ind] = remain_qb[i]
        return new_order
