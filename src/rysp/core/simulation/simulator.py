import qutip
import warnings

import numpy as np
from qutip import Options, Qobj, propagator, ket2dm, vector_to_operator, operator_to_vector

from pathos.multiprocessing import ProcessingPool

from rysp.core.experiment import AtomSystem, ExperimentSetup
from rysp.core.simulation.qterm import QTerm, expand_operator, tensor
from rysp.core.simulation.hamiltonian import EvolutionHamiltonian
from rysp.core.circuit.operations import Measurement, Calculation, AtomTransport, CustomGate, ParametrizedOperation
from rysp.core.circuit.pulses import PulsedGate, ParametrizedPulsedGate
from rysp.core.circuit.quantumcircuit import CircuitComponent, QuantumCircuit

from rysp.core.physics.atom_wf_integrals import light_atom_integrate_3D_basis
from rysp.core.simulation.losses import loss_rates

import copy

DEBUG = False

# from RSP_old.Hamiltonian.calc_matrix_elements import light_atom_integrate_3D_basis
# from RSP_old.Hamiltonian.loss_rates.post_process_loss import integrated_loss_time, loss_rates, loss_psi, loss_rates_anti_trapping


def _measure_and_project(rng: np.random.Generator, state: Qobj, projector: Qobj, identity: Qobj) -> tuple[Qobj, int, float]:
    '''
    Projects a `state` onto a given projector
    returns the projected state together with the projection result and measure probability
    Inputs:
        rng - numpy random generator (eg: np.random.default_rng())
        state - state to be projected
        projector - projection operator
    returns:
        psi, result, probability
    '''
    counter_projector = identity-projector

    if state.type == 'ket':
        measure_prob = (state.dag() * projector * state).tr()
        # result is true if projected into 'projector'
        result = rng.random() < measure_prob
        if result:
            return projector * state, int(result), measure_prob
        else:
            return counter_projector * state, int(result), measure_prob
    elif state.type == 'oper':
        measure_prob = (state * projector).tr()
        result = rng.random() < measure_prob
        if result:
            return projector * state * projector, int(result), measure_prob
        else:
            return counter_projector * state * counter_projector, int(result), measure_prob
    else:
        raise ValueError(f"State type invalid: {state.type=}")


class Simulator:  # TODO: Simulator docstring
    '''
    Simulates the quantum circuit for different simulation levels.
    Allows the user to define different sources of noise, and how the noise is managed.
    '''

    def __init__(self,
                 experiment: ExperimentSetup,
                 atomsystem: AtomSystem,
                 sim_config={
                     'loss': {
                         ('0', '1'): {'0': 0, '1': 0},
                         ('1', 'r'): {'1': 0, 'r': 0}},
                     'motional energy': 0,
                     'dephase': [
                         #  ('0', '1'),
                         #  ('1', 'r')
                     ]},
                 backend_obj=QTerm) -> None:  # TODO: consider motional energy
        """
        __init__ _summary_

        _extended_summary_

        Parameters
        ----------
        experiment : ExperimentSetup
            Experimental setup object defining the physics and contraints of the system
        atomsystem : AtomSystem
            AtomSystem loaded with atoms and corresponding positions to run the simulation on
        sim_config : dict, optional
            dict specifying the simulation parameters., by default { 'loss': { ('0', '1'): {'0': 0, '1': 0}, ('1', 'r'): {'1': 0, 'r': 0}}, 'motional energy': 0, 'dephase': [ ]}
        backend_obj : _type_, optional
            _description_, by default QTerm
        """
        self._atom_system = atomsystem
        self._experiment = experiment
        self._config = sim_config
        self.states = [*self._experiment.atom['states'].keys()]
        self.ryd_states = [st for st in self.states if st not in {'0', '1'}]

        # list of tags for all the atoms in the system
        self.atom_list: list[str] = [*self._atom_system.atomsetup.keys()]
        self._state_order: list[str] = [
            *self._experiment.atom['states'].keys()]
        self.vdw_ham = self._atom_system.get_interaction_hamiltonian_info()

        self.available_transitions = self._experiment.available_transitions

        self.transition_offresonance_scattering = self._experiment.transition_offresonance_scattering

        self.rabi_freq = self._experiment.rabi_freq
        self.dephase_rates = self._experiment.dephase_rates

        identity = self._atom_system[(self._atom_system.atom_labels[0], '_id')]
        self.dims = identity.dims
        self.label_idx = 0

        self.idle_propagators = {}
        self.idle_integrators = {}
        self.custom_drift_call = {}
        self.custom_drift_evaluated = {}

    def __deepcopy__(self, memo):
        not_there = []
        existing = memo.get(self, not_there)
        if existing is not not_there:  # avoid copy recursion
            return existing

        sim = Simulator(self._experiment, copy.deepcopy(
            self._atom_system), copy.deepcopy(self._config))
        sim.saved_states = copy.deepcopy(self.saved_states)
        sim.saved_states_times = copy.deepcopy(self.saved_states)
        sim.atom_loss = copy.deepcopy(self.atom_loss)
        sim.psi = copy.deepcopy(self.psi)
        return sim

    def softcopy(self):
        return Simulator(self._experiment, copy.deepcopy(
            self._atom_system), copy.deepcopy(self._config))

    def update_config(self, new_config={}):
        """
        update_config _summary_

        _extended_summary_

        Parameters
        ----------
        new_config : dict, optional
            _description_, by default {}
        """
        for k, v in new_config.items():
            self._config[k] = v

    def _update_atom_interactions(self):
        '''
        Updates the Van der Waals terms (in case the atom positions have changed)
        '''
        self.vdw_ham = self._atom_system.get_interaction_hamiltonian_info()

        for key, val_func in self.custom_drift_call.items():
            if len(key) == 3:  # one atom term
                self.custom_drift_evaluated[key] = val_func(
                    self._atom_system.atomsetup[key[0]].pos)  # atom0.pos
            elif len(key) == 6:  # two atom term
                self.custom_drift_evaluated[key] = val_func(
                    self._atom_system.atomsetup[key[0]].pos,  # atom0.pos
                    self._atom_system.atomsetup[key[3]].pos)  # atom1.pos

    def _customize_drift(self, interaction_func, atom1: str, state1_in: str, state1_out: str, atom2=None, state2_in=None, state2_out=None, hermitian=True):
        """
        _customize_drift Adds a custom drift term to the Hamiltonian. The interaction strength will be calculated at run time, given the atomic positions. It is considered a two atom interaction if atom2 is not None.

        _extended_summary_

        Parameters
        ----------
        interaction_func : Callable (returns complex)
            function that returns the interaction stregth. It must take one or two arguments, which correspond to the atom positions.
        atom1 : str
            Atom label for first atom
        state1_in : _type_
            term with bra: < state1_in |_atom1
        state1_out : _type_
            term with ket: | state1_in >_atom1
        atom2 : str, optional
            Atom label for second atom, by default None
        state2_in : str, optional
            term with bra: < state1_in |_atom2, by default None
        state2_out : str, optional
            term with ket: | state1_in >_atom2, by default None
        """
        if atom2:
            state2_in = state2_in or state1_in
            state2_out = state2_out or state1_out
            self.custom_drift_call[(atom1, atom2, state1_in, state1_out, state2_in, state2_out)
                                   ] = lambda a1_pos, a2_pos: interaction_func(a1_pos, a2_pos)
            if hermitian:  # adds dagger term
                self.custom_drift_call[(
                    atom1, atom2, state1_out, state1_in, state2_out, state2_in)] = lambda a1_pos, a2_pos: np.conj(interaction_func(a1_pos, a2_pos))
        else:
            self.custom_drift_call[(atom1, state1_in, state1_out)
                                   ] = lambda a1_pos: interaction_func(a1_pos)
            if hermitian:  # adds dagger term
                self.custom_drift_call[(atom1, state1_out, state1_in)
                                       ] = lambda a1_pos: np.conj(interaction_func(a1_pos))

    def _add_drift_hamiltonians(self, ham_evol: EvolutionHamiltonian):
        '''
        Adding the Van der Waals forces to the EvolutionHamiltonian, readying for propagation
        '''

        # Adding VdW Hamiltonian terms
        for (a1, a2, r1, r2), V in self.vdw_ham.items():
            # a1: atom 1
            # a2: atom 2
            # r1, r2: rydberg states
            # V: VdW interatomic potential
            # Getting projector from BasisControl
            # TODO: VdW for motional states
            n1 = self._atom_system[(a1, r1, r2)]
            n2 = self._atom_system[(a2, r1, r2)]

            n12 = n1*n2
            ham_evol.add(
                Hamiltonian=n12,
                htype='drift',
                label=f'DRyd_{self.label_idx}_{(a1,a2)}',
                Control_Func_obj=V,
                operator_type='hamiltonian'
            )
            self.label_idx += 1
        for key, interaction in self.custom_drift_evaluated.items():
            if len(key) == 3:  # one atom
                a, r1, r2 = key
                n1 = self._atom_system[(a, r1, r2)]
                ham_evol.add(
                    Hamiltonian=n1,
                    htype='drift',
                    label=f'CtmDrift_{self.label_idx}_{key[0]}_{(r1,r2)}',
                    Control_Func_obj=interaction,
                    operator_type='hamiltonian'
                )
            elif len(key) == 6:  # two atoms
                a1, r11, r12, a2, r21, r22 = key
                n1 = self._atom_system[(a1, r11, r12)]
                n2 = self._atom_system[(a2, r21, r22)]
                ham_evol.add(
                    Hamiltonian=n1*n2,
                    htype='drift',
                    label=f'CtmDrift_{self.label_idx}_{(a1, a2)}_{(r11, r12, r21, r22)}',
                    Control_Func_obj=interaction,
                    operator_type='hamiltonian'
                )
            self.label_idx += 1

    def _add_losses(self, ham_evol: EvolutionHamiltonian):
        '''
        Add loss terms to the Evolution_Hamiltonian 
        Implemented: 
            off resonance scattering
        '''

        # Adding off resonance scattering
        for tr, states_scatt in self._config['loss'].items():
            for st, rate_mult in states_scatt.items():

                decay_rate = self.transition_offresonance_scattering[tr][st]

                if rate_mult*decay_rate != 0:
                    for a in self.atom_list:
                        ham_evol.add(Hamiltonian=self._atom_system[(a, st)],
                                     htype='drift',
                                     label=f"L_{self.label_idx}_scatt{''.join(tr)}_{a}",
                                     Control_Func_obj=rate_mult*decay_rate,
                                     operator_type='loss')
                        self.label_idx += 1

    def _add_cops(self, ham_evol: EvolutionHamiltonian):
        '''
        Add collapse operators to the Evolution Hamiltonian
        Implemented:
            Dephasing
        '''

        for atom in self.atom_list:
            for tr, dephase in self.dephase_rates.items():
                if tr in self._config['dephase']:
                    # Adds a dephasing term, with the projector relative to the second state
                    ham_evol.add(
                        Hamiltonian=self._atom_system[(atom, tr[1])],
                        htype='drift',
                        label=f"Co_{self.label_idx}_{''.join(tr)}",
                        Control_Func_obj=dephase,
                        operator_type='cops'
                    )
                    self.label_idx += 1

    def _post_process_losses(self, ham_evol: EvolutionHamiltonian):
        """
        _post_process_losses _summary_

        _extended_summary_

        Parameters
        ----------
        ham_evol : EvolutionHamiltonian
            _description_
        """
        # TODO: post_process_losses
        pass

    def _add_control_hamiltonians(self,
                                  ham_evol: EvolutionHamiltonian,
                                  pulse: PulsedGate,
                                  target_mapping: dict[int | str, str]):
        '''
        pulses:
            key: tuple(transition, atom)
            value: control function (in units of the max rabi frequency)
        target mapping: maps the pulse targets onto atoms of AtomSystem
        '''
        if len(pulse.pulses) == 0:
            warnings.warn("Control Hamiltonian: adding empty pulse!")

        for (channel, target), cfunc in pulse.pulses.items():

            transition = pulse.channel_config[channel]
            s0, s1 = transition
            atom = target_mapping[target]
            atom_it = self._atom_system.atomsetup[atom]  # Atom_in_Trap object
            h01, h10, h11 = light_atom_integrate_3D_basis(
                transition=(
                    self._state_order.index(s0),
                    self._state_order.index(s1),
                    len(self._state_order)
                ),
                motional_basis=atom_it.motional_basis,
                w=atom_it.trap_frequencies[0, :],
                wave_vec=self._experiment.qubit_lasers[transition]['direction'],
                mass=atom_it.m,
                lambdicke_regime=self._experiment.qubit_lasers[transition]['Lamb Dicke'],
                ignore_motional=True
            )
            ham_evol.add_list(
                Hamiltonians=[
                    Qobj(expand_operator(h10*self.rabi_freq[transition], len(self.atom_list),
                                         self.atom_list.index(atom), dims=self.dims[0])),
                    Qobj(expand_operator(h01*self.rabi_freq[transition], len(self.atom_list),
                                         self.atom_list.index(atom), dims=self.dims[0])),
                    Qobj(expand_operator(h11*self.rabi_freq[transition], len(self.atom_list),
                                         self.atom_list.index(atom), dims=self.dims[0]))
                ],
                htypes=['control']*3,
                labels=[f'PC_{self.label_idx+i}' for i in range(3)],
                Control_Funcs=[
                    (cfunc['amp'], pulse.max_time),
                    (np.conj(cfunc['amp']), pulse.max_time),
                    (-cfunc['det'], pulse.max_time)
                ],
                operator_types=['hamiltonian']*3
            )
            self.label_idx += 3

    def _func_H_to_array(self, H, tlist, args={}):
        """
        _func_H_to_array _summary_

        _extended_summary_

        Parameters
        ----------
        H : _type_
            _description_
        tlist : _type_
            _description_
        args : dict, optional
            _description_, by default {}

        Returns
        -------
        _type_
            _description_
        """
        new_H = []
        for hi in H:
            if isinstance(hi, list) and callable(hi[1]):
                new_H.append(
                    [hi[0], np.array([hi[1](t, args) for t in tlist])])
                # print("WARNING: H is callable. Shennanigans inc...")
            else:
                new_H.append(hi)
        return new_H

    def _run_evolution(self,
                       eH: EvolutionHamiltonian,
                       evol_time: float,
                       save_states=False,
                       time_list: np.ndarray | None = None,
                       ham_time_step_size=1e-8):
        '''
            Runs standard evolution of some evolution Hamiltonian, 

            cache_evolution: 
                [0] bool: do caching of the evolution operator 
                [1] str: associated cachelabel 
        '''
        self._add_drift_hamiltonians(eH)
        self._add_cops(eH)
        self._add_losses(eH)
        if time_list is None:
            intvs = int(np.ceil(evol_time/ham_time_step_size))
            evol_time_list = np.linspace(
                0, evol_time, intvs)
        else:
            intvs = time_list[1] - time_list[0]
            evol_time = time_list[-1] - time_list[0]
            evol_time_list = time_list - time_list[0]

        H, cops, hlabel, clabel = eH.create_total_Hamiltonian()

        opt = Options(tidy=False)
        opt.use_openmp = False

        # print(f"h={H}")
        if DEBUG:
            print(
                f"Running full space evolution. time gap: {evol_time_list[1]-evol_time_list[0]}, evol time {evol_time_list[-1]-evol_time_list[0]}")
        U = propagator(H=H,
                       t=evol_time_list,
                       c_op_list=cops,
                       parallel=False,
                       options=opt,
                       args={"_step_func_coeff": False})
        if len(cops) > 0:  # Then it runs a Lindbladian
            if self.psi.type == 'ket':  # ket state # type: ignore
                rhoS = ket2dm(self.psi)
            elif self.psi.type == 'oper':  # density operator # type: ignore
                rhoS = self.psi
            else:
                raise ValueError("Psi state has wrong type")
            if DEBUG:
                print('Lindbladian evolution')
            rhoSvec = operator_to_vector(rhoS)
            rhovec = U * rhoSvec
            psi_list: list[Qobj] = [
                Qobj(vector_to_operator(rho_k)) for rho_k in rhovec]
            exp_loss, loss_cumm, lbls = loss_rates(psi_list, eH, evol_time)

            def map_loss_label_to_atom(loss_label: str):
                labels = loss_label.split(sep='_')
                try:
                    lbl = labels[3]
                    if lbl in self.atom_list:
                        return lbl
                    else:
                        raise ValueError(
                            f"run_circuit: Label of loss Hamiltonian not defined properly{loss_label}")
                except IndexError:
                    raise ValueError(
                        f"run_circuit: Label of loss Hamiltonian not defined properly: {loss_label}")

            for a, prob in zip([map_loss_label_to_atom(lb)
                                for lb in lbls], exp_loss):
                self.atom_loss[a] = 1 - (1-prob)*(1-self.atom_loss[a])

            if save_states:
                self.saved_states += psi_list
                self.saved_states_times += [self.run_time + dt
                                            for dt in evol_time_list]
            self.psi = psi_list[-1]

        else:  # Schrodinger evolution
            if DEBUG:
                print('Schrodinger evolution')
            psi_list = U*self.psi
            self.psi = psi_list[-1]
            if save_states:
                self.saved_states += [*psi_list]
                self.saved_states_times += [self.run_time + dt
                                            for dt in evol_time_list]

        self.run_time += evol_time

    def _run_subspace_propagation(self,
                                  pulse: PulsedGate | None,
                                  targets: list[str] | None,
                                  time_list: np.ndarray | None = None,
                                  state_projectors: list[str] = ['r'],
                                  lattice_match_tolerance: float = 1e-2,
                                  ham_time_step_size: float = 1e-8,
                                  evol_time: float | None = None):
        '''
        Propagates the pulsed gate by first evolving the propagator on a small subsystem. 
        No crosstalk is considered for the idle qubits, but single qubit gates (like cops) are ran 
        '''
        # print("Running subspace propagation...")
        if time_list is not None:
            if pulse is not None:
                pulse._convert_pulse_list(time_list)
        elif pulse is not None:
            if pulse.time_list is None:
                pulse._compile_pulses(time_resolution=ham_time_step_size)
            time_list = pulse.time_list
        elif evol_time is not None:
            time_list = np.arange(
                0, evol_time+ham_time_step_size, ham_time_step_size)
        else:
            raise RuntimeError(
                "_run_subspace_propagation: pulse and time_list/evol_time cannot be None at the same time")

        # if time_list is not None and len(time_list) == 0:
        #     print(f"{ham_time_step_size=}, {evol_time=}")
        assert (time_list is not None and len(time_list) > 0)

        for atom in self.atom_list:
            idle_evol_time = 1e-6

            if atom not in self.idle_propagators or atom not in self.idle_integrators:
                idle_time_list = np.arange(
                    0, idle_evol_time+ham_time_step_size, ham_time_step_size)
                # print(
                #     f"Generating integrator for atom '{atom}', {idle_evol_time=}")
                U, intg = Simulator.cache_gate_propagator(
                    gate=None,
                    lattice=self._atom_system,
                    experiment=self._experiment,
                    time_list=idle_time_list,
                    targets=[atom],
                    sim_config=self._config,
                    state_projectors=state_projectors
                )
                # assert(U.check_isunitary())
                # FIXME: bad time conversion, need to apply power of matrix
                self.idle_propagators[atom] = (
                    U, (idle_time_list[-1]-idle_time_list[0]))
                # normalize cached integrator to account for different pulse durations
                self.idle_integrators[atom] = {
                    st: intg[st]/(idle_time_list[-1]-idle_time_list[0]) for st in state_projectors}

        if pulse is not None:
            if targets is None:
                raise ValueError(
                    "targets cannot be None while pulse is defined ")
            # assert(self.psi is not None)
            lattice_compat = (pulse.is_cached and pulse.cached_lattice is not None) and self._atom_system._check_lattice_compatibility(
                pulse.cached_lattice, targets, lattice_match_tolerance)
            settings_compat = (self._config == pulse.cache_settings)
            if pulse.is_cached and not lattice_compat:
                print(
                    f"Cached lattice does not match .... {pulse.cached_lattice=}")
            if pulse.is_cached and not settings_compat:
                print(
                    f"Cached settings do not match .... {self._config=}, {pulse.cache_settings=}")
            if not pulse.is_cached or not lattice_compat or not settings_compat:
                if DEBUG:
                    print("Caching pulsed gate...")
                Simulator.cache_gate_propagator(
                    pulse,
                    self._atom_system,
                    self._experiment,
                    time_list=time_list,
                    sim_config=self._config,
                    state_projectors=state_projectors
                )

            # check if idle propagators/integrators have been cached. If not, cache them
            # TODO: verify if cached settings are the same as at runtime

            assert (pulse.cached_integrator is not None)
            labels = [*targets]
            op_list = [pulse.cached_propagator]
            intg_list = {st: [pulse.cached_integrator[st]]
                         for st in state_projectors}
            appended = False
            for atom in self.atom_list:
                if atom not in labels:

                    op_list.append(self.idle_propagators[atom][0])
                    labels.append(atom)
                    for st in state_projectors:
                        intg_list[st].append(
                            self.idle_integrators[atom][st] * (time_list[-1]-time_list[0]))
                    appended = True
            permute_list = self._atom_system.get_permutation_list(targets)
        else:
            labels = []
            op_list = []
            intg_list = {st: [] for st in state_projectors}
            appended = False

            # FIXME: set time for idle propagators
            for atom in self.atom_list:
                op_list.append(self.idle_propagators[atom][0])
                labels.append(atom)
                for st in state_projectors:
                    intg_list[st].append(
                        self.idle_integrators[atom][st] * (time_list[-1]-time_list[0]))
                appended = True
            permute_list = None

        if appended:
            if permute_list:

                big_op = tensor(op_list).permute(permute_list)
                big_int = {st: tensor(intg_list[st]).permute(
                    permute_list) for st in state_projectors}
            else:
                big_op = tensor(op_list)
                big_int = {st: tensor(intg_list[st])
                           for st in state_projectors}
        else:
            big_op = op_list[0]
            big_int = {st: intg_list[st][0] for st in state_projectors}

        integration_res = {}
        for st in state_projectors:
            intg = (self.psi.dag() * big_int[st] * self.psi).tr()
            # if np.abs(np.imag(intg)) > 1e-9:
            #     raise RuntimeError(f"Integrated time is complex .. something went wrong... {intg=} ")
            if np.real(intg) < 0:
                raise RuntimeError(
                    f"State time integration failed. State time integral should be positive. It got {intg} for state {st}")
            integration_res[st] = intg
        # print(f"BigOp: {big_op}")

        # TODO: Calculate anti-trapping losses from integrated times ...
        self.psi = Qobj(big_op * self.psi)
        self.run_time += (time_list[-1]-time_list[0])

    def run_circuit(self,
                    circuit: 'QuantumCircuit',
                    psi0: Qobj | str | None = None,
                    ham_time_step_size=1e-8,
                    save_states=False,
                    use_cached=True,
                    lattice_match_tolerance=1e-2,
                    stochastic_projection=True,
                    runtime_caching=False,
                    reset_positions=True,
                    return_state=False):
        """
        run_circuit _summary_

        _extended_summary_

        Parameters
        ----------
        circuit : QuantumCircuit
            _description_
        psi0 : Qobj | str | None, optional
            _description_, by default None
        ham_time_step_size : _type_, optional
            _description_, by default 1e-8
        save_states : bool, optional
            _description_, by default False
        fake_evol : bool, optional
            _description_, by default False
        use_cached : bool, optional
            _description_, by default True
        lattice_match_tolerance : _type_, optional
            _description_, by default 1e-2
        stochastic_projection : bool, optional
            _description_, by default True
        runtime_caching : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        NotImplementedError
            _description_
        ValueError
            _description_
        NotImplementedError
            _description_
        NotImplementedError
            _description_
        """

        rng = np.random.default_rng()
        identity = self._atom_system[(self._atom_system.atom_labels[0], '_id')]
        self.dims = identity.dims

        if psi0 is None:
            psi0 = '0'*len(self._atom_system.atom_labels)

        if isinstance(psi0, str):
            self.psi: Qobj = self._atom_system.ket_str(psi0)
        else:
            self.psi: Qobj = psi0

        assert (self.psi.dims is not None)

        if self.psi.dims[0] != self.dims[0]:
            raise ValueError(
                f"Simulator.run_circuit: psi has wrong dimensions! {self.psi.dims[0]=}, {self.dims[0]=}")  # type: ignore

        self.atom_loss: dict[str, float] = {atom: 0 for atom in self.atom_list}

        skip_idle = True
        # 'loss': {
        #      ('0', '1'): {'0': 0, '1': 0},
        #      ('1', 'r'): {'1': 0, 'r': 0}},
        #  'motional energy': 0,
        #  'dephase': [
        #      #  ('0', '1'),
        #      #  ('1', 'r')
        #  ]

        if len(self._config['dephase']) > 0:
            skip_idle = False
        for tr, val in self._config['loss'].items():
            if skip_idle:
                for s, rate in val.items():
                    if rate > 0:
                        skip_idle = False
                        break
            else:
                break

        self._update_atom_interactions()

        if save_states:
            self.saved_states = [self.psi]
            self.saved_states_times = [0.0]

        self.label_idx = 0
        self.run_time = 0

        def run_idle(evol_time, step_size=ham_time_step_size):
            if evol_time > 0:
                if DEBUG:
                    print("Idle evolution")
                if not use_cached:
                    eH = EvolutionHamiltonian(self.dims)
                    self._run_evolution(
                        eH, evol_time, save_states=save_states,
                        ham_time_step_size=step_size)
                else:
                    self._run_subspace_propagation(
                        None, None, ham_time_step_size=step_size,
                        evol_time=evol_time)
                    if save_states:
                        self.saved_states += [self.psi]
                        self.saved_states_times += [self.run_time]

        starting_position = self._atom_system.get_atom_positions(
            self.atom_list)

        runtime_sequence = [*circuit.operation_sequence]

        runtime_vars = copy.deepcopy(circuit.variables)

        # Running the operation sequence .........
        for i, component in enumerate(runtime_sequence):
            if DEBUG:
                print(f"Component {str(component)}")
            # Atom Transport
            if isinstance(component.operation, AtomTransport) and self._experiment.simulation_level == 'pulsed':
                new_positions = component.operation.new_positions
                old_positions = self._atom_system.get_atom_positions(
                    [*new_positions.keys()])

                travel_times = component.operation.get_travel_times(
                    old_positions)

                self._atom_system.remap_atom_positions(new_positions)
                self._update_atom_interactions()
                if self._experiment._data['parallel transport']:
                    evol_time = float(
                        np.max([tt for tt, _ in travel_times.values()]))
                else:
                    evol_time = float(
                        np.sum([tt for tt, _ in travel_times.values()]))
                for a, v in travel_times.items():
                    prob = v[1]
                    self.atom_loss[a] = 1 - (1-prob)*(1-self.atom_loss[a])
                if not skip_idle:
                    run_idle(evol_time, step_size=1e-6)

            # Parametrized
            elif isinstance(component.operation, ParametrizedOperation) and self._experiment.simulation_level == 'pulsed':
                arg_dict = {}
                for arg in component.needs_variables:
                    arg_dict[arg] = runtime_vars[arg]
                opp = component.evaluate(**arg_dict)

                # load operation
                if opp is not None:
                    if opp.iID not in circuit.id_counter:
                        circuit.id_counter[opp.iID] = 0

                    inner_id = opp.iID + '' + \
                        str(circuit.id_counter[opp.iID]+1)
                    circuit.id_counter[opp.iID] += 1
                    # print(f"Adding operation: {str(operation)}, iid={inner_id}")
                    runtime_sequence.insert(
                        i+1, CircuitComponent(operation=opp,
                                              targets=component.targets,
                                              controls=component.controls,
                                              args_dict={},
                                              simulation_level='pulsed',
                                              name=f'{str(opp)}_{inner_id}')
                    )

            # Pulsed Gate
            elif isinstance(component.operation, PulsedGate) and self._experiment.simulation_level == 'pulsed':
                evol_time = component.operation.duration

                run_cached_flag = use_cached and (
                    component.operation.is_cached or runtime_caching)

                if not run_cached_flag:
                    # run full evolution
                    warnings.warn("Running full evolution,"
                                  + " cache checks failed")
                    target_mapping: dict[int | str, str] = {targets: atoms for targets, atoms in zip(
                        component.operation.targets, component.targets)}

                    if component.operation.time_list is None:
                        component.operation._compile_pulses(
                            time_resolution=ham_time_step_size)
                    eH = EvolutionHamiltonian(self.dims)
                    # Adds control Hamiltonians
                    self._add_control_hamiltonians(eH,
                                                   pulse=component.operation,
                                                   target_mapping=target_mapping)
                    self._run_evolution(eH,
                                        evol_time=0,  # gets computed automatically from time list
                                        time_list=component.operation.time_list,
                                        save_states=save_states,
                                        ham_time_step_size=ham_time_step_size)
                else:  # run cached operation
                    # if subspace_evolution:
                    if DEBUG:
                        print("Subspace propagation")
                    self._run_subspace_propagation(pulse=component.operation,
                                                   targets=component.targets,
                                                   time_list=component.operation.time_list,
                                                   lattice_match_tolerance=lattice_match_tolerance,
                                                   ham_time_step_size=ham_time_step_size)

                    if save_states:
                        self.saved_states += [self.psi]
                        self.saved_states_times += [self.run_time]

            # Custom Gate
            elif isinstance(component.operation, CustomGate) and self._experiment.simulation_level == 'digital':
                # TODO: custom gate on run circuit
                raise NotImplementedError()

            # Measurement
            elif isinstance(component.operation, Measurement):
                targets = component.targets
                if component.operation.method_name == 'simple':
                    params = component.operation.method_parameters
                    wait_time = params['wait time']
                    state_project = params['target state']
                    if component.variable is None:
                        raise ValueError(
                            "Simulator - run_circuit: Measurement variable is None. Please define a measurement variable for the operation")

                    if not skip_idle:
                        run_idle(wait_time)

                    meas_output_variable = component.variable

                    projection_result = {}  # dict with {atoms: overlap}
                    # assert(isinstance(self.psi, Qobj))

                    if stochastic_projection:
                        for atom in targets:
                            self.psi, result, prob = _measure_and_project(
                                rng, self.psi, self._atom_system[(atom, state_project)], identity)
                            norm = self.psi.norm()
                            if norm == 0:
                                print(
                                    f"State norm is 0 !? {self.psi=}, {result=}, {prob=}")
                            self.psi: Qobj = Qobj(self.psi.unit())
                            projection_result[atom] = component.operation.map_result[result]
                    else:
                        raise NotImplementedError(
                            "Not implemented non-stochastic projection evolution.")

                    if component.operation.flatten:
                        if len(projection_result) == 1:
                            runtime_vars[meas_output_variable] = [
                                *projection_result.values()][0]
                        else:
                            runtime_vars[meas_output_variable] = [
                                *projection_result.values()]
                    else:
                        runtime_vars[meas_output_variable] = projection_result

            # Calculation
            elif type(component.operation) == Calculation:
                arg_dict = {}
                for arg in component.needs_variables:
                    arg_dict[arg] = runtime_vars[arg]['value']
                exec_time, result = component.evaluate(**arg_dict)

                for k, v in result.items():  # loading the results back into the stored variables
                    runtime_vars[k] = {'value': v}
                if not skip_idle:
                    run_idle(exec_time)

            else:
                raise NotImplementedError(
                    f"Simulator does not know what to do with such operation: {type(component.operation)}.")
            # etc ...
            pass

        if reset_positions:
            self._atom_system.remap_atom_positions(starting_position)
        if return_state:
            return runtime_vars, self.psi
        else:
            return runtime_vars

    def run_circuit_statistical(self,
                                circuit: 'QuantumCircuit',
                                num_runs=100,
                                psi0: Qobj | str | None = None,
                                ham_time_step_size=1e-8,
                                save_states=False,
                                use_cached=True,
                                lattice_match_tolerance=1e-2,
                                stochastic_projection=True,
                                runtime_caching=False,
                                parallel_runs=True,
                                track_end_state=True):
        """
        run_circuit_statistical _summary_

        _extended_summary_

        Parameters
        ----------
        circuit : QuantumCircuit
            _description_
        num_runs : int, optional
            _description_, by default 100
        psi0 : Qobj | str | None, optional
            _description_, by default None
        ham_time_step_size : _type_, optional
            _description_, by default 1e-8
        save_states : bool, optional
            _description_, by default False
        fake_evol : bool, optional
            _description_, by default False
        use_cached : bool, optional
            _description_, by default True
        lattice_match_tolerance : _type_, optional
            _description_, by default 1e-2
        stochastic_projection : bool, optional
            _description_, by default True
        runtime_caching : bool, optional
            _description_, by default False
        parallel_runs : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        self.stat_vars = {}
        self.stat_start_indexes = []
        run_params = {'psi0': psi0, 'ham_time_step_size': ham_time_step_size,
                      'save_states': save_states,
                      'use_cached': use_cached, 'lattice_match_tolerance': lattice_match_tolerance,
                      'stochastic_projection': stochastic_projection, 'runtime_caching': runtime_caching}
        end_states = []

        def process_runs(out):
            for qvars in out:
                if track_end_state:
                    circ_vars = qvars[0]
                    end_states.append(qvars[1])
                else:
                    circ_vars = qvars
                for varname, varvalue in circ_vars.items():
                    if varname not in self.stat_vars:
                        self.stat_vars[varname] = [varvalue]
                    else:
                        self.stat_vars[varname].append(varvalue)
        if not parallel_runs:
            qvars_arr = []
            for i in range(num_runs):
                if save_states:
                    if len(self.stat_start_indexes) == 0:
                        self.stat_start_indexes = [0]
                    else:
                        self.stat_start_indexes.append(len(self.saved_states))
                qcvars = self.run_circuit(circuit, *run_params)
                qvars_arr.append(qcvars)
            process_runs(qvars_arr)

        else:
            if DEBUG:
                print("Running first circuit")
            # running once without paralelization to cache gates
            r0 = self.run_circuit(circuit, **run_params,
                                  return_state=track_end_state)

            def run_simulation(circuit, run_params, track_end_state):
                sim = self.softcopy()
                result = sim.run_circuit(
                    circuit, **run_params, return_state=track_end_state)
                return result

            if DEBUG:
                print("Running the rest")
            qvars_arr = ProcessingPool().map(lambda args: run_simulation(
                circuit, run_params, track_end_state=track_end_state), [None]*(num_runs-1))
            if qvars_arr is not None:
                process_runs([r0, *qvars_arr])
            else:
                print("Paralelization failed.....")
        if track_end_state:
            return self.stat_vars, end_states
        else:
            return self.stat_vars

    @staticmethod
    def cache_gate_propagator(gate: PulsedGate | None,
                              lattice: AtomSystem,
                              experiment: ExperimentSetup,
                              time_list: np.ndarray | list | None = None,
                              targets: list[str] = [],
                              interval: float | str = 1e-8,
                              sim_config: dict = {'loss': {
                                  ('0', '1'): {'0': 0, '1': 0},
                                  ('1', 'r'): {'1': 0, 'r': 0}},
                                  'motional energy': 0, 'dephase': []},
                              state_projectors: list[str] = ['r']) -> tuple[Qobj, dict[str, Qobj]]:
        """
        cache_gate_propagator Static method that generates the propagator for the gate, caching the evolution into a unitary operator. In case the simulation is noisy, then the cached propagator corresponds to a superoperator.

        Parameters
        ----------
        gate : PulsedGate | None
            PulsedGate to cache. If None, then generates the propagator for the idle qubits.
        lattice : AtomSystem
            Atom lattice where to run the propagation on, as the evolution is sensitive to interatomic distances. 
        experiment : ExperimentSetup
            Experiment configuration that defines the physical quantities.
        time_list : np.ndarray | list | None, optional
            List of times where to compile the pulse sequence and to evolve the propagator. If None, and `interval` is a float, then considers a time_list with corresponding time intervals. If `interval=='suggested'` then it uses the time_list suggested by the gate (), by default None
        targets : list[str], optional
            List of targets on the lattice to run the PulsedGate on. If empty, then assumes natural ordering (first atom on the lattice is first target, etc), by default []
        interval : float | str, optional
            Time interval to generate the time_list. If 'suggested', then uses the time_list suggested by the gate, by default 1e-9
        sim_config : _type_, optional
            Simulation configuration to be considered. Here the user defines parameters of the noise simulation, by default {'loss':{}, 'motional energy':0, 'dephase':[]}
        state_projectors : list[str], optional
            For idle atoms, find the time operator for the given states. Often only the rydberg state time is needed, by default ['r']

        Returns
        -------
        tuple[Qobj,dict[str, Qobj]]
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        """

        if isinstance(gate, ParametrizedPulsedGate):
            raise ValueError(
                "Cannot cache  parametrized gates, as the parameters are variable")

        step_coef = False
        opt = Options(tidy=False, use_openmp=False)
        if gate is not None:  # Active pulse mode
            if len(targets) == 0:  # get default targets
                # pruning the lattice to have the right number of targets
                local_asys = AtomSystem({})
                for i, (label, atom) in enumerate(lattice.atomsetup.items()):
                    if i == len(gate.targets):
                        break
                    local_asys.add_atom(label, atom, atom.pos)
                target_map = {tgt: local_asys.atom_labels[i]
                              for i, tgt in enumerate(gate.targets)}
            else:  # specify targets on the lattice
                # pruning the lattice to have the right number of targets
                local_asys = AtomSystem({})
                for tgt in targets:
                    local_asys.add_atom(
                        tgt, lattice.atomsetup[tgt], lattice.atomsetup[tgt].pos)
                target_map = {tgt: targets[i]
                              for i, tgt in enumerate(gate.targets)}
            if time_list is None:
                if type(interval) == float and interval > 0:
                    time_list = np.arange(
                        0, gate.duration+interval, interval)
                    gate._convert_pulse_list(time_list)
                elif (type(interval) == str) and (interval == 'suggested'):
                    time_list = np.array(gate.get_pulse_switch_times())
                    gate._convert_pulse_list(time_list)
                    step_coef = True
                else:
                    raise ValueError(
                        f"Argument `interval` not properly defined. Has to either be a float>0 or 'suggested'. {interval=}")
            sim1 = Simulator(experiment, local_asys, sim_config=sim_config)
            eH = EvolutionHamiltonian(sim1.dims)
            sim1._add_drift_hamiltonians(eH)
            sim1._add_cops(eH)
            sim1._add_control_hamiltonians(eH, gate, target_mapping=target_map)

        elif len(targets) > 0:  # idle targets
            if time_list is None:
                raise ValueError(
                    "cache_gate_propagator: time_list cannot be None when pulsed gate is not defined")
            # print(f"time list gap: {time_list[1]-time_list[0]:e}")
            local_asys = AtomSystem({})
            for tgt in targets:
                local_asys.add_atom(
                    tgt, lattice.atomsetup[tgt], lattice.atomsetup[tgt].pos)
            sim1 = Simulator(experiment, local_asys, sim_config=sim_config)
            eH = EvolutionHamiltonian(sim1.dims)
            sim1._add_drift_hamiltonians(eH)
            sim1._add_cops(eH)
        else:
            raise ValueError("Gate cannot be none without defining targets...")

        H, cops, hlab, clab = eH.create_total_Hamiltonian()

        U = propagator(H=H,
                       t=time_list,
                       c_op_list=cops,
                       parallel=False,
                       options=opt,
                       args={"_step_func_coeff": step_coef})

        integrators = {}

        for state in state_projectors:
            proj = 0  # add state projectors for all atoms
            for i, atom in enumerate(local_asys.atomsetup):
                if i == 0:
                    proj = local_asys[(atom, state, state)]
                else:
                    proj += local_asys[(atom, state, state)]
            # integrating:  int_0^T U^+(t) P_r U(t) dt
            for i, u in enumerate(U):
                if i == 0:
                    integrators[state] = u.dag() * proj * u
                else:
                    integrators[state] += u.dag() * proj * u
        if gate is not None:
            gate.cache_gate(U[-1], local_asys, integrators, sim_config)

        # if not U[-1].check_isunitary(): # then shit happened
        #     raise RuntimeError(f"Unitary check failed: {[*zip(hlab, H)]}, {time_list=}")

        return U[-1], integrators
