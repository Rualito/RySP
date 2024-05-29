

from pulser.waveforms import ConstantWaveform, BlackmanWaveform, CompositeWaveform
from pulser.pulse import Pulse
import numpy as np
import functools

from rysp.core.circuit import QuantumCircuit, AtomSystem, Measurement, PulsedGate, ParametrizedPulsedGate
from rysp.core.experiment import ExperimentSetup
from rysp.core.simulation import Simulator


from pathos.multiprocessing import ProcessingPool


class GateDictionary:
    """
    Dictionary of gates to be used by the quantum circuit. 
    Inherit from this class and overload methods to define custom ways of implementing gates.
    """
    _complex_parametrized_gates = ['CRX', 'CRZ', 'CRY']

    def __init__(self, experiment: ExperimentSetup, time_factor=2e-2):
        self._experiment = experiment
        # Duration of a rabi flop
        self.time_factor = time_factor
        self.duration_clock = 1/experiment.rabi_freq[('0', '1')] * 1e9
        self.duration_ryd = 1/experiment.rabi_freq[('1', 'r')] * 1e9

    def _add_constant_pulse(self, pgate: PulsedGate, channel, duration, phase, tgt=0, detuning=0, time_factor=None):
        # rabi intensity is at most 1, but area must be kept constant
        # detuning is a multiple of the amplitude
        time_factor = time_factor or self.time_factor
        area = duration * phase
        int_duration = int(area) or int(duration)

        if phase != 0:
            pgate.add_pulse(pulse=Pulse(amplitude=ConstantWaveform(int_duration, 1),
                                        detuning=ConstantWaveform(int_duration, detuning), phase=0),
                            channel=channel, target=tgt,
                            time_scale=time_factor*int_duration*1e-9)

        else:
            pgate.add_pulse(pulse=Pulse(amplitude=ConstantWaveform(int_duration, 0),
                                        detuning=ConstantWaveform(int_duration, detuning), phase=0),
                            channel=channel, target=tgt,
                            time_scale=time_factor*int_duration*1e-9)

        return pgate

    def RX(self, transition, phi, size=1, p: PulsedGate | None = None, target=0, align=False) -> PulsedGate:
        """
        RX _summary_

        _extended_summary_

        Parameters
        ----------
        transition : _type_
            _description_
        phi : _type_
            _description_
        size : int, optional
            _description_, by default 1
        p : PulsedGate | None, optional
            _description_, by default None
        target : int, optional
            _description_, by default 0

        Returns
        -------
        PulsedGate
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if p is None:
            p = PulsedGate()
            p.new_channel('clock', ('0', '1'))
            p.new_channel('ryd', ('1', 'r'))
            if target > 0:
                p.targets = [*range(target)]
        # print(phi)
        if phi == 0:
            return p
        if phi < 0:
            phi = phi % (2*np.pi)
        transition_map = {'01': 'clock', 'clock': 'clock', ('0', '1'): 'clock',
                          '1r': 'ryd', 'ryd': 'ryd', ('1', 'r'): 'ryd'}
        duration_map = {'01': self.duration_clock, 'clock': self.duration_clock, ('0', '1'): self.duration_clock,
                        '1r': self.duration_ryd, 'ryd': self.duration_ryd, ('1', 'r'): self.duration_ryd}
        if transition not in transition_map:
            raise ValueError(
                f"Transition not defined: {transition}")

        for i in range(size):
            p = self._add_constant_pulse(
                pgate=p, channel=transition_map[transition],
                duration=duration_map[transition],
                phase=phi, tgt=target if size == 1 else i)
        if align:
            p.shift_pulses(-p.max_time)
        return p

    def RY(self, transition, phi, size=1, p: PulsedGate | None = None, target=0, align=False) -> PulsedGate:
        raise NotImplementedError()

    def RZ(self, transition, phi, size=1, p: PulsedGate | None = None, target=0, align=False) -> PulsedGate:
        """
        RZ _summary_

        _extended_summary_

        Parameters
        ----------
        transition : _type_
            _description_
        phi : _type_
            _description_
        size : int, optional
            _description_, by default 1
        p : PulsedGate | None, optional
            _description_, by default None
        target : int, optional
            _description_, by default 0

        Returns
        -------
        PulsedGate
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        # phi = np.mod(phi, 2*np.pi)
        if p is None:
            p = PulsedGate()
            p.new_channel('clock', ('0', '1'))
            p.new_channel('ryd', ('1', 'r'))
            if target > 0:
                p.targets = [*range(target)]
        if phi == 0:
            return p
        if phi < 0:
            phi = phi % (2*np.pi)
        transition_map = {'01': 'clock', 'clock': 'clock', ('0', '1'): 'clock',
                          '1r': 'ryd', 'ryd': 'ryd', ('1', 'r'): 'ryd'}

        duration_map = {'01': self.duration_clock, 'clock': self.duration_clock, ('0', '1'): self.duration_clock,
                        '1r': self.duration_ryd, 'ryd': self.duration_ryd, ('1', 'r'): self.duration_ryd}
        if transition not in transition_map:
            raise ValueError(
                f"Transition not defined: {transition}")

        for i in range(size):
            p = self._add_constant_pulse(pgate=p, channel=transition_map[transition], duration=duration_map[transition],
                                         phase=0, detuning=phi, tgt=target if size == 1 else i)
        if align:
            p.shift_pulses(-p.max_time)
        return p

    def X(self, transition, size=1, p: PulsedGate | None = None, target=0, align=False) -> PulsedGate:
        """
        X _summary_

        _extended_summary_

        Parameters
        ----------
        transition : _type_
            _description_
        size : int, optional
            _description_, by default 1
        p : PulsedGate | None, optional
            _description_, by default None
        target : int, optional
            _description_, by default 0

        Returns
        -------
        PulsedGate
            _description_
        """
        return self.RX(transition, np.pi, size, p, target, align)

    def SX(self, transition, size=1, p: PulsedGate | None = None, target=0, align=False) -> PulsedGate:
        """
        SX _summary_

        _extended_summary_

        Parameters
        ----------
        transition : _type_
            _description_
        size : int, optional
            _description_, by default 1
        p : PulsedGate | None, optional
            _description_, by default None
        target : int, optional
            _description_, by default 0

        Returns
        -------
        PulsedGate
            _description_
        """
        return self.RX(transition, np.pi/2, size, p, target, align)

    def Z(self, transition, size=1, p: PulsedGate | None = None, target=0, align=False) -> PulsedGate:
        """
        Z _summary_

        _extended_summary_

        Parameters
        ----------
        transition : _type_
            _description_
        size : int, optional
            _description_, by default 1
        p : PulsedGate | None, optional
            _description_, by default None
        target : int, optional
            _description_, by default 0

        Returns
        -------
        PulsedGate
            _description_
        """
        return self.RZ(transition, np.pi, size, p, target, align)

    def Y(self, transition, size=1, p: PulsedGate | None = None, target=0, align=False) -> PulsedGate:
        raise NotImplementedError()

    def H(self, transition, size=1, p: PulsedGate | None = None, target=0) -> PulsedGate:
        """
        H _summary_

        _extended_summary_

        Parameters
        ----------
        transition : _type_
            _description_
        size : int, optional
            _description_, by default 1
        p : PulsedGate | None, optional
            _description_, by default None
        target : int, optional
            _description_, by default 0

        Returns
        -------
        PulsedGate
            _description_
        """
        p = self.S(transition, size, p, target, True)
        p = self.SX(transition, size, p, target, True)
        p = self.S(transition, size, p, target, True)
        return p

    def T(self, transition, size=1, p: PulsedGate | None = None, target=0, align=False) -> PulsedGate:
        return self.RZ(transition, np.pi/4, size, p, target, align)

    def S(self, transition, size=1, p: PulsedGate | None = None, target=0, align=False) -> PulsedGate:
        return self.RZ(transition, np.pi/2, size, p, target, align)

    def CZ(self, p: PulsedGate | None = None, target=[0, 1]) -> PulsedGate:
        """
        CZ default implementation of Rydberg CZ gate 

        _extended_summary_

        Parameters
        ----------
        p : PulsedGate | None, optional
            _description_, by default None
        target : list, optional
            _description_, by default [0, 1]

        Returns
        -------
        PulsedGate
            _description_
        """

        p = self.X('01', size=2, p=p, align=True)
        p = self.RX('1r', phi=np.pi, p=p, target=target[0], align=True)
        p = self.RX('1r', phi=2*np.pi, p=p, target=target[1], align=True)
        p = self.RX('1r', phi=np.pi, p=p, target=target[0], align=True)
        p = self.X('01', size=2, p=p, align=True)

        return p

    def CX(self, p: PulsedGate | None = None, target=[0, 1]) -> PulsedGate:
        """
        CX _summary_

        _extended_summary_

        Parameters
        ----------
        p : PulsedGate | None, optional
            _description_, by default None
        target : list, optional
            _description_, by default [0, 1]

        Returns
        -------
        PulsedGate
            _description_
        """
        p = self.H('01', p=p, target=target[1])
        p = self.CZ(p=p, target=target)
        p = self.H('01', p=p, target=target[1])
        return p

    def CRX(self, phi: float | None = None, qc: QuantumCircuit | None = None,  target=[0, 1]) -> QuantumCircuit:
        """
        CRX _summary_

        _extended_summary_

        Parameters
        ----------
        phi : float
            _description_
        qc : QuantumCircuit | None, optional
            _description_, by default None
        target : list, optional
            _description_, by default [0, 1]

        Returns
        -------
        QuantumCircuit
            _description_
        """
        crx_prep = self.X('01', target=target[0])  # control flip

        # remapping
        crx_prep = self.X('1r', target=target[1], p=crx_prep, align=True)
        crx_prep = self.X('01', target=target[1], p=crx_prep, align=True)

        # Controlled blockade
        crx_prep = self.X('1r', target=target[0], p=crx_prep, align=True)

        qc = qc or QuantumCircuit('pulsed')

        qc.add_operation(crx_prep, target)

        # parametrized operation
        crx_g = ParametrizedPulsedGate(
            lambda th: self.RX('1r', phi=th, target=target[1]), ['th'])
        qc.add_operation(crx_g, target, args={'th': phi})

        # reverting the remapping
        crx_post = self.X('01', target=target[1], align=True)
        crx_post = self.X('1r', target=target[1], p=crx_post, align=True)

        # deactivating the control
        crx_post = self.X('1r', target=target[0], p=crx_post, align=True)

        # Negative control
        crx_post = self.X('01', target=target[0], p=crx_post, align=True)

        qc.add_operation(crx_post, target)

        return qc

    def CRZ(self, qc: QuantumCircuit | None = None, target=[0, 1]) -> QuantumCircuit:
        raise NotImplementedError()

    def getGate(self, gt: str) -> tuple[PulsedGate, str]:
        if gt == 'X':
            return self.X('01'), 'box {$X$} {target};'
        elif gt == 'S':
            return self.S('01'), 'box {$S$} {target};'
        elif gt == 'Z':
            return self.Z('01'), 'box {$Z$} {target};'
        elif gt == 'H':
            return self.H('01'), 'box {$H$} {target};'
        elif gt == 'T':
            return self.T('01'), 'box {$T$} {target};'
        elif gt == 'Xr':
            return self.X('1r'), 'box {$X_r$} {target};'
        elif gt == 'CX':
            return self.CX(), 'cnot {target} | {control};'
        elif gt == 'CZ':
            return self.CZ(), 'zz {target} | {control};'
        elif gt == 'CRX':
            pass
        elif gt == 'CRZ':
            pass
        else:
            raise ValueError(f"GateDictionary: Gate {gt} not known")

    def load_gate(self, qc: QuantumCircuit, gt: str):
        gateop, yqant_latex = self.getGate(gt)
        gateop.custom_repr = yqant_latex
        gateop.override_custom_when_show_pulse = False
        qc.add_to_dictionary(gateop, gt, 'pulsed', None)

    def load_cache_gate(self,
                        qc: QuantumCircuit,
                        gt: str,
                        lattice: AtomSystem,
                        interval: float = 1e-9,
                        sim_config={'loss': {}, 'motional energy': 0, 'dephase': []}):
        gateop, yqant_latex = self.getGate(gt)
        gateop.custom_repr = yqant_latex
        gateop.override_custom_when_show_pulse = False
        gateop._compile_pulses(interval)
        Simulator.cache_gate_propagator(
            gateop, lattice, self._experiment,
            gateop.time_list, sim_config=sim_config)
        qc.add_to_dictionary(gateop, gt, 'pulsed', None)

    def load_cached_qc(self,
                       qc: QuantumCircuit,
                       gates: list[str],
                       lattice: AtomSystem | None,
                       lattice_unit: int | None = None,
                       interval: float = 1e-9,
                       sim_config={'loss':  {('0', '1'): {'0': 0, '1': 0},
                                             ('1', 'r'): {'1': 0, 'r': 0}},
                                   'motional energy': 0, 'dephase': []}):  # TODO: Simulator.load_cached_qc docstring
        """
        load_cached_qc _summary_

        _extended_summary_

        Parameters
        ----------
        qc : QuantumCircuit
            _description_
        gates : list[str]
            _description_
        experiment : ExperimentSetup
            _description_
        lattice : AtomSystem | None
            _description_
        lattice_unit : int | None, optional
            _description_, by default None
        interval : float, optional
            _description_, by default 1e-9
        sim_config : dict, optional
            _description_, by default {'loss':  {('0', '1'): {'0': 0, '1': 0}, ('1', 'r'): {'1': 0, 'r': 0}}, 'motional energy': 0, 'dephase': []}

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if lattice is None:
            # Auto generate lattice based on lattice_unit
            if lattice_unit is None:
                raise ValueError(
                    "Cannot have lattice unit None while lattice is not determined")
            lattice = AtomSystem()
            default_positions = lattice_unit * \
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [-1, 0, 0], [0, -1, 0]])
            for i, pos in enumerate(default_positions):
                atom = self._experiment.get_atom()
                lattice.add_atom(str(i), atom, pos)

        def gen_cached_gate(gateop):
            if isinstance(gateop, PulsedGate):
                gateop._compile_pulses(interval)
                Simulator.cache_gate_propagator(
                    gateop,
                    lattice, self._experiment,
                    gateop.time_list,
                    sim_config=sim_config)
                return gateop
            elif isinstance(gateop, QuantumCircuit):
                for component in gateop.operation_sequence:
                    if type(component.operation) == PulsedGate:
                        if component.operation.compare_cached(lattice, lattice.atom_labels, 1e-2, sim_config):
                            continue  # if the cached settings match, then continue for the next operations
                        component.operation._compile_pulses(interval)

                        Simulator.cache_gate_propagator(
                            component.operation,
                            lattice, self._experiment,
                            component.operation.time_list,
                            sim_config=sim_config
                        )

        # expand gates with partial gates

        # create quantum circuit as operation

        precache = {gt: self.getGate(gt) for gt in gates
                    if gt not in GateDictionary._complex_parametrized_gates}

        gates_op = ProcessingPool().map(
            lambda gt: gen_cached_gate(precache[gt][0]), gates)
        for gt, (gate) in zip(gates, gates_op):
            gate.custom_repr = precache[gt][1]
            qc.add_to_dictionary(gate, gt, 'pulsed', None)
