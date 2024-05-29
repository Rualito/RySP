import pathlib
import numpy as np
from rysp.core.experiment import ExperimentSetup

from rysp.core.circuit.pulses import Pulse_from_file, PulsedGate
from rysp.core.circuit.gatedictionary import GateDictionary

import os

folder = pathlib.Path(__file__).parent

rabi = np.loadtxt(f'{folder}/rabi.dat')
det = np.loadtxt(f'{folder}/det.dat')

class CustomGateDictionary(GateDictionary):
    def __init__(self, experiment: ExperimentSetup, time_factor=2e-2):
        super().__init__(experiment, time_factor)
        self.pulseMadhav = Pulse_from_file(rabi, det, 10*self.duration_ryd)

    def H(self, transition, size=1, p: PulsedGate | None = None, target=0, align=True) -> PulsedGate:
        if p is None:
            p = PulsedGate()
            p.new_channel('clock', ('0', '1'))
            p.new_channel('ryd', ('1', 'r'))
        transition_map = {'01': 'clock', 'clock': 'clock', ('0', '1'): 'clock',
                          '1r': 'ryd', 'ryd': 'ryd', ('1', 'r'): 'ryd'}
        duration_map = {'01': self.duration_clock, 'clock': self.duration_clock, ('0', '1'): self.duration_clock,
                        '1r': self.duration_ryd, 'ryd': self.duration_ryd, ('1', 'r'): self.duration_ryd}

        for i in range(size):
            p = self._add_constant_pulse(p,
                                         transition_map[transition],
                                         duration_map[transition]/np.sqrt(2), phase=np.pi, detuning=1,
                                         tgt=target if size == 1 else i)
        if align:
            p.shift_pulses(-p.max_time)
        return p

    def RZ(self, transition, phi, size=1, p: PulsedGate | None = None, target=0, align=False) -> PulsedGate:
        p = self.H(transition, size, p, target, True)
        p = self.RX(transition, phi, size, p, target, True)
        p = self.H(transition, size, p, target, True)
        return p

    def CZ(self, p: PulsedGate | None = None, target=[0, 1]) -> PulsedGate:
        """
        CZ implementation of Rydberg CZ gate according to https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.033052

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
        if p is None:
            p = PulsedGate()
            p.new_channel('clock', ('0', '1'))
            p.new_channel('ryd', ('1', 'r'))

        p = self.X('01', 2, p, align=True)

        for tg in target:
            p.add_pulse(self.pulseMadhav, 'ryd', tg, t0=0, time_scale=1e-9)

        p.shift_pulses(-p.max_time)
        # angle to correct after madhav pulse
        phi_c = -0.11349061892726953
        p = self.RZ('01', phi_c, 2, p)
        p.shift_pulses(-p.max_time)

        p = self.X('01', 2, p)
        p.shift_pulses(-p.max_time)

        return p
