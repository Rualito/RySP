
import arc
import numpy as np
from .atom import AtomInTrap
import json
from ..physics import units
from ..physics import rabicalc


class ExperimentSetup:
    '''
    Defines hardware parameters depending on the configuration file
    '''

    def __init__(self, json_str: str, simulation_method='qutip', _atom_type=AtomInTrap) -> None:
        self._data = json.loads(json_str)
        self._atomtype = _atom_type

        self.name = self._data['name']
        self.simulation_level = self._data['simulation_level']

        self.base_gates = None
        if self.simulation_level == 'digital':
            self.base_gates = self._data['base gates']
            self.coupling_map = self._data['coupling map']
        elif self.simulation_level == 'pulsed':
            self.atom = self._data['atom']
            self.magnetic_field = self._data['magnetic field']
            self.env_temp = self._data['environment temp']

            self.qubit_lasers = {}

            for laser in self._data['lasers']:
                if laser['type'] == 'trap':
                    self.trap_laser_info = laser
                elif laser['type'] == 'qubit':
                    self.qubit_lasers[tuple(laser['transition'])] = laser
        self.available_transitions = [*self.qubit_lasers.keys()]

        def transition_to_freq(s0: str, s1: str, det: float):
            '''
            Convert transition to laser frequency
            s0, s1: state labels
            det: standard laser detuning
            '''
            # State as string to atomic parameters
            satom0 = self.atom['states'][s0][:4]
            satom1 = self.atom['states'][s1][:4]

            ediff = rabicalc.get_energy_diff([*satom0, *satom1])
            wavelength = units.freq_to_wavelength(ediff + det)
            freq = units.wavelength_to_freq(wavelength)

            return freq
        self.transition_offresonance_scattering = {}
        self.rabi_freq = {}

        for transition, laser in self.qubit_lasers.items():
            satom0 = list(self.atom['states'][transition[0]]
                          [:3])+[self.atom['states'][transition[0]][4]]
            satom1 = list(self.atom['states'][transition[1]]
                          [:3])+[self.atom['states'][transition[1]][4]]

            intensity = 2*laser['power']/(np.pi*laser['waist']**2)

            rate0, rate1, s1, s2 = rabicalc.total_off_resonance_scattering_rate_calc(
                transition=[*satom0, *satom1],
                frequency=transition_to_freq(
                    transition[0], transition[1], laser['detuning']),
                I=intensity
            )

            self.transition_offresonance_scattering[tuple(transition)] = {
                transition[0]: rate0,
                transition[1]: rate1
            }

            if laser['name'] == 'clock':
                self.rabi_freq[tuple(transition)] = rabicalc.rabi_frequency_calc_clock(
                    intensity, self.magnetic_field)
            else:
                self.rabi_freq[tuple(transition)] = rabicalc.rabi_frequency_calc_trans(
                    [*satom0, *satom1], intensity)

        self.dephase_rates = {}
        for tr in self.available_transitions:
            self.dephase_rates[tr] = np.sqrt(
                self.qubit_lasers[tr]['line width'])  # doppler broadening

        self.simulation_method = simulation_method

    @classmethod
    def fromFile(cls, filename, simulation_method='qutip', _atom_type=AtomInTrap):

        with open(filename, 'r') as file:
            data = file.read().replace('\n', '')
        return ExperimentSetup(data, simulation_method, _atom_type)

    def validate_gate(self, gate, targets):
        raise NotImplementedError(
            "Cannot validate gate, hardware method not yet implemented.")

    def get_trap_params(self):
        w0 = self.trap_laser_info['waist']
        p0 = self.trap_laser_info['power']
        wavelength = self.trap_laser_info['wavelength']
        nref = self.trap_laser_info['refractive index']
        zR = np.pi*w0**2*nref/(wavelength)
        I0 = 2*p0/(np.pi*w0**2)

        def waistf(z): return w0*np.sqrt(1+(z/zR)**2)

        return {'laser wavelength': wavelength,
                'laser intensity': lambda z, r: I0*(w0/waistf(z))**2*np.exp(-2*(r/waistf(z))**2),
                'laser w0': w0, 'zR': zR}

    def get_atom(self):
        if self.atom['species'] == 'Sr88':
            atom = arc.Strontium88()
        else:
            raise ValueError(
                f"Hardware setup: Atom unknown: {self.atom['species']}")

        return self._atomtype({'name': atom.elementName, 'mass': atom.mass},
                              trap_site=self.get_trap_params(),
                              state="ground_state",
                              spin_basis=[*self.atom['states'].values()],
                              spin_basis_labels=[*self.atom['states'].keys()],
                              motional_basis=[*self.atom['motional'].values()],
                              motional_basis_labels=[
                                  *self.atom['motional'].keys()],
                              Temp=self.atom['temperature'],
                              interact_states=['r']
                              )
