
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

    def __init__(self, json_str: str, simulation_method='qutip') -> None:
        self._data = json.loads(json_str)
        # self._atomtype = _atom_type

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
            satom0 = self.atom['states'][s0]
            satom1 = self.atom['states'][s1]

            ediff = rabicalc.get_energy_diff(tuple([*satom0, *satom1]))

            return ediff + det
        self.transition_offresonance_scattering = {}
        self.rabi_freq = {}

        for transition, laser in self.qubit_lasers.items():
            satom0 = self.atom['states'][transition[0]]
            satom1 = self.atom['states'][transition[1]]

            intensity = 2*laser['power']/(np.pi*laser['waist']**2)

            rate0, rate1, s0, s1 = rabicalc.total_off_resonance_scattering_rate_calc(
                transition=[*satom0, *satom1],
                frequency=transition_to_freq(
                    transition[0], transition[1], laser['detuning']*0),
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
                    tuple([*satom0, *satom1]), intensity)

        self.dephase_rates = {}
        for tr in self.available_transitions:
            s0, s1 = tr[0], tr[1]
            satom0 = self.atom['states'][s0]  # atom state 0
            satom1 = self.atom['states'][s1]  # atom state 1
            decay_rate = rabicalc.decay_rate_state(satom1)
            self.dephase_rates[tr] = np.sqrt(decay_rate)

            # np.sqrt(
            #   self.qubit_lasers[tr]['line width'])

        self.loss_rates = {st: 0.0 for st in self.atom['states']}

        self.default_C6 = 0.0

        self.simulation_method = simulation_method

    @classmethod
    def fromFile(cls, filename, simulation_method='qutip'):

        with open(filename, 'r') as file:
            data = file.read().replace('\n', '')
        return cls(data, simulation_method)

    def validate_gate(self, gate, targets):
        raise NotImplementedError(
            "Cannot validate gate, hardware method not yet implemented.")

    def show_physical_parameters(self):
        """
        show_physical_parameters Shows the estimated physical parameters for simulation and gives indications on how to customize them

        """
        for transition in self.qubit_lasers:
            print(f"\nTransition: {transition} ")
            print(f"\tRabi Frequency: {self.rabi_freq[(*transition,)]:g} Hz")
            print(
                "\t\tdefines the maximum Rabi frequency associated with the max laser intensity. \n\t\tIt gives the units for the transition and detuning terms")
            print(
                f"\t\tchange it with the attribute exp.rabi_freq[{(*transition,)}]")
            print(
                f"\tDecay rate Γ: {self.dephase_rates[(*transition,)]**2:g} Hz")
            print(
                f"\tDephase rate γ (√Γ): {self.dephase_rates[(*transition,)]:g} √Hz")
            print("\t\tIntroduces collapse operators on the Lindbladian evolution.")
            print(
                f"\t\tchange it with the attribute exp.dephase_rates[{(*transition,)}]")
            print("\tOffresonance scattering rates:")
            for st in transition:
                print(
                    f"\t\tfrom state {st}: {self.transition_offresonance_scattering[(*transition,)][st]:g} Hz")
            print(
                "\t\tAdds loss terms to the Lindbladian that are active during a pulse,\n\t\t and are integrated a posteriori.")
            print(
                f"\t\tchange it with the attribute exp.transition_offresonance_scattering[{(*transition,)}]['{transition[0]}' or '{transition[1]}']")

        print("\nThe C6 terms are defined on a per atom basis.\n If you want to customize the |rrXrr| interaction term (C6_rr) you may set exp.default_C6")

        print("\nState loss rate")
        for st in self.atom['states']:
            print(f"\tState '{st}': {self.loss_rates[st]:g} Hz")
        print("\t\tThese are 0 by default. They define rates of transition to states outside the computational basis, and can be computed a posteriori.")
        print(f"\t\tChange it with exp.loss_rates[ 'state' ]")

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

        atom = AtomInTrap({'name': atom.elementName, 'mass': atom.mass},
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

        if self.default_C6 == 0.0:
            atom.c6_coef_dict[('r', 'r')] = self.default_C6
        return atom
