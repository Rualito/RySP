# %% Initial config

import numpy as np

import rysp
from rysp.core.circuit import QuantumCircuit, AtomTransport, PulsedGate, ParametrizedGate, Calculation, Measurement
from rysp.core.circuit.gatedictionary import GateDictionary
from rysp.core.experiment import ExperimentSetup, AtomSystem
from rysp.core.simulation import Simulator

# %% main

if __name__ == '__main__':

    exp = ExperimentSetup.fromFile('template_hardware_pulsed.json')
    gd = GateDictionary(exp)

    qc = QuantumCircuit()

    lattice_unit = 3e-6

    gd.load_cached_qc(qc, ['X', 'Xr', 'H', 'Z', 'CX',
                      'CZ'], exp, None, lattice_unit)

    qc.add_to_dictionary(Measurement(), 'M', 'pulsed', 'm')

    positions = {'A': lattice_unit * np.array([0, 0, 0]),
                 'B': lattice_unit * np.array([1, 0, 0]),
                 'C': lattice_unit * np.array([0.5, 1/np.sqrt(2), 0])}

    atom = exp.get_atom()
    atomsys = AtomSystem()
    for label, pos in positions.items():
        atomsys.add_atom(label, atom, pos)

    # %% circuit
    qc.reset()
    qc.add_operation('H', ['A'])


# %%
