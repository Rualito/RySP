import unittest
import unittest.mock

import rysp


from rysp.core.experiment import AtomInTrap, ExperimentSetup

json_str = '''{
    "name": "template_pulsed",
    "simulation_level": "pulsed",
    "atom" : {
        "species": "Sr88",
        "states": {
            "0": [5,0,0,0,0],
            "1": [5,1,0,0,1],
            "r": [61,0,1,0,1]
        },
        "motional": {
            "000" : [0,0,0]
        },
        "temperature": 0
    },
    "parallel transport": true,
    "magnetic field": 0.01,
    "environment temp":0,
    "lasers" : [
        {
            "name": "trap",
            "type": "trap",
            "efficiency": 0.1,
            "decrease factor": 0.02,
            "power": 8,
            "waist": 1e-6,
            "wavelength": 813.035e-9,
            "refractive index": 1,
            "polarization": 0,
            "z": 0
        },
        {
            "name": "clock",
            "type":"qubit",
            "transition": ["0", "1"],
            "power": 0.2,
            "decrease factor": 0.2654,
            "waist": 1e-6,
            "polarization": 0,
            "Lamb Dicke": false,
            "direction": [0,0,1],
            "line width": 1e6,
            "concurrence":1,
            "detuning": 1e6
        },
        {
            "name": "rydberg",
            "type":"qubit",
            "transition": ["1", "r"],
            "power": 4e-5,
            "decrease factor": 0.2654,
            "waist": 1e-6,
            "polarization": 0,
            "Lamb Dicke": true,
            "direction": [0,0,1],
            "line width": 1e6,
            "concurrence":1,
            "detuning": 1e7
        }
    ]
}'''


class TestExpSetup(unittest.TestCase):
    def test_atom(self):
        exp = ExperimentSetup(json_str)
        self.assertIsInstance(exp.get_atom(), AtomInTrap)
