import numpy as np
from typing import (
    Any,
    Union,
    Callable
)
from abc import ABC, abstractmethod
import inspect
import functools


class Operation(ABC):
    @abstractmethod
    def __init__(self, simulation_level='') -> None:
        super().__init__()
        self._simulation_level = simulation_level
        self.num_args = 0
        self.iID = 'Op'  # inner ID
        self.custom_repr = None

    def __str__(self):
        return 'Op'

    def set_representation(self, latex_str):
        """
        set_representation Changes the latex representation to a custom one

        _extended_summary_

        Parameters
        ----------
        latex_str : _type_
            _description_
        """
        # changes the latex representation of the operation
        if type(latex_str) == str:
            self.custom_repr = latex_str

    def _to_latex_yquant(self, **kwargs):
        if self.custom_repr == None:
            # default representation
            return f"box {{ ${str(self)}$ }} ({{targets}});"
        else:
            return str(self.custom_repr)

    @abstractmethod
    def evaluate(self, *args: Any, **kwds: Any) -> Any:
        return 0


class AtomTransport(Operation):
    """
    AtomTransport The inputs are of the kind 'transport to this lattice'. In essence, you define the new lattice configuration for the atoms you want to move, independent of the old positions.

    _extended_summary_

    Parameters
    ----------
    Operation : _type_
        _description_
    """

    def __init__(self, new_positions={}, transport_method={'method': 'direct', 'speed': 0.5}, simulation_level='pulsed') -> None:
        '''
        old_positions: key=atom label, value=atom position
        '''
        # self.old_positions = old_positions
        self.new_positions = new_positions
        self.method = transport_method
        super().__init__(simulation_level)
        self.iID = 'AT'

    def evaluate(self, *args: Any, **kwds: Any) -> Any:
        return 0

    def get_mapping(self):
        return self.new_positions

    # def get_travel_distances(self):
    #     distances = []
    #     for lbl, oldpos in self.old_positions.items():
    #         newpos = self.new_positions[lbl]
    #         distances += [np.linalg.norm(oldpos-newpos)]
    #     return distances

    def __str__(self):
        return f"AT"

    def _to_latex_yquant(self, **kwargs):
        if self.custom_repr is None:
            return "box {AT} {targets};"
        return self.custom_repr

    # TODO: make get_travel_times use AtomSystem instead
    # Remove old_positions requirement
    def get_travel_times(self, old_positions) -> dict[str, tuple[float, float]]:
        '''
        Returns the transport times for each of the atoms, in the form of a dictionary: key=atom label, value=(transport time, loss probability)
        '''
        if self.method['method'] == 'direct':
            times = {}
            for lbl, oldpos in old_positions.items():
                newpos = self.new_positions[lbl]
                dist = np.linalg.norm(oldpos-newpos)
                if self.method['method'] == 'direct':
                    # time, loss probability
                    times[lbl] = (dist/self.method['speed'], 0)
            return times
        else:
            raise NotImplementedError(
                f"AtomTransport: transport method '{self.method['method']}' not implemented.")


class CustomGate(Operation):
    '''CustomGate _summary_

    :param Operation: _description_
    :type Operation: _type_
    '''

    def __init__(self, name, qobj, arg_labels=None):
        self.operator = qobj
        if qobj is not None:
            self.size = len(qobj.dims[0])
        # super().__init__(name, arg_label=arg_labels, targets=[*range(self.size)])

        self._simulation_level = 'digital'
        self.num_args = len(arg_labels) if arg_labels is not None else 0
        self.iID = 'G'

    def evaluate(self, *args, **kwargs):
        '''
        Returns the gate implementation for the given parameters
        '''
        # this object is not parametrized
        return self.operator

    def validate_gate(self, exp_setup, targets):
        '''validate_gate Validates the gate against a given experimental setup.

        :param exp_setup: _description_
        :type exp_setup: ExperimentalSetup
        :param targets: _description_
        :type targets: list[str]
        :return: _description_
        :rtype: bool
        '''  # TODO: validate gate on Custom Gate
        return exp_setup.validate_gate(self, targets)


class Measurement(Operation):
    def __init__(self,
                 method: tuple[str, dict[str, Any]] = ('simple', {'wait time': 1e-3,
                                                                  'target state': '0',
                                                                  'map result': {1: 0, 0: 1}}),
                 flatten=True) -> None:
        '''
        Measures the qubit, indicating the target state

        method: tuple, where first corresponds to the method name, the second are the parameters of the method
        '''
        super().__init__()
        self._simulation_level = 'any'
        self.measured_bit = 0
        self.variable = None
        self.iID = 'M'
        self.flatten = flatten  # whether to flatten the measurement output

        self.method_name = method[0]
        self.method_parameters = method[1]
        self.map_result = method[1]['map result']
        # Since the target state is '0', the measurement result will be 1 when measures the state '0'. Thus this mapping has to be considered

    def evaluate(self, *args: Any, **kwds: Any) -> Any:
        return 0

    def __str__(self):
        return f"Measurement(target={self.method_parameters['target state']})"

    def _to_latex_yquant(self, show_vars: bool = False, variable: str = '?', **kwargs):
        if self.custom_repr is None:
            if show_vars:
                return '[direct control] measure {target};'+f"box {{{variable}}} var | {{target}};"
            return 'measure {target};'
        return self.custom_repr


class Calculation(Operation):
    '''
    Operation object representing a classical computation on the measured bits.
    Output result can adjust the circuit parameters in following components 
    '''

    def __init__(self, func, input_vars: list, wait_time=0, cache=False) -> None:
        '''
        func: Calculation function should output a dictionary, indicating which circuit parameters to update according to the result.
        ex: {'a': 0.5, 'b':1}
        input_vars: renaming of the arguments of func, in accordance to the variables in QuantumCircuit 
        '''
        super().__init__()

        self._func = func
        self._simulation_level = 'any'
        self.wait_time = wait_time
        self.input_vars = input_vars
        self.func_args = inspect.getfullargspec(func).args
        self.defaultsf = inspect.getfullargspec(func).defaults
        self.defaults = self.defaultsf if self.defaultsf is not None else {}
        if len(self.input_vars) < len(self.func_args)-len(self.defaults):
            raise ValueError("Not enough input variables have been specified")

        self.variable_mapping = {}
        for fa, iv in zip(self.func_args, self.input_vars):
            self.variable_mapping[iv] = fa
        self.iID = 'Cal'
        if cache:
            self._func = functools.lru_cache()(func)

    def __str__(self):
        return f"Calc"

    def _to_latex_yquant(self, show_vars, **kwargs):
        if self.custom_repr is None:
            if show_vars:
                wait_str = ''
                if self.wait_time > 0:
                    wait_str = f'["{self.wait_time:.2e} s" below]'
                return wait_str + f'box {{f({", ".join(self.input_vars)})}} var;'
            else:
                return ''  # don't show anything
        return self.custom_repr

    def evaluate(self, **kwargs) -> tuple[float, tuple[Any, ...]]:
        '''
        Calculation function should output a dictionary, indicating which circuit parameters to update according to the result.
        '''
        dic2 = {}
        for key, val in kwargs.items():  # remapping the arguments to the original names
            dic2[self.variable_mapping[key]] = val
        self.result = self._func(**dic2)
        return self.wait_time, self.result


class StateOperation(Calculation):
    """
    StateOperation Does some operation/calculation on the state of the system, including circuit variables

    Parameters
    ----------
    Calculation : _type_
        _description_
    """

    def __init__(self, func, input_vars: list) -> None:
        super().__init__(func, input_vars, 0, False)

    def _to_latex_yquant(self, show_vars, **kwargs):
        return super()._to_latex_yquant(show_vars, **kwargs)

    def evaluate(self, **kwargs) -> tuple[float, tuple[Any, ...]]:
        return super().evaluate(**kwargs)


class ParametrizedOperation(Calculation):
    def __init__(self, opfunc, input_vars: list[str], cache=False):
        super().__init__(opfunc, input_vars, 0, cache)
        self.iID = 'pO'

    def _to_latex_yquant(self, show_vars: bool = False):
        if show_vars:
            return r"box {$Op_{\theta}$} {targets} | var;"
        return r"box {$Op_{\theta}$} {targets};"

    def evaluate(self, **kwargs) -> Operation:
        dic2 = {}
        for key, val in kwargs.items():  # remapping the arguments to the original names
            dic2[self.variable_mapping[key]] = val
        return self._func(**dic2)


class ParametrizedGate(ParametrizedOperation):
    '''
    Defines a parametrized gate (digital). 

    qobj_func - some callback function that returns a Qobj that describes the digital operation between qubits 
    '''

    def __init__(self,
                 gate_func: Callable[[Any], CustomGate],
                 operator_dim: int,
                 arg_labels: list[str] = []):
        '''
        gate_func: callable that retuns CustomGate 
        operator_dim: dimension of the CustomGate (how many qubits it acts on)
        arg_labels: list of strings indicating the variable names associated with the callable function
            These strings are defined as circuit variables that map onto the gate_func, in the order they are defined
        '''
        super().__init__(opfunc=gate_func, input_vars=arg_labels)

        self.iID = 'prG'

    def _to_latex_yquant(self, show_vars: bool, **kwargs):
        if self.custom_repr is None:
            if show_vars:
                return '["' + ', '.join(self.input_vars) + '" below]box {$P_{\\theta}$} {targets} | var;'
            return r"box {$P_{\theta}$} {targets};"
        return self.custom_repr

    def evaluate(self, **kwargs) -> CustomGate:
        '''
        Evaluates the operator function and returns the CustomGate object
        '''
        dic2 = {}
        for key, val in kwargs:  # remapping the arguments to the original names
            dic2[self.variable_mapping[key]] = val
        # self.update_args(**dic2)
        return self._func(**dic2)


# class OperationSequence(Calculation):

#     def __init__(self, operations, input_vars: list) -> None:
#         super().__init__(lambda x: 0, input_vars, 0)

#         self.operations = operations
#         self.iID = 'OSeq'

#     # def __init__(self, operations: list[Operation], simulation_level='pulsed') -> None:
#     #     super().__init__(simulation_level)
#     #     self.operations = operations
#     #     self.iID = 'OSeq'
