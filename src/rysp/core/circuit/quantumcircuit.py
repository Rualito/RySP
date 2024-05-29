from typing import Union, Any

from rysp.core.circuit.operations import ParametrizedOperation, Calculation, Operation, Measurement, CustomGate, ParametrizedGate, AtomTransport
from rysp.core.circuit.pulses import PulsedGate
import re
import warnings


class CircuitComponent(Operation):
    """
    CircuitComponent _summary_

    _extended_summary_

    Parameters
    ----------
    Operation : _type_
        _description_
    """

    def __init__(self, operation, simulation_level, targets, controls=None, args_dict={}, name="default") -> None:
        '''
        operation: Operation object to execute
        simulation_level: digital or pulsed
        targets:
        controls:
        args_dict: 
            keys: specify which variables are needed at runtime
            values: can be set to None or preloaded with defaults
        '''

        self.operation = operation
        self.targets = targets
        self.controls = controls
        self.init_args = args_dict
        self.name = name
        if (simulation_level == 'pulsed') and (controls is not None):
            raise ValueError(
                "CircuitComponent: Can only define control atoms in digital mode")
        self._simulation_level = simulation_level
        self.needs_variables = [*args_dict.keys()]  # circuit variables

        if isinstance(operation, ParametrizedOperation):
            opfunc = operation._func
            self.func_args = operation.func_args
            defaults = operation.defaults
            if len(self.needs_variables) < len(self.func_args)-len(defaults):
                raise ValueError(
                    "Not enough input variables have been specified")

            self.variable_mapping = {}
            for fa, iv in zip(self.func_args, self.needs_variables):
                self.variable_mapping[iv] = fa
        if type(operation) == Calculation:
            self.func_args = operation.func_args
            self.needs_variables = [*args_dict['in']]
            self.out_vars = [*args_dict['out']]
            self.variable_mapping = {}
            for fa, iv in zip(self.func_args, self.needs_variables):
                self.variable_mapping[iv] = fa

        if isinstance(operation, Measurement):
            if 'measurement variable' not in args_dict:
                raise ValueError(
                    "CircuitComponent: measurement operation requires output variable")
            self.variable = args_dict['measurement variable']

    def __str__(self):
        """
        __str__ _summary_

        _extended_summary_

        Returns
        -------
        _type_
            _description_
        """
        return f"CircuitComponent( {self.name}, {str(self.operation)}, {self.targets}, {self.controls})"

    def _to_latex_yquant(self, target_registers, replace_plots=[], **kwargs):
        gate_latex = self.operation._to_latex_yquant(
            target_registers=target_registers, **kwargs)
        args = re.findall(r'(?<=(?<!\{)\{)[^{}]*(?=\}(?!\}))', gate_latex)

        format_dict = {a: f'{{{a}}}' for a in args}
        target_is_control_index = 0
        if 'control' in args:
            if self.controls is not None:
                control_str = f'{target_registers[self.controls[0]]}'
            else:
                target_is_control_index = 1
                control_str = f'{target_registers[self.targets[0]]}'
            # print(f"{gate_latex=}, {control_str=}")
            format_dict['control'] = control_str
            # gate_latex = gate_latex.format(control=control_str)
        elif 'controls' in args:
            if self.controls is not None:
                control_str = f"{', '.join([target_registers[tg] for tg in self.controls])}"
            else:  # assume all but the last is control
                control_str = f"{', '.join([target_registers[tg] for tg in self.targets[:-1]])}"
                target_is_control_index = len(self.targets)-1
            format_dict['controls'] = control_str
            # gate_latex = gate_latex.format(controls=control_str)
        # print(f"{target_registers=}")
        if 'target' in args or 'targets' in args:
            if self.targets is None:
                raise ValueError(
                    f"Targets is None, while it is required... operation: {str(self.operation)}")

        if 'target' in args:
            target_str = f'{target_registers[self.targets[target_is_control_index]]}'
            # gate_latex = gate_latex.format(target=target_str)
            format_dict['target'] = target_str
        elif 'targets' in args:
            target_str = f"{', '.join([target_registers[tg] for tg in self.targets[target_is_control_index:]])}"
            format_dict['targets'] = target_str

        if len(replace_plots) > 0:
            for i, pltfig in enumerate(replace_plots):
                if f'plot{i}' in args:
                    format_dict[f'plot{i}'] = f'{{\\includegraphics[height=0.05\\textwidth]{{{pltfig}}}}}'
                    format_dict[f'target{i}'] = target_registers[self.targets[i]]
        # print(format_dict)
        return gate_latex.format(**format_dict)

    def missing_variables(self):
        varss = []
        for var in self.needs_variables:
            if (var not in self.init_args) or (self.init_args[var] is None):
                varss.append(var)
        return varss

    def get_required_args(self):
        return self.needs_variables

    def evaluate(self, **kwargs):
        """
        evaluate _summary_

        Returns
        -------
        _type_
            _description_
        """
        # if isinstance(self.operation, ParametrizedGate):
        #     # self.operation.update_args(**kwargs)
        #     if not self.operation.validate_inputs():
        #         args = self.operation.func_args
        #         raise ValueError(f"CircuitComponent: Gate does not have all parameters defined. func_args: {args}")

        if isinstance(self.operation, ParametrizedOperation) or type(self.operation) == Calculation:
            if len(self.variable_mapping) > 0:
                dic1 = {}
                for k, v in kwargs.items():
                    dic1[self.variable_mapping[k]] = v
            else:
                dic1 = kwargs

            operation = self.operation.evaluate(**dic1)
            # Returns operation after parameter evaluation, if necessary
            if type(self.operation) == Calculation:
                wait_time, result = operation  # type: ignore
                return wait_time, {lbl: res for lbl, res in zip(self.out_vars, result)}
        else:
            operation = self.operation
        return operation


class QuantumCircuit(Operation):  # TODO: QuantumCircuit docstring
    """
     _summary_

    _extended_summary_

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
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    NotImplementedError
        _description_
    NotImplementedError
        _description_
    """

    def __init__(self, simulation_level='pulsed'):
        '''
        simulation_level: digital or pulsed. 
            'digital': 2 states - 0 or 1; 
            'pulsed': Hamiltonian simulation with realistic laser pulses.  
        '''
        super().__init__(simulation_level)
        # super().__init__(N, input_states, output_states, reverse_states, user_gates, dims, num_cbits)
        self.known_gates = {}
        self.gate_objects = {}
        self.simulation_level = simulation_level
        self.reset()
        self.iID = 'QCirc'
        self.targets = []

    # def validate_circuit(self, exp_setup:HardwareSetup):
    #     # TODO: QuantumCircuit.validate_circuit
    #     pass

    def evaluate(self, *args: Any, **kwds: Any) -> Any:
        return super().evaluate(*args, **kwds)

    def add_to_dictionary(self, operation, name, level='pulsed', arg_names=None):
        '''
        arg_names: standard argument names for parametrized operation
        '''
        if level not in ('digital', 'pulsed', 'any'):
            raise ValueError(
                f' Simulation level - {level} - has to be either digital, pulsed or any')

        if operation._simulation_level not in {level, 'any'}:
            raise ValueError(
                "Operation type has to be compatible with simulation level. Digital gates are not available for pulse level simulation")

        if name not in self.known_gates:
            self.known_gates[name] = {
                'args': arg_names, 'level': [level]}
        elif level not in self.known_gates[name]['level']:
            self.known_gates[name]['level'].append(level)

        self.gate_objects[(name, level)] = operation
        #

    def add_operation(self, operation: Union[str, Operation], targets: list[str | int] | None = None, controls=None, args: str | dict | None = None, index: int | None = None):
        '''
        Add operation to circuit sequence. 

        ----------
        Parameters:
            args_dict
                keys: strings defining the tags for the arguments of the operation object. These should be unique for the whole circuit.
                values: the values themselves for the arguments. may be set to None in case of a ParametrizedGate.
            If operation is a quantum circuit, then args is a mapping from the old variables to the new ones
        '''
        arguments = {}
        # Preload arguments at circuit creation level
        if type(args) == dict:
            arguments = args.copy()

        if targets is not None:  # keep track of the targets of the circuit
            for tgt in targets:  # Helps to have a standard target ordering
                if tgt not in self.targets:
                    self.targets.append(tgt)

        if isinstance(operation, str):
            if (operation, self.simulation_level) not in self.gate_objects:
                raise ValueError(
                    f"Operation {operation} is not defined in current simulation level {self.simulation_level}. Please add compatible implementation to the dictionary.")
            operation_obj = self.gate_objects[(
                operation, self.simulation_level)]

        elif isinstance(operation, Operation):
            operation_obj = operation
            if operation_obj._simulation_level not in {self.simulation_level, 'any'}:
                raise ValueError(
                    f"Operation {operation} is not defined in current simulation level {self.simulation_level}.")

        if isinstance(operation_obj, CustomGate) and (targets is None):
            raise ValueError("Need to define target for gate object")

        if operation_obj.iID not in self.id_counter:
            self.id_counter[operation_obj.iID] = 0
        inner_id = operation_obj.iID + '' + \
            str(self.id_counter[operation_obj.iID]+1)

        self.id_counter[operation_obj.iID] += 1

        prepend_var = ''
        if type(operation_obj) == Calculation:
            prepend_var = inner_id + '_'
        for arg_tag, arg_val in arguments.items():
            temp_arg_rag = prepend_var + arg_tag
            if temp_arg_rag not in self.variables:
                self.variables[temp_arg_rag] = arg_val
            else:
                if arg_val is not None:
                    self.variables[temp_arg_rag] = arg_val

        if isinstance(operation_obj, ParametrizedGate):
            if not isinstance(args, dict) or args is None:
                raise ValueError(
                    "Parametrized gate needs a dictionary as arg definition")
            if (operation_obj.num_args != len(args)):
                raise ValueError(
                    f"Not enough argument tags provided (need {operation_obj.num_args}, while {len(args)} arg names were provided). ")
            elif len(args) != operation_obj.num_args:
                raise ValueError(
                    f"Not enough arguments provided (need {operation_obj.num_args}, while {len(args)} arg names were provided). ")

        elif isinstance(operation_obj, PulsedGate):
            # if len(operation_obj.pulses) == 0: Only compile at runtime
            #     operation_obj._compile_pulses()
            if ((len(operation_obj.targets) > 0) and (targets is None)) or ((targets is not None) and (len(operation_obj.targets) != len(targets))):
                raise ValueError(
                    f"Wrong number of targets for this operation. It requires {len(operation_obj.targets)}, while only {len(targets) if targets is not None else 0} have been defined")

        elif isinstance(operation_obj, Measurement):
            if args is None:
                raise ValueError(
                    "QuantumCircuit: need to add variable name to circuit measurement variable")
            if type(args) == str:
                arguments = {'measurement variable': args}
            elif type(args) == dict:
                arguments = {'measurement variable': [*args.values()][0]}

        elif isinstance(operation_obj, AtomTransport):
            targets = [*operation_obj.new_positions.keys()]

        elif isinstance(operation_obj, QuantumCircuit):
            self.append_circuit(operation_obj, targets,
                                controls, args, _iid=inner_id)
            self.components[inner_id] = operation_obj
            return

        self.components[inner_id] = CircuitComponent(
            operation_obj,
            simulation_level=self.simulation_level,
            targets=targets, controls=controls,
            args_dict=arguments,
            name=f"{str(operation)}_{inner_id}"
        )

        if index is None:
            self.operation_sequence.append(self.components[inner_id])
        else:
            self.operation_sequence.insert(index, self.components[inner_id])

    def add_parallel_operations(self, operations, targetss, controlss, arg_dicts, indexs):
        """
        add_parallel_operations _summary_

        _extended_summary_

        Parameters
        ----------
        operations : _type_
            _description_
        targetss : _type_
            _description_
        controlss : _type_
            _description_
        arg_dicts : _type_
            _description_
        indexs : _type_
            _description_
        """
        pass
        # TODO: paralell pulsed gatess

    def load_arguments(self, args_dict={}):
        '''
        Updates the values of the variables of the circuit 
        '''
        for k, v in args_dict.items():
            self.variables[k] = v

    def append_circuit(self,
                       circuit: 'QuantumCircuit',
                       targets: list[str] | None = None,
                       controls=None,
                       args: dict | None = None,
                       *, _iid=None):
        """
        append_circuit _summary_

        _extended_summary_

        Parameters
        ----------
        circuit : QuantumCircuit
            _description_
        targets : list[str] | None, optional
            _description_, by default None
        controls : _type_, optional
            _description_, by default None
        args : dict | None, optional
            _description_, by default None
        _iid : _type_, optional
            _description_, by default None
        """
        # defining the latex representation
        if circuit.custom_repr is not None:  # quantum circuit is represented by first component
            circuit.operation_sequence[0].operation.custom_repr = circuit.custom_repr
            for component in circuit.operation_sequence:
                component.operation.custom_repr = ''  # the rest is invisible

        target_mapping = {old_target: new_target
                          for old_target, new_target in zip(circuit.targets, self.targets)}  # assumes same target ordering

        for old_component in circuit.operation_sequence:
            # unroll the operation sequence onto the circuit

            circuit_id = _iid or 'QCirc'
            old_args = old_component.init_args
            new_args = {args[old_key]: old_value for old_key,
                        old_value in old_args.items()}
            refresh_comp = CircuitComponent(
                old_component.operation,
                simulation_level=self.simulation_level,
                targets=[target_mapping[tgt] for tgt in targets],
                controls=[target_mapping[ctl] for ctl in controls],
                args_dict=new_args,
                name=f"{circuit_id}_{old_component.name}"
            )
            refresh_comp.operation.custom_repr = old_component.custom_repr
            self.operation_sequence.append(refresh_comp)

    def reset(self):
        """
        reset Resets circuit operations, keeping the dictionary
        """
        self.operation_sequence = []
        self.variables = {}
        self.id_counter = {}
        self.components = {}
        self.targets = []
