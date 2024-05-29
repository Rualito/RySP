

from typing import Any

import numpy as np
from qutip import Qobj
from pulser.pulse import Pulse
from pulser.waveforms import InterpolatedWaveform, ConstantWaveform

from rysp.core.circuit.operations import CustomGate, ParametrizedOperation
from ..experiment.atomsystem import AtomSystem

from numba import njit


@njit
def _evaluate_array_interpolation(arr: np.ndarray, t0: float, tf: float, t: float):
    # Assuming linear time scalling
    idx = (t-t0)/(tf-t0) * (len(arr)-1)
    if t < t0:
        return 0
    if t > tf:
        return 0

    x1 = max(0, min(int(np.ceil(idx)), len(arr)-1))
    x0 = max(0, min(int(np.floor(idx)), len(arr)-1))
    if x0 == x1:
        return arr[x0]
    else:
        y1 = arr[x1]
        y0 = arr[x0]
        output = (y1-y0) * (idx % 1) + y0

        return output


@njit
def pulse_compile(time_list, start_times, durations, amps_list, dets_list, phases):
    total_steps = len(time_list)
    samples_amp = np.zeros(total_steps, dtype=np.complex64)
    samples_det = np.zeros(total_steps, dtype=np.float64)
    t0 = start_times[0]
    # print("Compiling pulse")
    for i, ti in enumerate(time_list):
        # ti = i*self.time_resolution
        if ti > t0:
            for st, dr, amps, dets, phase in zip(start_times, durations, amps_list, dets_list, phases):
                # if pulse is still active
                tp0 = st  # pulse start time
                tpf = st+dr  # pulse end time

                if (ti > tp0) and (ti < tpf):
                    # Add amplitude samples
                    samples_amp[i] += _evaluate_array_interpolation(
                        amps, tp0, tpf, ti) * np.exp(1.0j * phase)

                    # Can just add the detunings
                    samples_det[i] += _evaluate_array_interpolation(
                        dets, tp0, tpf, ti)
    return samples_amp, samples_det


@njit
def get_compatible_timelist(intervals):
    # total_steps = len(time_list)
    # find start time
    # find end time
    # find keytimes
    # start time, final time, time scale
    key_times = []
    for t0, tf, ts in intervals:
        key_times.append(t0)
        key_times.append(tf)
        # if (t0,tf) not in scales:
        #     scales[(t0, tf)] = []
    #     scales[t0] = (ts, 0)
    #     scales[tf]
    key_times = sorted(list(set(key_times)))  # sort the list of unique times
    # print(key_times)
    scales = []
    for k, tk in enumerate(key_times[:-1]):
        valid_scales = []
        for t0, tf, ts in intervals:
            if (tk >= t0) and (tk < tf):
                valid_scales.append(ts)
        if len(valid_scales) == 0:
            scales.append((scales[-1]+key_times[k+1])/2)
            continue

        # print(f"Scale: ", valid_scales)
        dt = key_times[k+1] - tk
        steps = max(2, int(dt/min(valid_scales)))
        time_list = np.linspace(tk, key_times[k+1], steps)
        if k == 0:
            for j, tj in enumerate(time_list):
                scales.append(tj)
        elif scales[-1] == time_list[0]:
            for j, tj in enumerate(time_list[1:]):
                scales.append(tj)
        else:
            scales.append((scales[-1]+time_list[0])/2)
            for j, tj in enumerate(time_list):
                scales.append(tj)
    return np.array(scales)


class PulseSequence:  # TODO: PulseSequence docstring
    '''
    Pulse sequence class. 
    Groups pulses into separate channels in a single compact object.
    '''

    def __init__(self):
        """Initializes a new pulse sequence."""
        self.sequence = {}
        self.channel_config = {}
        self.start_times = []
        self.pulses = {}
        self.max_time = 0
        self.targets = []
        self.start_time = 0
        self.duration = 0

        self.pulse_intervals = []
        self.pulse_intervals_pos = []
        self.time_list = None

    def new_channel(self, name, transition):
        '''
        transition: tuple
            - indicates which atomic transition does the channel target
        '''
        self.channel_config[name] = transition

    def validate_sequence(self, experiment):
        '''validate_sequence _summary_

        :param experiment: _description_
        :type experiment: ExperimentSetup
        :raises ValueError: _description_
        '''
        if experiment.simulation_level != 'pulsed':
            raise ValueError(
                "PulseSequence: Provided experiment config is not pulse based.")

    def add_pulse(self, pulse,
                  channel: str,
                  target: str | int,
                  time_scale=1e-9,
                  t0=0):
        '''
        :param pulse: Pulse to add to the channel
        :type pulse: Pulse
        :param channel: channel name declared, indicating the targetted transition.
        :type channel: str
        :param target: _description_
        :type target: str | int
        :param t0: starting time to execute the pulse (in seconds), defaults to 0
        :type t0: int, optional
        '''
        if target not in self.targets:
            self.targets += [target]

        if (channel, target) not in self.sequence:
            self.sequence[(channel, target)] = []
        self.sequence[(channel, target)].append(
            (t0, pulse.duration*1e-9, pulse))
        self.start_times.append(t0)

        if self.max_time < t0 + pulse.duration*1e-9:
            self.max_time = t0 + pulse.duration*1e-9
        self.duration = self.max_time - self.start_time

        self.pulse_intervals += [(t0, t0+pulse.duration*1e-9, time_scale)]
        self.pulse_intervals_pos += [(channel, target,
                                      len(self.sequence[(channel, target)])-1)]

    def shift_pulses(self, dt: float):
        '''
        shifts pulse times backwards or forwards in time according to `dt`. Needs to be compiled afterwards

        dt: time in seconds to shift the time list
        '''
        for key in self.sequence:
            for i, el in enumerate(self.sequence[key]):
                self.sequence[key][i] = (el[0]+dt, el[1], el[2])
        for i, st in enumerate(self.start_times):
            self.start_times[i] = st+dt
        for i, (t0, tf, ts) in enumerate(self.pulse_intervals):
            self.pulse_intervals[i] = (t0+dt, tf+dt, ts)
        self.start_time = self.start_time + dt
        self.max_time += dt

    def _convert_pulse_list(self, time_list: np.ndarray):
        '''
        Converts pulse sequence values to be evaluated at times defined by `time_list`. Acts as pulse compiler.

        time_list: np.ndarray[float]
        '''
        self._prepare_compile()
        self.pulses = {}
        total_steps = len(time_list)
        delta = time_list[1] - time_list[0]
        # adding safe padding to both sides
        self.time_list = time_list
        # np.pad(time_list, (0,1), mode='constant',
        #                         constant_values=[-delta, time_list[-1]+delta])+delta
        self.time_resolution = None
        for ch, tgt in self.sequence:
            ordered_pulses = sorted(
                self.sequence[(ch, tgt)], key=lambda x: x[0])
            samples_amp = np.zeros(total_steps, dtype=np.complex64)
            samples_det = np.zeros(total_steps, dtype=np.float64)
            t0 = ordered_pulses[0][0]  # start time
            samples_amp, samples_det = pulse_compile(time_list,
                                                     [op[0]
                                                         for op in ordered_pulses],
                                                     [op[1]
                                                         for op in ordered_pulses],
                                                     [op[2].amplitude.samples for op in ordered_pulses],
                                                     [op[2].detuning.samples for op in ordered_pulses],
                                                     [op[2].phase for op in ordered_pulses])

            self.pulses[(ch, tgt)] = {'amp': samples_amp,
                                      'det': samples_det}

    def get_pulse_switch_times(self) -> np.ndarray:
        switch_times = []
        for t0, t1 in self.pulse_intervals:
            switch_times += [t0, t1]
        switch_times: list[float] = sorted([*set(switch_times)])
        if switch_times[-1] < self.max_time:
            switch_times.append(self.max_time)
        elif switch_times[-1] > self.max_time:
            raise ValueError(
                "PulseSequence: Error in defining pulse intervals. max_time is not being tracked correctly")
        return np.array(switch_times)

    def _prepare_compile(self):
        st0: float = min(self.start_times)
        if st0 != 0:
            self.shift_pulses(-st0)

        st0: float = min(self.start_times)

        if self.max_time != self.duration:
            raise ValueError(
                "PulseSequence: max_time != duration, internal error!!")
        if st0 != self.start_time:
            print(f"{st0=}, {self.start_time=}, {self.max_time=}, {self.duration=}")
            print(f"{self.start_times=}")
            raise ValueError(
                "PulseSequence: st0 != start_time, internal error!!")

    def _compile_pulses(self, time_resolution=1e-8, adapt=True):  # ns resolution
        '''
        For every channel/target, get the compiled waveform from the corresponding samples.
        Compiles the pulse for a standard time scale
        time_resolution: defines the time discretization step size, in ns
        '''
        self._prepare_compile()

        # self.time_resolution = time_resolution

        # warnings.warn("Cannot add complex phases in the current implementation.
        # Output has constant phase, taken to be the average over the whole pulse.")
        if adapt:
            timelist = get_compatible_timelist(self.pulse_intervals)
        else:
            num_steps = int(np.ceil(self.duration/time_resolution))
            timelist = np.linspace(0, self.duration, num_steps)

        self._convert_pulse_list(timelist)


class PulsedGate(PulseSequence, CustomGate):  # TODO: PulsedGate docstring
    """ 
    PulsedGate _summary_

    _extended_summary_

    Parameters
    ----------
    PulseSequence : _type_
        _description_
    CustomGate : _type_
        _description_
    """

    def __init__(self):
        '''
        pulse_sequence: defines the targets for each pulse, as well as the amplitudes and detunings 
        '''
        super().__init__()
        # self.name = name
        self._simulation_level = 'pulsed'
        # self.pulse_sequence = pulse_sequence
        self.iID = 'pG'

        self.is_cached = False
        self.custom_repr = None
        self.override_custom_when_show_pulse = True

        self.cached_propagator: Qobj | None = None
        self.cached_lattice: AtomSystem | None = None
        self.cached_integrator: dict[str, Qobj] | None = None
        self.cache_settings: dict[str, Any] | None = None

    def __str__(self):
        return f"PulsedGate"

    def cache_gate(self,
                   propagator: Qobj,
                   lattice: AtomSystem,
                   integrators: dict[str, Qobj],
                   sim_config: dict):
        self.cached_propagator = propagator
        self.cached_lattice = lattice
        self.cached_integrator = integrators
        self.cache_settings = sim_config

        self.is_cached = True

    def compare_cached(self,
                       lattice: AtomSystem,
                       targets: list[str],
                       lattice_match_tolerance: float,
                       #    integrators: dict[str, Qobj],
                       sim_config: dict) -> bool:
        """
        compare_cached Checks if the current gate caching is compatible with the given settings

        _extended_summary_

        Parameters
        ----------
        lattice : AtomSystem
            _description_
        targets : list[str]
            _description_
        lattice_match_tolerance : float
            _description_
        integrators : dict[str, Qobj]
            _description_
        sim_config : dict
            _description_

        Returns
        -------
        bool
            _description_
        """
        test = self.is_cached
        test = test and self.cached_lattice._check_lattice_compatibility(
            lattice, targets, lattice_match_tolerance)
        test = test and (self.cache_settings == sim_config)

        return test

    def set_representation(self, latex_str, override_when_pulsed=False):
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
            self.override_custom_when_show_pulse = override_when_pulsed

    def _to_latex_yquant(self, show_pulsed, **kwargs):
        if (self.custom_repr is not None) and not (self.override_custom_when_show_pulse and show_pulsed):
            return self.custom_repr
        if show_pulsed:
            return 'align {targets};\n'+''.join([f"[inner xsep=1pt, inner ysep=1pt]box {{plot{i}}} ({{target{i}}});\n" for i in range(len(self.targets))]) + 'align {targets};'
        return r"box {$P$} ({targets});"


def Pulse_from_file(amplitudes: np.ndarray,
                    detunings: np.ndarray,
                    duration: float,
                    interpkind: str = 'linear',
                    normalize=True):
    amps = amplitudes
    dets = detunings
    if normalize:
        amps = amps/np.max(amplitudes)
        dets = dets/np.max(amplitudes)
    if np.max(np.abs(dets)) < 1e-8:
        return Pulse(amplitude=InterpolatedWaveform(int(duration), amps,
                                                    interpolator='interp1d', kind=interpkind),
                     detuning=ConstantWaveform(int(duration), 0),
                     phase=0)
    else:
        return Pulse(amplitude=InterpolatedWaveform(int(duration), amps,
                                                    interpolator='interp1d', kind=interpkind),
                     detuning=InterpolatedWaveform(int(duration), dets,
                                                   interpolator='interp1d', kind=interpkind),
                     phase=0)


class ParametrizedPulsedGate(ParametrizedOperation):
    '''
    Constructs pulse sequence later, as a callback
    '''

    def __init__(self, pulsegen_func, input_vars: list[str], cache=False):

        super().__init__(pulsegen_func, input_vars, cache)

    def __str__(self):
        return f"ParametrizedPulsedGate"

    def evaluate(self, **kwargs) -> PulsedGate:
        dic2 = {}
        for key, val in kwargs.items():  # remapping the arguments to the original names
            dic2[self.variable_mapping[key]] = val
        # print(f"{dic2=}")
        return self._func(**dic2)

    def _to_latex_yquant(self, show_vars, **kwargs):
        if self.custom_repr is None:
            if show_vars:
                return '["' + ', '.join(self.input_vars) + r'" below]box {{$P_{{\vec\theta}}$}} {targets} | var;'
            return r"box {{$P_{{\vec\theta}}$}} {targets};"
        else:
            return self.custom_repr
