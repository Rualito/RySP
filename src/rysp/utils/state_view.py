
import warnings
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from qutip import ket, Qobj
import seaborn as sns
from ipywidgets import FloatSlider, interactive, IntSlider

# TODO: improve algorithm of state vizualization (scales exponentially currently)


def print_state(state: Qobj):
    Nqb = len(state.dims[0])  # type: ignore
    dim: int = state.dims[0][0]  # type: ignore
    st_list = []
    ci_list = []
    for st in itertools.product([str(i) for i in range(dim)], repeat=Nqb):
        ketc = ket(''.join(st), dim)
        ci = (ketc.dag() * state).tr()

        if np.abs(ci) != 0:
            ci_list.append((ci, "".join(st)))
            st_list.append(f'{ci}|{"".join(st)}>')
    ci_list = sorted(ci_list, key=lambda x: np.abs(x[0]), reverse=True)
    return ' + '.join([f"{ci:.2e}|{st}>" for ci, st in ci_list])


def show_state(state: Qobj):
    Nqb = len(state.dims[0])  # type: ignore
    dim: int = state.dims[0][0]  # type: ignore
    st_list = []
    ci_list = []
    df = pd.DataFrame({})
    for st in itertools.product([str(i) for i in range(dim)], repeat=Nqb):
        ketc = ket(''.join(st), dim)
        ci = (ketc.dag() * state).tr()
        # df.append({'state': f'|{"".join(st)}>', 'c':ci}, ignore_index=True)
        df = pd.concat([df, pd.DataFrame(
            {'state': f'|{"".join(st)}>', 'c': np.abs(ci)**2}, index=[0])], ignore_index=True)

        if np.abs(ci) != 0:
            ci_list.append((ci, "".join(st)))
            st_list.append(f'{ci}|{"".join(st)}>')

    # ci_list = sorted(ci_list, key=lambda x: np.abs(x[0]), reverse=True)
    return df
# @interact(ti=IntSlider(min=0, max=len(sim.saved_states)))


def interactive_state_plot(saved_states, saved_state_times, threshold=1e-4):
    def slider_state_func(ti):
        df = show_state(saved_states[ti])
        sns.barplot(df[df['c'] > threshold], x='state', y='c', fill=True)
        plt.show()
        print(f"{saved_state_times[ti]:.2e}: {print_state(saved_states[ti])}")
    warnings.filterwarnings('ignore')
    interactive_plot = interactive(
        slider_state_func, ti=IntSlider(min=0, max=len(saved_states)-1))

    output = interactive_plot.children[-1]
    output.layout.height = '600px'  # type: ignore

    return interactive_plot


def print_oper(rho: Qobj, trim=1e-3):
    # Prints string representing states with largest probabilities

    Nqb = len(rho.dims[0])  # type: ignore
    dim = rho.dims[0][0]  # type: ignore
    st_list = []

    # type: ignore
    for st0 in itertools.product([str(i) for i in range(dim)], repeat=Nqb):
        ketc = ket(''.join(st0), rho.dims[0])  # type: ignore
        # type: ignore
        for st1 in itertools.product([str(i) for i in range(dim)], repeat=Nqb):
            ketd = ket(''.join(st1), rho.dims[1])  # type: ignore

            ci = (ketc.dag() * rho * ketd).norm()
            if ci != 0 and np.abs(ci) > trim:
                st_list.append(f'{ci:.2e}|{"".join(st0)}X{"".join(st1)}|')
    return ' + '.join(st_list)


def show_state_evol(times, state_history, overlap_states, asys):
    for st in overlap_states:
        ket_ov = asys.ket_str(st)
        st_rr = [np.abs((sti.overlap(ket_ov))) for sti in state_history]
        plt.plot(times, st_rr, label=f'|{st}>')
    plt.legend()


def measure_and_project(rng: np.random.Generator, state: Qobj, projector: Qobj, identity: Qobj) -> tuple[Qobj, int, float]:
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
