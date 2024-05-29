import os

import matplotlib.pyplot as plt
from ..core.experiment.atomsystem import AtomSystem
from ..core.circuit.quantumcircuit import QuantumCircuit
from ..core.circuit.pulses import PulsedGate
from ..core.circuit.operations import AtomTransport, Measurement, Operation

import numpy as np
import networkx as nx
import subprocess

import matplotlib.pyplot as plt

# TODO: pulser-like visualization of the evolution? with circles representing the overlap with the state through time?


def plot_pulsed_gate(pulsed: 'PulsedGate'):
    if not pulsed.pulses:
        pulsed._compile_pulses()

    time_list = pulsed.time_list
    pulses = pulsed.pulses

    # transition_colors = {('0', '1'): 'r', ('1', 'r'): 'tab:purple'}

    pulses_tgt = []
    for i in range(len(pulsed.targets)):
        pulses_tgt += [[[None, None], [None, None]]]
    min_y = 1000
    max_y = -1000
    for (ch, tgt) in pulses:
        index = pulsed.targets.index(tgt)
        if ch in {'clock', '01'}:
            pulses_tgt[index][0][0] = np.copy(
                np.real(pulses[(ch, tgt)]['amp']))
            pulses_tgt[index][0][1] = np.copy(
                np.real(pulses[(ch, tgt)]['det']))
        elif ch in {'ryd', '1r'}:
            pulses_tgt[index][1][0] = np.copy(
                np.real(pulses[(ch, tgt)]['amp']))
            pulses_tgt[index][1][1] = np.copy(
                np.real(pulses[(ch, tgt)]['det']))
        min_y = np.min([min_y,
                       *np.real(pulses[(ch, tgt)]['amp']),
                       *np.real(pulses[(ch, tgt)]['det'])])
        max_y = np.max([max_y,
                       *np.real(pulses[(ch, tgt)]['amp']),
                       *np.real(pulses[(ch, tgt)]['det'])])
    plt.show()
    fig, axs = plt.subplots(len(pulsed.targets),
                            sharex=True,
                            sharey=True)
    if isinstance(axs, plt.Axes):
        axs = [axs]
    fig.suptitle("Pulse shapes on the various targets")
    # print(pulses_tgt)
    for i, pulse_shape in enumerate(pulses_tgt):
        # 01 transition
        if pulse_shape[0][0] is not None:
            # axs[i].plot(time_list, pulse_shape[0][0],
            #             c='r', linestyle='-', label='clock amp')  # amplitude
            axs[i].fill_between(time_list, pulse_shape[0][0],
                                color='b', alpha=0.5, label='clock amp')
        if pulse_shape[0][1] is not None:
            axs[i].plot(time_list, pulse_shape[0][1],
                        c='b', linestyle='--', label='clock det')  # detuning

        # 1r transition
        if pulse_shape[1][0] is not None:
            # axs[i].plot(time_list, pulse_shape[1][0],
            #             c='tab:purple', linestyle='-', label='ryd amp')  # amplitude

            axs[i].fill_between(time_list, pulse_shape[1][0],
                                color='r', alpha=0.5, label='ryd amp')
        if pulse_shape[1][1] is not None:
            axs[i].plot(time_list, pulse_shape[1][1],
                        c='r', linestyle='--', label='ryd det')  # detuning
        axs[i].set_ylabel(f'Target {i}')
    plt.xlabel('time (s)')
    plt.ylim((min_y-0.1*np.abs(min_y), max_y + 0.1*np.abs(max_y)))


def get_pulse_shapes(pulsed: 'PulsedGate'):
    if not pulsed.pulses:
        pulsed._compile_pulses()
    pulse_shapes = {}
    map_ch = {'ryd': '1r', '1r': '1r', 'clock': '01', '01': '01'}
    for (ch, tgt), profiles in pulsed.pulses.items():
        if tgt not in pulse_shapes:
            pulse_shapes[tgt] = {}
        pulse_shapes[tgt][map_ch[ch]] = profiles  # {'amp':..., 'det':...}

    return pulse_shapes


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


def plot_atom_lattice(lattice: 'AtomSystem',
                      node_size=1e-6,
                      screen=(12, 16),
                      color=(0, 0, 0),
                      threshold_ratio=0.5):
    # TODO: change color scheme to improve graphics
    nodes = {}
    nodes_dist = {}
    max_dist = 0
    min_dist = -1
    G = nx.Graph()
    for i, (lbl0, atom0) in enumerate(lattice.atomsetup.items()):
        nodes[lbl0] = atom0.pos
        G.add_node(lbl0,
                   pos=atom0.pos[:2])
        for j, (lbl1, atom1) in enumerate(lattice.atomsetup.items()):
            if lbl1 != lbl0:
                adist = np.linalg.norm(atom0.pos-atom1.pos)
                nodes_dist[(lbl0, lbl1)] = adist
                max_dist = max(max_dist, adist)
                if min_dist == -1 or adist < min_dist:
                    min_dist = adist
    if np.abs(min_dist-max_dist)/max_dist < 1e-2:
        max_dist = 3*min_dist
    threshold = (max_dist-min_dist)*threshold_ratio + min_dist

    def color_func(dist):
        color_target = np.array(color)
        color_min = np.array((160, 160, 160))
        if dist > threshold:
            return rgb_to_hex(tuple(color_min))
        else:
            return rgb_to_hex(tuple(np.uint8((color_min-color_target)/(threshold-min_dist) * (dist-min_dist) + color_target)))

    def weight_func(dist):
        weight_target = 5
        weight_blank = 0
        # print(f'{dist=}, {threshold=}, {min_dist=}, {max_dist=}')
        if dist < threshold:
            return weight_target
        else:
            return 0.5*((weight_blank-weight_target)/(max_dist-threshold) * (dist-threshold) + weight_target)

    for l0 in nodes:
        for l1 in nodes:
            if l0 != l1:
                G.add_edge(l0,
                           l1,
                           color=color_func(nodes_dist[(l0, l1)]),
                           weight=weight_func(nodes_dist[(l0, l1)]))

    colors = [*nx.get_edge_attributes(G, 'color').values()]
    weights = [*nx.get_edge_attributes(G, 'weight').values()]

    # labels = [*nx.get_node_attributes(G, 'label').values()]
    # print(f"{colors=}, \n{weights=} \n{list(G)=}")

    nx.draw(G, nx.get_node_attributes(G, 'pos'),
            edge_color=colors,
            width=list(weights),
            with_labels=True,
            node_color='lightblue')


def plot_quantum_circuit(qc: 'QuantumCircuit',
                         atom_to_qubit_repr: dict,
                         img_dir='_generated_images',
                         filename='circuit',
                         dpi=600,
                         keepfiles=False,
                         show_pulsed=False,
                         show_vars=False,
                         debug_mode=False):
    assert (type(dpi) == int)
    register_tag = {a: f'$|{q}\\rangle$' for a,
                    q in atom_to_qubit_repr.items()}

    register_latex = [
        f"qubit {{{register_tag[a]}}} {atom_to_qubit_repr[a]};" for a in atom_to_qubit_repr]
    latex_kwargs = {'target_registers': atom_to_qubit_repr,
                    'show_pulsed': show_pulsed,
                    'show_vars': show_vars,
                    'replace_plots': []}
    tex_file = f'{img_dir}/{filename}.tex'
    pdf_name = f'{img_dir}/{filename}.pdf'
    png_name = f'{img_dir}/{filename}.png'
    os.makedirs(os.path.dirname(tex_file), exist_ok=True)

    if show_vars:
        register_latex += ["qubits {vars} var;"]
    circuit_latex = []
    plot_index = 0
    ch_color = {'01': 'b', '1r': 'r'}
    for cc in qc.operation_sequence:
        latex_kwargs['replace_plots'] = []
        if isinstance(cc.operation, Measurement):
            latex_kwargs['variable'] = cc.variable
        if show_pulsed and type(cc.operation) == PulsedGate \
            and (cc.operation.custom_repr is None
                 or cc.operation.override_custom_when_show_pulse):
            # print('Operation is PulsedGate')
            # generate plots
            pulse_shapes = get_pulse_shapes(cc.operation)
            time_list = cc.operation.time_list
            plot_files = []
            max_y, min_y = None, None
            for i, target in enumerate(pulse_shapes):
                for k in ('01', '1r'):
                    if k in pulse_shapes[target]:
                        for l in ('amp', 'det'):
                            arr = pulse_shapes[target][k][l]
                            maxr = np.max(arr)
                            minr = np.min(arr)
                            if max_y == None:
                                max_y = maxr
                            if min_y == None:
                                min_y = minr
                            max_y = max_y if max_y > maxr else maxr
                            min_y = min_y if min_y < minr else minr

            for i, target in enumerate(pulse_shapes):
                # print("pulse shapes is not empty")
                profile = pulse_shapes[target]
                fig = plt.figure(figsize=(9, 3))
                # plt.ioff()
                for k in ('01', '1r'):
                    # clock, ryd
                    if k in profile:
                        plt.fill_between(time_list, np.real(
                            profile[k]['amp']), color=ch_color[k], alpha=0.6)  # clock amp
                        plt.plot(time_list, np.real(
                            profile[k]['det']), c=ch_color[k], linestyle='--')  # clock det
                plt.axis('off')
                plt.ylim((min_y, max_y))
                plt.xlim((time_list[0], time_list[-1]))
                plot_name = f'plot-{plot_index}.png'
                plt.savefig(f'{img_dir}/{plot_name}', bbox_inches='tight')
                plot_files.append(plot_name)
                plt.close()
                if debug_mode:
                    print(f"generated file {img_dir}/{plot_name}")

                plot_index += 1
            latex_kwargs['replace_plots'] = plot_files
        circuit_latex.append(cc._to_latex_yquant(**latex_kwargs))

    latex_doc_str = \
        r'''\documentclass[10pt, margin=0.1in]{standalone}
\usepackage{braket}
\usepackage[compat=0.4]{yquant}
\usetikzlibrary{quotes}
\begin{document}
    \begin{tikzpicture}
        \begin{yquant*}
'''
    for reg in register_latex:
        latex_doc_str += '\t'*2+reg + '\n'

    # if show_vars:
    #     pass
    for circ in circuit_latex:
        latex_doc_str += '\t'*2+circ + '\n'
    latex_doc_str += \
        r'''
        \end{yquant*}
    \end{tikzpicture}
\end{document}
'''

    with open(tex_file, 'w') as f:
        f.write(latex_doc_str)
        if debug_mode:
            print(f'Latex doc string: {latex_doc_str}')

    if not debug_mode:
        with open(os.devnull, 'w') as FNULL:
            subprocess.call(
                ['pdflatex', f'-output-directory={img_dir}', tex_file], stdout=FNULL)
            subprocess.call(['rm', f'{img_dir}/{filename}.log',
                             f'{img_dir}/{filename}.aux'],
                            stdout=FNULL)
            if not keepfiles:
                subprocess.call(['rm', tex_file],
                                stdout=FNULL)
                if len(latex_kwargs['replace_plots']) > 0 and not keepfiles:
                    subprocess.call(['rm', *latex_kwargs['replace_plots']],
                                    stdout=FNULL)
            subprocess.call(['pdftoppm', pdf_name,
                            f'{img_dir}/{filename}',
                             '-png', '-r', str(dpi),
                             '-singlefile'])

        return png_name
    else:
        return None


def plot_atom_transport(at: 'AtomTransport'):
    pass
