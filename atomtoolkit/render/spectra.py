"""
Tools for drawing transition spectra using matplotlib
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors
from atomtoolkit.atom import BaseTransition, Transition, HFTransition, ZTransition
import colorsys
from .lineshapes import LineShape, LorentzianLineShape
from typing import List, Type
import itertools

# FIXME: Zeeman transitions kind of work, but it's a bit of a bodge and the colors are off.

def plot_transitions(transitions: Type[BaseTransition] or List[Type[BaseTransition]],
                     lineshape: Type[LineShape], **kwargs):
    if 'color_func' not in kwargs:
        def color_func(t):
            return None
    else:
        color_func = kwargs['color_func']
    label_func = kwargs.pop('label_func', lambda t: f"{t.E_lower.term.short_name} → {t.E_upper.term.short_name}")
    total = kwargs.pop('total', True)
    unit = kwargs.pop('unit', 'GHz')

    all_x, all_y = [], []
    for transition in transitions:
        fGHz = transition.freq.to(unit).magnitude
        ampl = transition.rel_strength
        if 'ampl_dict' in kwargs:
            ampl *= kwargs['ampl_dict'][transition.name]
        x_values, y_values = lineshape.compute(fGHz, ampl=ampl, **kwargs)
        all_x.append(x_values)
        all_y.append(y_values)

        label = label_func(transition)

        col = color_func(transition)

        plt.plot(x_values, y_values, label=label, c=col)
    if total:
        x_values = []
        for x in all_x:
            x_values.append(x)
        x_values = np.sort(np.array(x_values).flatten())
        y_values = np.zeros_like(x_values)
        for curve, x in zip(all_y, all_x):
            y_values += np.interp(x_values, x, curve)
        plt.plot(x_values, y_values, "k:", label="Total", alpha=0.5)
    plt.ylim(0, None)
    plt.xlabel("Frequency (GHz)")
    plt.legend()

def plot_hyperfine_spectrum(transitions: Type[BaseTransition] or List[Type[BaseTransition]],
                            lineshape: Type[LineShape] = LorentzianLineShape, coloring='l', **kwargs):
    # TODO: reconsider the name, since this is theoretically able to plot *any* sub-spectrum
    if isinstance(transitions, list):
        lines = list(itertools.chain.from_iterable(list(t.values() for t in transitions)))
    else:
        lines = list(transitions.values())

    lower_Fs, upper_Fs = [], []
    for t in lines:
        lower_Fs.append(t.E_lower.term.F)
        upper_Fs.append(t.E_upper.term.F)
    num_lower_Fs = {F: lower_Fs.count(F) for F in np.unique(lower_Fs)}
    num_upper_Fs = {F: upper_Fs.count(F) for F in np.unique(upper_Fs)}
    F_pairs = list(zip(lower_Fs, upper_Fs))
    min_upper_Fs = {Flo: min([Fp[1] for Fp in F_pairs if Fp[0] == Flo]) for Flo in lower_Fs}
    min_lower_Fs = {Fhi: min([Fp[0] for Fp in F_pairs if Fp[1] == Fhi]) for Fhi in upper_Fs}

    def color_by_F(line, cmap="tab10"):
        colormap = plt.get_cmap(cmap)

        # depending on coloring by upper/lower, one set of Fs determines color, while the other determines shade
        if not (coloring == 'u'):
            color_num_Fs = num_lower_Fs
            color_F = line.E_lower.term.F
            shade_F = line.E_upper.term.F
            min_shade_Fs = min_upper_Fs
        else:
            color_num_Fs = num_upper_Fs
            color_F = line.E_upper.term.F
            shade_F = line.E_lower.term.F
            min_shade_Fs = min_lower_Fs

        # pick a color depending on the color F
        col = colormap(list(color_num_Fs.keys()).index(color_F))
        # convert it to HLS
        col = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(col))
        # set the lightness depending on the shade F and convert back to RGB
        lightness = 0.2 + 0.6 * (shade_F - min_shade_Fs[color_F] + 1) / (color_num_Fs[color_F] + 1)
        col = colorsys.hls_to_rgb(col[0], lightness, col[2])
        return col

    if coloring == 'l' or coloring == 'u':
        color_func = color_by_F
    else:
        color_func = None

    plot_transitions(lines, lineshape=lineshape, color_func=color_func, unit="GHz", **kwargs)

def plot_zeeman_spectrum(transitions: Type[BaseTransition] or List[Type[BaseTransition]],
                         lineshape: Type[LineShape] = LorentzianLineShape, **kwargs):
    # TODO: Currently all mFs get the same color. Generalize the colorize function in
    #  plot_hyperfine_spectrum to color by any parameter
    if isinstance(transitions, Transition):
        lines = list(transitions.values())
    elif isinstance(transitions, list):
        lines = list(itertools.chain.from_iterable(list(t.values() for t in transitions)))
    else:
        lines = transitions

    def label_func(transition):
        F0 = transition.E_lower.term.F_frac
        F1 = transition.E_upper.term.F_frac
        mF0 = transition.E_lower.term.mF_frac
        mF1 = transition.E_upper.term.mF_frac
        return f'F={F0}, mF={mF0} → F={F1}, mF={mF1}'

    plot_hyperfine_spectrum(lines, lineshape=lineshape, label_func=label_func, **kwargs)
