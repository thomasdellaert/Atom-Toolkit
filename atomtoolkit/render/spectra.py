"""
Tools for drawing transition spectra using matplotlib
"""
from __future__ import annotations
import colorsys

from matplotlib import pyplot as plt

import matplotlib.colors
from atomtoolkit.atom import *
from .lineshapes import *


# TODO: Make this use the axis method instead of pyplot

def plot_transitions(transitions: BaseTransition or List[BaseTransition],
                     lineshape: LineShape, **kwargs):
    if 'color_func' not in kwargs:
        def color_func(t): return 0, 0, 0, 1
    else:
        color_func = kwargs['color_func']
    label_func = kwargs.pop('label_func', lambda t: f"{t.E_lower.term.short_name} → {t.E_upper.term.short_name}")
    total = kwargs.pop('total', True)
    unit = kwargs.pop('unit', 'GHz')

    all_x, all_y = [], []
    for transition in transitions:
        fGHz = transition.freq.to(unit).magnitude
        ampl = transition.rel_strength
        ampl *= kwargs.get('ampl_dict', {transition.name: 1.0})[transition.name]
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


def plot_hyperfine_spectrum(transitions: Transition or List[Transition] or List[HFTransition],
                            lineshape: LineShape = LorentzianLineShape, coloring: str = 'l', **kwargs):
    # FIXME: lineshape parameter doesn't make sense rn
    if isinstance(transitions, list):
        lines = list(itertools.chain.from_iterable(list(t.values() for t in transitions)))
    else:
        lines = list(transitions.values())

    plot_spectrum(lines, lineshape=lineshape, coloring=coloring, unit="GHz", **kwargs)


def plot_zeeman_spectrum(transitions: BaseTransition or List[BaseTransition],
                         lineshape: LineShape = LorentzianLineShape, coloring: str = 'transition_type', **kwargs):
    if isinstance(transitions, HFTransition):
        lines = list(transitions.values())
    elif isinstance(transitions, Transition):
        lines = list(itertools.chain.from_iterable(list(t.values()) for t in list(transitions.values())))
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

    if coloring == 'l':
        color_dict = color_table_from_property(lines,
                                               property_color=lambda t: t.E_lower.term.mF,
                                               property_shade=lambda t: t.E_upper.term.mF,
                                               cmap=kwargs.pop('cmap', 'tab10'))
        color_func = lambda t: color_dict[t]
    elif coloring == 'u':
        color_dict = color_table_from_property(lines,
                                               property_color=lambda t: t.E_upper.term.mF,
                                               property_shade=lambda t: t.E_lower.term.mF,
                                               cmap=kwargs.pop('cmap', 'tab10'))
        color_func = lambda t: color_dict[t]
    elif coloring == 'transition_type':
        color_dict = color_table_from_property(lines,
                                               property_color=lambda t: t.E_upper.term.mF - t.E_lower.term.mF,
                                               property_shade=lambda t: t.E_lower.term.mF,
                                               cmap=kwargs.pop('cmap', 'tab10'))
        color_func = lambda t: color_dict[t]
    elif isinstance(coloring, Callable):
        color_func = coloring
    else:
        color_func = None

    plot_transitions(lines, lineshape=lineshape, color_func=color_func, unit="GHz", **kwargs)


def color_table_from_property(items: List[BaseTransition],
                              property_color: Callable, property_shade: Callable = None, cmap="tab10"):
    main_props, secondary_props = [], []
    for t in items:
        main_props.append(property_color(t))
        secondary_props.append(property_shade(t))
    # the number of colors that will be needed
    num_main_props = {F: main_props.count(F) for F in np.unique(main_props)}
    prop_pairs = list(zip(main_props, secondary_props))
    # a dict that yields the lowest secondary property associated with each main property
    min_main_props = {p0: min([Fp[1] for Fp in prop_pairs if Fp[0] == p0]) for p0 in main_props}

    def color_by_properties(t):
        colormap = plt.get_cmap(cmap)

        color_prop = property_color(t)
        shade_prop = property_shade(t)

        # pick a color depending on the color property
        col = colormap(list(num_main_props.keys()).index(color_prop))
        # convert it to HLS
        col = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(col))
        # set the lightness depending on the shade property and convert back to RGB
        lightness = 0.2 + 0.6 * (shade_prop - min_main_props[color_prop] + 1) / (num_main_props[color_prop] + 1)
        col = colorsys.hls_to_rgb(col[0], lightness, col[2])
        return col

    return {t: color_by_properties(t) for t in items}


def plot_spectrum(transitions: BaseTransition or List[BaseTransition],
                  lineshape: str or LineShape = None,
                  coloring: str or Callable = None,
                  temp: pint.Quantity = None,
                  laser_linewidth: pint.Quantity = Q_(0, 'Hz'),
                  mass: pint.Quantity = None,
                  ampl_dict: Dict = None,
                  polarization: Tuple = None,
                  **kwargs
                  ):
    lines = transitions
    if not isinstance(transitions, list):
        lines = list(transitions.values())
    # determine the lineshape:
    if lineshape is not None:
        lineshape_func = lineshape
    elif temp.magnitude == 0:
        lineshape_func = LorentzianLineShape(gamma=laser_linewidth.to(GHz).magnitude)
    else:
        assert mass is not None, "thermal lineshape requires mass to be given"
        lineshape_func = ThermalLineShape(T=temp, m=mass, gamma=laser_linewidth.to(GHz).magnitude)

    # parse the color function CONSIDER: move this to a separate function for cleaner code?
    def color_func(t): pass
    if type(lines[0]) == Transition:
        def color_func(t): return (0, 0, 0, 1)
    elif type(transitions[0]) == HFTransition:
        if coloring == 'l':
            color_dict = color_table_from_property(lines,
                                                   property_color=lambda t: t.E_lower.term.F,
                                                   property_shade=lambda t: t.E_upper.term.F,
                                                   cmap=kwargs.pop('cmap', 'tab10'))

            def color_func(t): return color_dict[t]
        elif coloring == 'u':
            color_dict = color_table_from_property(lines,
                                                   property_color=lambda t: t.E_upper.term.F,
                                                   property_shade=lambda t: t.E_lower.term.F,
                                                   cmap=kwargs.pop('cmap', 'tab10'))
    elif type(transitions[0]) == ZTransition:
        if coloring == 'l':
            color_dict = color_table_from_property(lines,
                                                   property_color=lambda t: t.E_lower.term.mF,
                                                   property_shade=lambda t: t.E_upper.term.mF,
                                                   cmap=kwargs.pop('cmap', 'tab10'))

            def color_func(t): return color_dict[t]
        elif coloring == 'u':
            color_dict = color_table_from_property(lines,
                                                   property_color=lambda t: t.E_upper.term.mF,
                                                   property_shade=lambda t: t.E_lower.term.mF,
                                                   cmap=kwargs.pop('cmap', 'tab10'))

            def color_func(t): return color_dict[t]
        elif coloring == 'transition_type':
            color_dict = color_table_from_property(lines,
                                                   property_color=lambda t: t.E_upper.term.mF - t.E_lower.term.mF,
                                                   property_shade=lambda t: t.E_lower.term.mF,
                                                   cmap=kwargs.pop('cmap', 'tab10'))

            def color_func(t): return color_dict[t]

    plot_transitions(transitions, lineshape_func, ampl_dict=ampl_dict, color_func=color_func)
