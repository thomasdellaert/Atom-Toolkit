"""
Eventually meant for drawing Grotrian diagrams, perhaps even modifiable ones
"""

import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import pandas as pd
from atomtoolkit.atom import BaseLevel
import atomtoolkit.util as util


def draw_levels(atom, plot_type='norm', **kwargs):
    """
    Draws a quick and dirty grotrian diagram.
    Placeholder for a more complete grotrian functionality
    """
    posdict = {l.name: (l.term.J, l.level_Hz/1e12) for l in atom.levels.values()}
    if plot_type == 'hf':
        model = atom._hfModel.copy()
    elif plot_type == 'z':
        model = atom._zModel.copy()
    else:
        model = atom._levelsModel.copy()
    for node in model.nodes():
        try:
            model.remove_edge(node, node)
        except nx.NetworkXError:
            pass
    nx.draw(model, pos=posdict, node_shape="_", with_labels=True, font_size=8, edge_color=(0., 0., 0., 0.2), **kwargs)


RENDER_METHODS = {'matplotlib', 'bokeh'}
RENDERER = 'matplotlib'


def gross_level_table(caller, level: BaseLevel, **kwargs):
    level = level.manifold

    width = kwargs.pop('width', caller.level_width)
    x0 = kwargs.pop('x0', caller.calculate_x0(level))
    x1 = kwargs.pop('x1', 0)
    y = level.manifold.level_Hz
    return [[(x0 + x1 - width / 2, y), (x0 + x1 + width / 2, y), kwargs]]


def hf_level_table(caller, level: BaseLevel, **kwargs):
    level = level.manifold

    W = kwargs.pop('width', caller.level_width)
    x0 = kwargs.pop('x0', caller.calculate_x0(level))
    x1 = kwargs.pop('x1', 0)
    offset = kwargs.pop('offset', 0.0)
    F_null = kwargs.pop('F_null', 3.0)
    hf_scale = kwargs.pop('hf_scale', 10000.0)

    b = kwargs.pop('spacing', 1.0)
    if b == 'realistic':
        b = 1.0
    elif b == 'schematic':
        b = 0.0
    else:
        assert 0 <= b <= 1

    a = kwargs.pop('sublevel_width', 1.0)
    if a == 'degeneracy':
        a = 1.0
    elif a == 'schematic':
        a = 0.0
    else:
        assert 0 <= a <= 1

    Fs = [s.term.F for s in level.sublevels()]
    N = len(Fs)

    lvls = []
    for i, sublevel in enumerate(level.sublevels()):
        wF = W * (2 * sublevel.term.F + 1) / (2 * F_null + 1)
        w_null = W - (N - 1) * offset * W
        span = abs(level[f'F={util.float_to_frac(max(Fs))}'].shift_Hz - level[f'F={util.float_to_frac(min(Fs))}'].shift_Hz)
        y = level.level_Hz + hf_scale * (b * sublevel.shift_Hz + (1 - b) * (i - N / 2) / span)
        dx = (1 - a) * w_null / 2 + a * wF / 2
        lvls.append(
            [(offset * W * i - dx + x1 + x0, y),
             (offset * W * i + dx + x1 + x0, y),
             kwargs])
    return lvls


def zeeman_level_table(caller, level: BaseLevel, **kwargs):
    level = level.manifold

    width = kwargs.pop('width', caller.level_width)
    x0 = kwargs.pop('x0', caller.calculate_x0(level))
    x1 = kwargs.pop('x1', 0)
    hf_scale = kwargs.pop('hf_scale', 10000.0)
    z_scale = kwargs.pop('z_scale', 10.0)
    fill_factor = kwargs.pop('fill_factor', 0.9)
    lvls = []

    deg_max = int(2 * (level.term.J + level.atom.I) + 1)

    sublevel_width = fill_factor * width / deg_max
    space_width = (1 - fill_factor) * width / (deg_max - 1)

    for sublevel in level.sublevels():
        for z_level in sublevel.sublevels():
            y = level.level_Hz + hf_scale * sublevel.shift_Hz + z_scale * hf_scale * z_level.shift_Hz
            lvls.append([
                ((x0 + x1) + z_level.term.mF * (sublevel_width + space_width) - sublevel_width / 2, y),
                ((x0 + x1) + z_level.term.mF * (sublevel_width + space_width) + sublevel_width / 2, y),
                kwargs])

    return lvls


class Grotrian:
    level_color = 'k'
    transition_color = 'spectrum'
    level_ordering = 'J'
    level_width = 0.8
    add_level_strategy = gross_level_table
    level_width_fluid = False

    def __init__(self):
        self.level_color = Grotrian.level_color
        self.transition_color = Grotrian.transition_color
        self.level_ordering = Grotrian.level_ordering
        self.level_width = Grotrian.level_width
        self.add_level_strategy = Grotrian.add_level_strategy
        self.level_width_fluid = Grotrian.level_width_fluid

        self.levels_df = pd.DataFrame(columns=['level', 'strategy', 'w', 'x1', 'x0', 'kwargs'])
        self.transition_df = pd.DataFrame(columns=['l0', 'l1', 'substructure', 'bold', 'color_func', 'kwargs'])

    def add_level(self, level, strategy=None, **kwargs):
        if strategy is None:
            strategy = self.add_level_strategy
        width = kwargs.get('width', self.level_width)
        x1 = kwargs.get('position_offset', 0.0)
        x0 = kwargs.get('position_override', None)

        args = {'level': [level], 'strategy': [strategy],
                'w': [width], 'x1': [x1], 'x0': [x0],
                'kwargs': [kwargs]}

        self.levels_df = pd.concat([self.levels_df, pd.DataFrame(args)], ignore_index=True)

    def add_transition(self, l0, l1,
                       substructure=False,
                       bold=False,
                       color_func=None,
                       **kwargs):
        if color_func is None:
            color_func = self.transition_color

        args = {'l0': l0, 'l1': l1,
                'substructure': substructure,
                'bold': bold,
                'color_func': color_func,
                'kwargs': kwargs}
        pd.concat([self.transition_df, pd.DataFrame(args)])

    def calculate_x0(self, level):
        if self.level_ordering == 'J':
            return level.term.J


class MPLGrotrianRenderer:
    @classmethod
    def render(cls, grotrian, axes=None, **kwargs):
        if axes is None:
            fig, axes = plt.subplots()
        cls.render_levels(grotrian, axes)
        cls.render_transitions()

    @classmethod
    def render_levels(cls, grotrian, axes):
        cols = ['p0', 'p1', 'kwargs']
        df = pd.DataFrame(columns=cols)
        for i, row in grotrian.levels_df.iterrows():
            df = pd.concat([df, pd.DataFrame(data=row['strategy'](grotrian, row['level'], **row['kwargs']), columns=cols)])
        coords = list(zip(df['p0'].tolist(), df['p1'].tolist()))

        lc = matplotlib.collections.LineCollection(coords)
        axes.add_collection(lc)

    @classmethod
    def render_transitions(cls):
        pass
