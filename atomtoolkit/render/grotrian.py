"""
Eventually meant for drawing Grotrian diagrams, perhaps even modifiable ones
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib

from atomtoolkit.atom import *
import atomtoolkit.util as util


def draw_levels(atom: Atom, plot_type: str = 'norm', **kwargs):
    """
    Draws a quick and dirty grotrian diagram.
    Placeholder for a more complete grotrian functionality
    """
    positions = {l.name: (l.term.J, l.level_Hz/1e12) for l in atom.levels.values()}
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
    nx.draw(model, pos=positions, node_shape="_", with_labels=True, font_size=8, edge_color=(0, 0, 0, 0.2), **kwargs)


RENDER_METHODS = {'matplotlib', 'bokeh'}
RENDERER = 'matplotlib'

CoordinateList = List[List[Union[Tuple[float, float], Dict]]]


def _process_kwargs(caller: Grotrian, level: BaseLevel, kwargs: Dict):
    W = kwargs.pop('width', caller.level_width)
    x0 = kwargs.pop('x0', caller.calculate_x0(level))
    x1 = kwargs.pop('x1', 0)
    offset = kwargs.pop('offset', 0.0)

    squeeze = kwargs.pop('squeeze', int((level.term.J + level.atom.I)))
    squeeze = 2 * squeeze + 1

    hf_scale = kwargs.pop('hf_scale', 10000.0)
    z_scale = kwargs.pop('z_scale', 5.0)

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

    return W, x0, x1, offset, squeeze, hf_scale, z_scale, a, b, kwargs


def bbox(l: CoordinateList) -> Tuple[float, ...]:
    """returns the bounding box of a given CoordinateList"""
    xs = [row[0][0] for row in l] + [row[1][0] for row in l]
    ys = [row[0][1] for row in l] + [row[1][1] for row in l]
    return min(xs), min(ys), max(xs), max(ys)


def gross_level_table(caller: Grotrian, level: BaseLevel, **kwargs) -> CoordinateList:
    level = level.manifold

    width, x0, x1, _, _, _, _, _, _, kwargs = _process_kwargs(caller, level, kwargs)

    y = level.manifold.level_Hz
    return [[(x0 + x1 - width / 2, y), (x0 + x1 + width / 2, y), kwargs]]


def hf_level_table(caller: Grotrian, level: BaseLevel, **kwargs) -> CoordinateList:
    W, x0, x1, offset, squeeze, hf_scale, _, a, b, kwargs = _process_kwargs(caller, level, kwargs)

    Fs = [s.term.F for s in level.manifold.sublevels()]
    N = len(Fs)
    span = abs(level.manifold[f'F={util.float_to_frac(max(Fs))}'].shift_Hz -
               level.manifold[f'F={util.float_to_frac(min(Fs))}'].shift_Hz)
    w_null = W - (N - 1) * offset * W

    table = []
    if isinstance(level, EnergyLevel):
        for i, sublevel in enumerate(level.sublevels()):
            wF = W * (2 * sublevel.term.F + 1) / (2 * squeeze + 1)
            dx = (1 - a) * w_null / 2 + a * wF / 2
            y = level.level_Hz + hf_scale * (b * sublevel.shift_Hz + (1 - b) * (i - N / 2) * span / N)
            table.append(
                [(offset * W * i - dx + x1 + x0, y),
                 (offset * W * i + dx + x1 + x0, y),
                 kwargs])
    elif isinstance(level, HFLevel):
        i = level.term.F - min(Fs)
        wF = W * (2 * level.term.F + 1) / (2 * squeeze + 1)
        dx = (1 - a) * w_null / 2 + a * wF / 2
        y = level.manifold.level_Hz + hf_scale * (b * level.shift_Hz + (1 - b) * (i - N / 2) * span / N)
        table.append(
            [(offset * W * i - dx + x1 + x0, y),
             (offset * W * i + dx + x1 + x0, y),
             kwargs])
    return table


def zeeman_level_table(caller: Grotrian, level: BaseLevel, **kwargs) -> CoordinateList:
    fill_factor = kwargs.pop('fill_factor', 0.9)
    W, x0, x1, offset, squeeze, hf_scale, z_scale, a, b, kwargs = _process_kwargs(caller, level, kwargs)

    sublevel_width = fill_factor * W / squeeze
    space_width = (1 - fill_factor) * W / (squeeze - 1)

    Fs = [s.term.F for s in level.manifold.sublevels()]
    N = len(Fs)
    span = abs(level.manifold[f'F={util.float_to_frac(max(Fs))}'].shift_Hz -
               level.manifold[f'F={util.float_to_frac(min(Fs))}'].shift_Hz)

    table = []
    # TODO: how to reduce code duplication here without a weird one-off function?
    if isinstance(level, EnergyLevel):
        for i, sublevel in enumerate(level.sublevels()):
            for z_level in sublevel.sublevels():
                y = level.level_Hz + hf_scale * \
                    (b * (sublevel.shift_Hz + z_scale * z_level.shift_Hz) + (1 - b) * (i - N / 2) * span / N)
                table.append([
                    ((x0 + x1) + z_level.term.mF * (sublevel_width + space_width) - sublevel_width / 2, y),
                    ((x0 + x1) + z_level.term.mF * (sublevel_width + space_width) + sublevel_width / 2, y),
                    kwargs])
    elif type(level) == HFLevel:
        i = level.term.F - min(Fs)
        for z_level in level.sublevels():
            y = level.manifold.level_Hz + hf_scale * \
                (b * (level.shift_Hz + z_scale * z_level.shift_Hz) + (1 - b) * (i - N / 2) * span / N)
            table.append([
                ((x0 + x1) + z_level.term.mF * (sublevel_width + space_width) - sublevel_width / 2, y),
                ((x0 + x1) + z_level.term.mF * (sublevel_width + space_width) + sublevel_width / 2, y),
                kwargs])
    elif type(level) == ZLevel:
        i = level.term.F - min(Fs)
        y = level.manifold.level_Hz + hf_scale * \
            (b * (level.parent.shift_Hz + z_scale * level.shift_Hz) + (1 - b) * (i - N / 2) * span / N)
        table.append([
            ((x0 + x1) + level.term.mF * (sublevel_width + space_width) - sublevel_width / 2, y),
            ((x0 + x1) + level.term.mF * (sublevel_width + space_width) + sublevel_width / 2, y),
            kwargs])

    return table


class Grotrian:
    level_color = (0, 0, 0, 1)
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

    def add_level(self, level: BaseLevel, strategy: Callable = None, **kwargs):
        if strategy is None:
            strategy = self.add_level_strategy
        width = kwargs.get('width', self.level_width)
        x1 = kwargs.get('position_offset', 0.0)
        x0 = kwargs.get('position_override', None)

        args = {'level': [level], 'strategy': [strategy],
                'w': [width], 'x1': [x1], 'x0': [x0],
                'kwargs': [kwargs]}

        self.levels_df = pd.concat([self.levels_df, pd.DataFrame(args)], ignore_index=True)

    def add_transition(self, t: BaseTransition,
                       substructure: bool = False,
                       bold: bool = False,
                       color_func: Callable = None,
                       **kwargs):
        if color_func is None:
            color_func = self.transition_color

        args = {'t': t,
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
    def render(cls, grotrian: Grotrian, axes: plt.Axes = None, **kwargs):
        if axes is None:
            fig, axes = plt.subplots()
        cls.render_levels(grotrian, axes, **kwargs)
        cls.render_transitions()

    @classmethod
    def render_levels(cls, grotrian, axes, **kwargs):
        cols = ['p0', 'p1', 'kwargs']
        df = pd.DataFrame(columns=cols)
        for i, row in grotrian.levels_df.iterrows():
            df = pd.concat([df, pd.DataFrame(data=row['strategy'](grotrian, row['level'], **row['kwargs']),
                                             columns=cols)], ignore_index=True)
        segments = list(zip(df['p0'].tolist(), df['p1'].tolist()))
        colors = [d.get('color', (0, 0, 0, 1)) for d in df['kwargs']]
        linewidths = [d.get('linewidth', 2) for d in df['kwargs']]

        lc = matplotlib.collections.LineCollection(segments, colors=colors, linewidths=linewidths)
        axes.add_collection(lc)

        # TODO: Bounding boxes (either as boxes or brackets)
        # TODO: Labels (either as box-levels or labels of individual lines, at the gross, F, and mF levels)

    @classmethod
    def render_transitions(cls):
        pass
