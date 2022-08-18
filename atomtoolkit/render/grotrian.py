"""
Eventually meant for drawing Geotrian diagrams, perhaps even modifiable ones
"""

import networkx as nx
import pandas as pd


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


class Grotrian:
    level_color = 'k'
    transition_color = 'spectrum'
    level_ordering = 'J'
    level_width = 0.8
    add_level_behavior = 'gross'
    level_width_fluid = False

    def __init__(self):
        self.level_color = Grotrian.level_color
        self.transition_color = Grotrian.transition_color
        self.level_ordering = Grotrian.level_ordering
        self.level_width = Grotrian.level_width
        self.add_level_behavior = Grotrian.add_level_behavior
        self.level_width_fluid = Grotrian.level_width_fluid

        self._add_level_behavior = {'hf': False, 'z': False}
        if self.add_level_behavior == 'hyperfine':
            self._add_level_behavior = {'hf': True, 'z': False}
        elif self.add_level_behavior == 'zeeman':
            self._add_level_behavior = {'hf': True, 'z': True}

        self.levels_df = pd.DataFrame(columns=['level', 'hf', 'hfs', 'z', 'zs', 'w', 'x0', 'x1', 'kwargs'])
        self.transition_df = pd.DataFrame(columns=['l0', 'l1', 'substructure', 'bold', 'color_func', 'kwargs'])

    def add_level(self, level,
                  hyperfine=None, hf_scale_factor=1e4,
                  zeeman=None, z_scale_factor=1e3,
                  width=None,
                  position_offset=(0., 0.),
                  position_override=None,
                  **kwargs):
        if hyperfine is None: hyperfine = self._add_level_behavior['hf']
        if zeeman is None: zeeman = self._add_level_behavior['z']
        if width is None: width = self.level_width

        args = {'level': level,
                'hf': hyperfine, 'hfs': hf_scale_factor,
                'z': zeeman, 'zs': z_scale_factor,
                'w': width,
                'x1': position_offset, 'x0': position_override,
                'kwargs': kwargs}
        pd.concat(self.levels_df, pd.DataFrame(args))

    def add_transition(self, l0, l1,
                       substructure=False,
                       bold=False,
                       color_func=None,
                       **kwargs):
        if color_func is None: color_func = self.transition_color

        args = {'l0': l0, 'l1': l1,
                'substructure': substructure,
                'bold': bold,
                'color_func': color_func,
                'kwargs': kwargs}
        pd.concat(self.transition_df, pd.DataFrame(args))

    def calculate_x0(self, level):
        if self.level_ordering == 'J':
            return level.J

    def compute_lines_gross(self, level, x0, x1, width):
        level = level.manifold
        if x0 is None:
            x0 = self.calculate_x0(level)

        return [[level.manifold.level_Hz, (x0 + x1) - width / 2, (x0 + x1) + width / 2]]

    def compute_lines_hf(self, level, x0, x1, width, scale_factor):
        level = level.manifold
        lvls = []
        if x0 is None:
            x0 = self.calculate_x0(level.manifold)

        for sublevel in level.sublevels():
            lvls.append([level.level_Hz + scale_factor * sublevel.shift_Hz, (x0 + x1) - width / 2, (x0 + x1) + width / 2])

        return lvls

    def compute_lines_z(self, level, x0, x1, width, fill_factor, hf_scale_factor, z_scale_factor):
        level = level.manifold
        lvls = []
        if x0 is None:
            x0 = self.calculate_x0(level.manifold)

        deg_max = int(2 * (level.term.J + level.atom.I) + 1)

        sublevel_width = fill_factor * width / deg_max
        space_width = (1 - fill_factor) * width / (deg_max - 1)

        for sublevel in level.sublevels():
            for z_level in sublevel.sublevels():
                lvls.append([
                    level.level_Hz + hf_scale_factor * sublevel.shift_Hz + z_scale_factor * z_level.shift_Hz,
                    (x0 + x1) + z_level.term.mF * (sublevel_width + space_width) - sublevel_width / 2,
                    (x0 + x1) + z_level.term.mF * (sublevel_width + space_width) + sublevel_width / 2
                ])

        return lvls


class MPLGrotrianRenderer:
    def render(self, grotrian, fig=None, axes=None, **kwargs):
        self.render_levels()
        self.render_transitions()

    def render_levels(self):
        pass

    def render_transitions(self):
        pass
