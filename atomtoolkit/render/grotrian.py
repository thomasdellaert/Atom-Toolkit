"""
Eventually meant for drawing Geotrian diagrams, perhaps even modifiable ones
"""

import matplotlib
import networkx as nx
import pandas as pd

def draw_levels(atom, plot_type='norm', **kwargs):
    """
    Draws a quick and dirty grotrian diagram.
    Placeholder for a more complete grotrian functionality
    """
    posdict = {l.name: (l.term.J, l.level_Hz) for l in atom.levels.values()}
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

RENDER_METHODS={'matplotlib', 'bokeh'}
RENDERER = 'matplotlib'

class Grotrian:
    level_color = 'k'
    transition_color = 'spectrum'
    level_ordering = 'J'
    level_width = 0.8
    add_level_behavior = 'gross'

    def __init__(self):
        self.level_color = Grotrian.level_color
        self.transition_color = Grotrian.transition_color
        self.level_ordering = Grotrian.level_ordering
        self.level_width = Grotrian.level_width
        self.add_level_behavior = Grotrian.add_level_behavior

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
                  position_override=None, **kwargs):
        if hyperfine is None: hyperfine = self._add_level_behavior['hf']
        if zeeman is None: zeeman = self._add_level_behavior['z']
        if width is None: width = self.level_width

        args = {'level': level,
                'hf':hyperfine, 'hfs': hf_scale_factor,
                'z': zeeman, 'zs': z_scale_factor,
                'w': width,
                'x1': position_offset, 'x0': position_override,
                'kwargs': kwargs}
        pd.concat(self.levels_df, pd.DataFrame(args))

    def add_transition(self, l0, l1, substructure=False, bold=False, color_func=None, **kwargs):
        if color_func is None: color_func = self.transition_color

        args = {'l0': l0, 'l1': l1,
                'substructure':substructure,
                'bold': bold,
                'color_func': color_func,
                'kwargs': kwargs}
        pd.concat(self.transition_df, pd.DataFrame(args))


class MPLGrotrianRenderer:
    def render(self, grotrian, fig=None, axes=None, **kwargs):
        self.render_levels()
        self.render_transitions()
