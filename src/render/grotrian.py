"""
Eventually meant for drawing Geotrian diagrams, perhaps even modifiable ones
"""

import matplotlib.pyplot as plt
import networkx as nx

def draw_levels(atom, plot_type='norm', **kwargs):
    """
    Draws a quick and dirty grotrian diagram.
    Placeholder for a more complete grotrian functionality
    """
    posdict = {l.name: (l.term.J, l.level_Hz) for l in atom.levels.values()}
    if plot_type == 'hf':
        model = atom.hfModel.copy()
    elif plot_type == 'z':
        model = atom.zModel.copy()
    else:
        model = atom.levelsModel.copy()
    for node in model.nodes():
        try:
            model.remove_edge(node, node)
        except KeyError:
            pass
    nx.draw(model, pos=posdict, node_shape="_", with_labels=True, font_size=8, edge_color=(0., 0., 0., 0.2), **kwargs)
    plt.show()


# class Grotrian:
#     def __init__(self):
#         self.ax = plt.ax
#
#
# class LevelRenderer:
#
#     def __init__(self, level, parent):
#         self.level = level
#         self.parent = parent
#         self.ax = self.parent.ax
#         self.modes = {self.draw_level, self.draw_hf, self.draw_z, self.draw_abstract}
#         self.draw_mode = self.draw_level
#
#     def mode(self, mode):
#         if mode in self.modes:
#             self.draw_mode = mode
#         else:
#             raise KeyError(f"mode not permitted. please use a mode in {self.modes}")
#
#     def draw_level(self):
#         self.ax.hlines(self.level.level_hz, self.level.term.J-0.5, self.level.term.J+0.5)
#
#     def draw_hf(self):
#         self.ax.hlines([hflevel.level_hz for hflevel in self.level.sublevels], self.level.term.J - 0.5, self.level.term.J + 0.5)
#
#     # def draw_z(self):
#     #     for hflevel in self.level.sublevels:
#     #         self.ax.hlines([zlevel.level_hz for zlevel in hflevel.sublevels], )
#
#
