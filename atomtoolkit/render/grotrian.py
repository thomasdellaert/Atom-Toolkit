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


# RENDER_METHODS = {'matplotlib', 'bokeh'}
# RENDERER = 'matplotlib'

CoordinateList = List[List[Union[Tuple[float, float], Dict]]]


class Grotrian:
    level_color = (0, 0, 0, 1)
    transition_color = 'spectrum'
    level_ordering = 'J'
    level_width = 0.8
    add_level_strategy = 'gross'
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

    def add_level(self, level: BaseLevel, strategy: str = None, **kwargs):
        """
        :param level: the level to be added
        :param strategy: the strategy to employ - whether to draw just the gross structure, the hyperfine structure,
            or the full Zeeman structure
        :param kwargs: see below

        :keyword float W: the overall width of the drawing. default: 0.8
        :keyword float x0: override the absolute x-position of the level to be drawn. By default, done in plot
            coordinates but (I think) can be changed to absolute coordinates with matplotlib's transform. default: None
        :keyword float x1: shift the absolute x-position of the level to be drawn. default: 0
        :keyword float y0: override the absolute y-position of the level to be drawn. By default, done in plot
            coordinates but (I think) can be changed to absolute coordinates with matplotlib's transform. default: None
        :keyword float y1: shift the absolute y-position of the level to be drawn. default: 0
        :keyword float offset: if plotting HF structure, the amount by which to horizontally shift each level,
            as a fraction of the total width. default: 0 CONSIDER: renaming 'offset' keyword
        :keyword float hf_scale: the factor by which to scale the hyperfine structure. default: 10000.0
        :keyword float z_scale: the factor by which to *additionally* scale the Zeeman structure. default: 5.0
        :keyword float|bool squeeze: whether to squeeze the hyperfine/zeeman structure into the width. By default,
            all hyperfine structure will have width W. If squeeze is set, the F=[squeeze] hyperfine sublevel will
            have width W. If squeeze is False, all Zeeman sublevels will have the same width as the ground state
        :keyword float|str spacing: how to space out the sublevels. default: 'realistic'
            'realistic': plot an accurate representation of the structure
            'schematic': plot the sublevels as being equally-spaced
            float: interpolate between realistic (1.0) and schematic (0.0) spacings
        :keyword float|str sublevel_width: what widths to draw hyperfine sublevels. default: 'degeneracy'
            'degeneracy': the width of sublevel F = 2F+1
            'schematic': all sublevel widths are equal to W
            float: interpolate between degeneracy (1.0) and schematic (0.0) widths
        :keyword str bbox: if given, a bounding box will be drawn around the level. default: None
            'rect' gives a rectangular bounding box
            'bracket' draws square brackets around the level
        :keyword int linewidth: the linewidth which which to draw the level. default: 2
        :keyword tuple color: an RGBA tuple to set the color of the level default (black): (0, 0, 0, 1)
        :keyword float bbox_pad: the amount of extra space to pad the bbox with. this corresponds to the exact value
            in the x axis, and the percentage value of hyperfine splitting in the y axis
        :keyword float rect_alpha: the transparency of the rectangle bbox. default: 0.2
        :keyword tuple rect_color: tne color to draw the rectangle bbox. default: None [i.e. matplotlib handles it]
        :keyword bool rect_fill: whether to fill the rectangle bbox. default: True
        :keyword float bracket_h: the length of the bracket 'ticks'. default: 0.05
        :keyword int bracket_lw: the linewidth wo draw the bracket with. default: 1
        :keyword tuple bracket_color: the bracket color. default: (0, 0, 0, 0.8)

        :return:
        """
        if strategy is None:
            strategy = self.add_level_strategy
        if strategy == 'g' or strategy == 'gross':
            strategy = self.gross_level_table
        elif strategy == 'hf' or strategy == 'hyperfine':
            strategy = self.hf_level_table
        elif strategy == 'z' or strategy == 'zeeman':
            strategy = self.zeeman_level_table

        width = kwargs.get('width', self.level_width)
        x1 = kwargs.get('x1', 0.0)
        x0 = kwargs.get('x0', None)
        y1 = kwargs.get('y1', 0.0)
        y0 = kwargs.get('y0', None)

        args = {'level': [level], 'strategy': [strategy],
                'w': [width], 'x1': [x1], 'x0': [x0], 'y1': [y1], 'y0': [y0],
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

    def render(self, axes: plt.Axes = None, **kwargs):
        if axes is None:
            fig, axes = plt.subplots()
        self.render_levels(axes, **kwargs)
        self.render_transitions()

    def render_levels(self, axes, **kwargs):

        def bbox(d: pd.DataFrame, pad=.05) -> Tuple[float, ...]:
            """returns the bounding box of a given set of points"""
            xs = [x[0] for x in d['p0']] + [x[0] for x in d['p1']]
            ys = [x[1] for x in d['p0']] + [x[1] for x in d['p1']]
            y_pad = (max(ys) - min(ys)) * pad
            return min(xs)-pad, min(ys)-y_pad, max(xs)+pad, max(ys)+y_pad

        def brackets(min_x, min_y, max_x, max_y, h=.05):
            return [[(min_x + h, min_y), (min_x, min_y), (min_x, max_y), (min_x + h, max_y)],
                    [(max_x - h, min_y), (max_x, min_y), (max_x, max_y), (max_x - h, max_y)]]

        # generate the dataframe for the level drawing
        cols = ['p0', 'p1', 'kwargs']
        lines_df = pd.DataFrame(columns=cols)
        rects_list = []
        bracket_list = []
        for i, row in self.levels_df.iterrows():
            draw_df = pd.DataFrame(data=row['strategy'](row['level'], **row['kwargs']), columns=cols)
            lines_df = pd.concat([lines_df, draw_df], ignore_index=True)

        # make bounding boxes, if desired
            bbox_setting = row['kwargs'].get('bbox', None)
            if bbox_setting is not None:
                x_min, y_min, x_max, y_max = bbox(draw_df, pad=row['kwargs'].get('bbox_pad', 0.05))

                if bbox_setting == 'rect' or bbox_setting == 'rectangle':
                    from matplotlib.patches import Rectangle
                    rects_list.append(Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                                alpha=row['kwargs'].get('rect_alpha', 0.2),
                                                color=row['kwargs'].get('rect_color', None),
                                                fill=row['kwargs'].get('rect_fill', True)
                                                ))
                elif bbox_setting == 'bracket':
                    bracket_list += (brackets(x_min, y_min, x_max, y_max,
                                              h=row['kwargs'].get('bracket_h', 0.05)))

        # draw the brackets
        bracket_colors = [d.get('bracket_color', (0, 0, 0, .8)) for d in lines_df['kwargs'] if d.get('bbox', False)]
        bracket_lws = [d.get('bracket_lw', 1) for d in lines_df['kwargs'] if d.get('bbox', False)]
        brackets = matplotlib.collections.LineCollection(bracket_list, colors=bracket_colors, linewidths=bracket_lws)
        axes.add_collection(brackets)
        # draw the rectangles
        rects = matplotlib.collections.PatchCollection(rects_list, match_original=True)
        axes.add_collection(rects)
        # draw the levels
        segments = list(zip(lines_df['p0'].tolist(), lines_df['p1'].tolist()))
        colors = [d.get('color', (0, 0, 0, 1)) for d in lines_df['kwargs']]
        linewidths = [d.get('linewidth', 2) for d in lines_df['kwargs']]
        lines = matplotlib.collections.LineCollection(segments, colors=colors, linewidths=linewidths)
        axes.add_collection(lines)

        # TODO: Labels (either as box-levels or labels of individual lines, at the gross, F, and mF levels)

    def render_transitions(self):
        pass

    def _process_kwargs(self, level: BaseLevel, kwargs: Dict):
        """
        Handle all the keyword arguments that the Grotrian coordinate generating functions require, and pass the rest
        along. For an explanation of the kwargs, see the add_level docstring
        :param self: the Grotrian object that is being pulled from. Used for accessing defaults
        :param level: the level being considered

        :return: all the keyword arguments, as the generating functions expect
        """
        W = kwargs.get('width', self.level_width)
        x0 = kwargs.get('x0', self.calculate_x0(level))
        x1 = kwargs.get('x1', 0)
        y0 = kwargs.get('y0', level.manifold.level_Hz)
        y1 = kwargs.get('y1', 0)
        offset = kwargs.get('offset', 0.0)

        squeeze = kwargs.get('squeeze', int((level.term.J + level.atom.I)))
        if squeeze is True:
            squeeze = int((level.term.J + level.atom.I))
        elif squeeze is False:
            squeeze = max(l.term.F for l in level.atom.levels[0].values())
        squeeze = int(2 * squeeze + 1)

        hf_scale = kwargs.get('hf_scale', 10000.0)
        z_scale = kwargs.get('z_scale', 5.0)

        b = kwargs.get('spacing', 1.0)
        if b == 'realistic':
            b = 1.0
        elif b == 'schematic':
            b = 0.0
        else:
            assert 0 <= b <= 1

        a = kwargs.get('sublevel_width', 1.0)
        if a == 'degeneracy':
            a = 1.0
        elif a == 'schematic':
            a = 0.0
        else:
            assert 0 <= a <= 1

        return W, x0, x1, y0, y1, offset, squeeze, hf_scale, z_scale, a, b, kwargs

    def compute_level_position(self, level: EnergyLevel, **kwargs):
        width, x0, x1, y0, y1, _, _, _, _, _, _, kwargs = self._process_kwargs(level, kwargs)

        y = y0 + y1
        return [(x0 + x1 - width / 2, y), (x0 + x1 + width / 2, y), kwargs]

    def compute_zlevel_position(self, level: ZLevel, **kwargs):
        W, x0, x1, y0, y1, _, squeeze, hf_scale, z_scale, a, b, kwargs = self._process_kwargs(level, kwargs)
        fill_factor = kwargs.pop('fill_factor', 0.9)

        sublevel_width = fill_factor * W / squeeze
        space_width = (1 - fill_factor) * W / (squeeze - 1)

        Fs = [s.term.F for s in level.manifold.sublevels()]
        N = len(Fs)
        span = abs(level.manifold[f'F={util.float_to_frac(max(Fs))}'].shift_Hz -
                   level.manifold[f'F={util.float_to_frac(min(Fs))}'].shift_Hz)
        i = level.term.F - min(Fs)

        y = y0 + y1 + hf_scale * \
            (b * (level.parent.shift_Hz + z_scale * level.shift_Hz) + (1 - b) * (i - N / 2) * span / N)
        return [((x0 + x1) + level.term.mF * (sublevel_width + space_width) - sublevel_width / 2, y),
                ((x0 + x1) + level.term.mF * (sublevel_width + space_width) + sublevel_width / 2, y),
                kwargs]

    def compute_hflevel_position(self, level: HFLevel, **kwargs):
        W, x0, x1, y0, y1, offset, squeeze, hf_scale, _, a, b, kwargs = self._process_kwargs(level, kwargs)
        Fs = [s.term.F for s in level.manifold.sublevels()]
        N = len(Fs)
        span = abs(level.manifold[f'F={util.float_to_frac(max(Fs))}'].shift_Hz -
                   level.manifold[f'F={util.float_to_frac(min(Fs))}'].shift_Hz)
        w_null = W - (N - 1) * offset * W
        i = level.term.F - min(Fs)
        wF = W * (2 * level.term.F + 1) / (2 * squeeze + 1)
        dx = (1 - a) * w_null / 2 + a * wF / 2
        y = y0 + y1 + hf_scale * (b * level.shift_Hz + (1 - b) * (i - N / 2) * span / N)
        return [(offset * W * i - dx + x1 + x0, y),
                (offset * W * i + dx + x1 + x0, y),
                kwargs]

    def gross_level_table(self, level: BaseLevel, **kwargs) -> CoordinateList:
        """
        Generate the points that plot the gross fine structure energy level (a single horizontal line)
        - 2P3/2
        :param self: the Grotrian object that is being pulled from. Used for accessing defaults
        :param level: the level being considered
        :param kwargs: forwarded to compute_level_position
        """
        level = level.manifold

        return [self.compute_level_position(level, **kwargs)]

    def hf_level_table(self, level: BaseLevel, **kwargs) -> CoordinateList:
        """
        Generate the points that plot the hyperfine structure of the energy level (generally I+J - |I-J| horizontal lines)
        - 2P3/2 F=3
        - 2P3/2 F=2
        - 2P3/2 F=1
        - 2P3/2 F=0
        :param self: the Grotrian object that is being pulled from. Used for accessing defaults
        :param level: the level being considered
        :param kwargs: forwarded to compute_hflevel_position

        :return: a list containing pairs of points to be rendered, plus any residual keyword arguments
        """

        table = []
        if isinstance(level, EnergyLevel):
            for i, sublevel in enumerate(level.sublevels()):
                table.append(self.compute_hflevel_position(sublevel, **kwargs))
        elif isinstance(level, HFLevel):
            table.append(self.compute_hflevel_position(level, **kwargs))
        return table

    def zeeman_level_table(self, level: BaseLevel, **kwargs) -> CoordinateList:
        """
        Generate the points to plot the Zeeman structure of the level (2F+1 horizontal lines per hyperfine sublevel)
        ------- 2P3/2 F=3 mF = -3, -2, -1, 0, 1, 2, 3
         -----  2P3/2 F=2 mF = -2, -1, 0, 1, 2
          ---   2P3/2 F=1 mF = -1, 0, 1
           -    2P3/2 F=0 mF = 0
        :param self: the Grotrian object that is being pulled from. Used for accessing defaults
        :param level: the level being considered
        :param kwargs: forwarded to compute_zlevel_position

        :return: a list containing pairs of points to be rendered, plus any residual keyword arguments
        """

        table = []
        if isinstance(level, EnergyLevel):
            for sublevel in level.sublevels():
                for z_level in sublevel.sublevels():
                    table.append(self.compute_zlevel_position(z_level, **kwargs))
        elif type(level) == HFLevel:
            for z_level in level.sublevels():
                table.append(self.compute_zlevel_position(z_level, **kwargs))
        elif type(level) == ZLevel:
            table.append(self.compute_zlevel_position(level, **kwargs))

        return table
