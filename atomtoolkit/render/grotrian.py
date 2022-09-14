from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

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

        self.level_prompts = pd.DataFrame()
        self.transition_prompts = pd.DataFrame()

        self.lines_df = pd.DataFrame(columns=['level', 'show', 'p0', 'p1', 'bbox', 'kwargs'])
        self.transitions_df = pd.DataFrame()

    def add_level(self, level: BaseLevel, strategy: str or Callable = None, **kwargs):
        """
        :param level: the level to be added
        :param strategy: the strategy to employ when drawing the level
            'g' or 'gross': plot the gross structure
            'hf' or 'hyperfine': plot the hyperfine structure
            'z' or 'zeeman': plot the Zeeman structure
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

        args = {'level': [level], 'strategy': [strategy], 'kwargs': [kwargs]}

        self.level_prompts = pd.concat([self.level_prompts, pd.DataFrame(args)], ignore_index=True)

    def add_transition(self, t: BaseTransition, substructure: bool = False, color_func: str = None, **kwargs):
        """

        :param t:
        :param substructure:
        :param color_func:
        :param kwargs:

        :keyword str style:
            'bold'
            'wavy'
        :keyword int linewidth:
        :keyword bool dash:
        :keyword int|str start_anchor:
        1  2  3
        4--5--6
        7  8  9
        :keyword int|str end_anchor:
        :keyword float start_x1:
        :keyword float end_x1:
        :keyword float start_y1:
        :keyword float end_y1:
        CONSIDER: general mpl arrow specs
        :return:
        """
        if color_func is None:
            color_func = self.transition_color

        args = {'t': [t],
                'substructure': [substructure],
                'color_func': [color_func],
                'kwargs': [kwargs]}

        self.transition_prompts = pd.concat([self.transition_prompts, pd.DataFrame(args)])

    def calculate_x0(self, level):
        if self.level_ordering == 'J':
            return level.term.J

    def render(self, axes: plt.Axes = None, **kwargs):
        if axes is None:
            fig, axes = plt.subplots()
        self.render_levels(axes)
        self.render_transitions(axes)
        axes.autoscale()

    @staticmethod
    def bbox(pts: List[Tuple], pad=.05) -> Tuple[float, ...]:
        """returns the bounding box of a given set of points"""
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        y_pad = (max(ys) - min(ys)) * pad
        return min(xs) - pad, min(ys) - y_pad, max(xs) + pad, max(ys) + y_pad

    def render_levels(self, axes):

        def brackets(min_x, min_y, max_x, max_y, h=.05):
            return [[(min_x + h, min_y), (min_x, min_y), (min_x, max_y), (min_x + h, max_y)],
                    [(max_x - h, min_y), (max_x, min_y), (max_x, max_y), (max_x - h, max_y)]]

        # build up a table of all the relevant levels AND sublevels (for bbox/transition reasons)
        for i, row in self.level_prompts.iterrows():
            self.lines_df = pd.concat([self.lines_df, pd.DataFrame(row['strategy'](row['level'], **row['kwargs']))],
                                      ignore_index=True)
        # make bounding boxes and brackets, if desired
        rects_list = []
        bracket_list = []
        for i, row in self.lines_df.iterrows():
            bbox_setting = row['kwargs'].get('bbox', None)
            if bbox_setting is not None:
                x_min, y_min, x_max, y_max = row['bbox']
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
        bracket_colors = [d.get('bracket_color', (0, 0, 0, .8)) for d in self.lines_df['kwargs'] if d.get('bbox', False)]
        bracket_lws = [d.get('bracket_lw', 1) for d in self.lines_df['kwargs'] if d.get('bbox', False)]
        brackets = matplotlib.collections.LineCollection(bracket_list, colors=bracket_colors, linewidths=bracket_lws)
        axes.add_collection(brackets)

        # draw the rectangles
        rects = matplotlib.collections.PatchCollection(rects_list, match_original=True)
        axes.add_collection(rects)

        # draw the levels
        filtered_df = self.lines_df[self.lines_df['show']]
        segments = list(zip(filtered_df['p0'].tolist(), filtered_df['p1'].tolist()))
        colors = [d.get('color', (0, 0, 0, 1)) for d in filtered_df['kwargs']]
        linewidths = [d.get('linewidth', 2) for d in filtered_df['kwargs']]
        lines = matplotlib.collections.LineCollection(segments, colors=colors, linewidths=linewidths)
        axes.add_collection(lines)

        # TODO: Labels (either as box-levels or labels of individual lines, at the gross, F, and mF levels)

    def render_transitions(self, axes):
        for i, row in self.transition_prompts.iterrows():
            axes.add_patch(self.compute_arrow(row['t'], **row['kwargs']))

    def compute_arrow(self, t: BaseTransition or Tuple, **kwargs):
        from matplotlib.patches import FancyArrowPatch, ArrowStyle, Arrow
        if isinstance(t, BaseTransition):
            t = (t.E_lower, t.E_upper)
        a_0 = kwargs.pop('start_anchor', 5)
        a_1 = kwargs.pop('end_anchor', 5)
        x1_0 = kwargs.pop('start_x1', 0)
        x1_1 = kwargs.pop('end_x1', 0)
        y1_0 = kwargs.pop('start_y1', 0)
        y1_1 = kwargs.pop('end_y1', 0)
        bbox_overrides = kwargs.get('override_bbox', (False, False))

        def compute_bbox_pos(bbox: Tuple, anchor: int or Tuple, offset: Tuple):
            pos_dict = {1: (1, 0, 0, 1),   2: (.5, 0, .5, 1),   3: (0, 0, 1, 1),
                        4: (1, .5, 0, .5), 5: (.5, .5, .5, .5), 6: (0, .5, 1, .5),
                        7: (1, 1, 0, 0),   8: (.5, 1, .5, 0),   9: (0, 1, 1, 0)}
            if type(anchor) == int:
                anchor_vec = pos_dict[anchor]
            else:
                anchor_vec = anchor
            return (bbox[0]*anchor_vec[0] + bbox[2]*anchor_vec[2] + offset[0],
                    bbox[1]*anchor_vec[1] + bbox[3]*anchor_vec[3] + offset[1])

        def compute_pos_from_level(level, a, x1, y1, bbox_override=False):
            pos = None
            if type(level) == ZLevel:
                if level.manifold in list(self.lines_df['level']) or bbox_override in {'manifold', 'gross', 'g'}:
                    pos = compute_bbox_pos(list(self.lines_df[self.lines_df['level'] == level.manifold]['bbox'])[0], a, (x1, y1))
                if level.parent in list(self.lines_df['level']) or bbox_override in {'hf', 'hyperfine'}:
                    pos = compute_bbox_pos(list(self.lines_df[self.lines_df['level'] == level.parent]['bbox'])[0], a, (x1, y1))
                if level in list(self.lines_df['level']) and bbox_override is False:
                    pos = compute_bbox_pos(list(self.lines_df[self.lines_df['level'] == level]['bbox'])[0], a, (x1, y1))
            elif type(level) == HFLevel:
                if level.manifold in list(self.lines_df['level']) or bbox_override in {'manifold', 'gross', 'g'}:
                    pos = compute_bbox_pos(list(self.lines_df[self.lines_df['level'] == level.manifold]['bbox'])[0], a, (x1, y1))
                if level in list(self.lines_df['level']) and bbox_override is False:
                    pos = compute_bbox_pos(list(self.lines_df[self.lines_df['level'] == level]['bbox'])[0], a, (x1, y1))
            else:
                pos = compute_bbox_pos(list(self.lines_df[self.lines_df['level'] == level]['bbox'])[0], a, (x1, y1))
            return pos

        pos_a = compute_pos_from_level(t[0], a_0, x1_0, y1_0, bbox_overrides[0])
        pos_b = compute_pos_from_level(t[1], a_1, x1_1, y1_1, bbox_overrides[1])
        # TODO: allow for some common default ArrowStyles. The default FancyArrowPatch settings are horrendous
        return FancyArrowPatch(posA=pos_a, posB=pos_b, **kwargs)

    def _process_level_kwargs(self, level: BaseLevel, kwargs: Dict) -> Tuple:
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
        y0 = kwargs.get('y0', level.manifold.level_Hz/1e12)
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

    def compute_level_row(self, level: EnergyLevel, bbox_override=None, show=True, **kwargs) -> pd.DataFrame:
        width, x0, x1, y0, y1, _, _, _, _, _, _, kwargs = self._process_level_kwargs(level, kwargs)

        y = y0 + y1
        p0 = (x0 + x1 - width / 2, y)
        p1 = (x0 + x1 + width / 2, y)

        if bbox_override is None:
            bbox = self.bbox([p0, p1], pad=kwargs.get('bbox_pad', 0.05))
        else:
            bbox = bbox_override

        return pd.DataFrame({
            'level': [level],
            'show': [show],
            'p0': [p0],
            'p1': [p1],
            'bbox': [bbox],
            'kwargs': [kwargs]})

    def compute_zlevel_row(self, level: ZLevel, show=True, **kwargs) -> pd.DataFrame:
        W, x0, x1, y0, y1, _, squeeze, hf_scale, z_scale, a, b, kwargs = self._process_level_kwargs(level, kwargs)
        fill_factor = kwargs.pop('fill_factor', 0.9)

        sublevel_width = fill_factor * W / squeeze
        space_width = (1 - fill_factor) * W / (squeeze - 1)

        Fs = [s.term.F for s in level.manifold.sublevels()]
        N = len(Fs)
        span = abs(level.manifold[f'F={util.float_to_frac(max(Fs))}'].shift_Hz -
                   level.manifold[f'F={util.float_to_frac(min(Fs))}'].shift_Hz)
        i = level.term.F - min(Fs)

        y = y0 + y1 + hf_scale/1e12 * \
            (b * (level.parent.shift_Hz + z_scale * level.shift_Hz) + (1 - b) * (i - N / 2) * span / N)
        p0 = ((x0 + x1) + level.term.mF * (sublevel_width + space_width) - sublevel_width / 2, y)
        p1 = ((x0 + x1) + level.term.mF * (sublevel_width + space_width) + sublevel_width / 2, y)
        return pd.DataFrame({
            'level': [level],
            'show': [show],
            'p0': [p0],
            'p1': [p1],
            'bbox': [self.bbox([p0, p1], pad=kwargs.get('bbox_pad', 0.05))],
            'kwargs': [kwargs]})

    def compute_hflevel_row(self, level: HFLevel, bbox_override=None, show=True, **kwargs) -> pd.DataFrame:
        W, x0, x1, y0, y1, offset, squeeze, hf_scale, _, a, b, kwargs = self._process_level_kwargs(level, kwargs)

        Fs = [s.term.F for s in level.manifold.sublevels()]
        N = len(Fs)
        span = abs(level.manifold[f'F={util.float_to_frac(max(Fs))}'].shift_Hz -
                   level.manifold[f'F={util.float_to_frac(min(Fs))}'].shift_Hz)
        i = level.term.F - min(Fs)

        w_null = W - (N - 1) * offset * W
        wF = W * (2 * level.term.F + 1) / (2 * squeeze + 1)
        dx = (1 - a) * w_null / 2 + a * wF / 2
        y = y0 + y1 + hf_scale/1e12 * (b * level.shift_Hz + (1 - b) * (i - N / 2) * span / N)
        p0 = (offset * W * i - dx + x1 + x0, y)
        p1 = (offset * W * i + dx + x1 + x0, y)

        if bbox_override is None:
            bbox = self.bbox([p0, p1], pad=kwargs.get('bbox_pad', 0.05))
        else:
            bbox = bbox_override

        return pd.DataFrame({
            'level': [level],
            'show': [show],
            'p0': [p0],
            'p1': [p1],
            'bbox': [bbox],
            'kwargs': [kwargs]})

    def gross_level_table(self, level: BaseLevel, **kwargs) -> pd.DataFrame:
        """
        Generate the points that plot the gross fine structure energy level (a single horizontal line)
        - 2P3/2
        :param self: the Grotrian object that is being pulled from. Used for accessing defaults
        :param level: the level being considered
        :param kwargs: forwarded to compute_level_position
        """
        level = level.manifold

        return self.compute_level_row(level, **kwargs)

    def hf_level_table(self, level: BaseLevel, **kwargs) -> pd.DataFrame:
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

        if isinstance(level, EnergyLevel):
            sub_tables = []
            for i, sublevel in enumerate(level.sublevels()):
                sub_kwargs = kwargs.copy()
                sub_kwargs['bbox'] = None
                sub_tables.append(self.hf_level_table(sublevel, **sub_kwargs))
            table = pd.concat(sub_tables, ignore_index=True)
            table = pd.concat([table,
                               self.compute_level_row(level,
                                                      bbox_override=self.bbox(
                                                          (list(table['p0'])+list(table['p1']))),
                                                      show=False,
                                                      **kwargs)], ignore_index=True)
        elif isinstance(level, HFLevel):
            table = self.compute_hflevel_row(level, **kwargs)

        return table

    def zeeman_level_table(self, level: BaseLevel, **kwargs) -> pd.DataFrame:
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
        if isinstance(level, EnergyLevel):
            sub_tables = []
            for sublevel in level.sublevels():
                sub_kwargs = kwargs.copy()
                sub_kwargs['bbox'] = None
                sub_tables.append(self.zeeman_level_table(sublevel, **sub_kwargs))
            table = pd.concat(sub_tables, ignore_index=True)
            table = pd.concat([table,
                               self.compute_level_row(level,
                                                      bbox_override=self.bbox(
                                                          (list(table['p0']) + list(table['p1']))),
                                                      show=False,
                                                      **kwargs)],
                              ignore_index=True)

        elif type(level) == HFLevel:
            sub_tables = []
            for z_level in level.sublevels():
                sub_kwargs = kwargs.copy()
                sub_kwargs['bbox'] = None
                sub_tables.append(self.zeeman_level_table(z_level, **sub_kwargs))
            table = pd.concat(sub_tables, ignore_index=True)
            table = pd.concat([table,
                               self.compute_hflevel_row(level,
                                                        bbox_override=self.bbox(
                                                            (list(table['p0']) + list(table['p1']))),
                                                        show=False,
                                                        **kwargs)],
                              ignore_index=True)
        elif type(level) == ZLevel:
            table = self.compute_zlevel_row(level, **kwargs)

        return table
