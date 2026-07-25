"""Emit a :class:`~.layout.Scene` as SVG.

Deliberately dumb: no decisions are made here, only element emission. That keeps
golden-file tests meaningful -- a change in output means a change in layout, not a change
in drawing taste.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import svgwrite

from .layout import Scene

__all__ = ['render_svg']

#: Baseline offsets for super/subscripts, in em. SVG `dy` is cumulative, so each run
#: shifts *relative* to the previous one and we track the running offset.
_SHIFT = {'normal': 0.0, 'super': -0.42, 'sub': 0.30}
_SIZE = {'normal': 1.0, 'super': 0.72, 'sub': 0.72}


def _add_runs(text_element, runs: Sequence[Tuple[str, str]], size: float) -> None:
    """Append (text, role) runs to an SVG <text> as <tspan>s with baseline shifts."""
    current = 0.0
    for content, role in runs:
        target = _SHIFT.get(role, 0.0)
        span = svgwrite.text.TSpan(content, dy=[f'{(target - current) * size:.2f}'])
        span.update({'font-size': f'{size * _SIZE.get(role, 1.0):.2f}px'})
        text_element.add(span)
        current = target
    if current:                     # return to the baseline so following text is unaffected
        tail = svgwrite.text.TSpan('', dy=[f'{-current * size:.2f}'])
        text_element.add(tail)


def render_svg(scene: Scene, precision: int = 2) -> str:
    """Render a scene to an SVG document string.

    :param scene: the positioned diagram
    :param precision: decimal places for coordinates
    :return: a complete standalone ``<svg>`` document
    """
    theme = scene.theme
    dwg = svgwrite.Drawing(size=(f'{scene.width:.0f}px', f'{scene.height:.0f}px'),
                           viewBox=f'0 0 {scene.width:.0f} {scene.height:.0f}',
                           profile='full')
    dwg.attribs['font-family'] = theme.font_family

    def r(value: float) -> float:
        return round(value, precision)

    if theme.background:
        dwg.add(dwg.rect(insert=(0, 0), size=(scene.width, scene.height),
                         fill=theme.background))

    if scene.title:
        dwg.add(dwg.text(scene.title,
                         insert=(r(theme.margin_left), r(theme.margin_top - 34)),
                         font_size=f'{theme.title_size}px',
                         fill=theme.label_color,
                         font_weight='600'))

    # --- axis --------------------------------------------------------------
    if theme.show_axis and scene.ticks:
        axis = dwg.g(stroke=theme.axis_color, fill=theme.axis_color,
                     stroke_width=1)
        axis.add(dwg.line(start=(r(scene.axis_x), r(scene.plot_top)),
                          end=(r(scene.axis_x), r(scene.plot_bottom))))
        for tick in scene.ticks:
            axis.add(dwg.line(start=(r(scene.axis_x - 5), r(tick.y)),
                              end=(r(scene.axis_x), r(tick.y))))
            axis.add(dwg.text(tick.label,
                              insert=(r(scene.axis_x - 8), r(tick.y + 3.6)),
                              text_anchor='end', stroke='none',
                              font_size=f'{theme.axis_size}px'))
            if theme.show_grid:
                axis.add(dwg.line(start=(r(scene.axis_x), r(tick.y)),
                                  end=(r(scene.width - theme.margin_right), r(tick.y)),
                                  stroke=theme.grid_color))
        dwg.add(axis)

        label_y = (scene.plot_top + scene.plot_bottom) / 2
        label_x = scene.axis_x - 46
        dwg.add(dwg.text(scene.axis_label,
                         insert=(r(label_x), r(label_y)),
                         text_anchor='middle',
                         fill=theme.axis_color,
                         font_size=f'{theme.axis_size}px',
                         transform=f'rotate(-90 {r(label_x)} {r(label_y)})'))

    # --- column headers ----------------------------------------------------
    for header in scene.headers:
        dwg.add(dwg.text(header.text,
                         insert=(r(header.x), r(header.y)),
                         text_anchor='middle',
                         fill=theme.header_color,
                         font_size=f'{theme.header_size}px'))

    # --- leaders (under the lines, so they never cover them) ---------------
    for leader in scene.leaders:
        dwg.add(dwg.line(start=(r(leader.x0), r(leader.y0)),
                         end=(r(leader.x1), r(leader.y1)),
                         stroke=leader.color, stroke_width=0.8))

    # --- hyperfine brackets ------------------------------------------------
    for bracket in scene.brackets:
        # a square bracket: |‾  spanning the manifold, opening left toward the lines
        path = (f'M {r(bracket.x + bracket.width)},{r(bracket.y0)} '
                f'L {r(bracket.x)},{r(bracket.y0)} '
                f'L {r(bracket.x)},{r(bracket.y1)} '
                f'L {r(bracket.x + bracket.width)},{r(bracket.y1)}')
        element = dwg.path(d=path, fill='none', stroke=bracket.color, stroke_width=0.9)
        if bracket.dashed:
            # dashed means the spacing is schematic: no hyperfine data for this manifold
            element.update({'stroke-dasharray': '2 2'})
        dwg.add(element)

    # --- level lines -------------------------------------------------------
    lines = dwg.g(stroke_linecap='round')
    for line in scene.lines:
        element = dwg.line(start=(r(line.x0), r(line.y)), end=(r(line.x1), r(line.y)),
                           stroke=line.color, stroke_width=line.linewidth)
        if line.opacity != 1.0:
            element.update({'stroke-opacity': f'{line.opacity:.3f}'})
        if line.dashes:
            element.update({'stroke-dasharray': line.dashes})
        if line.title:
            element.set_desc(title=line.title)
        lines.add(element)
    dwg.add(lines)

    # --- labels ------------------------------------------------------------
    for label in scene.labels:
        element = dwg.text('', insert=(r(label.x), r(label.y)),
                           fill=label.color,
                           font_size=f'{label.size}px',
                           text_anchor=label.anchor)
        _add_runs(element, label.runs, label.size)
        dwg.add(element)

    # --- hyperfine scale bar -----------------------------------------------
    bar = scene.scalebar
    if bar is not None:
        group = dwg.g(stroke=bar.color, fill=bar.color, stroke_width=1)
        group.add(dwg.line(start=(r(bar.x), r(bar.y0)), end=(r(bar.x), r(bar.y1))))
        for y in (bar.y0, bar.y1):
            group.add(dwg.line(start=(r(bar.x - 4), r(y)), end=(r(bar.x + 4), r(y))))
        group.add(dwg.text(bar.label,
                           insert=(r(bar.x + 10), r((bar.y0 + bar.y1) / 2 + 3.6)),
                           stroke='none', font_size=f'{theme.axis_size}px'))
        dwg.add(group)

    return dwg.tostring()
