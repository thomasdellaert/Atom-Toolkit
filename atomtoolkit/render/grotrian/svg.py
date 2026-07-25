"""Emit a :class:`~.layout.Scene` as SVG.

Deliberately dumb: no decisions are made here, only element emission. That keeps
golden-file tests meaningful -- a change in output means a change in layout, not a change
in drawing taste.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import svgwrite

from .layout import Arrow, Scene

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


def _quadratic_point(p0, pc, p1, t: float) -> Tuple[float, float]:
    """Point at parameter t on a quadratic Bezier."""
    u = 1 - t
    return (u * u * p0[0] + 2 * u * t * pc[0] + t * t * p1[0],
            u * u * p0[1] + 2 * u * t * pc[1] + t * t * p1[1])


def _wavy_points(arrow: Arrow, amplitude: float, period: float,
                 head_gap: float) -> List[Tuple[float, float]]:
    """Sample a squiggle along the arrow's path.

    A wavy line is the usual convention for spontaneous emission. The wave is laid out
    perpendicular to the local path direction, and damped to zero at both ends so the
    tail meets the level line and the head meets the arrowhead cleanly.
    """
    p0 = (arrow.x0, arrow.y0)
    p1 = (arrow.x1, arrow.y1)
    pc = arrow.control or ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if length < 1e-6:
        return [p0, p1]
    stop = max(0.0, 1.0 - head_gap / length)
    steps = max(int(length / 2.0), 12)
    cycles = max(length / period, 1.0)

    points = []
    for i in range(steps + 1):
        t = stop * i / steps
        x, y = _quadratic_point(p0, pc, p1, t)
        # tangent of the quadratic, for the perpendicular offset
        tx = 2 * (1 - t) * (pc[0] - p0[0]) + 2 * t * (p1[0] - pc[0])
        ty = 2 * (1 - t) * (pc[1] - p0[1]) + 2 * t * (p1[1] - pc[1])
        norm = math.hypot(tx, ty) or 1.0
        damp = math.sin(math.pi * min(t / stop if stop else 0.0, 1.0)) ** 0.5
        offset = amplitude * damp * math.sin(2 * math.pi * cycles * (t / stop if stop else 0))
        points.append((x - ty / norm * offset, y + tx / norm * offset))
    return points


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

    # --- transitions (under the labels, over the level lines) --------------
    if scene.arrows:
        markers = {}
        for arrow in scene.arrows:
            if arrow.head == 'none' or arrow.polygon is not None:
                continue
            key = (arrow.color, round(arrow.width, 2))
            if key in markers:
                continue
            size = theme.transition_head
            # svgwrite hands out incrementing auto-ids, which would make two renders of
            # the same scene differ byte-for-byte. render() is documented as pure, so the
            # id is derived deterministically from what the marker actually looks like.
            marker_id = f'head-{arrow.color.lstrip("#")}-{round(arrow.width * 100):d}'
            marker = dwg.marker(id=marker_id, insert=(size * 0.9, size / 2),
                                size=(size, size), orient='auto', markerUnits='userSpaceOnUse')
            marker.add(dwg.path(d=f'M0,0 L{size},{size / 2} L0,{size} z', fill=arrow.color))
            dwg.defs.add(marker)
            markers[key] = marker

        # block arrows go down first: a thin stroked arrow is often drawn on top of one
        for arrow in scene.arrows:
            if arrow.polygon is None:
                continue
            points = [(r(x), r(y)) for x, y in arrow.polygon]
            element = dwg.polygon(points=points, fill=arrow.color, stroke='none')
            if arrow.opacity != 1.0:
                element.update({'fill-opacity': f'{arrow.opacity:.3f}'})
            if arrow.title:
                element.set_desc(title=arrow.title)
            dwg.add(element)

        for arrow in scene.arrows:
            if arrow.polygon is not None:
                continue
            key = (arrow.color, round(arrow.width, 2))
            head_gap = theme.transition_head * 0.9 if arrow.head != 'none' else 0.0
            if arrow.wavy:
                points = _wavy_points(arrow, theme.wavy_amplitude, theme.wavy_period,
                                      head_gap)
                d = 'M ' + ' L '.join(f'{r(x)},{r(y)}' for x, y in points)
            elif arrow.control is not None:
                d = (f'M {r(arrow.x0)},{r(arrow.y0)} '
                     f'Q {r(arrow.control[0])},{r(arrow.control[1])} '
                     f'{r(arrow.x1)},{r(arrow.y1)}')
            else:
                d = f'M {r(arrow.x0)},{r(arrow.y0)} L {r(arrow.x1)},{r(arrow.y1)}'
            element = dwg.path(d=d, fill='none', stroke=arrow.color,
                               stroke_width=arrow.width, stroke_linecap='round')
            if arrow.dashed:
                element.update({'stroke-dasharray': '5 3'})
            if arrow.opacity != 1.0:
                element.update({'stroke-opacity': f'{arrow.opacity:.3f}'})
            if arrow.head in ('end', 'both'):
                element.set_markers((None, None, markers[key]))
            if arrow.head == 'both':
                element.update({'marker-start': markers[key].get_funciri()})
            if arrow.title:
                element.set_desc(title=arrow.title)
            dwg.add(element)

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
