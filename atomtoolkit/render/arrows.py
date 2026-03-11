from matplotlib.patches import FancyArrowPatch, ArrowStyle, Arrow

def ThickArrow(posA, posB, **kwargs):
    scale = kwargs.get('scale', 100)
    color = kwargs.get('color', (0., 0., 0., 1.))
    edgecolor = kwargs.get('edgecolor', None)
    lw = kwargs.get('lw', 0)
    kwargs.pop('arrowstyle')
    return FancyArrowPatch(posA, posB,
                           mutation_scale=scale,
                           color=color,
                           arrowstyle='simple',
                           edgecolor=edgecolor,
                           lw=lw,
                           **kwargs)


def SimpleArrow(posA, posB, **kwargs):
    scale = kwargs.get('scale', 50)
    color = kwargs.get('color', (0., 0., 0., 1.))
    lw = kwargs.get('lw', 0)
    arrowstyle = kwargs.get('arrowstyle', '->')
    assert arrowstyle in {'<->', '->', '<-', '<|-|>', '-|>', '<|-', '-'}
    kwargs.pop('arrowstyle')
    return FancyArrowPatch(posA, posB,
                           mutation_scale=scale,
                           color=color,
                           arrowstyle=arrowstyle,
                           lw=lw,
                           **kwargs)

def WigglyArrow(posA, posB, **kwargs):
