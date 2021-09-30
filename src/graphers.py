import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors
from .atom import Transition
import colorsys
from .lineshapes import LineShape

def plot_spectrum(transition: Transition, lineshape: LineShape, coloring='l'):
    lines = transition.subtransitions.values()
    lo_Fs, hi_Fs = [], []
    for t in lines:
        lo_Fs.append(t.E_lower.term.F)
        hi_Fs.append(t.E_upper.term.F)
    num_lo_Fs = {F: lo_Fs.count(F) for F in np.unique(lo_Fs)}
    num_hi_Fs = {F: hi_Fs.count(F) for F in np.unique(hi_Fs)}
    F_pairs = list(zip(lo_Fs, hi_Fs))
    min_hi_Fs = {Flo: min([Fp[1] for Fp in F_pairs if Fp[0] == Flo]) for Flo in lo_Fs}
    min_lo_Fs = {Fhi: min([Fp[0] for Fp in F_pairs if Fp[1] == Fhi]) for Fhi in hi_Fs}

    cmap = plt.get_cmap("tab10")

    all_x, all_y = [], []
    for line in lines:
        fGHz = line.freq.to("GHz").magnitude
        x_values = np.linspace(float(fGHz - lineshape.width_func()), float(fGHz + lineshape.width_func()), 1000)
        y_values = np.array([lineshape.shape_func(x_values[i], x0=fGHz, A_coeff=line.A) for i in range(len(x_values))])
        all_x.append(x_values)
        all_y.append(y_values)

        # depending on coloring by upper/lower, one ser of Fs determines color, while the other determines shade
        if not (coloring == 'u'):
            color_num_Fs = num_lo_Fs
            color_F = line.E_lower.term.F
            shade_F = line.E_upper.term.F
            min_shade_Fs = min_hi_Fs
        else:
            color_num_Fs = num_hi_Fs
            color_F = line.E_upper.term.F
            shade_F = line.E_lower.term.F
            min_shade_Fs = min_lo_Fs

        # pick a color depending on the color F
        col = cmap(list(color_num_Fs.keys()).index(color_F))
        # convert it to HLS
        col = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(col))
        # set the lightness depending on the shade F and convert back to RGB
        lightness = 0.2+0.6*(shade_F-min_shade_Fs[color_F]+1)/(color_num_Fs[color_F]+1)
        col = colorsys.hls_to_rgb(col[0], lightness, col[2])
        if (coloring == 'l') or (coloring == 'u'):
            plt.plot(x_values, y_values, label=f"{line.E_lower.term.F_frac} → {line.E_upper.term.F_frac}", c=col)
        else:
            plt.plot(x_values, y_values, label=f"{line.E_lower.term.F_frac} → {line.E_upper.term.F_frac}")

    x_values = []
    for x in all_x:
        x_values.append(x)
    x_values = np.sort(np.array(x_values).flatten())
    y_values = np.zeros_like(x_values)
    for curve, x in zip(all_y, all_x):
        y_values += np.interp(x_values, x, curve)

    plt.plot(x_values, y_values, "k:", label="Total", alpha=0.5)
    plt.ylim(0, None)
    plt.xlabel("Frequency (GHz)")
    plt.legend()
