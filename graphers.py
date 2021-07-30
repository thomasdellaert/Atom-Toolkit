import numpy as np
from matplotlib import pyplot as plt
from atom import Transition

def plot_spectrum(transition: Transition, laserwidth=0.01, colorbyupper=False):
    def lorentzian(x, x0, ampl, gamma):
        return ampl * gamma**2 / (4 * (x - x0)**2 + gamma**2)

    lines = transition.subtransitions.values()
    freqs, lo_Fs, hi_Fs, widths = [], [], [], []
    for t in lines:
        freqs.append(t.freq_Hz/1e9)

    if colorbyupper:
        pass  # TODO color by upper

    plt.figure(figsize=(16, 9))

    all_x, all_y = [], []

    for line in lines:
        fGHz = line.freq.to("GHz").magnitude
        totwidth = laserwidth + line.A/(1e6*2*np.pi)
        x_vals = np.linspace(float(fGHz - 20*totwidth), float(fGHz + 20*totwidth), 1000)
        y_vals = np.array([float(lorentzian(x_vals[i], fGHz, 1, totwidth)) for i in range(len(x_vals))])
        all_x.append(x_vals)
        all_y.append(y_vals)

        plt.plot(x_vals, y_vals, label=f"{line.E_lower.term.F_frac} → {line.E_upper.term.F_frac}")

    x_vals = []
    for x in all_x:
        x_vals.append(x)
    x_vals = np.sort(np.array(x_vals).flatten())
    y_vals = np.zeros_like(x_vals)
    for curve, x in zip(all_y, all_x):
        y_vals += np.interp(x_vals, x, curve)

    plt.plot(x_vals, y_vals, "k:", label="Total", alpha=0.5)
    plt.ylim(0, None)
    plt.xlabel("Frequency (GHz)")
    plt.legend()
