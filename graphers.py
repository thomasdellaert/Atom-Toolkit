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

    num_points = int(max(freqs) - min(freqs) + 2 * 100 / laserwidth)
    x_vals_tot = np.linspace(min(freqs) - 1, max(freqs) + 1, num_points)
    tot = np.zeros(num_points)

    if colorbyupper:
        pass #TODO color by upper

    plt.figure(figsize=(16, 9))
    for line in lines:
        fGHz = line.freq.to("GHz").magnitude
        totwidth = laserwidth + line.A/(1e6*2*np.pi)
        x_vals = np.linspace([fGHz - 15*totwidth], [fGHz + 15*totwidth], 200)
        y_vals = [lorentzian(x_vals[i], fGHz, 1, totwidth) for i in range(len(x_vals))]
        # tot = tot + y_vals
        #TODO: Figure out how to add the multiple lorentzians into a total.
        plt.plot(x_vals, y_vals, label=f"{line.E_lower.term.F_frac} → {line.E_upper.term.F_frac}")
    # plt.plot(x_vals, tot, "k:", label="Total", alpha=0.5)
    plt.ylim(0, None)
    plt.xlabel("Frequency (GHz)")
    plt.legend()
