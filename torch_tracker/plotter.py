import h5py
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
#plt.rcParams.update({'font.size': 12,'font.family':'serif','text.usetex':True})
mpl.rcParams['mathtext.fontset'] = 'cm'    # Computer Modern
mpl.rcParams['font.family'] = 'serif'

class Plotter:
    def __init__(self, analysis_file, quantity_labels=None):
        self.analysis_file = analysis_file
        self.quantity_labels = quantity_labels or {}
        self._load()

    def _load(self):
        import h5py
        self.f = h5py.File(self.analysis_file, "r")
        self.snapshot = self.f["snapshots/snapshot"][:]
        self.time = self.f["snapshots/time"][:]
        self.quantities = {q: self.f[f"quantities/{q}"][:] for q in self.f["quantities"]}

    def available_quantities(self):
        return list(self.quantities.keys())

    def plot(self, quantity, xaxis="time", ylog=True, outdir="plots", show=False, title=None):
        import matplotlib.pyplot as plt
        import numpy as np

        if quantity not in self.quantities:
            raise ValueError(f"Unknown quantity '{quantity}'.")

        if xaxis == "time":
            x = self.time
            xlabel = "Time [Myr]"
        elif xaxis == "snapshot":
            x = self.snapshot
            xlabel = "Snapshot"
        else:
            raise ValueError("xaxis must be 'time' or 'snapshot'")

        plot_order = np.argsort(x)
        y = self.quantities[quantity]

        # Pad NaNs if length mismatch
        if len(x) != len(y):
            y_full = np.full_like(x, np.nan, dtype=float)
            y_full[:len(y)] = y
            y = y_full

        plt.figure()
        plt.plot(x[plot_order], y[plot_order], 'k-', lw=0.75)
        plt.xlabel(xlabel)
        ylabel = self.quantity_labels.get(quantity, quantity.replace("_", " "))
        plt.ylabel(ylabel)
        plt.title(title)
        if ylog:
            plt.yscale("log")

        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"{quantity}.png"), dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
        print(f"Saved {quantity}.png")

    def plot_num_runaways(self):
        x = self.time
        xlabel = "Time [Myr]"
        for q in self.quantities['unbound_star_ids']:
            print(q)

        plt.plot(x,y,'k-',lw=0.75)
        plt.xlabel(xlabel)
        plt.ylabel(r"$N_{\rm runaways}$")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "num_runaways.png"), dpi=200, bbox_inches="tight")



    def plot_multiple(self, quantities, labels, ylabel, cmap, savename, yscale="log", xaxis="time", outdir="plots", show=False, title=None):
        import numpy as np

        if xaxis == "time":
            x = self.time
            xlabel = "Time [Myr]"
        elif xaxis == "snapshot":
            x = self.snapshot
            xlabel = "Snapshot"
        else:
            raise ValueError("xaxis must be 'time' or 'snapshot'")
        plot_order = np.argsort(x)
        colors = mpl.cm.get_cmap(cmap)
        for p,q in enumerate(quantities):
            y = self.quantities[q]
            plt.plot(x[plot_order],y[plot_order],'-',lw=0.75, color=colors(p/len(quantities)), label=labels[p])

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.yscale("log")
        plt.legend(framealpha=1.0, fancybox=False, edgecolor='k')

        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, savename), dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
        print(f"Saved "+savename)
