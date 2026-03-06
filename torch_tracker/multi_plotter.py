import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class MultiPlotter:
    """
    Plot quantities from multiple analysis HDF5 files on the same plot.

    Example usage:
        mp = MultiPlotter(
            files=["inter_MC.h5","out_LC.h5","out_MC.h5"],
            labels=["Inter MC","Out LC","Out MC"],
            quantity_labels={"sfe_roi":"SFE$_{ROI}$"}
        )
        mp.plot("sfe_roi", xaxis="time", cmap="viridis")
    """

    def __init__(self, files, labels=None, quantity_labels=None):
        if labels is None:
            labels = [os.path.basename(f) for f in files]
        if len(files) != len(labels):
            raise ValueError("Number of files and labels must match")

        self.files = files
        self.labels = labels
        self.quantity_labels = quantity_labels or {}

        # Load all datasets
        self.data = []
        for f in files:
            with h5py.File(f, "r") as h:
                snap = h["snapshots/snapshot"][:]
                time = h["snapshots/time"][:]
                quantities = {q: h[f"quantities/{q}"][:] for q in h["quantities"]}
                self.data.append({"snapshot": snap, "time": time, "quantities": quantities})

    # -----------------------------
    # Plot a single quantity for all simulations
    # -----------------------------
    def plot(self, quantity, xaxis="time", cmap="viridis", outdir="plots", show=False, ylog=True):
        if not self.data:
            print("No data loaded")
            return

        colors = cm.get_cmap(cmap)
        plt.figure()

        for i, sim in enumerate(self.data):
            if quantity not in sim["quantities"]:
                print(f"Quantity {quantity} not in file {self.files[i]}")
                continue

            x = sim["time"] if xaxis=="time" else sim["snapshot"]
            y = sim["quantities"][quantity]

            plot_order = np.argsort(x)

            plt.plot(x[plot_order], y[plot_order], '-', label=self.labels[i], color=colors(i/len(self.data)))

        xlabel = "Time [Myr]" if xaxis=="time" else "Snapshot"
        plt.xlabel(xlabel)
        ylabel = self.quantity_labels.get(quantity, quantity.replace("_"," "))
        plt.ylabel(ylabel)
        if ylog:
            plt.yscale("log")
        plt.legend(framealpha=1.0, fancybox=False, edgecolor='k')
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"{quantity}_multi.png"), dpi=200)
        if show:
            plt.show()
        else:
            plt.close()
        print(f"Saved {quantity}_multi.png")


    def plot_multiple(self, quantities, labels, lstyles, ylabel, cmap, savename, yscale="log", xaxis="time", outdir="plots", show=False, title=None):
        import numpy as np

        if not self.data:
            print("No data loaded")
            return

        colors = cm.get_cmap(cmap)
        plt.figure()

        for i, sim in enumerate(self.data):

            x = sim["time"] if xaxis=="time" else sim["snapshot"]

            for p,q in enumerate(quantities):
                if i == 0:
                    plt.plot([None], ls=lstyles[p], color=colors(i/len(self.data)), label=labels[p])
                plot_order = np.argsort(x)
                y = sim["quantities"][q]
                plt.plot(x[plot_order],y[plot_order], ls=lstyles[p], color=colors(i/len(self.data)))
            plt.plot([None], ls=lstyles[0], color=colors(i/len(self.data)), label=self.labels[i])

        xlabel = "Time [Myr]" if xaxis=="time" else "Snapshot"
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

