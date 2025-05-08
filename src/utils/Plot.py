# -*- coding: utf-8 -*-
"""
plot.py

Visualization module for SSCFLP instances, solutions, regions, and kernel search components.

"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils.FLPspatialpattern.GetCoordinates import GetCoordinates

class SSCFLPPlot:
    """
    Class for visualizing capacitated facility location problems (SSCFLP) instances, 
    solutions, kernels, regions, and bucket structures.
    """

    # Default colors
    COLOR_I_CANDIDATE = "tab:olive"
    COLOR_J = "tab:blue"
    COLOR_I_OPEN = "tab:green"
    COLOR_I_CLOSED = "tab:gray"
    COLOR_KERNEL = "red"

    COLOR_LIST_FOR_REGIONS = [
        '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', 'mediumturquoise', 'purple',
        'palegreen', 'firebrick', 'pink', 'darkblue', 'hotpink', 'darksalmon', 'yellow',
        'sienna', 'blueviolet', 'slategrey', 'sandybrown', 'lime', 'orange', 'green',
        'skyblue', 'tomato', 'slateblue', 'olive', 'coral', 'goldenrod', 'cyan',
        'lemonchiffon', 'steelblue', 'peru', 'indigo', 'crimson', 'royalblue', 'gold',
        'aqua', 'teal', 'fuchsia', 'lavenderblush', 'maroon', 'peachpuff', 'lightgray',
        'darkolivegreen', 'deeppink', 'springgreen', 'darkkhaki', 'oldlace',
        'darkslateblue', 'cadetblue', 'plum', 'aliceblue', 'orangered', 'powderblue',
        'darkmagenta', 'beige'
    ]

    def __init__(self, size: float, **params):
        """
        Initialize a plotting object for SSCFLP visualizations.

        Parameters
        ----------
        size : float
            Fraction of LaTeX text width used to scale the figure size. For example,
            0.5 will create a figure that is half the LaTeX text width.
        **params : dict, optional
            Additional plotting parameters to store for custom use (e.g., colors, markers).
        """

        self.size = size
        self.params = params
        self.lgd = None
        self.MYTEXTWIDTH = 427.4315  # pt
        self.fig = plt.figure(figsize=self._set_size(size))

        # Configure matplotlib
        plt.rcParams["text.usetex"] = False
        plt.rcParams['figure.dpi'] = 1000
        plt.rcParams['savefig.dpi'] = 1000

    def _set_size(self, fraction: float):
        """
        Compute figure dimensions in inches from a fraction of LaTeX text width.

        Parameters
        ----------
        fraction : float
            Fraction of the text width to scale the figure to (e.g., 0.5 means half the width).

        Returns
        -------
        tuple
            A tuple (width, height) in inches, assuming a square figure.
        """
        inches_per_pt = 1 / 72.27
        fig_width_in = self.MYTEXTWIDTH * fraction * inches_per_pt
        return fig_width_in, fig_width_in

    def _save(self, path: str, close: bool = True, png: bool = False):
        """
        Save the plot to a file in PNG or PDF format.

        Parameters
        ----------
        path : str
            File path (without extension) where the figure should be saved.
        close : bool, optional
            If True, closes the figure after saving to free memory. Default is True.
        png : bool, optional
            If True, saves the figure as a PNG file. Otherwise saves as PDF.
        """
        plt.rcParams.update({'figure.subplot.left': 0, 'figure.subplot.bottom': 0,
                             'figure.subplot.right': 1, 'figure.subplot.top': 1})
        if not self.lgd:
            if png:
                self.fig.savefig(path + ".png", bbox_inches="tight")
            else:
                self.fig.savefig(path + ".pdf", bbox_inches="tight")
        else:
            title = self.ax.get_title()
            self.ax.set_title("")
            fig = self.lgd.figure
            fig.canvas.draw()
            bbox = self.lgd.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            if png:
                fig.savefig(path + "_legend.png", dpi="figure", bbox_inches=bbox)
            else:
                fig.savefig(path + "_legend.pdf", dpi="figure", bbox_inches=bbox)
            self.ax.legend([], [], loc="upper left", frameon=False)
            self.ax.set_title(title)
            if png:
                self.fig.savefig(path + ".png", bbox_inches="tight")
            else:
                self.fig.savefig(path + ".pdf", bbox_inches="tight")
        if close:
            plt.close(self.fig)

    def _add_subplot(self, instance, solution=None, regions=None, **kwargs):
        """
        Add a subplot displaying the instance layout and optionally the solution or regions.

        Parameters
        ----------
        instance : Instance
            The instance object containing facility and customer data.
        solution : Solution, optional
            A solution object with open facilities and assignments.
        regions : dict, optional
            Optional region coloring/grouping to be shown.
        **kwargs :
            Additional plotting arguments, e.g.:
            - legend (bool): Whether to include a legend (default: True)
            - title (str): Title of the subplot
            - no_customers (bool): If True, suppress customer plot
        """
        self.params.update(kwargs)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        if "plot" not in instance.data:
            print("Retrieve coordinates...")
            mytime = time.time()
            instance.data["plot"] = {}
            getcoord = GetCoordinates(np.asarray(instance.data["params"]["c_ij"]))
            instance.data["plot"]["xloc"] = getcoord.coord[instance.data["params"]["J"]:, 0].tolist()
            instance.data["plot"]["yloc"] = getcoord.coord[instance.data["params"]["J"]:, 1].tolist()
            instance.data["plot"]["xcus"] = getcoord.coord[:instance.data["params"]["J"], 0].tolist()
            instance.data["plot"]["ycus"] = getcoord.coord[:instance.data["params"]["J"], 1].tolist()
            print(f"Coordinates computed in {time.time() - mytime:.2f} seconds.")
            instance.store()

        self._compute_marker_sizes(instance)

        if regions:
            self._add_regions(instance, regions, **kwargs)
            if solution:
                self._add_assignments(instance, solution)
        elif solution:
            self._add_customers(instance)
            self._add_facilities_open_closed(instance, solution)
            self._add_assignments(instance, solution)
            if kwargs.get("legend", True):
                self._add_solution_legend()
        else:
            self._add_customers(instance)
            self._add_facilities_candidates(instance)
            if kwargs.get("legend", True):
                self._add_instance_legend()

        if kwargs.get("title"):
            self.ax.set_title(kwargs["title"])
        return self.ax

    def _compute_marker_sizes(self, instance):
        """
        Compute relative marker sizes for plotting facilities and customers.

        Marker sizes are scaled proportionally to each facility's capacity (Q_i)
        and each customer's demand (D_j), using a figure-width-based scaling.

        Parameters
        ----------
        instance : Instance
            The instance object containing parameters 'Q_i' (capacities) and 'D_j' (demands).
        """
        width_per_marker_pt = self.MYTEXTWIDTH / (instance.data["params"]["I"] + instance.data["params"]["J"])
        total_I_pt = (width_per_marker_pt * instance.data["params"]["I"])**2 * 1.5
        self.size_I = [total_I_pt * q / sum(instance.data["params"]["Q_i"]) for q in instance.data["params"]["Q_i"]]
        total_J_pt = (width_per_marker_pt * instance.data["params"]["J"])**2 * 0.2
        self.size_J = [total_J_pt * d / sum(instance.data["params"]["D_j"]) for d in instance.data["params"]["D_j"]]
        self.linewidth = 0.75 * (self.size / 0.75)

# =============================================================================
# Internal Plotting Methods
# =============================================================================

    def _add_customers(self, instance):
        """Add customer nodes."""
        self.ax.scatter(
            instance.data["plot"]["xcus"],
            instance.data["plot"]["ycus"],
            s=self.size_J,
            marker="^",
            color=self.COLOR_J,
            alpha=self.params.get("alpha", 0.6),
            zorder=5
        )

    def _add_facilities_candidates(self, instance):
        """Add candidate facility nodes."""
        self.ax.scatter(
            instance.data["plot"]["xloc"],
            instance.data["plot"]["yloc"],
            s=self.size_I,
            marker=".",
            color=self.COLOR_I_CANDIDATE,
            alpha=self.params.get("alpha", 1),
            zorder=2
        )

    def _add_facilities_open_closed(self, instance, solution):
        """Add open and closed facilities based on solution."""
        open_indices = [i for i, val in solution.data['dvars']['y'].items() if val > 0]
        closed_indices = [i for i, val in solution.data['dvars']['y'].items() if val <= 0]

        # Open facilities
        self.ax.scatter(
            np.array(instance.data["plot"]["xloc"])[open_indices],
            np.array(instance.data["plot"]["yloc"])[open_indices],
            s=[self.size_I[i] * 0.3 for i in open_indices],
            marker="s",
            color=self.COLOR_I_OPEN,
            alpha=0.9,
            zorder=10
        )

        # Closed facilities
        self.ax.scatter(
            np.array(instance.data["plot"]["xloc"])[closed_indices],
            np.array(instance.data["plot"]["yloc"])[closed_indices],
            s=[self.size_I[i] for i in closed_indices],
            marker=".",
            color=self.COLOR_I_CLOSED,
            alpha=0.8,
            zorder=2
        )

    def _add_assignments(self, instance, solution):
        """Draw assignment lines between facilities and customers."""
        for ij, val in solution.data['dvars']['x'].items():
            if val > 0:
                i, j = map(int, ij)
                self.ax.plot(
                    [instance.data["plot"]["xloc"][i], instance.data["plot"]["xcus"][j]],
                    [instance.data["plot"]["yloc"][i], instance.data["plot"]["ycus"][j]],
                    linestyle=self.params.get('linestyle', "-"),
                    linewidth=self.params.get('linewidth', self.linewidth),
                    color=self.params.get('linecolor', 'k'),
                    zorder=4
                )

    def _add_regions(self, instance, regions, **kwargs):
        """Highlight regions for facilities and customers."""
        for r_index, (facilities, customers) in regions.items():
            color = self.COLOR_LIST_FOR_REGIONS[r_index] if r_index > -1 else "gray"
            zorder = 20 if r_index > -1 else 5

            if not kwargs.get("regional_customers_only", False):
                # Facilities in region
                self.ax.scatter(
                    [instance.data["plot"]["xloc"][i] for i in facilities],
                    [instance.data["plot"]["yloc"][i] for i in facilities],
                    s=[self.size_I[i] for i in facilities],
                    marker=".",
                    color=color,
                    alpha=0.9,
                    zorder=zorder
                )

            # Customers in region
            self.ax.scatter(
                [instance.data["plot"]["xcus"][j] for j in customers],
                [instance.data["plot"]["ycus"][j] for j in customers],
                s=[self.size_J[j] for j in customers],
                marker="^",
                color=color,
                alpha=0.9,
                zorder=zorder
            )

    def _add_instance_legend(self):
        """Legend for basic instance visualization."""
        legend_elements = [
            Line2D([0], [0], marker="^", color='w', markerfacecolor=self.COLOR_J, label="Customers $J$", markersize=10),
            Line2D([0], [0], marker=".", color='w', markerfacecolor=self.COLOR_I_CANDIDATE, label="Candidate facilities $I$", markersize=10)
        ]
        self.lgd = self.ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.5),
                                  ncol=2, frameon=False)

    def _add_solution_legend(self):
        """Legend for solution visualization."""
        legend_elements = [
            Line2D([0], [0], marker="^", color='w', markerfacecolor=self.COLOR_J, label="Customers", markersize=10),
            Line2D([0], [0], marker="s", color='w', markerfacecolor=self.COLOR_I_OPEN, label="Open facilities", markersize=10),
            Line2D([0], [0], marker=".", color='w', markerfacecolor=self.COLOR_I_CLOSED, label="Closed facilities", markersize=10)
        ]
        self.lgd = self.ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.5),
                                  ncol=3, frameon=self.params.get("frameon", False))

