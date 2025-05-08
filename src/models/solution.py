# -*- coding: utf-8 -*-
"""
solution.py

Defines the Solution class for storing, loading, and visualizing solutions 
to the SSCFLP.
"""

import json
from utils.plot import SSCFLPPlot  

class Solution:
    """
    Class representing a solution to the SSCFLP.
    """

    def __init__(self, 
                 instance,
                 path: str = None, 
                 from_dict: dict = None,
                 **params):
        """
        Initialize a Solution object.

        This constructor allows loading a solution from a file, from a dictionary,
        or creating it directly from provided keyword arguments.

        Parameters
        ----------
        instance : object
            The instance to which the solution belongs.
        path : str, optional
            Path to a JSON file containing a previously stored solution.
        from_dict : dict, optional
            A dictionary containing preloaded solution data.
        **params : keyword arguments
            Parameters defining a new solution (e.g., 'obj', 'dvars', 'time').
        """
        self.instance = instance
        
        if path is not None:
            with open(path, 'r') as json_file:
                self.data = json.load(json_file)
        elif from_dict is not None:
            self.data = from_dict
        else:
            self.data = dict()
            for kw, arg in params.items():
                self.data[kw] = arg

    def store(self, path: str) -> None:
        """
        Store the solution data to a JSON file.

        Parameters
        ----------
        path : str
            Output file path (without file extension); '.json' will be added automatically.
        """
        # Convert tuple keys to strings for JSON serialization
        self.data["dvars"]["x"] = {str(key): value for key, value in self.data["dvars"]["x"].items()}
        if "reduced_costs" in self.data:
            self.data["reduced_costs"]["x"] = {str(key): value for key, value in self.data["reduced_costs"]["x"].items()}

        with open(path + ".json", "w") as json_file:
            json.dump(self.data, json_file, indent=1)

        # Re-convert string keys back to tuples for internal use
        self.data["dvars"]["x"] = {eval(key): value for key, value in self.data["dvars"]["x"].items()}
        if "reduced_costs" in self.data:
            self.data["reduced_costs"]["x"] = {eval(key): value for key, value in self.data["reduced_costs"]["x"].items()}

    def visualize(self, 
                  path: str = None, 
                  size: float = 0.5, 
                  I_IDs: list = None, 
                  J_IDs: list = None, 
                  title: str = "Optimal solution",
                  legend: bool = True,
                  png: bool = False,
                  no_assignments: bool = False) -> SSCFLPPlot:
        """
        Visualize the solution using a problem-specific plotting class.

        This method creates a visual representation of the facility location solution, 
        with optional customization of displayed facilities, customers, and connections.
        It supports both interactive display and export to file (PDF or PNG).

        Parameters
        ----------
        path : str, optional
            File path to save the visualization. If None, the plot is shown interactively.
        size : float, optional
            Scaling factor for the figure size (default is 0.5).
        I_IDs : list of int, optional
            Subset of facility IDs to plot (default is all).
        J_IDs : list of int, optional
            Subset of customer IDs to plot (default is all).
        title : str, optional
            Title of the plot (default is "Optimal solution").
        legend : bool, optional
            Whether to display the legend (default is True).
        png : bool, optional
            If True, save plot as PNG; otherwise, save as PDF.
        no_assignments : bool, optional
            If True, omit the assignment arcs between facilities and customers.

        Returns
        -------
        SSCFLPPlot
            The plot object, which may be useful for further customization or reuse.
        """
        plot = SSCFLPPlot(size=size)
        plot._add_subplot(instance=self.instance, 
                            solution=self, 
                            I_IDs=I_IDs, 
                            J_IDs=J_IDs, 
                            title=title, 
                            legend=legend, 
                            no_assignments=no_assignments)

        if path is not None:
            plot._save(path, png=png)
        else:
            plot.show()

        return plot
