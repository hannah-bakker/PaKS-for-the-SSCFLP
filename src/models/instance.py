# -*- coding: utf-8 -*-
"""
instance.py

This module defines the Instance class for loading, visualizing, and storing
benchmark instances for facility location problems.

"""

import json
from ..utils.plot import SSCFLPPlot  

class Instance:
    """
    Class representing a facility location instance loaded from a JSON file.
    """

    def __init__(self, path: str):
        """
        Initialize an instance by loading data from a JSON file.
        
        Parameters
        ----------
        path : str, optional
            Path to the JSON file containing the instance data.
        """
        with open(path, 'r') as json_file:
            self.data = json.load(json_file)

        # Ensure that the path is stored inside the instance information
        self.data["info"]["path"] = path

    def visualize(self, path: str = None, size: float = 0.5,
                  I_IDs: list = None, J_IDs: list = None, title: str = "Instance",
                  png: bool = False, legend: bool = True, no_customers: bool = False):
        """
        Visualize the SSCFLP instance with optional facility and customer selection.

        Uses the SSCFLPPlot utility to create a visual representation of the instance.
        Allows selective plotting and configurable output.

        Parameters
        ----------
        path : str, optional
            If provided, saves the plot to this file path. Otherwise, displays the plot interactively.
        size : float, optional
            Scaling factor for the plot dimensions (default is 0.5).
        I_IDs : list of int, optional
            Subset of facility indices to display. If None, all are shown.
        J_IDs : list of int, optional
            Subset of customer indices to display. If None, all are shown.
        title : str, optional
            Title displayed on the plot (default is "Instance").
        png : bool, optional
            If True, saves the plot as a PNG file. If False, saves as PDF.
        legend : bool, optional
            Whether to display a legend (default is True).
        no_customers : bool, optional
            If True, do not plot customer nodes (default is False).

        Returns
        -------
        None
        """
        plot = SSCFLPPlot(size=size)
        plot._add_subplot(self, I_IDs=I_IDs, J_IDs=J_IDs,
                              title=title, legend=legend, no_customers=no_customers)
        
        if path is not None:
            plot._save(path, png=png)
        else:
            plot.show()

    def store(self, path: str = None):
        """
        Store the instance data to a JSON file.

        Parameters
        ----------
        path : str, optional
            Destination file path to store the instance. If None, uses the original
            path from the instance metadata ('info' field in self.data).
        """
        if path is None:
            path = self.data["info"]["path"]
        
        with open(path, "w") as json_file:
            json.dump(self.data, json_file, indent=1)

