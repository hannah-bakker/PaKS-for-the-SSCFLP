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

        Args:
            path (str): Path to the JSON file containing the instance data.
        """
        with open(path, 'r') as json_file:
            self.data = json.load(json_file)

        # Ensure that the path is stored inside the instance information
        self.data["info"]["path"] = path

    def visualize(self, path: str = None, size: float = 0.5,
                  I_IDs: list = None, J_IDs: list = None, title: str = "Instance",
                  png: bool = False, legend: bool = True, no_customers: bool = False):
        """
        Visualize the instance using plotting utilities.

        Args:
            path (str, optional): Path where the plot should be saved. If None, the plot is shown.
            size (float, optional): Plot size scaling factor.
            I_IDs (list, optional): Subset of facility IDs to visualize.
            J_IDs (list, optional): Subset of customer IDs to visualize.
            title (str, optional): Title for the plot.
            png (bool, optional): Save as PNG if True, else save as PDF.
            legend (bool, optional): Show legend if True.
            no_customers (bool, optional): Do not plot customers if True.
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
        Store the instance data back into a JSON file.

        Args:
            path (str, optional): Path where the instance should be stored.
                                  If None, uses the original loaded path.
        """
        if path is None:
            path = self.data["info"]["path"]
        
        with open(path, "w") as json_file:
            json.dump(self.data, json_file, indent=1)

