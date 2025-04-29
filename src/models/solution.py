# -*- coding: utf-8 -*-
"""
solution.py

Defines the Solution class for storing, loading, and visualizing solutions 
to facility location problems.

"""

import json
from ..utils.plot import SSCFLPPlot  

class Solution:
    """
    Class representing a solution for a facility location problem instance.
    """

    def __init__(self, 
                 instance,
                 path: str = None, 
                 from_dict: dict = None,
                 **params):
        """
        Initialize a Solution object.

        Args:
            instance (Instance): The instance object related to this solution.
            path (str, optional): Path to load an existing solution from JSON.
            from_dict (dict, optional): Dictionary to load solution from.
            params: Additional parameters to create a new solution.
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
        Store the solution in a JSON file.

        Args:
            path (str): Output path (without ".json" extension).
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
                  problem_type: str = "Type1",
                  I_IDs: list = None, 
                  J_IDs: list = None, 
                  title: str = "Optimal solution",
                  legend: bool = True,
                  png: bool = False,
                  no_assignments: bool = False) -> Type1Plot:
        """
        Visualize the solution.

        Args:
            path (str, optional): Path to save the figure. If None, displays the figure.
            size (float, optional): Size scaling factor.
            problem_type (str, optional): Problem type for visualization (default is 'Type1').
            I_IDs (list, optional): Subset of facility IDs to visualize.
            J_IDs (list, optional): Subset of customer IDs to visualize.
            title (str, optional): Title for the plot.
            legend (bool, optional): Show legend if True.
            png (bool, optional): Save as PNG if True, otherwise as PDF.
            no_assignments (bool, optional): Do not plot assignment lines if True.

        Returns:
            Type1Plot: Plotting object.
        """
        if problem_type == "Type1":
            plot = Type1Plot(size=size)
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
