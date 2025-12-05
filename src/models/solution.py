# -*- coding: utf-8 -*-
import json

class Solution:
    """
    Lightweight wrapper for solution data.
    """
    
    def __init__(
        self,
        instance,
        path=None,
        from_dict=None,
        **params,
    ):
        """
        Initialize the solution object.

        Parameters
        ----------
        instance : object
            The corresponding problem instance.
        path : str, optional
            Path to a JSON solution file to load.
        from_dict : dict, optional
            Dictionary with solution data.
        **params :
            Additional keyword arguments used to build a new solution
            dictionary if no file or dictionary is provided.
        """
        self.instance = instance

        if path is not None:
            # Load solution from JSON file
            with open(path) as f:
                self.data = json.load(f)
        elif from_dict is not None:
            # Initialize directly from dictionary
            self.data = from_dict
        else:
            # Build a new solution from keyword parameters
            self.data = {}
            for kw, arg in params.items():
                self.data[kw] = arg

    # ------------------------------------------------------------------
    # Store to disk
    # ------------------------------------------------------------------
    def store(self, path):
        """
        Write the solution to a JSON file.

        During serialization, tuple keys in dvars["x"] (and optionally
        reduced_costs["x"]) are converted to strings to make them JSON
        compatible and then converted back afterwards.

        Parameters
        ----------
        path : str
            Output path without the `.json` extension.

        Returns
        -------
        None
        """
        # Convert tuple keys to strings for JSON
        self.data["dvars"]["x"] = {
            str(key): value for key, value in self.data["dvars"]["x"].items()
        }
        if "reduced_costs" in self.data:
            self.data["reduced_costs"]["x"] = {
                str(key): value
                for key, value in self.data["dvars"]["x"].items()
            }

        with open(path + ".json", "w") as f:
            json.dump(self.data, f, indent=1)

        # Convert string keys back to tuples
        self.data["dvars"]["x"] = {
            eval(key): value for key, value in self.data["dvars"]["x"].items()
        }
        if "reduced_costs" in self.data:
            self.data["reduced_costs"]["x"] = {
                eval(key): value
                for key, value in self.data["reduced_costs"]["x"].items()
            }