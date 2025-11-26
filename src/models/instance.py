# -*- coding: utf-8 -*-
import json 

class Instance:
    """
    Lightweight wrapper around a JSON instance file.
    Provides: (i) loading, (ii) storing
    """

    def __init__(self, path: str):
        """
        Load an instance from a JSON file.

        Parameters
        ----------
        path : str
            Path to the JSON instance file.
        """
        with open(path) as f:
            self.data = json.load(f)

        # Record original file path inside the metadata
        self.data.setdefault("info", {})
        self.data["info"]["path"] = path
            
    # ------------------------------------------------------------------
    # Store to disk
    # ------------------------------------------------------------------
    def store(self, path=None):
        """
        Write the instance back to a JSON file.

        Parameters
        ----------
        path : str, optional
            Output path. If None, overwrite the original file.

        Returns
        -------
        None
        """
        if path is None:
            path = self.data["info"]["path"]

        with open(path, "w") as f:
            json.dump(self.data, f, indent=1)
     