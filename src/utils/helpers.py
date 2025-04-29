# -*- coding: utf-8 -*-
"""
helpers.py

Utility functions for timing methods, handling dataframes, 
storing/loading solution sets, and computing set overlaps.

"""

import time
import json
import ast
import pandas as pd
from ..model.solution import Solution  # relative import for consistency

def time_step(func):
    """
    Decorator to time a method and store its execution time in an instance attribute `method_times`.
    """
    def wrapper(*args, **kwargs):
        instance = args[0]
        if not hasattr(instance, "method_times"):
            instance.method_times = {}

        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        instance.method_times[f"time_{func.__name__}"] = elapsed_time

        if getattr(instance, "verbose", False):
            print(f"{func.__name__} took {elapsed_time:.4f} seconds")

        return result
    return wrapper

def add_line_to_df(df: pd.DataFrame, line: dict, index: int = None) -> pd.DataFrame:
    """
    Add a line (row) to a DataFrame.

    Args:
        df (pd.DataFrame): Target DataFrame.
        line (dict): Line to add (column:value pairs).
        index (int, optional): Row index. Defaults to next available index.

    Returns:
        pd.DataFrame: Updated DataFrame.
    """
    index = index or (len(df) + 1)
    if df.empty:
        df = pd.DataFrame(pd.Series(line), columns=[index]).transpose()
    else:
        df.loc[index] = pd.Series(line)
    return df

def concat_to_string_name(mylist: list) -> str:
    """
    Concatenate a list of elements into a string separated by underscores.

    Args:
        mylist (list): List of elements.

    Returns:
        str: Concatenated string.
    """
    return "_".join(str(e) for e in mylist)

def convert_keys_to_tuples(data: dict) -> dict:
    """
    Reconvert string-represented tuple keys (after JSON storage) back to actual tuples.

    Args:
        data (dict): Dictionary with string keys.

    Returns:
        dict: Dictionary with tuple keys.
    """
    new_data = {}
    for key, value in data.items():
        try:
            new_key = ast.literal_eval(key)
            if isinstance(new_key, tuple):
                new_key = tuple(
                    int(x) if isinstance(x, str) and x.isdigit() else x
                    for x in new_key
                )
            new_data[new_key] = value
        except (ValueError, SyntaxError):
            print(f"Key '{key}' is not a valid tuple string.")
            new_data[key] = value
    return new_data

def store_S(S: dict, path: str) -> None:
    """
    Store a dictionary of Solutions as a JSON file.

    Args:
        S (dict): Dictionary of Solution objects.
        path (str): File path (without extension).
    """
    for s in S.values():
        s.data["dvars"]["x"] = {str(key): value for key, value in s.data["dvars"]["x"].items()}
        if "reduced_costs" in s.data:
            s.data["reduced_costs"]["x"] = {str(key): value for key, value in s.data["reduced_costs"]["x"].items()}

    store_dict = {s: S[s].data for s in S.keys()}
    with open(path + ".json", "w") as json_file:
        json.dump(store_dict, json_file, indent=1)

    for s in S.values():
        s.data["dvars"]["x"] = {eval(key): value for key, value in s.data["dvars"]["x"].items()}
        if "reduced_costs" in s.data:
            s.data["reduced_costs"]["x"] = {eval(key): value for key, value in s.data["reduced_costs"]["x"].items()}

def load_S(path: str, instance) -> dict:
    """
    Load a dictionary of Solutions from a JSON file.

    Args:
        path (str): File path.
        instance (Instance): Associated instance.

    Returns:
        dict: Dictionary of Solution objects.
    """
    with open(path) as json_file:
        data = json.load(json_file)

    S = {
        int(s): Solution(instance=instance, from_dict=solution_vals)
        for s, solution_vals in data.items()
    }

    for solution in S.values():
        for decision_variable in solution.data["dvars"]:
            solution.data["dvars"][decision_variable] = convert_keys_to_tuples(solution.data["dvars"][decision_variable])

    return S

def overlap(A: list, B: list) -> float:
    """
    Calculate normalized overlap between two lists: (|A ∩ B| / min(|A|, |B|)).

    Args:
        A (list): List of indices.
        B (list): List of indices.

    Returns:
        float: Overlap coefficient in [0, 1].
    """
    A, B = set(A), set(B)
    intersection = A.intersection(B)
    min_card = min(len(A), len(B))
    return round(len(intersection) / min_card, 2) if min_card > 0 else 0

def Jacc(A: list, B: list) -> float:
    """
    Calculate Jaccard similarity between two lists: (|A ∩ B| / |A ∪ B|).

    Args:
        A (list): List of indices.
        B (list): List of indices.

    Returns:
        float: Jaccard index in [0, 1].
    """
    A, B = set(A), set(B)
    intersection = A.intersection(B)
    union = A.union(B)
    return round(len(intersection) / len(union), 2) if len(union) > 0 else 0
