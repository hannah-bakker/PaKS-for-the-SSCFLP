# -*- coding: utf-8 -*-
"""
helpers.py

Utility functions for timing methods, handling dataframes, 
storing/loading solution sets, and computing set overlaps.

"""

import time
import ast
import pandas as pd

def time_step(func):
    """
    Decorator to measure and store the execution time of a method.

    The elapsed time is stored in the instance attribute `method_times` under the key "time_<method_name>".
    If the instance has `verbose=True`, it prints the duration to the console.

    Returns
    -------
    The result of the original method call, unaffected by the timing logic.
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
    Append or insert a new row to a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The target DataFrame to which the row should be added.
    line : dict
        Dictionary containing column-value pairs for the new row.
    index : int, optional
        Index at which to insert the row. Defaults to the next available integer index.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame including the new row.
    """
    index = index or (len(df) + 1)
    if df.empty:
        df = pd.DataFrame(pd.Series(line), columns=[index]).transpose()
    else:
        df.loc[index] = pd.Series(line)
    return df

def concat_to_string_name(mylist: list) -> str:
    """
    Convert a list of elements into a single underscore-separated string.

    Parameters
    ----------
    mylist : list
        List of elements to concatenate. Elements are converted to strings if not already.

    Returns
    -------
    str
        A single string formed by joining the elements with underscores.
        Example: [1, "a", 3] -> "1_a_3"
    """
    return "_".join(str(e) for e in mylist)

def convert_keys_to_tuples(data: dict) -> dict:
    """
    Convert dictionary keys that are string representations of tuples 
    back into actual tuple keys. This is useful after loading JSON files
    where tuple keys were stringified.

    Parameters
    ----------
    data : dict
        Dictionary with keys as strings that may represent tuples (e.g., "(1, 2)").

    Returns
    -------
    dict
        Dictionary with tuple keys where applicable.
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

