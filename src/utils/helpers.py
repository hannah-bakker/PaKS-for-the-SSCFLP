# -*- coding: utf-8 -*-
import time
import json
import ast
import pandas as pd
from models.solution import Solution

def time_step(func):
    """
    Decorator to time a function and store its execution time in a central dictionary.
    """
    def wrapper(*args, **kwargs):
        # Access the instance (self) from the args
        instance = args[0]
        if not hasattr(instance, "method_times"):
            instance.method_times = {}  # Initialize a dictionary to store times

        # Start timing
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        # Store elapsed time
        instance.method_times["time_"+str(func.__name__)] = elapsed_time

        # Optional: Print if verbose
        if getattr(instance, "verbose", False):
            print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        
        return result
    return wrapper

def add_line_to_df(df, line, index = None):
    if index is None: 
        index = len(df)+1
    if df.empty:
        df = pd.DataFrame(pd.Series(line), columns=[index]).transpose()
    else:
        df.loc[index] = pd.Series(line)                       

    return df

def concat_to_string_name(mylist):
    name = str(mylist[0])
    for e in mylist[1:]:
        name+="_" +str(e)
    return name

def convert_keys_to_tuples(data):
    """
    When storing the data  as a json, the tuples are converted to strings. In order to be 
    able to access them as tuples again, we have to reconvert

    Parameters
    ----------
    data : dict
        dictionary with tuples as strings keys.

    Returns
    -------
    new_data : dict
        dictionary with tuple keys.

    """
    new_data = {}
    for key, value in data.items():
        try:
            # Use ast.literal_eval to convert string representation of tuple back to tuple
            new_key = ast.literal_eval(key)
            if isinstance(new_key, tuple):
                new_key = tuple(int(x) if isinstance(x, str) and x.isdigit() else x for x in new_key)

            new_data[new_key] = value
        except (ValueError, SyntaxError):
            # Handle cases where the key is not a tuple or is not a valid string representation
            print(f"Key '{key}' is not a valid tuple string.")
            new_data[key] = value
    return new_data


def store_S(S, path):
     """
     Store set of solutions

     Parameters
     ----------
     S : dictionary of Solutions
     
     path : String
         Path+filename.

     Returns
     -------
     None.
     """
     
     for s in S.values():
        s.data["dvars"]["x"] = {str(key): value for key, value in s.data["dvars"]["x"].items()} #convert tuple to string
        if "reduced_costs" in s.data:
            s.data["reduced_costs"]["x"] = {str(key): value for key, value in s.data["reduced_costs"]["x"].items()} #convert tuple to string

     store_dict = {s:S[s].data for s in S.keys()}
     with open(path+".json", "w") as json_file:
         json.dump(store_dict, json_file, indent=1)
         json_file.close()
         
     for s in S.values():
        s.data["dvars"]["x"] = {eval(key): value for key, value in s.data["dvars"]["x"].items()} #convert tuple to string
        if "reduced_costs" in s.data:
                s.data["reduced_costs"]["x"] = {eval(key): value for key, value in s.data["reduced_costs"]["x"].items()} #convert tuple to string
           
def load_S(path, instance):
      """
      Load set of solutions

      Parameters
      ----------
      path : String
          Path+filename.

      Returns
      -------
      S : dictionary of Solutions
      """
      with open(path) as json_file:
            data = json.load(json_file)
            json_file.close()
     
      S = {int(s):Solution(instance = instance, from_dict = solution_vals) for s, solution_vals in data.items()}
      
      for s, solution in S.items():
          for decision_variable in solution.data["dvars"]:
              solution.data["dvars"][decision_variable] = convert_keys_to_tuples(solution.data["dvars"][decision_variable])
      return S
 
    
def overlap(A, B):
    """
        Overlap between two lists. (A \cap B)/(A \cup B)

    Parameters
    ----------
    A : list
        List of indices
    B : list
        List of indices

    Returns
    -------
    o : float
        Overlap coefficient of the two lists, in [0,1].

    """
    # Convert lists to sets to remove duplicates and allow for set operations
    A = set(A)
    B = set(B)
    
    # Calculate the intersection and union
    intersection = A.intersection(B)
    min_card = min(len(A), len(B))
    
    # Compute the ratio of the length of intersection to the length of union
    o = round(len(intersection) / min_card,2) if min_card > 0 else 0
    return o

def Jacc(A, B):
    """
        Overlap between two lists. (A \cap B)/(A \cup B)

    Parameters
    ----------
    A : list
        List of indices
    B : list
        List of indices

    Returns
    -------
    o : float
        Overlap coefficient of the two lists, in [0,1].

    """
    # Convert lists to sets to remove duplicates and allow for set operations
    
    A = set(A)
    B = set(B)
    
    # Calculate the intersection and union
    intersection = A.intersection(B)
    union = A.union(B)
    
    # Compute the ratio of the length of intersection to the length of union
    o = round(len(intersection) / len(union),2) if len(union) > 0 else 0
   # print(f"{A}, {B}: {o}")
    return o