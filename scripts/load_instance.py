# -*- coding: utf-8 -*-
"""
load_instance.py

Converts benchmark instances from original formats (.txt or .plc)
into a unified JSON format for reproducibility.

Supports:
- OR4 instances from the OR-Library (.txt format)
- Instances from Avella and Boccia (.plc format)

Usage:
    python load_instance.py <folder_path> <name> <test_set> [<capacity>]

Example:
    python load_instance.py ../raw_data capa OR4 5000
"""

import os
import sys
import json
import numpy as np

# Add the src directory to the Python path (if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

def convert(folder_path, name, test_set, capacity=0):
    """
    Converts a benchmark instance file into a unified JSON format.

    Args:
        folder_path (str): Path to the folder containing the input file.
        name (str): Name of the instance (without extension).
        test_set (str): Either "OR4" or another set ("B1", "B2", etc.).
        capacity (float, optional): Fixed facility capacity for OR4 instances.
    """
    params = {
        "I": 0,
        "J": 0,
        "D_j": [],
        "Q_i": [],
        "F_i": [],
        "c_ij": []
    }

    if test_set == "OR4":
        input_file = os.path.join(folder_path, name + ".txt")
        with open(input_file, 'r') as file:
            line_counter = 0
            for line in file:
                line = line.strip()
                if not line:
                    continue

                if line_counter == 0:
                    # First line contains |J| and |I|, but swapped compared to our model
                    params["I"], params["J"] = map(int, line.split())
                    c_ij = np.zeros((params["I"], params["J"]))
                    j = 0
                    cost_column = []
                elif 1 <= line_counter <= params["I"]:
                    # Facility data: fixed capacity and opening cost
                    Q_i_value, F_i_value = line.split()
                    params["Q_i"].append(float(capacity))  # Overwrite Q_i by user-specified capacity
                    params["F_i"].append(float(F_i_value))
                elif line_counter > params["I"]:
                    # Demand and transportation costs
                    if len(params["D_j"]) < j + 1:
                        D = float(line.split()[0])
                        params["D_j"].append(D)
                    else:
                        cost_column.extend(map(float, line.split()))
                        if len(cost_column) == params["I"]:
                            c_ij[:, j] = cost_column
                            j += 1
                            cost_column = []
                line_counter += 1

            # Normalize transportation costs by demand
            c_ij = c_ij / np.array(params["D_j"])
            c_ij = np.round(c_ij, 8)
            params["c_ij"] = c_ij.tolist()

    else:
        input_file = os.path.join(folder_path, name + ".plc")
        with open(input_file, 'r') as file:
            line_counter = 0
            for line in file:
                line = line.strip()
                if not line:
                    continue

                if line_counter == 0:
                    # First line contains |J| and |I| (swapped order compared to our model)
                    parts = line.split()
                    params["I"] = int(parts[1])
                    params["J"] = int(parts[0])
                else:
                    # Read demands, capacities, fixed costs, and transportation costs sequentially
                    if len(params["D_j"]) < params["J"]:
                        params["D_j"].extend(map(float, line.split()))
                    elif len(params["Q_i"]) < params["I"]:
                        params["Q_i"].extend(map(float, line.split()))
                    elif len(params["F_i"]) < params["I"]:
                        params["F_i"].extend(map(float, line.split()))
                    else:
                        params["c_ij"].extend(map(float, line.split()))
                line_counter += 1

            # Reshape transportation cost list into a 2D array
            array = np.array(params["c_ij"])
            params["c_ij"] = array.reshape(params["I"], params["J"]).tolist()

    # Create the final instance dictionary
    instance = {
        "info": {
            "name": name
        },
        "params": params
    }

    # Ensure output folder exists
    output_folder = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(output_folder, exist_ok=True)
    
    output_path = os.path.join(output_folder, name + ".json")
    with open(output_path, "w") as outfile:
        json.dump(instance, outfile, indent=4)

    print(f"Instance '{name}' converted successfully and saved to '{output_path}'.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python load_instance.py <folder_path> <name> <test_set> [<capacity>]")
        sys.exit(1)

    folder_path = sys.argv[1]
    name = sys.argv[2]
    test_set = sys.argv[3]

    if len(sys.argv) > 4:
        capacity = float(sys.argv[4])
        convert(folder_path, name, test_set, capacity)
    else:
        convert(folder_path, name, test_set)
