# Source Code (`src/`)

## Overview

This folder contains all core Python modules used to model and solve the **Single-Source Capacitated Facility Location Problem (SSCFLP)** using both:

- The proposed **Pattern-based Kernel Search (PaKS)**;
- The conventional Kernel Search heuristic `KS14` presented in Guastaroba, G., & Speranza, M. G. (2014). [A heuristic for BILP problems: The Single Source Capacitated Facility Location Problem](hhttps://doi.org/10.1016/j.ejor.2014.04.007). *European Journal of Operational Research, 238(2), 438-450.* 

Both algorithms share a common structure, with configurable behavior controlled via parameter dictionaries.

---

## Folder Structure

### `kernel_search/`

Implements the main logic for both the **KS14** and **PaKS** metaheuristics.

- `kernel_search.py`  
  Core implementation of the kernel search algorithm.  
  Depending on the configuration passed, it executes either `KS14` or `PaKS`, with modular components shared between both.

- `biclustering.py`  
  Provides pattern recognition methods used in the **first phase of PaKS**.

- `configurations.py` 
  Defines configuration dictionaries for both algorithms:  
  - PaKS: parameters described in **Table 3** of the manuscript.
  - KS14: parameters as specified in the original publication.

---

### `models/`

Encapsulates instance data handling, problem definition, and solution representation.

- `instance.py`  
  Loads and stores instance data from `.json` files in a structured format.

- `ss_cflp.py`  
  Defines the **SSCFLP model**, including decision variables, objective function, and constraints.

- `solution.py`  
  Stores solution information (objective value, assignments, timing) and prepares output for export or analysis.

- `general_mip.py` 
  Provides generic MIP routines shared across problem variants.

---

### `utils/`

Auxiliary tools used across the project.

- `helpers.py`  
  A collection of helper functions for data processing.

- `plot.py`  
  Functions for visualizing instances and solutions to the SSCFLP.

- `FLPspatialpattern/GetCoordinates.py`  
   Functionality to approximate coordinates of candidates and customers for visualization.

---

## Notes

- All modules are imported via the `scripts/` interface (e.g., `main.py`) and are not designed to be run standalone.
- Paths are relative to the project root, with `src/` added to `sys.path` during execution.


