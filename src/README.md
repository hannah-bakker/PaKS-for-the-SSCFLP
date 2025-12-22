# Source Code (`src/`)

## Overview

This folder contains all core Python modules used to model and solve the **Single-Source Capacitated Facility Location Problem (SSCFLP)** using the proposed **Pattern-based Kernel Search (PaKS)**.

---

## Folder Structure

### `algs/`

Implements the main logic for **PaKS** metaheuristics.

- `paks.py`  
  Core implementation of the pattern-based kernel search algorithm.  

- `paks_configs.py` 
  Defines configuration dictionaries.  
  - default_configuration: parameters described in **Table 3** of the manuscript.

---

### `models/`

Encapsulates instance data handling, problem definition, and solution representation.

- `cb_report.py`  
  Callback reporting intermediate solutions in regular intervals.

- `instance.py`  
  Loads and stores instance data from `.json` files in a structured format.

- `sscflp.py`  
  Defines the **SSCFLP model**, including decision variables, objective function, and constraints.

- `solution.py`  
  Stores solution information (objective value, assignments, timing) and prepares output for export or analysis.

- `mip.py` 
  Provides generic MIP routines shared across problem variants.

---

### `utils/`

Auxiliary tools used across the project.

- `helpers.py`  
  A collection of helper functions for data processing.

- `biclustering.py`  
  Implements the biclustering step in the pattern-recognition phase.

---

## Notes

- All modules are imported via the `scripts/` interface (e.g., `main.py`) and are not designed to be run standalone.
- Paths are relative to the project root, with `src/` added to `sys.path` during execution.


