# Models Module

This module defines the core components for modeling and solving facility location problems
using mixed-integer programming (MIP) techniques and CPLEX.

It provides flexible model classes that can load instance data, build models, solve them using 
advanced MIP strategies, and support methods for linear relaxations and decomposition techniques 
useful for large-scale problems.

---

## Structure

| File | Description |
|:-----|:------------|
| `instance.py` | Handles loading, storing, and visualizing facility location instances from JSON files. |
| `general_mip.py` | Defines a base class for general MIP models, providing solve functionality with support for CPLEX callbacks. |
| `solution.py` | Defines the solution object, allowing for storing and visualization of optimized model solutions. |
| `ss_cflp.py` | Implements a specific single-source capacitated facility location problem (SSCFLP) model, extending the general MIP class with problem-specific constraints and valid inequalities. |

---

## Main Classes

### `Instance`
- Loads and manages instance data.
- Supports visualization of facilities and customers.
- Can store modified instances back to disk.

### `MIP`
- Abstract base class for MIP models.
- Handles model creation, parameter setting, solving, and callback registration.
- Provides a generic framework that can be extended by specific models.

### `Solution`
- Represents a solved instance of a model.
- Supports storing solutions in JSON format.
- Supports visualization of optimal solutions.

### `SS_CFLP`
- Specialized model class for the Single-Source Capacitated Facility Location Problem.
- Adds valid inequalities, kernel search mechanisms, LP relaxations, and bucket-based heuristics.
- Implements customized constraint handling for large-scale model decomposition.

---

## Dependencies

The model classes rely on:

- [DOcplex](https://ibmdecisionoptimization.github.io/docplex-doc/) (for modeling and solving MIP problems)
- [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio) (solver backend required)
- `numpy` (for array handling and reshaping)
- `matplotlib` (optional, only needed for visualization functions)

Make sure that DOcplex and CPLEX are installed and properly configured in your environment.

---

## Notes

- All model and solution files are structured for easy integration with external scripts in the `/scripts/` directory.
- Relative imports are used inside `/src/` to maintain modularity and package structure.
- Designed to support reproducible computational experiments for scientific publications.

---

## Example Usage

```python
from src.model.instance import Instance
from src.model.ss_cflp import SS_CFLP

# Load an instance
instance = Instance("path/to/instance.json")

# Build and solve the model
model = SS_CFLP(instance)
solution = model.solve(timelimit=600, mipgap=0.001)

# Visualize the solution
solution.visualize()
