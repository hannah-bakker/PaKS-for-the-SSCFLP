# Utils

This folder provides helper modules for solution handling, clustering, and visualization tasks used across the project.

## Folder Structure

- `helpers.py`  
  General-purpose helper functions including:
  - Timing decorators
  - Saving and loading solutions
  - List overlap computations (e.g., Jaccard index)

- `biclustering.py`  
  Functions and classes for:
  - Generating feature matrices from solutions
  - Performing biclustering of facilities and customers
  - Computing region overlap metrics (`l_inter`, `l_intra`)