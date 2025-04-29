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

- `SSCFLPPlot.py`  
  A comprehensive plotting class (`SSCFLPPlot`) for:
  - Visualizing instances
  - Plotting solutions and assignments
  - Displaying kernel search regions and buckets

- `FLPspatialpattern/GetCoordinates.py`  
  Utility to reconstruct 2D coordinates for facility and customer locations based on the cost matrix.
