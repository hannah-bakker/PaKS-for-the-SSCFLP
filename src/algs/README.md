# PaKS-for-the-SSCFLP

# Biclustering Module

This module provides a framework for identifying *regions* in the solution space of the **Single-Source Capacitated Facility Location Problem (SSCFLP)** using biclustering techniques. It is particularly useful for *kernel-based heuristics*, such as PaKS, to detect coherent groups of facilities and customers based on patterns in LP-relaxation solutions.

---

## 📌 Purpose

The primary goal of this script is to **construct a region-based partitioning** of facilities and customers by:
- Aggregating assignment decisions across multiple LP-relaxation solutions.
- Applying **Spectral Co-clustering** to identify biclusters (facility–customer regions).
- Evaluating region coherence using the `l_inter` metric.
- Iteratively increasing the number of biclusters until region coherence exceeds a given threshold.

---

## 📂 Functions and Classes

### `alpha(S: dict) -> pd.DataFrame`
Aggregates a dictionary of `Solution` objects into an I×J matrix representing the number of (rounded-up) assignments between facilities and customers.

### `remove_zeros(A: pd.DataFrame) -> pd.DataFrame`
Cleans the matrix by removing rows and columns (facilities/customers) that were never assigned in any solution.

### `l_inter(Rcal: dict, A: pd.DataFrame) -> float`
Computes the inter-region coherence metric: the proportion of assignment mass that remains within its assigned region. Higher values imply more coherent regions.

### `bicluster(A: pd.DataFrame, r: int) -> dict`
Performs **Spectral Co-clustering** to identify `r` biclusters, each represented as a tuple of facility and customer indices.

### `class Biclustering`
Main class that orchestrates the biclustering process. Given a set of solutions and a coherence threshold `theta`, it:
- Builds the feature matrix (`alpha`)
- Repeatedly clusters until `l_inter` exceeds `theta`
- Stores the final region structure in `self.Rcal`

---

## ✅ Example Use

```python
from biclustering import Biclustering

# S is a dictionary of LP-based Solution objects
bic = Biclustering(S, theta=0.85)

# Access bicluster-based regions
regions = bic.Rcal