# -*- coding: utf-8 -*-
"""
biclustering.py

Defines functions and a class for performing biclustering based on feature matrices 
derived from a set of solutions to the SSCFLP.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import SpectralCoclustering
from sklearn.exceptions import ConvergenceWarning

def alpha(S: dict) -> pd.DataFrame:
    """
    Aggregate assignment decisions across a set of solutions into an I×J count matrix.

    Parameters
    ----------
    S : dict
        Dictionary of Solution objects. Each must contain 'dvars' with 'x' decisions.

    Returns
    -------
    pd.DataFrame
        I x J count matrix with the total number of assignments (after rounding) across solutions.
    """
    # Infer problem size from first solution
    I = len(S[0].data["dvars"]["y"])
    J = int(len(S[0].data["dvars"]["x"]) / I)
    A = np.zeros((I, J))

    for sol in S.values():
        x_vals = sol.data["dvars"]["x"]
        x_s = np.zeros((I, J))
        for (i, j), v in x_vals.items():
            x_s[i, j] = np.ceil(v)
        A += x_s

    return pd.DataFrame(A)

def remove_zeros(A: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all-zero rows (facilities) from the feature matrix.

    Parameters
    ----------
    A : pd.DataFrame
        I x J feature matrix where each entry (i, j) reflects the number of times
        facility i was assigned to customer j across solutions.

    Returns
    -------
    pd.DataFrame
        Reduced feature matrix without zero-only rows (facilities) or columns (customers).
    """
    # Remove all-zero rows (facilities)
    A = A.loc[(A != 0).any(axis=1), :]
    return A

def l_inter(Rcal: dict, A: pd.DataFrame) -> float:
    """
    Compute the metric l_inter.

    Parameters
    ----------
    Rcal : dict
        Regions.
    A : pd.DataFrame
        Numeric feature matrix.

    Returns
    -------
    float
        l_inter value in the range [0, 1], representing the fraction of the
        out-of-region-allocations.
    """
    total_sum = 0
    for R in Rcal.values():
        # Extract the row indices and column indices
        row_indices = R[0]
        col_indices_to_exclude = R[1]

        # Get the column indices to include by excluding the specified columns
        col_indices_to_include = [col for col in A.columns if col not in col_indices_to_exclude]

        # Use NumPy to sum the selected elements efficiently
        total_sum += A.iloc[row_indices][col_indices_to_include].values.sum()
    return total_sum/A.values.sum()

def bicluster(A: pd.DataFrame, r: int) -> dict:
    """
    Perform Spectral Co-clustering to identify biclusters in a binary or weighted feature matrix.

    Parameters
    ----------
    A : pd.DataFrame
        Feature matrix.
    r : int
        Number of biclusters to identify (must be > 0).

    Returns
    -------
    dict
        A dictionary of biclusters in the format:
            {cluster_id: (row_indices, column_indices), ...}
    """
    I = [i for i in A.index] 
    J = [j for j in A.columns] 
    
    A = remove_zeros(A)

    biclustering = SpectralCoclustering(n_clusters=r, 
                                n_init = 10,
                                random_state=np.random.RandomState(12051991),
                                )
    biclustering.fit(A)  
    row_labels = [-1 for i in I]
    column_labels = [-1 for j in J]  
    
    for position, i in enumerate(A.index):
        row_labels[i] = biclustering.row_labels_[position]
        
    for position, j in enumerate(A.columns):
        column_labels[j] = biclustering.column_labels_[position]     
   
    Rcal = dict()
    for r_label in range(r):
        Rcal[r_label] = ([i for i in I if row_labels[i] == r_label], [j for j in J if column_labels[j] == r_label])
    Rcal[-1] = ([i for i in I if row_labels[i] == -1], [j for j in J if column_labels[j] == -1])
    
    return Rcal

class Biclustering:
    """
    Biclustering framework for identifying coherent regions from LP-relaxation solutions.

    Uses Spectral Co-clustering to group facilities and customers into assignment regions,
    stopping when inter-region coherence exceeds a defined threshold (theta).   
    """

    def __init__(self, S: dict, theta: float):
        """
        Initialize and run biclustering until the inter-region overlap drops below theta.

        Parameters
        ----------
        S : dict
            Dictionary of Solution objects (indexed by integer keys).
        theta : float
            Threshold on the inter-region overlap score to stop clustering.
        """
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        self.A = alpha(S)
        self.l_inter = 0.0

        r = 2
        self.Rcal = bicluster(self.A, r)
        self.l_inter = l_inter(self.Rcal, self.A)

        # Increase the number of clusters until inter-region coherence (l_inter) exceeds threshold
        while self.l_inter < theta:
                r += 1
                Rcal = bicluster(self.A, r=r)
                self.l_inter = l_inter(Rcal, self.A)
                if self.l_inter <= theta:
                    self.Rcal = Rcal
