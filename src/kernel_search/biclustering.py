# -*- coding: utf-8 -*-
"""
biclustering.py

Defines functions and a class for performing biclustering based on feature matrices 
derived from facility location solutions.

Supports various aggregation schemes, removal of sparse regions, 
and computation of intra- and inter-region overlap metrics.

"""

import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import SpectralCoclustering
from sklearn.exceptions import ConvergenceWarning

def alpha(S: dict) -> pd.DataFrame:
    """
    Aggregate a set of solution vectors into a binary feature matrix.

    This function computes an I x J matrix where each entry (i, j) counts how often
    assignment variable x_ij is active (rounded up to 1) across all solutions in S.

    Parameters
    ----------
    S : dict
        Dictionary of Solution objects, where each Solution includes 'dvars' with keys 'x' and 'y'.

    Returns
    -------
    pd.DataFrame
        I x J binary feature matrix representing the union of assignment decisions.
    """

    I = len(S[0].data["dvars"]["y"])
    J = int(len(S[0].data["dvars"]["x"]) / I)
    A = np.zeros((I, J))

    for s, sol in S.items():
        rows, cols = zip(*sol.data["dvars"]["x"].keys())
        values = list(sol.data["dvars"]["x"].values())
        x_s = np.zeros((I, J))
        x_s[rows, cols] = values
        x_s[rows, cols] = [np.ceil(v) for v in values]
        A += x_s
        
    return pd.DataFrame(A)

def remove_zeros(A: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all-zero rows and columns from a binary or numeric feature matrix.

    Parameters
    ----------
    A : pd.DataFrame
        Input feature matrix where rows typically represent facilities and
        columns represent customers or features.

    Returns
    -------
    pd.DataFrame
        Reduced feature matrix with only non-zero rows and columns.
    """
    A_dash = A.loc[:, (A != 0).any(axis=0)]
    A_dash = A_dash.loc[(A_dash != 0).any(axis=1), :]
    return A_dash

def l_inter(Rcal: dict, A: pd.DataFrame) -> float:
    """
    Compute the inter-region overlap metric l_inter.

    This metric quantifies the proportion of nonzero activity in a feature matrix A
    that lies outside the explicitly excluded columns defined by Rcal.

    Parameters
    ----------
    Rcal : dict
        Dictionary mapping region identifiers to tuples of the form (rows, excluded_cols),
        where:
            - rows: list of row indices in A belonging to a region.
            - excluded_cols: list of column indices to be excluded for that region.

    A : pd.DataFrame
        Binary or numeric feature matrix (e.g., facility-customer interactions).

    Returns
    -------
    float
        Normalized l_inter value in the range [0, 1], representing the fraction of the
        total matrix sum that lies *outside* the excluded columns for each region.
    """
    total_sum = 0
    for rows, excluded_cols in Rcal.values():
        included_cols = [col for col in A.columns if col not in excluded_cols]
        total_sum += A.iloc[rows][included_cols].values.sum()
    return total_sum / A.values.sum()

def bicluster(A: pd.DataFrame, r: int) -> dict:
    """
    Perform Spectral Co-clustering to identify biclusters in a binary or weighted feature matrix.

    Parameters
    ----------
    A : pd.DataFrame
        Feature matrix (e.g., facility-customer associations). Should be non-empty and non-trivial.
    r : int
        Number of biclusters to identify (must be > 0).

    Returns
    -------
    dict
        A dictionary of biclusters in the format:
            {cluster_id: (row_indices, column_indices), ...}
        Includes an additional entry with key -1 for any rows/columns not assigned to a cluster
        (though this is unlikely with SpectralCoclustering).
    """
    I = list(A.index)
    J = list(A.columns)
    
    A = remove_zeros(A)
    model = SpectralCoclustering(n_clusters=r, n_init=10, random_state=12051991)
    model.fit(A)

    row_labels = {i: model.row_labels_[pos] for pos, i in enumerate(A.index)}
    col_labels = {j: model.column_labels_[pos] for pos, j in enumerate(A.columns)}

    Rcal = {}
    for r_label in range(r):
        Rcal[r_label] = (
            [i for i in I if row_labels.get(i, -1) == r_label],
            [j for j in J if col_labels.get(j, -1) == r_label]
        )
    Rcal[-1] = (
        [i for i in I if row_labels.get(i, -1) == -1],
        [j for j in J if col_labels.get(j, -1) == -1]
    )

    return Rcal

class Biclustering:
    """
    Biclustering framework to derive bicluster-based regions from multiple LP-based solutions.

    This class computes an aggregated feature matrix from a dictionary of Solution objects,
    and uses spectral co-clustering to identify overlapping regions of facility-customer interactions.

    Attributes
    ----------
    A : pd.DataFrame
        Aggregated feature matrix of binary assignment variables.
    Rcal : dict
        Dictionary of identified bicluster regions (rows, excluded columns).
    Rcal_set : dict
        Unused (placeholder for extensions, e.g., tracking iterations).
    l_inter : float
        Inter-region overlap score, indicating region separation quality.
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
            Lower theta implies less overlap between identified regions (more separated clusters).
        """
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        self.A = alpha(S)
        self.Rcal_set = {}
        self.l_inter = 0.0

        r = 2
        self.Rcal = bicluster(self.A, r)
        self.l_inter = l_inter(self.Rcal, self.A)

        while self.l_inter < theta:
            r += 1
            Rcal_candidate = bicluster(self.A, r)
            l_inter_candidate = l_inter(Rcal_candidate, self.A)
            if l_inter_candidate <= theta:
                self.Rcal = Rcal_candidate
                self.l_inter = l_inter_candidate
