# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
from sklearn.cluster import SpectralCoclustering
from sklearn.exceptions import ConvergenceWarning

def alpha(S):
    """
    Aggregate a set of solutions into a feature matrix A ∈ R^{I×J}.

    Parameters
    ----------
    S : dict[int, Solution]
        Dictionary of Solution objects, keyed by integer index
        (assumed 0, 1, ..., |S|-1). Each solution must provide
        `data["dvars"]["x"]` and `data["dvars"]["y"]`.

    Returns
    -------
    pd.DataFrame
        Feature matrix A with shape (I, J).
    """
    # Weights: first solution has full weight, others share weight 1
    lambda_s = [1] + [1 / (len(S) - 1) for _ in list(S.keys())[:-1]]

    # Infer dimensions I and J from the first solution
    I = len(S[0].data["dvars"]["y"].keys())
    J = int(len(S[0].data["dvars"]["x"].keys()) / I)
    A = np.zeros((I, J))
    
    for s, sol in S.items():
        # Build dense x_s matrix for this solution
        rows, cols = zip(*sol.data["dvars"]["x"].keys())
        values = list(sol.data["dvars"]["x"].values())
        x_s = np.zeros((I, J))
        x_s[rows, cols] = values

        # Work with ceil(x_ij) as a count-like quantity
        values = [np.ceil(v) for v in values]
        x_s[rows, cols] = values

        # Weighted accumulation into A
        A += x_s * lambda_s[s]

    return pd.DataFrame(A)

def remove_zeros(A):
    """
    Remove all rows and columns with only zero entries to avoid errors in biclustering.

    Parameters
    ----------
    A : pd.DataFrame
        Feature matrix.

    Returns
    -------
    pd.DataFrame
        Feature matrix of reduced dimensions.
    """
    A_dash = pd.DataFrame(A) #make I the columns
    A_dash = A.loc[:, (A != 0).any(axis=0)]
    A_dash = A.loc[(A != 0).any(axis=1), :]   
    return A_dash
        
def l_inter(Rcal, A):
    """
    Calculate the inter-region overlap metric.

    Parameters
    ----------
    Rcal : dict
        Dictionary of regions, with regionID as keys and (row_indices, column_indices) as values.
    A : pd.DataFrame
        Feature matrix.

    Returns
    -------
    float
        l_inter value in the range [0, 1].
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

def l_intra_R(R, A):
    """
    Calculate the intra-region overlap metric for a single region.

    Parameters
    ----------
    R : tuple
        One region represented as (row_indices, column_indices).
    A : pd.DataFrame
        Feature matrix.

    Returns
    -------
    float
        l_intra_R value in the range [0, 1].
    """
    intra_sum = 0
    # Get the column indices to include by excluding the specified columns
    col_indices_to_include = [col for col in A.columns if col not in R[1]]

    # Use NumPy to sum the selected elements efficiently
    intra_sum += A.iloc[R[0]][col_indices_to_include].values.sum()
    
    total_sum= A.iloc[R[0]].values.sum()
    if total_sum>0:
        return intra_sum/total_sum
    else:
        return 0.0
    
def bicluster(A, r):
    """
    Run spectral co-clustering on a feature matrix and build region assignments.

    Parameters
    ----------
    A : pd.DataFrame
        Feature matrix (rows and columns will be clustered).
    r : int
        Number of biclusters (regions).

    Returns
    -------
    dict[int, tuple[list[int], list[int]]]
        Mapping from region ID to (row_indices, column_indices).
        An additional key -1 collects indices not assigned because their
        row/column was removed during zero-filtering.
    """
    # Original row/column indices (used for mapping back)
    I = list(A.index)
    J = list(A.columns)

    # Remove all-zero rows/columns before clustering
    A_reduced = remove_zeros(A)

    # Spectral co-clustering model
    biclustering = SpectralCoclustering(
        n_clusters=r,
        n_init=10,
        random_state=np.random.RandomState(12051991),
    )
    biclustering.fit(A_reduced)

    # Initialize all labels as -1 (for rows/cols that may have been removed)
    row_labels = [-1] * len(I)
    col_labels = [-1] * len(J)
    
    # Map back row labels
    for pos, i in enumerate(A_reduced.index):
        row_labels[i] = biclustering.row_labels_[pos]

    # Map back column labels
    for pos, j in enumerate(A_reduced.columns):
        col_labels[j] = biclustering.column_labels_[pos]
     
    # Build region dictionary
    Rcal = {}
    for r_label in range(r):
        Rcal[r_label] = (
            [i for i in I if row_labels[i] == r_label],
            [j for j in J if col_labels[j] == r_label],
        )
    # Group everything that remained label -1
    Rcal[-1] = (
        [i for i in I if row_labels[i] == -1],
        [j for j in J if col_labels[j] == -1],
    )
    
    return Rcal
    
class Biclustering():
    """
    Simple biclustering wrapper driven by an l_inter threshold.
    """
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    
    def __init__(self, S, theta, logger = None):
        """
        Build the feature matrix and choose the number of biclusters r
        such that l_inter ≥ theta (as tightly as possible from below).

        Parameters
        ----------
        S : dict[int, Solution]
            Dictionary of solutions used to construct the feature matrix.
        theta : float
            Target threshold for the inter-region overlap metric l_inter.
        """
        # Aggregate solutions into feature matrix A
        self.A = alpha(S)

        # Start with r = 2 and increase until l_inter reaches theta
        r = 2
        self.Rcal = bicluster(self.A, r=r)
        self.l_inter = l_inter(self.Rcal, self.A)
        if logger: logger.debug(f"With {r} regions, l_inter = {self.l_inter}.")
        # Generate the feature matrix using the alpha function
        self.A = alpha(S)
        
        while self.l_inter < theta:
            r += 1
            Rcal_candidate = bicluster(self.A, r=r)
            l_inter_candidate = l_inter(Rcal_candidate, self.A)
            # Recall Rcal_candidate will have r+1 region - 1 with the non-open facilities that is added past the biclustering step.
            if logger: logger.debug(f"With {len(Rcal_candidate)} regions, l_inter = {l_inter_candidate}; theta = {theta}.")
            if l_inter_candidate <= theta:
                self.Rcal = Rcal_candidate
                self.l_inter = l_inter_candidate
            else:
                break
        if logger: logger.debug(f"Terminate with {len(self.Rcal)} regions, l_inter = {self.l_inter}.")