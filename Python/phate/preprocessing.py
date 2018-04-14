# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from __future__ import print_function, division
from sklearn.preprocessing import normalize
import numpy as np
from scipy import sparse

try:
    import pandas as pd
except ImportError:
    pass


def library_size_normalize(data, verbose=False):
    """Performs L1 normalization on input data
    Performs L1 normalization on input data such that the sum of expression values for each cell sums to 1
    then returns normalized matrix to the metric space using median UMI count per
    cell effectively scaling all cells as if they were sampled evenly.

    Parameters
    ----------
    data : ndarray [n,p]
        2 dimensional input data array with n cells and p dimensions

    Returns
    -------
    data_norm : ndarray [n, p]
        2 dimensional array with normalized gene expression values
    """
    if verbose:
        print("Normalizing library sizes for %s cells" % (data.shape[0]))

    try:
        if isinstance(data, pd.core.sparse.frame.SparseDataFrame):
            data = data.to_coo()
        elif isinstance(data, pd.DataFrame):
            data = np.array(data)
    except NameError:
        pass
    median_transcript_count = np.median(np.array(data.sum(axis=1)))
    try:
        data_norm = normalize(data, norm='l1', axis=1)
    except MemoryError:
        data_norm = sparse.vstack([normalize(
            data[i, :], 'l1', axis=1) for i in range(data.shape[0])])

    # norm = 'l1' computes the L1 norm which computes the
    # axis = 1 independently normalizes each sample

    data_norm = data_norm * median_transcript_count
    return data_norm
