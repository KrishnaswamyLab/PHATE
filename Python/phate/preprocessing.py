# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from __future__ import print_function, division
import sklearn.preprocessing
import sklearn.decomposition
import numpy as np
import scipy.sparse as sp


def pca_reduce(data, n_components=100, solver='sparse', verbose=False):
    """PCA dimensionality reduction
    Reduces input matrix and saves n_components. If solver='sparse', then
    `sklearn.decomposition.TruncatedSVD` is used for dimensionality reduction.
    This is faster than performing SVD on a dense matrix for sparse data. If PCA on a dense
    matrix is required, then use 'out', 'svd', or 'random'.

    Parameters
    ----------
    data: array-like [n, p]
        2 dimensional input data array-like with n cells and p dimensions

    n_components : int, optional, default: 100
        number of components to keep

    solver : string, optional, default: 'sparse'
        If solver='sparse', then `sklearn.decomposition.TruncatedSVD` is used for dimensionality reduction. (Fast)
        Othervise, value is passed to sklearn.decomposition.PCA()
        allowable values: ['auto', 'svd', 'random', 'sparse']

    Returns
    -------
    data_reduced : ndarray [n, n_components]
        input data reduced to desired number of dimensions
    """

    if verbose:
        print('Running PCA to %s dimensions using %s PCA...' %
              (n_components, solver))
    if solver == 'sparse':
        if not sp.issparse(data):
            try:
                data = sp.csc_matrix(np.array(data))
            except TypeError:
                raise TypeError("Input data must be castable as np.array().")

        pca_solver = sklearn.decomposition.TruncatedSVD(
            n_components=n_components)
        data_reduced = pca_solver.fit_transform(data)
    else:
        pca_solver = sklearn.decomposition.PCA(
            n_components=n_components, svd_solver=solver)
        data_reduced = pca_solver.fit_transform(data)

    return data_reduced


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
    data_norm = sklearn.preprocessing.normalize(data, norm='l1', axis=1)
    # norm = 'l1' computes the L1 norm which computes the
    # axis = 1 independently normalizes each sample

    median_transcript_count = np.median(data.sum(axis=1))
    data_norm = data_norm * median_transcript_count
    return data_norm
