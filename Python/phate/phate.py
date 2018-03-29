"""
Potential of Heat-diffusion for Affinity-based Trajectory Embedding (PHATE)
"""

# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

import time
import numpy as np
import sys
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.linalg import svd

# sdfasdf
from .mds import embed_MDS


def calculate_kernel(M, knn_dist, k, a, verbose=True):
    if verbose:
        print("Building kNN graph and diffusion operator...")
    try:
        pdx = squareform(pdist(M, metric=knn_dist))
        knn_dist = np.sort(pdx, axis=1)
        # bandwidth(x) = distance to k-th neighbor of x
        epsilon = knn_dist[:, k]
        pdx = (pdx / epsilon).T  # autotuning d(x,:) using epsilon(x).
    except RuntimeWarning:
        raise ValueError(
            'It looks like you have at least k identifical data points.'
            ' Try removing dupliates.')

    gs_ker = np.exp(-1 * (pdx ** a))  # not really Gaussian kernel
    gs_ker = gs_ker + gs_ker.T  # symmetriziation
    return gs_ker


def calculate_potential(data, a=10, k=5, t=30, knn_dist='euclidean',
                        gs_ker=None, diff_op=None,
                        diff_potential=None, njobs=1, verbose=True):
    """
    Calculate the diffusion potential

    Parameters
    ----------
    data : ndarray [n, p]
        2 dimensional input data array with n cells and p dimensions

    a : int, optional, default: 10
        sets decay rate of kernel tails

    k : int, optional, default: 5
        used to set epsilon while autotuning kernel bandwidth

    t : int, optional, default: 30
        power to which the diffusion operator is powered
        sets the level of diffusion

    knn_dist : string, optional, default: 'euclidean'
        reccomended values: 'eucliean' and 'cosine'
        Any metric from scipy.spatial.distance can be used
        distance metric for building kNN graph

    gs_ker : array-like, shape [n_samples, n_samples]
        Precomputed graph kernel

    diff_op : ndarray, optional [n, n], default: None
        Precomputed diffusion operator

    diff_potential : ndarray, optional [n, n], default: None
        Precomputed diffusion potential

    verbose : boolean, optional, default: True
        Print updates during PHATE embedding

    Returns
    -------
    gs_ker : array-like, shape [n_samples, n_samples]
        The graph kernel built on the input data
        Only necessary for calculating Von Neumann Entropy

    diff_op : array-like, shape [n_samples, n_samples]
        The diffusion operator fit on the input data

    diff_potential : array-like, shape [n_samples, n_samples]
        Precomputed diffusion potential
    """
    # print('Imported numpy: %s'%np.__file__)
    # if nothing is precomputed
    if (gs_ker is None and
            (diff_op is not None or diff_potential is not None)) \
            or (diff_op is None and diff_potential is not None):
        print("Warning: incomplete precomputed matrices provided, recomputing.")
        diff_op = None
        diff_potential = None
    tic = time.time()
    if gs_ker is None:
        gs_ker = calculate_kernel(data, knn_dist, k, a, verbose=verbose)
    if diff_op is None:
        diff_op = gs_ker / gs_ker.sum(axis=1)[:, None]  # row stochastic
        if verbose:
            print("Built graph and diffusion operator in %.2f seconds." %
                  (time.time() - tic))
    else:
        if verbose:
            print("Using precomputed diffusion operator...")

    if diff_potential is None:
        tic = time.time()
        if verbose:
            print("Calculating diffusion potential...")
        # transforming X
        # print('Diffusion operator • %s:'%t)
        # print(diff_op)
        X = np.linalg.matrix_power(diff_op, t)  # diffused diffusion operator
        # print('X:')
        # print(X)
        X[X == 0] = np.finfo(float).eps  # handling zeros
        X[X <= np.finfo(float).eps] = np.finfo(
            float).eps  # handling small values
        diff_potential = -1 * np.log(X)  # diffusion potential
        if verbose:
            print("Calculated diffusion potential in %.2f seconds." %
                  (time.time() - tic))
    # if diffusion potential is precomputed (i.e. 'mds' or 'mds_dist' has
    # changed on PHATE object)
    else:
        if verbose:
            print("Using precomputed diffusion potential...")

    return gs_ker, diff_op, diff_potential


def embed_mds(diff_potential, n_components=2, mds='metric',
              mds_dist='euclidean', njobs=1, random_state=None, verbose=True):
    """
    Create the MDS embedding from the diffusion potential

    Parameters
    ----------
    diff_potential : ndarray, optional [n, n], default: None
        Diffusion potential

    n_components : int, optional, default: 2
        number of dimensions in which the data will be embedded

    mds : string, optional, default: 'metric'
        choose from ['classic', 'metric', 'nonmetric']
        which multidimensional scaling algorithm is used for dimensionality
        reduction

    mds_dist : string, optional, default: 'euclidean'
        reccomended values: 'eucliean' and 'cosine'
        Any metric from scipy.spatial.distance can be used
        distance metric for MDS

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global numpy random number generator

    verbose : boolean, optional, default: True
        Print updates during PHATE embedding

    Returns
    -------
    embedding : ndarray [n_samples, n_components]
        PHATE embedding in low dimensional space.
    """

    tic = time.time()
    if verbose:
        print("Embedding data using %s MDS..." % (mds))
    embedding = embed_MDS(diff_potential, ndim=n_components, how=mds,
                          distance_metric=mds_dist, njobs=njobs,
                          seed=random_state)
    if verbose:
        print("Embedded data in %.2f seconds." % (time.time() - tic))
    return embedding


def embed_phate(data, n_components=2, a=10, k=5, t=30, mds='metric',
                knn_dist='euclidean', mds_dist='euclidean', diff_op=None,
                gs_ker=None, diff_deg=None, diff_potential=None, njobs=1,
                random_state=None, verbose=True):
    """
    Embeds high dimensional single-cell data into two or three dimensions for
    visualization of biological progressions.

    Parameters
    ----------
    data : ndarray [n, p]
        2 dimensional input data array with n cells and p dimensions

    n_components : int, optional, default: 2
        number of dimensions in which the data will be embedded

    a : int, optional, default: 10
        sets decay rate of kernel tails

    k : int, optional, default: 5
        used to set epsilon while autotuning kernel bandwidth

    t : int, optional, default: 30
        power to which the diffusion operator is powered
        sets the level of diffusion

    mds : string, optional, default: 'metric'
        choose from ['classic', 'metric', 'nonmetric']
        which multidimensional scaling algorithm is used for dimensionality
        reduction

    knn_dist : string, optional, default: 'euclidean'
        reccomended values: 'eucliean' and 'cosine'
        Any metric from scipy.spatial.distance can be used
        distance metric for building kNN graph

    mds_dist : string, optional, default: 'euclidean'
        reccomended values: 'eucliean' and 'cosine'
        Any metric from scipy.spatial.distance can be used
        distance metric for MDS

    diff_op : ndarray, optional [n, n], default: None
        Precomputed diffusion operator

    diff_potential : ndarray, optional [n, n], default: None
        Precomputed diffusion potential

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global numpy random number generator

    verbose : boolean, optional, default: True
        Print updates during PHATE embedding

    Returns
    -------
    embedding : ndarray [n_samples, n_components]
        PHATE embedding in low dimensional space.

    gs_ker : array-like, shape [n_samples, n_samples]
        The graph kernel built on the input data
        Only necessary for calculating Von Neumann Entropy

    diff_op : array-like, shape [n_samples, n_samples]
        The diffusion operator fit on the input data

    diff_potential : array-like, shape [n_samples, n_samples]
        Precomputed diffusion potential

    References
    ----------
    .. [1] `Moon KR, van Dijk D, Zheng W, et al. (2017). "PHATE: A
       Dimensionality Reduction Method for Visualizing Trajectory Structures in
       High-Dimensional Biological Data". Biorxiv.
       <http://biorxiv.org/content/early/2017/03/24/120378>`_
    """
    gs_ker, diff_op, diff_potential = calculate_potential(
        data, a=a, k=k, t=t, knn_dist=knn_dist, gs_ker=gs_ker,
        diff_op=diff_op, diff_potential=diff_potential,
        njobs=njobs, verbose=verbose)
    embedding = embed_mds(
        diff_potential, n_components=n_components, mds=mds, mds_dist=mds_dist,
        njobs=njobs, random_state=random_state, verbose=verbose)
    return embedding, gs_ker, diff_op, diff_potential


class PHATE(BaseEstimator):
    """Potential of Heat-diffusion for Affinity-based Trajectory Embedding (PHATE)
    Embeds high dimensional single-cell data into two or three dimensions for
    visualization of biological progressions.

    Parameters
    ----------
    data : ndarray [n, p]
        2 dimensional input data array with n cells and p dimensions

    n_components : int, optional, default: 2
        number of dimensions in which the data will be embedded

    a : int, optional, default: 10
        sets decay rate of kernel tails

    k : int, optional, default: 5
        used to set epsilon while autotuning kernel bandwidth

    t : int, optional, default: 30
        power to which the diffusion operator is powered
        sets the level of diffusion

    mds : string, optional, default: 'metric'
        choose from ['classic', 'metric', 'nonmetric']
        which MDS algorithm is used for dimensionality reduction

    knn_dist : string, optional, default: 'euclidean'
        reccomended values: 'eucliean' and 'cosine'
        Any metric from scipy.spatial.distance can be used
        distance metric for building kNN graph

    mds_dist : string, optional, default: 'euclidean'
        reccomended values: 'eucliean' and 'cosine'
        Any metric from scipy.spatial.distance can be used
        distance metric for MDS

    njobs : integer, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global numpy random number generator

    Attributes
    ----------

    embedding : array-like, shape [n_samples, n_dimensions]
        Stores the position of the dataset in the embedding space

    gs_ker : array-like, shape [n_samples, n_samples]
        The graph kernel built on the input data
        Only necessary for calculating Von Neumann Entropy

    diff_op : array-like, shape [n_samples, n_samples]
        The diffusion operator fit on the input data

    diff_potential : array-like, shape [n_samples, n_samples]
        Precomputed diffusion potential

    References
    ----------
    .. [1] `Moon KR, van Dijk D, Zheng W, et al. (2017). "PHATE: A
       Dimensionality Reduction Method for Visualizing Trajectory Structures in
       High-Dimensional Biological Data". Biorxiv.
       <http://biorxiv.org/content/early/2017/03/24/120378>`_
    """

    def __init__(self, n_components=2, a=10, k=5, t=30, mds='metric',
                 knn_dist='euclidean', mds_dist='euclidean', njobs=1,
                 random_state=None, verbose=True):
        self.ndim = n_components
        self.a = a
        self.k = k
        self.t = t
        self.mds = mds
        self.knn_dist = knn_dist
        self.mds_dist = mds_dist
        self.njobs = 1
        self.random_state = random_state
        self.verbose = verbose

        self.gs_ker = None
        self.diff_op = None
        self.diff_potential = None
        self.embedding = None

    def reset_mds(self, n_components=None, mds=None, mds_dist=None):
        if n_components is not None:
            self.n_components = n_components
        if mds is not None:
            self.mds = mds
        if mds_dist is None:
            self.mds_dist = mds_dist
        self.embedding = None

    def reset_diffusion(self, t=None):
        if t is not None:
            self.t = t
        self.diff_potential = None

    def fit(self, X):
        """
        Computes the diffusion operator

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            Input data.

        Returns
        -------
        phate : PHATE
        The estimator object
        """
        self.X = X
        self.gs_ker, self.diff_op, self.diff_potential = calculate_potential(
            X, a=self.a, k=self.k, t=self.t, knn_dist=self.knn_dist,
            njobs=self.njobs, gs_ker=self.gs_ker,
            diff_op=self.diff_op, diff_potential=self.diff_potential,
            verbose=self.verbose)
        return self

    def transform(self, X=None):
        """
        Computes the position of the cells in the embedding space

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            Input data.

        Returns
        -------
        embedding : array, shape=[n_samples, n_dimensions]
        The cells embedded in a lower dimensional space using PHATE
        """
        if X is not None and not np.all(X == self.X):
            """
            sklearn.BaseEstimator assumes out-of-sample transformations are
            possible. We explicitly test for this in case the user is not aware
            that reusing the same diffusion operator with a different X will
            not give different results.
            """
            raise RuntimeWarning("Pre-fit PHATE cannot be used to transform a "
                                 "new data matrix. Please fit PHATE to the new"
                                 " data by running 'reset_diffusion' and then "
                                 "'fit' with the new data.")
        if self.diff_potential is None:
            raise NotFittedError("This PHATE instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments before "
                                 "using this method.")
        self.embedding = embed_mds(
            self.diff_potential, n_components=self.ndim,
            mds=self.mds, mds_dist=self.mds_dist, njobs=self.njobs,
            random_state=self.random_state, verbose=self.verbose)
        return self.embedding

    def fit_transform(self, X):
        """
        Computes the diffusion operator and the position of the cells in the
        embedding space

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            Input data.

        diff_op : array, shape=[n_samples, n_samples], optional
            Precomputed diffusion operator

        Returns
        -------
        embedding : array, shape=[n_samples, n_dimensions]
        The cells embedded in a lower dimensional space using PHATE
        """
        start = time.time()
        self.fit(X)
        self.transform()
        if self.verbose:
            print("Finished PHATE embedding in %.2f seconds.\n" %
                  (time.time() - start))
        return self.embedding

    def von_neumann_entropy(self, t_max=100):
        """
        Determines the Von Neumann entropy of the diffusion affinities
        at varying levels of t. The user should select a value of t
        around the "knee" of the entropy curve.

        We require that 'fit' stores the values of gs_ker and diff_deg
        in order to calculate the Von Neumann entropy. Alternatively,
        we could recalculate them here.

        Parameters
        ----------
        t_max : int
            Maximum value of t to test

        Returns
        -------
        entropy : array, shape=[t_max]
        The entropy of the diffusion affinities for each value of t
        """
        if self.gs_ker is None:
            raise NotFittedError("This PHATE instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments before "
                                 "using this method.")
        diff_aff = np.diagflat(
            np.power(np.sum(self.gs_ker, axis=0), 1 / 2))
        diff_aff = np.matmul(np.matmul(diff_aff, self.gs_ker),
                             diff_aff)
        diff_aff = (diff_aff + diff_aff.T) / 2

        _, eigenvalues, _ = svd(diff_aff)
        entropy = []
        eigenvalues_t = np.copy(eigenvalues)
        for _ in range(t_max):
            prob = eigenvalues_t / np.sum(eigenvalues_t)
            prob = prob[prob > 0]
            entropy.append(-np.sum(prob * np.log(prob)))
            eigenvalues_t = eigenvalues_t * eigenvalues

        return np.array(entropy)
