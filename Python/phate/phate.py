"""
Potential of Heat-diffusion for Affinity-based Trajectory Embedding (PHATE)
"""

# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2
from __future__ import print_function, division, absolute_import

import time
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.utils.extmath import randomized_svd
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from .mds import embed_MDS
from .vne import compute_von_neumann_entropy, find_knee_point


def calculate_kernel(M, a=10, k=5, knn_dist='euclidean', verbose=True,
                     alpha_decay=True, ndim=100, random_state=None):
    if verbose:
        print("Building kNN graph and diffusion operator...")
    precomputed = isinstance(knn_dist, list) or \
        isinstance(knn_dist, np.ndarray)
    if not precomputed and ndim < M.shape[1]:
        pca = PCA(ndim, svd_solver='randomized', random_state=random_state)
        M = pca.fit_transform(M)
    if alpha_decay:
        try:
            if precomputed:
                pdx = knn_dist
            else:
                pdx = squareform(pdist(M, metric=knn_dist))
            knn_dist = np.partition(pdx, k, axis=1)[:, :k]
            # bandwidth(x) = distance to k-th neighbor of x
            epsilon = np.max(knn_dist, axis=1)
            pdx = (pdx / epsilon).T  # autotuning d(x,:) using epsilon(x).
        except RuntimeWarning:
            raise ValueError(
                'It looks like you have at least k identical data points. '
                'Try removing duplicates.')
        gs_ker = np.exp(-1 * (pdx ** a))  # not really Gaussian kernel
    else:
        if precomputed:
            pdx = knn_dist
            knn_idx = np.argpartition(pdx, k, axis=1)[:, :k]
        else:
            knn = NearestNeighbors(n_neighbors=k - 1).fit(M)
            _, knn_idx = knn.kneighbors(M)
        ind_ptr = np.arange(knn_idx.shape[0] + 1) * knn_idx.shape[1]
        col_ind = knn_idx.reshape(-1)
        data = np.repeat(1., len(col_ind))
        gs_ker = sparse.csr_matrix((data, col_ind, ind_ptr))
    gs_ker = gs_ker + gs_ker.T  # symmetrization
    return gs_ker


def calculate_landmark_operator(gs_ker, n_landmark=1000, n_svd=100,
                                random_state=None):
    is_sparse = sparse.issparse(gs_ker)
    diff_op = normalize(gs_ker, norm='l1', axis=1)  # row stochastic
    if n_landmark is not None and n_landmark < gs_ker.shape[0]:
        # spectral clustering
        U, S, _ = randomized_svd(diff_op,
                                 n_components=n_svd,
                                 random_state=random_state)
        kmeans = MiniBatchKMeans(n_landmark, init_size=3 * n_landmark,
                                 random_state=random_state)
        clusters = kmeans.fit_predict(np.matmul(U, np.diagflat(S)))
        landmarks = np.unique(clusters)

        # transition matrices
        if is_sparse:
            pmn = sparse.vstack(
                [sparse.csr_matrix(gs_ker[clusters == i, :].sum(axis=0)) for i in landmarks])
        else:
            pmn = np.array([np.sum(
                gs_ker[clusters == i, :], axis=0) for i in landmarks])
        # row normalize
        pnm = pmn.transpose()
        pmn = normalize(pmn, norm='l1', axis=1)
        pnm = normalize(pnm, norm='l1', axis=1)
        diff_op = pmn.dot(pnm)  # sparsity agnostic matrix multiplication
    else:
        pnm = None
    if is_sparse:
        diff_op = diff_op.todense()
    return diff_op, pnm


def calculate_operator(data, a=10, k=5, knn_dist='euclidean',
                       diff_op=None, landmark_transitions=None,
                       njobs=1, verbose=True, alpha_decay=True,
                       n_landmark=1000, n_svd=100,
                       random_state=None):
    """
    Calculate the diffusion operator

    Parameters
    ----------
    data : ndarray [n, p]
        2 dimensional input data array with n cells and p dimensions

    a : int, optional, default: 10
        sets decay rate of kernel tails

    k : int, optional, default: 5
        used to set epsilon while autotuning kernel bandwidth

    knn_dist : string, optional, default: 'euclidean'
        recommended values: 'euclidean' and 'cosine'
        Any metric from scipy.spatial.distance can be used
        distance metric for building kNN graph

    gs_ker : array-like, shape [n_samples, n_samples]
        Precomputed graph kernel

    diff_op : ndarray, optional [n, n], default: None
        Precomputed diffusion operator

    verbose : boolean, optional, default: True
        Print updates during PHATE embedding

    Returns
    -------
    gs_ker : array-like, shape [n_samples, n_samples]
        The graph kernel built on the input data
        Only necessary for calculating Von Neumann Entropy

    diff_op : array-like, shape [n_samples, n_samples]
        The diffusion operator fit on the input data
    """
    # print('Imported numpy: %s'%np.__file__)

    tic = time.time()
    if alpha_decay is None:
        if n_landmark is not None and len(data) > n_landmark:
            alpha_decay = False
            if a is not None:
                print("Warning: a is set, but alpha decay is not used "
                      "as n_landmark < len(X). To override this behavior,"
                      " set alpha_decay=True (increases memory requirements)"
                      " or n_landmark=None (increases memory and CPU requirements.)")
        else:
            alpha_decay = True
    if diff_op is None:
        gs_ker = calculate_kernel(data, a, k, knn_dist,
                                  verbose=verbose,
                                  alpha_decay=alpha_decay,
                                  random_state=random_state)
        diff_op, landmark_transitions = calculate_landmark_operator(
            gs_ker, n_landmark,
            random_state=random_state)
        if verbose:
            print("Built graph and diffusion operator in %.2f seconds." %
                  (time.time() - tic))
    else:
        if verbose:
            print("Using precomputed diffusion operator...")

    return diff_op, landmark_transitions


def embed_mds(diff_op, t=30, n_components=2, diff_potential=None,
              embedding=None, mds='metric', mds_dist='euclidean', njobs=1,
              potential_method='log', random_state=None, verbose=True,
              landmark_transitions=None):
    """
    Create the MDS embedding from the diffusion potential

    Parameters
    ----------

    diff_op : array-like, shape [n_samples, n_samples]
        The diffusion operator fit on the input data

    t : int, optional, default: 30
        power to which the diffusion operator is powered
        sets the level of diffusion

    n_components : int, optional, default: 2
        number of dimensions in which the data will be embedded

    diff_potential : ndarray, optional [n, n], default: None
        Precomputed diffusion potential

    potential_method : string, optional, default: 'log'
        choose from ['log', 'sqrt']

    mds : string, optional, default: 'metric'
        choose from ['classic', 'metric', 'nonmetric']
        which multidimensional scaling algorithm is used for dimensionality
        reduction

    mds_dist : string, optional, default: 'euclidean'
        recommended values: 'euclidean' and 'cosine'
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

    diff_potential : array-like, shape [n_samples, n_samples]
        Precomputed diffusion potential

    embedding : ndarray [n_samples, n_components]
        PHATE embedding in low dimensional space.
    """

    if diff_potential is None:
        embedding = None  # can't use precomputed embedding
        tic = time.time()
        if verbose:
            print("Calculating diffusion potential...")
        if landmark_transitions is not None:
            # landmark operator is doing diffusion twice
            t = np.floor(t / 2).astype(np.int8)

        X = np.linalg.matrix_power(diff_op, t)  # diffused diffusion operator

        if potential_method == 'log':
            X[X <= np.finfo(float).eps] = np.finfo(
                float).eps  # handling small values
            diff_potential = -1 * np.log(X)  # diffusion potential
        elif potential_method == 'sqrt':
            diff_potential = np.sqrt(X)  # diffusion potential
        else:
            raise ValueError("Allowable 'potential_method' values: 'log' or "
                             "'sqrt'. '%s' was passed." % (potential_method))

        if verbose:
            print("Calculated diffusion potential in %.2f seconds." %
                  (time.time() - tic))
    # if diffusion potential is precomputed (i.e. 'mds' or 'mds_dist' has
    # changed on PHATE object)
    else:
        if verbose:
            print("Using precomputed diffusion potential...")

    tic = time.time()
    if verbose:
        print("Embedding data using %s MDS..." % (mds))
    if embedding is None:
        embedding = embed_MDS(diff_potential, ndim=n_components, how=mds,
                              distance_metric=mds_dist, njobs=njobs,
                              seed=random_state)
        if landmark_transitions is not None:
            # return to ambient space
            embedding = landmark_transitions.dot(embedding)
        if verbose:
            print("Embedded data in %.2f seconds." % (time.time() - tic))
    else:
        if verbose:
            print("Using precomputed embedding...")
    return embedding, diff_potential


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
        recommended values: 'euclidean' and 'cosine'
        Any metric from scipy.spatial.distance can be used
        distance metric for building kNN graph

    mds_dist : string, optional, default: 'euclidean'
        recommended values: 'euclidean' and 'cosine'
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

    def __init__(self, n_components=2, a=None, k=5, t='auto', mds='metric',
                 knn_dist='euclidean', mds_dist='euclidean', njobs=1,
                 random_state=None, verbose=True, n_landmark=1000,
                 alpha_decay=None, potential_method='log'):
        self.ndim = n_components
        self.a = a
        self.k = k
        self.t = t
        self.n_landmark = n_landmark
        self.mds = mds
        self.knn_dist = knn_dist
        self.mds_dist = mds_dist
        self.njobs = 1
        self.random_state = random_state
        self.verbose = verbose
        self.potential_method = potential_method

        if a is None:
            alpha_decay = False
        self.alpha_decay = alpha_decay
        self.diff_op = None
        self.landmark_transitions = None
        self.diff_potential = None
        self.embedding = None
        self.X = None

    def reset_mds(self, n_components=None, mds=None, mds_dist=None):
        if n_components is not None:
            self.n_components = n_components
        if mds is not None:
            self.mds = mds
        if mds_dist is not None:
            self.mds_dist = mds_dist
        self.embedding = None

    def reset_potential(self, t=None, potential_method=None):
        if t is not None:
            self.t = t
        if potential_method is not None:
            self.potential_method = potential_method
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
        if self.X is not None and not np.all(X == self.X):
            """
            If the same data is used, we can reuse existing kernel and
            diffusion matrices. Otherwise we have to recompute.
            """
            self.diff_op = None
            self.landmark_transitions = None
            self.diff_potential = None
            self.embedding = None
        self.X = X

        if self.diff_op is None:
            self.diff_potential = None  # can't use precomputed potential
        self.diff_op, self.landmark_transitions = calculate_operator(
            X, a=self.a, k=self.k, knn_dist=self.knn_dist,
            njobs=self.njobs, n_landmark=self.n_landmark,
            diff_op=self.diff_op, verbose=self.verbose,
            landmark_transitions=self.landmark_transitions,
            alpha_decay=self.alpha_decay, random_state=self.random_state)
        return self

    def transform(self, X=None, t_max=100, plot_optimal_t=False, ax=None):
        """
        Computes the position of the cells in the embedding space

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            Input data.

        t : int, optional, default: 30
            power to which the diffusion operator is powered
            sets the level of diffusion

        Returns
        -------
        embedding : array, shape=[n_samples, n_dimensions]
        The cells embedded in a lower dimensional space using PHATE
        """
        if self.X is not None and X is not None and not np.all(X == self.X):
            """
            sklearn.BaseEstimator assumes out-of-sample transformations are
            possible. We explicitly test for this in case the user is not aware
            that reusing the same diffusion operator with a different X will
            not give different results.
            """
            raise RuntimeWarning("Pre-fit PHATE cannot be used to transform a "
                                 "new data matrix. Please fit PHATE to the new"
                                 " data by running 'fit' with the new data.")
        if self.diff_op is None:
            raise NotFittedError("This PHATE instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments before "
                                 "using this method.")
        if self.t == 'auto':
            t = self.optimal_t(t_max=t_max, plot=plot_optimal_t, ax=ax)
            print("Automatically selected t = {}".format(t))
        else:
            t = self.t
        self.embedding, self.diff_potential = embed_mds(
            self.diff_op,
            t=t,
            landmark_transitions=self.landmark_transitions,
            n_components=self.ndim,
            diff_potential=self.diff_potential,
            embedding=self.embedding,
            mds=self.mds, mds_dist=self.mds_dist,
            njobs=self.njobs,
            random_state=self.random_state,
            verbose=self.verbose,
            potential_method=self.potential_method)
        return self.embedding

    def fit_transform(self, X, **kwargs):
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
        self.transform(**kwargs)
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
        if self.diff_op is None:
            raise NotFittedError("This PHATE instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments before "
                                 "using this method.")
        if self.landmark_transitions is not None:
            # landmark operator is doing diffusion twice
            t_max = np.floor(t_max / 2).astype(np.int16)
            t = np.arange(t_max) * 2 + 1
        else:
            t = np.arange(t_max)

        return t, compute_von_neumann_entropy(self.diff_op, t_max=t_max)

    def optimal_t(self, t_max=100, plot=False, ax=None):

        t, h = self.von_neumann_entropy(t_max=t_max)
        t_opt = find_knee_point(y=h, x=t)

        if plot:
            if ax is None:
                fig, ax = plt.subplots()
                show = True
            else:
                show = False
            ax.plot(t, h)
            ax.scatter(t_opt, h[t == t_opt], marker='*', c='k', s=50)
            ax.set_xlabel("t")
            ax.set_ylabel("Von Neumann Entropy")
            if show:
                plt.show()

        return t_opt
