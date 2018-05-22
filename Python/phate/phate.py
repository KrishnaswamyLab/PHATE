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
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from scipy import sparse
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


import matplotlib.pyplot as plt

from .mds import embed_MDS
from .vne import compute_von_neumann_entropy, find_knee_point
from .logging import set_logging, log_start, log_complete, log_info

try:
    import anndata
except ImportError:
    # anndata not installed
    pass


def calculate_kernel(data, k=15, a=10, alpha_decay=True, knn_dist='euclidean',
                     ndim=100, random_state=None, n_jobs=1):
    """Calculate the alpha-decay or KNN kernel

    Parameters
    ----------
    data : array-like [n_samples, n_dimensions]
        2 dimensional input data array with n cells and p dimensions If
        `knn_dist` is 'precomputed', `data` should be a n_samples x n_samples
        distance matrix

    k : int, optional, default: 15
        used to set epsilon while autotuning kernel bandwidth

    a : int, optional, default: 10
        sets decay rate of kernel tails.

    alpha_decay : boolean, default: True
        If true, use the alpha decaying kernel

    knn_dist : string, optional, default: 'euclidean'
        recommended values: 'euclidean', 'cosine', 'precomputed'
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph. If 'precomputed',
        `data` should be an n_samples x n_samples distance matrix

    ndim : int, optional, default: 100
        Number of principal components to use for KNN calculation

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global numpy random number generator

    n_jobs : integer, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used

    Returns
    -------

    kernel : array-like [n_samples, n_samples]
        kernel matrix built from the input data
    """
    if knn_dist != 'precomputed' and ndim < data.shape[1]:
        log_start("PCA")
        if sparse.issparse(data):
            _, _, VT = randomized_svd(data, ndim,
                                      random_state=random_state)
            data = data.dot(VT.T)
        else:
            pca = PCA(ndim, svd_solver='randomized',
                      random_state=random_state)
            data = pca.fit_transform(data)
        log_complete("PCA")
    # kernel includes self as connection but not in k
    # actually search for k+1 neighbors including self
    k = k + 1
    log_start("KNN search")
    if alpha_decay and a is not None:
        try:
            if knn_dist == 'precomputed':
                pdx = data
            else:
                pdx = squareform(pdist(data, metric=knn_dist))
            knn_dist = np.partition(pdx, k, axis=1)[:, :k]
            # bandwidth(x) = distance to k-th neighbor of x
            epsilon = np.max(knn_dist, axis=1)
            pdx = (pdx / epsilon).T  # autotuning d(x,:) using epsilon(x).
        except RuntimeWarning:
            raise ValueError(
                'It looks like you have at least k identical data points. '
                'Try removing duplicates.')
        kernel = np.exp(-1 * (pdx ** a))  # not really Gaussian kernel
    else:
        if knn_dist == 'precomputed':
            # we already have pairwise distances
            pdx = knn_dist
            knn_idx = np.argpartition(pdx, k, axis=1)[:, :k]
            ind_ptr = np.arange(knn_idx.shape[0] + 1) * knn_idx.shape[1]
            col_ind = knn_idx.reshape(-1)
            ones = np.repeat(1., len(col_ind))
            kernel = sparse.csr_matrix((ones, col_ind, ind_ptr),
                                       shape=[data.shape[0], data.shape[0]])
        else:
            knn = NearestNeighbors(n_neighbors=k,
                                   n_jobs=n_jobs).fit(data)
            kernel = knn.kneighbors_graph(data, mode='connectivity')
    log_complete("KNN search")
    kernel = kernel + kernel.T  # symmetrization
    return kernel


def calculate_landmark_operator(kernel, n_landmark=2000,
                                random_state=None, n_svd=100):
    """
    Calculate the landmark operator

    Parameters
    ----------
    kernel : array-like [n_samples, n_samples]
        kernel matrix built from the input data

    n_landmark : int, optional, default: 2000
        number of landmarks to use in fast PHATE

    landmark_transitions : array-like, shape=[n_samples, n_landmarks], default: None
        Precomputed transition matrix between input data and landmarks

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global numpy random number generator

    n_svd : int, optional, default: 100
        Number of singular vectors to compute for spectral clustering
        if landmarks are used

    Returns
    -------

    diff_op : array-like, shape [n_samples, n_samples]
        The diffusion operator fit on the input data

    landmark_transitions : array-like, shape=[n_samples, n_landmarks]
        Transition matrix between input data and landmarks,
        if `n_landmark` is set, otherwise `None`
    """
    is_sparse = sparse.issparse(kernel)
    diff_op = normalize(kernel, norm='l1', axis=1)  # row stochastic
    if n_landmark is not None and n_landmark < kernel.shape[0]:
        # spectral clustering
        log_start("SVD")
        U, S, _ = randomized_svd(diff_op,
                                 n_components=n_svd,
                                 random_state=random_state)
        log_complete("SVD")
        log_start("KMeans")
        kmeans = MiniBatchKMeans(n_landmark,
                                 init_size=3 * n_landmark,
                                 batch_size=10000,
                                 random_state=random_state)
        clusters = kmeans.fit_predict(np.matmul(U, np.diagflat(S)))
        landmarks = np.unique(clusters)
        log_complete("KMeans")

        # transition matrices
        if is_sparse:
            pmn = sparse.vstack(
                [sparse.csr_matrix(kernel[clusters == i, :].sum(
                    axis=0)) for i in landmarks])
        else:
            pmn = np.array([np.sum(
                kernel[clusters == i, :], axis=0) for i in landmarks])
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


def calculate_operator(data, k=15, a=10, alpha_decay=True, n_landmark=2000,
                       knn_dist='euclidean', diff_op=None,
                       landmark_transitions=None, n_jobs=1,
                       random_state=None, n_pca=100, n_svd=100):
    """
    Calculate the diffusion operator

    Parameters
    ----------
    data : array-like [n_samples, n_dimensions]
        2 dimensional input data array with n cells and p dimensions. If
        `knn_dist` is 'precomputed', `data` should be a n_samples x n_samples
        distance or affinity matrix

    k : int, optional, default: 15
        used to set epsilon while autotuning kernel bandwidth

    a : int, optional, default: 10
        sets decay rate of kernel tails.

    alpha_decay : boolean, default: True
        If true, use the alpha decaying kernel

    n_landmark : int, optional, default: 2000
        number of landmarks to use in fast PHATE

    knn_dist : string, optional, default: 'euclidean'
        recommended values: 'euclidean', 'cosine', 'precomputed'
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph. If 'precomputed',
        `data` should be an n_samples x n_samples distance or
        affinity matrix

    diff_op : array-like, optional shape=[n_samples, n_samples], default: None
        Precomputed diffusion operator

    landmark_transitions : array-like, shape=[n_samples, n_landmarks], default: None
        Precomputed transition matrix between input data and landmarks

    n_jobs : integer, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global numpy random number generator

    n_svd : int, optional, default: 100
        Number of singular vectors to compute for spectral clustering
        if landmarks are used

    Returns
    -------

    diff_op : array-like, shape [n_samples, n_samples]
        The diffusion operator fit on the input data

    landmark_transitions : array-like, shape=[n_samples, n_landmarks]
        Transition matrix between input data and landmarks,
        if `n_landmark` is set, otherwise `None`
    """
    # print('Imported numpy: %s'%np.__file__)

    tic = time.time()
    if alpha_decay is None:
        if n_landmark is not None and len(data) > n_landmark:
            alpha_decay = False
            if a is not None:
                log_info("Alpha decay is not used as n_landmark < n_samples. "
                         "To override this behavior, set alpha_decay=True "
                         "(increases memory requirements) or n_landmark=None "
                         "(increases memory and CPU requirements.)")
        else:
            alpha_decay = True
    if diff_op is None:
        log_start("graph and diffusion operator")
        if knn_dist == 'precomputed' and np.all(np.diagonal(data) != 0):
            log_info("Using precomputed affinity matrix...")
            kernel = data
        else:
            if knn_dist == 'precomputed':
                if np.all(np.diagonal(data) == 0):
                    print("Using precomputed distance matrix...")
                else:
                    raise ValueError(
                        "Cannot determine precomputed data type. "
                        "Precomputed affinity matrices should have "
                        "only non-zero entries on the diagonal, and"
                        " precomputed distance matrices should have"
                        " only zero entries on the diagonal.")
            kernel = calculate_kernel(data, a=a, k=k, knn_dist=knn_dist,
                                      ndim=n_pca,
                                      alpha_decay=alpha_decay,
                                      random_state=random_state,
                                      n_jobs=n_jobs)
        diff_op, landmark_transitions = calculate_landmark_operator(
            kernel, n_landmark=n_landmark,
            random_state=random_state)
        log_complete("graph and diffusion operator")
    else:
        log_info("Using precomputed diffusion operator...")

    return diff_op, landmark_transitions


def embed_mds(diff_op, t=30, n_components=2, diff_potential=None,
              embedding=None, mds='metric', mds_dist='euclidean', n_jobs=1,
              potential_method='log', random_state=None,
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
        which transformation of the diffusional operator is used
        to compute the diffusion potential

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
        log_start("diffusion potential")
        if landmark_transitions is not None:
            # landmark operator is doing diffusion twice
            t = np.floor(t / 2).astype(np.int16)

        X = np.linalg.matrix_power(diff_op, t)  # diffused diffusion operator

        if potential_method == 'log':  # or potential_method == 1:
            # handling small values
            # X[X <= np.finfo(float).eps] = np.finfo(
            #     float).eps
            X = X + 1e-7
            diff_potential = -1 * np.log(X)  # diffusion potential
        elif potential_method == 'sqrt':
            diff_potential = np.sqrt(X)  # diffusion potential
        else:  # if isinstance(potential_method, str):
            raise ValueError(
                "Allowable 'potential_method' values: 'log' or "
                "'sqrt'. '{}' was passed.".format(potential_method))
        # else:
        #     # gamma
        #     print("Warning: gamma potential is not stable."
        #           " Recommended values: 'log' or 'sqrt'")
        #     if potential_method > 1 or potential_method < -1:
        #         raise ValueError(
        #             "Allowable 'potential_method' values between -1 and 1"
        #             " inclusive. '{}' was passed.".format(potential_method))
        #     elif potential_method != -1:
        #         diff_potential = 2 / (1 - potential_method) * \
        #             np.power(X, ((1 - potential_method) / 2))
        #     else:
        #         # gamma = -1 is just MDS on DM
        #         diff_potential = X

        log_complete("diffusion potential")
    # if diffusion potential is precomputed (i.e. 'mds' or 'mds_dist' has
    # changed on PHATE object)
    else:
        log_info("Using precomputed diffusion potential...")

    if embedding is None:
        log_start("{} MDS".format(mds))
        embedding = embed_MDS(diff_potential, ndim=n_components, how=mds,
                              distance_metric=mds_dist, n_jobs=n_jobs,
                              seed=random_state)
        if landmark_transitions is not None:
            # return to ambient space
            embedding = landmark_transitions.dot(embedding)
        log_complete("{} MDS".format(mds))
    else:
        log_info("Using precomputed embedding...")
    return embedding, diff_potential


class PHATE(BaseEstimator):
    """PHATE operator which performs dimensionality reduction.

    Potential of Heat-diffusion for Affinity-based Trajectory Embedding
    (PHATE) embeds high dimensional single-cell data into two or three
    dimensions for visualization of biological progressions as described
    in Moon et al, 2017 [1]_.

    Parameters
    ----------

    n_components : int, optional, default: 2
        number of dimensions in which the data will be embedded

    k : int, optional, default: 15
        number of nearest neighbors on which to build kernel

    a : int, optional, default: None
        sets decay rate of kernel tails.
        If None, alpha decaying kernel is not used

    alpha_decay : boolean, default: None
        forces the use of alpha decaying kernel
        If None, alpha decaying kernel is used for small inputs
        (n_samples < n_landmark) and not used otherwise

    n_landmark : int, optional, default: 2000
        number of landmarks to use in fast PHATE

    t : int, optional, default: 'auto'
        power to which the diffusion operator is powered
        sets the level of diffusion

    potential_method : string, optional, default: 'log'
        choose from ['log', 'sqrt']
        which transformation of the diffusional operator is used
        to compute the diffusion potential

    n_pca : int, optional, default: 100
        Number of principal components to use for calculating
        neighborhoods. For extremely large datasets, using
        n_pca < 20 allows neighborhoods to be calculated in
        log(n_samples) time.

    knn_dist : string, optional, default: 'euclidean'
        recommended values: 'euclidean', 'cosine', 'precomputed'
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph. If 'precomputed',
        `data` should be an n_samples x n_samples distance or
        affinity matrix

    mds_dist : string, optional, default: 'euclidean'
        recommended values: 'euclidean' and 'cosine'
        Any metric from `scipy.spatial.distance` can be used
        distance metric for MDS

    mds : string, optional, default: 'metric'
        choose from ['classic', 'metric', 'nonmetric']
        which MDS algorithm is used for dimensionality reduction

    n_jobs : integer, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used

    njobs : deprecated in favor of n_jobs to match `sklearn` standards

    random_state : integer or `numpy.RandomState`, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global `numpy` random number generator

    verbose : `int` or `boolean`, optional (default: 1)
        If `True` or `> 0`, print status messages

    Attributes
    ----------

    X : array-like, shape=[n_samples, n_dimensions]

    embedding : array-like, shape=[n_samples, n_components]
        Stores the position of the dataset in the embedding space

    diff_op : array-like, shape=[n_samples, n_samples] or [n_landmarks, n_landmarks]
        The diffusion operator fit on the input data

    diff_potential : array-like, shape=[n_samples, n_samples]
        Precomputed diffusion potential

    landmark_transitions : array-like, shape=[n_samples, n_landmarks]
        Transition matrix between input data and landmarks,
        if `n_landmark` is set, otherwise `None`

    Examples
    --------
    >>> import phate
    >>> import matplotlib.pyplot as plt
    >>> tree_data, tree_clusters = phate.tree.gen_dla(n_dim=100,
                                                      n_branch=20,
                                                      branch_length=100)
    >>> tree_data.shape
    (2000, 100)
    >>> phate_operator = phate.PHATE(k=5, a=20, t=150)
    >>> tree_phate = phate_operator.fit_transform(tree_data)
    >>> tree_phate.shape
    (2000, 2)
    >>> plt.scatter(tree_phate[:,0], tree_phate[:,1], c=tree_clusters)
    >>> plt.show()

    References
    ----------
    .. [1] Moon KR, van Dijk D, Zheng W, *et al.* (2017),
        *PHATE: A Dimensionality Reduction Method for Visualizing Trajectory
        Structures in High-Dimensional Biological Data*,
        `BioRxiv <http://biorxiv.org/content/early/2017/03/24/120378>`_.
    """

    def __init__(self, n_components=2, k=15, a=10, alpha_decay=None,
                 n_landmark=2000, t='auto', potential_method='log',
                 n_pca=100, knn_dist='euclidean', mds_dist='euclidean',
                 mds='metric', n_jobs=1, random_state=None, verbose=1,
                 njobs=None):
        self.ndim = n_components
        self.a = a
        self.k = k
        self.t = t
        self.n_landmark = n_landmark
        self.mds = mds
        self.n_pca = n_pca
        self.knn_dist = knn_dist
        self.mds_dist = mds_dist
        if njobs is not None:
            print("Warning: njobs is deprecated. Please use n_jobs in future.")
            n_jobs = njobs
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.potential_method = potential_method

        if a is None:
            alpha_decay = False
        self.alpha_decay = alpha_decay

        self.diff_op = None
        self.landmark_transitions = None
        self.diff_potential = None
        self.embedding = None
        self.X = None

        set_logging(verbose)

    def reset_mds(self, n_components=None, mds=None, mds_dist=None):
        """
        Reset parameters related to multidimensional scaling

        Parameters
        ----------
        n_components : int, optional, default: None
            If given, sets number of dimensions in which the data
            will be embedded

        mds : string, optional, default: None
            choose from ['classic', 'metric', 'nonmetric']
            If given, sets which MDS algorithm is used for
            dimensionality reduction

        mds_dist : string, optional, default: None
            recommended values: 'euclidean' and 'cosine'
            Any metric from scipy.spatial.distance can be used
            If given, sets the distance metric for MDS
        """
        if n_components is not None:
            self.n_components = n_components
        if mds is not None:
            self.mds = mds
        if mds_dist is not None:
            self.mds_dist = mds_dist
        self.embedding = None

    def reset_potential(self, t=None, potential_method=None):
        """
        Reset parameters related to the diffusion potential

        Parameters
        ----------
        t : int or 'auto', optional, default: None
            Power to which the diffusion operator is powered
            If given, sets the level of diffusion

        potential_method : string, optional, default: None
            choose from ['log', 'sqrt']
            If given, sets which transformation of the diffusional
            operator is used to compute the diffusion potential
        """
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
            input data with `n_samples` samples and `n_dimensions`
            dimensions. Accepted data types: `numpy.ndarray`,
            `scipy.sparse.spmatrix`, `pd.DataFrame`, `anndata.AnnData`. If
            `knn_dist` is 'precomputed', `data` should be a n_samples x
            n_samples distance or affinity matrix

        Returns
        -------
        phate_operator : PHATE
        The estimator object
        """
        try:
            if isinstance(X, anndata.AnnData):
                X = X.X
        except NameError:
            # anndata not installed
            pass
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
            n_jobs=self.n_jobs, n_landmark=self.n_landmark,
            diff_op=self.diff_op, n_pca=self.n_pca,
            landmark_transitions=self.landmark_transitions,
            alpha_decay=self.alpha_decay, random_state=self.random_state)
        return self

    def transform(self, X=None, t_max=100, plot_optimal_t=False, ax=None):
        """
        Computes the position of the cells in the embedding space

        Parameters
        ----------
        X : array, optional, shape=[n_samples, n_features]
            input data with `n_samples` samples and `n_dimensions`
            dimensions. Not required, since PHATE does not currently embed
            cells not given in the input matrix to `PHATE.fit()`.
            Accepted data types: `numpy.ndarray`,
            `scipy.sparse.spmatrix`, `pd.DataFrame`, `anndata.AnnData`. If
            `knn_dist` is 'precomputed', `data` should be a n_samples x
            n_samples distance or affinity matrix

        t_max : int, optional, default: 100
            maximum t to test if `t` is set to 'auto'

        plot_optimal_t : boolean, optional, default: False
            If true and `t` is set to 'auto', plot the Von Neumann
            entropy used to select t

        ax : matplotlib.axes.Axes, optional
            If given and `plot_optimal_t` is true, plot will be drawn
            on the given axis.

        Returns
        -------
        embedding : array, shape=[n_samples, n_dimensions]
        The cells embedded in a lower dimensional space using PHATE
        """
        if self.diff_op is None:
            raise NotFittedError("This PHATE instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments before "
                                 "using this method.")
        elif self.X is not None and X is not None and not np.all(X == self.X):
            """
            sklearn.BaseEstimator assumes out-of-sample transformations are
            possible. We explicitly test for this in case the user is not aware
            that reusing the same diffusion operator with a different X will
            not give different results.
            """
            raise RuntimeWarning("Pre-fit PHATE cannot be used to transform a "
                                 "new data matrix. Please fit PHATE to the new"
                                 " data by running 'fit' with the new data.")
        if self.t == 'auto':
            log_start("optimal t")
            t = self.optimal_t(t_max=t_max, plot=plot_optimal_t, ax=ax)
            log_complete("optimal t")
            log_info("Automatically selected t = {}".format(t))
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
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            potential_method=self.potential_method)
        return self.embedding

    def fit_transform(self, X, **kwargs):
        """Computes the diffusion operator and the position of the cells in the
        embedding space

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            input data with `n_samples` samples and `n_dimensions`
            dimensions. Accepted data types: `numpy.ndarray`,
            `scipy.sparse.spmatrix`, `pd.DataFrame`, `anndata.AnnData` If
            `knn_dist` is 'precomputed', `data` should be a n_samples x
            n_samples distance or affinity matrix

        kwargs : further arguments for `PHATE.transform()`
            Keyword arguments as specified in :func:`~phate.PHATE.transform`

        Returns
        -------
        embedding : array, shape=[n_samples, n_dimensions]
            The cells embedded in a lower dimensional space using PHATE
        """
        log_start('PHATE')
        self.fit(X)
        self.transform(**kwargs)
        log_complete('PHATE')
        return self.embedding

    def von_neumann_entropy(self, t_max=100):
        """
        Determines the Von Neumann entropy of the diffusion affinities
        at varying levels of `t`. The user should select a value of `t`
        around the "knee" of the entropy curve.

        We require that 'fit' stores the value of `PHATE.diff_op`
        in order to calculate the Von Neumann entropy. Alternatively,
        we could recalculate it here, but that is less desirable.

        Parameters
        ----------
        t_max : int, default: 100
            Maximum value of `t` to test

        Returns
        -------
        entropy : array, shape=[t_max]
            The entropy of the diffusion affinities for each value of `t`
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
        """
        Selects the optimal value of t based on the knee point of the
        Von Neumann Entropy of the diffusion operator.

        Parameters
        ----------
        t_max : int, default: 100
            Maximum value of t to test

        plot : boolean, default: False
            If true, plots the Von Neumann Entropy and knee point

        ax : matplotlib.Axes, default: None
            If plot=True and ax is not None, plots the VNE on the given axis
            Otherwise, creates a new axis and displays the plot

        Returns
        -------
        t_opt : int
            The optimal value of t
        """
        log_start("Von Neumann entropy")
        t, h = self.von_neumann_entropy(t_max=t_max)
        log_complete("Von Neumann entropy")
        log_start("optimal t")
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
            ax.set_title("Optimal t = {}".format(t_opt))
            if show:
                plt.show()

        return t_opt
