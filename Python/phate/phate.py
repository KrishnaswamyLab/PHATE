"""
Potential of Heat-diffusion for Affinity-based Trajectory Embedding (PHATE)
"""

# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2
from __future__ import print_function, division, absolute_import

import time
import numpy as np
import graphtools
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

import matplotlib.pyplot as plt

from .mds import embed_MDS
from .vne import compute_von_neumann_entropy, find_knee_point
from .utils import check_int, check_positive, check_between, check_in, check_if_not
from .logging import set_logging, log_start, log_complete, log_info

try:
    import anndata
except ImportError:
    # anndata not installed
    pass


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
>>>>>> > master


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

    a : int, optional, default: 10
        sets decay rate of kernel tails.
        If None, alpha decaying kernel is not used

    alpha_threshold : float in range [0, 1], optional, default: 1e-5
        Alpha decay kernel is truncated to zero below this value
        Use 0 for exact alpha decay, 1 for KNN kernel

    n_landmark : int, optional, default: 2000
        number of landmarks to use in fast PHATE

    t : int, optional, default: 'auto'
        power to which the diffusion operator is powered.
        This sets the level of diffusion. If 'auto', t is selected
        according to the knee point in the Von Neumann Entropy of
        the diffusion operator

    potential_method : string, optional, default: 'log'
        choose from ['log', 'sqrt'].
        Selects which transformation of the diffusional operator is used
        to compute the diffusion potential

    n_pca : int, optional, default: 100
        Number of principal components to use for calculating
        neighborhoods. For extremely large datasets, using
        n_pca < 20 allows neighborhoods to be calculated in
        roughly log(n_samples) time.

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
        choose from ['classic', 'metric', 'nonmetric'].
        Selects which MDS algorithm is used for dimensionality reduction

    n_jobs : integer, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used

    njobs : deprecated in favor of n_jobs to match `sklearn` standards

    random_state : integer or numpy.RandomState, optional, default: None
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

    graph : graphtools.BaseGraph
        The graph built on the input data

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

    def __init__(self, n_components=2, k=5, a=10, alpha_threshold=1e-5,
                 n_landmark=2000, t='auto', potential_method='log',
                 n_pca=100, knn_dist='euclidean', mds_dist='euclidean',
                 mds='metric', n_jobs=1, random_state=None, verbose=1,
                 njobs=None):
        self.n_components = n_components
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
            alpha_threshold = 1
        self.alpha_threshold = alpha_threshold

        self.graph = None
        self.diff_potential = None
        self.embedding = None
        self.X = None
        self._check_params()

    @property
    def diff_op(self):
        """The diffusion operator calculated from the data
        """
        if self.graph is not None:
            if isinstance(self.graph, graphtools.LandmarkGraph):
                return self.graph.landmark_op
            else:
                return self.graph.diff_op
        else:
            raise NotFittedError("This PHATE instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments before "
                                 "using this method.")

    def _check_params(self):
        """Check PHATE parameters

        This allows us to fail early - otherwise certain unacceptable
        parameter choices, such as mds='mmds', would only fail after
        minutes of runtime.

        Raises
        ------
        ValueError : unacceptable choice of parameters
        """
        check_positive(n_components=self.n_components,
                       k=self.k)
        check_int(n_components=self.n_components,
                  k=self.k,
                  n_jobs=self.n_jobs)
        check_between(0, 1, alpha_threshold=self.alpha_threshold)
        check_if_not(None, check_positive, a=self.a)
        check_if_not(None, check_positive, check_int,
                     n_landmark=self.n_landmark,
                     n_pca=self.n_pca)
        check_if_not('auto', check_positive, check_int,
                     t=self.t)
        check_in(['euclidean', 'cosine', 'correlation', 'cityblock',
                  'l1', 'l2', 'manhattan', 'braycurtis', 'canberra',
                  'chebyshev', 'dice', 'hamming', 'jaccard',
                  'kulsinski', 'mahalanobis', 'matching', 'minkowski',
                  'rogerstanimoto', 'russellrao', 'seuclidean',
                  'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'],
                 knn_dist=self.knn_dist)
        check_in(['euclidean', 'cosine', 'correlation', 'braycurtis',
                  'canberra', 'chebyshev', 'cityblock', 'dice', 'hamming',
                  'jaccard', 'kulsinski', 'mahalanobis', 'matching',
                  'minkowski', 'rogerstanimoto', 'russellrao',
                  'seuclidean', 'sokalmichener', 'sokalsneath',
                  'sqeuclidean', 'yule'],
                 mds_dist=self.mds_dist)
        check_in(['classic', 'metric', 'nonmetric'],
                 mds=self.mds)
        check_in(['log', 'sqrt'],
                 potential_method=self.potential_method)

    def _set_graph_params(self, **params):
        try:
            self.graph.set_params(**params)
        except AttributeError:
            # graph not defined
            pass

    def set_params(self, **params):
        """Set the parameters on this estimator.

        Any parameters not given as named arguments will be left at their
        current value.

        Parameters
        ----------
        n_components : int, optional, default: 2
            number of dimensions in which the data will be embedded

        k : int, optional, default: 5
            number of nearest neighbors on which to build kernel

        a : int, optional, default: 10
            sets decay rate of kernel tails.
            If None, alpha decaying kernel is not used

        alpha_threshold : float in range [0, 1], optional, default: 1e-5
            Alpha decay kernel is truncated to zero below this value
            Use 0 for exact alpha decay, 1 for KNN kernel

        n_landmark : int, optional, default: 2000
            number of landmarks to use in fast PHATE

        t : int, optional, default: 'auto'
            power to which the diffusion operator is powered.
            This sets the level of diffusion. If 'auto', t is selected
            according to the knee point in the Von Neumann Entropy of
            the diffusion operator

        potential_method : string, optional, default: 'log'
            choose from ['log', 'sqrt'].
            Selects which transformation of the diffusional operator is used
            to compute the diffusion potential

        n_pca : int, optional, default: 100
            Number of principal components to use for calculating
            neighborhoods. For extremely large datasets, using
            n_pca < 20 allows neighborhoods to be calculated in
            roughly log(n_samples) time.

        knn_dist : string, optional, default: 'euclidean'
            recommended values: 'euclidean', 'cosine', 'precomputed'
            Any metric from `sklearn.neighbors.NearestNeighbors` can be used.
            Distance metric for building kNN graph. If 'precomputed',
            data should be an [n_samples x n_samples] pairwise
            distance matrix

        mds_dist : string, optional, default: 'euclidean'
            recommended values: 'euclidean' and 'cosine'
            Any metric from `scipy.spatial.distance` can be used
            distance metric for MDS

        mds : string, optional, default: 'metric'
            choose from ['classic', 'metric', 'nonmetric'].
            Selects which MDS algorithm is used for dimensionality reduction

        n_jobs : integer, optional, default: 1
            The number of jobs to use for the computation.
            If -1 all CPUs are used. If 1 is given, no parallel computing code is
            used at all, which is useful for debugging.
            For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
            n_jobs = -2, all CPUs but one are used

        random_state : integer or numpy.RandomState, optional, default: None
            The generator used to initialize SMACOF (metric, nonmetric) MDS
            If an integer is given, it fixes the seed
            Defaults to the global numpy random number generator

        verbose : boolean, optional
            If true, print status messages

        Returns
        -------
        self
        """
        reset_kernel = False
        reset_potential = False
        reset_embedding = False

        # mds parameters
        if 'n_components' in params and params['n_components'] != self.n_components:
            self.n_components = params['n_components']
            reset_embedding = True
        if 'mds' in params and params['mds'] != self.mds:
            self.mds = params['mds']
            reset_embedding = True
        if 'mds_dist' in params and params['mds_dist'] != self.mds_dist:
            self.mds_dist = params['mds_dist']
            reset_embedding = True

        # diff potential parameters
        if 't' in params and params['t'] != self.t:
            self.t = params['t']
            reset_potential = True
        if 'potential_method' in params and \
                params['potential_method'] != self.potential_method:
            self.potential_method = params['potential_method']
            reset_potential = True

        # kernel parameters
        if 'k' in params and params['k'] != self.k:
            self.k = params['k']
            reset_kernel = True
        if 'a' in params and params['a'] != self.a:
            self.a = params['a']
            reset_kernel = True
        if 'alpha_threshold' in params and \
                params['alpha_threshold'] != self.alpha_threshold:
            self.alpha_threshold = params['alpha_threshold']
            reset_kernel = True
        if 'n_pca' in params and params['n_pca'] != self.n_pca:
            self.n_pca = params['n_pca']
            reset_kernel = True
        if 'knn_dist' in params and params['knn_dist'] != self.knn_dist:
            if self.knn_dist is 'precomputed' or \
                    params['knn_dist'] is 'precomputed':
                # need a different type of graph, reset entirely
                self.graph = None
            self.knn_dist = params['knn_dist']
            reset_kernel = True
        if 'n_landmark' in params and params['n_landmark'] != self.n_landmark:
            if self.n_landmark is None or params['n_landmark'] is None:
                # need a different type of graph, reset entirely
                self.graph = None
            else:
                self._set_graph_params(n_landmark=params['n_landmark'])
            self.n_landmark = params['n_landmark']

        # parameters that don't change the embedding
        if 'n_jobs' in params:
            self.n_jobs = params['n_jobs']
            self._set_graph_params(n_jobs=params['n_jobs'])
        if 'random_state' in params:
            self.random_state = params['random_state']
            self._set_graph_params(random_state=params['random_state'])
        if 'verbose' in params:
            self.verbose = params['verbose']
            self._set_graph_params(verbose=params['verbose'])

        if reset_kernel:
            # can't reset the graph kernel without making a new graph
            self.graph = None
            reset_potential = True
        if reset_potential:
            reset_embedding = True
            self.diff_potential = None
        if reset_embedding:
            self.embedding = None

        self._check_params()
        return self

        set_logging(verbose)

    def reset_mds(self, n_components=None, mds=None, mds_dist=None):
        """
        Deprecated. Reset parameters related to multidimensional scaling

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
        print("Warning: PHATE.reset_mds is deprecated. "
              "Please use PHATE.set_params in future.")
        if n_components is not None:
            self.n_components = n_components
        if mds is not None:
            self.mds = mds
        if mds_dist is not None:
            self.mds_dist = mds_dist
        self.embedding = None

    def reset_potential(self, t=None, potential_method=None):
        """
        Deprecated. Reset parameters related to the diffusion potential

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
        print("Warning: PHATE.reset_potential is deprecated. "
              "Please use PHATE.set_params in future.")
        if t is not None:
            self.t = t
        if potential_method is not None:
            self.potential_method = potential_method
        self.diff_potential = None

    def fit(self, X):
        """Computes the diffusion operator

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
            self.graph = None
        self.X = X

        if self.graph is None:
            if self.knn_dist == 'precomputed':
                if X[0, 0] == 0:
                    distance = "distance"
                else:
                    distance = "affinity"
            else:
                distance = self.knn_dist

            self.graph = graphtools.Graph(
                X,
                n_pca=self.n_pca,
                knn=self.k + 1,
                decay=self.a,
                thresh=self.alpha_threshold,
                n_landmark=self.n_landmark,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=self.random_state)
        else:
            # check the user hasn't changed parameters manually
            try:
                self.graph.set_params(
                    decay=self.a, knn=self.k + 1, distance=self.knn_dist,
                    n_jobs=self.n_jobs, verbose=self.verbose, n_pca=self.n_pca,
                    thresh=self.alpha_threshold, n_landmark=self.n_landmark,
                    random_state=self.random_state)
            except ValueError:
                # something changed that should have invalidated the graph
                self.graph = None
                return self.fit(X)
        return self

    def transform(self, X=None, t_max=100, plot_optimal_t=False, ax=None):
        """Computes the position of the cells in the embedding space

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
        if self.graph is None:
            raise NotFittedError("This PHATE instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments before "
                                 "using this method.")
        elif X is not None and not np.all(X == self.X):
            # fit to external data
            raise RuntimeWarning("Pre-fit PHATE cannot be used to transform a "
                                 "new data matrix. Please fit PHATE to the new"
                                 " data by running 'fit' with the new data.")
            if isinstance(self.graph, graphtools.TraditionalGraph):
                raise ValueError("Cannot transform additional data using a "
                                 "precomputed distance matrix.")
            else:
                transitions = self.graph.extend_to_data(X)
                return self.graph.interpolate(self.embedding,
                                              transitions)
        else:
            if self.t == 'auto':
                t = self.optimal_t(t_max=t_max, plot=plot_optimal_t, ax=ax)
                print("Automatically selected t = {}".format(t))
            else:
                t = self.t
            if isinstance(self.graph, graphtools.LandmarkGraph):
                t = t // 2
            if self.diff_potential is None:
                self.calculate_potential(self.diff_op, t)
            if self.embedding is None:
                self.embedding = embed_MDS(
                    self.diff_potential, ndim=self.n_components, how=self.mds,
                    distance_metric=self.mds_dist, n_jobs=self.n_jobs,
                    seed=self.random_state)
            if isinstance(self.graph, graphtools.LandmarkGraph):
                return self.graph.interpolate(self.embedding)
            else:
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
        embedding = self.transform(**kwargs)
        log_complete('PHATE')
        return embedding

    def calculate_potential(self, diff_op, t):
        """Calculates the diffusion potential

        Parameters
        ----------

        diff_op : array-like, shape=[n_samples, n_samples] or [n_landmarks, n_landmarks]
            The diffusion operator fit on the input data

        t : int
            power to which the diffusion operator is powered
            sets the level of diffusion

        Returns
        -------

        diff_potential : array-like, shape=[n_samples, n_samples]
            The diffusion potential fit on the input data
        """
        tic = time.time()
        if self.verbose:
            print("Calculating diffusion potential...")

        # diffused diffusion operator
        diff_op_t = np.linalg.matrix_power(diff_op, t)

        if self.potential_method == 'log':
            # handling small values
            diff_op_t = diff_op_t + 1e-3
            self.diff_potential = -1 * np.log(diff_op_t)
        elif self.potential_method == 'sqrt':
            self.diff_potential = np.sqrt(diff_op_t)
        else:
            raise ValueError("Allowable 'potential_method' values: 'log' or "
                             "'sqrt'. '{}' was passed.".format(
                                 self.potential_method))
        if self.verbose:
            print("Calculated diffusion potential in {:.2f} seconds.".format(
                time.time() - tic))

    def von_neumann_entropy(self, t_max=100):
        """Calculate Von Neumann Entropy

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
        if isinstance(self.graph, graphtools.LandmarkGraph):
            # landmark operator is doing diffusion twice
            t_max = np.floor(t_max / 2).astype(np.int16)
            t = np.arange(t_max) * 2 + 1
        else:
            t = np.arange(t_max)

        return t, compute_von_neumann_entropy(self.diff_op, t_max=t_max)

    def optimal_t(self, t_max=100, plot=False, ax=None):
        """Find the optimal value of t

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
            ax.set_title("Optimal t = {}".format(t_opt))
            if show:
                plt.show()

        return t_opt
