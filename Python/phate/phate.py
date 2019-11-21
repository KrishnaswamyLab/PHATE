"""
Potential of Heat-diffusion for Affinity-based Trajectory Embedding (PHATE)
"""

# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2
from __future__ import print_function, division, absolute_import

import numpy as np
import graphtools.estimator
import graphtools.utils
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from scipy import sparse
import warnings
import tasklogger

import matplotlib.pyplot as plt

from . import utils, vne, mds

try:
    import anndata
except ImportError:
    # anndata not installed
    pass

try:
    import pygsp
except ImportError:
    # anndata not installed
    pass

_logger = tasklogger.get_tasklogger("graphtools")


class PHATE(graphtools.estimator.GraphEstimator):
    """PHATE operator which performs dimensionality reduction.

    Potential of Heat-diffusion for Affinity-based Trajectory Embedding
    (PHATE) embeds high dimensional single-cell data into two or three
    dimensions for visualization of biological progressions as described
    in Moon et al, 2017 [1]_.

    Parameters
    ----------

    n_components : int, optional, default: 2
        number of dimensions in which the data will be embedded

    knn : int, optional, default: 5
        number of nearest neighbors on which to build kernel

    decay : int, optional, default: 40
        sets decay rate of kernel tails.
        If None, alpha decaying kernel is not used

    n_landmark : int, optional, default: 2000
        number of landmarks to use in fast PHATE

    t : int, optional, default: 'auto'
        power to which the diffusion operator is powered.
        This sets the level of diffusion. If 'auto', t is selected
        according to the knee point in the Von Neumann Entropy of
        the diffusion operator

    gamma : float, optional, default: 1
        Informational distance constant between -1 and 1.
        `gamma=1` gives the PHATE log potential, `gamma=0` gives
        a square root potential.

    n_pca : int, optional, default: 100
        Number of principal components to use for calculating
        neighborhoods. For extremely large datasets, using
        n_pca < 20 allows neighborhoods to be calculated in
        roughly log(n_samples) time.

    knn_dist : string, optional, default: 'euclidean'
        recommended values: 'euclidean', 'cosine', 'precomputed'
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph. Custom distance
        functions of form `f(x, y) = d` are also accepted. If 'precomputed',
        `data` should be an n_samples x n_samples distance or
        affinity matrix. Distance matrices are assumed to have zeros
        down the diagonal, while affinity matrices are assumed to have
        non-zero values down the diagonal. This is detected automatically using
        `data[0,0]`. You can override this detection with
        `knn_dist='precomputed_distance'` or `knn_dist='precomputed_affinity'`.

    mds_dist : string, optional, default: 'euclidean'
        Distance metric for MDS. Recommended values: 'euclidean' and 'cosine'
        Any metric from `scipy.spatial.distance` can be used. Custom distance
        functions of form `f(x, y) = d` are also accepted

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
        Defaults to the global `numpy` random number generator

    verbose : `int` or `boolean`, optional (default: 1)
        If `True` or `> 0`, print status messages

    potential_method : deprecated.
        Use `gamma=1` for log transformation and `gamma=0` for square root
        transformation.

    alpha_decay : deprecated.
        Use `decay=None` to disable alpha decay

    njobs : deprecated.
        Use n_jobs to match `sklearn` standards

    k : Deprecated for `knn`

    a : Deprecated for `decay`

    Attributes
    ----------

    X : array-like, shape=[n_samples, n_dimensions]

    embedding : array-like, shape=[n_samples, n_components]
        Stores the position of the dataset in the embedding space

    diff_op :  array-like, shape=[n_samples, n_samples] or [n_landmark, n_landmark]
        The diffusion operator built from the graph

    graph : graphtools.base.BaseGraph
        The graph built on the input data

    optimal_t : int
        The automatically selected t, when t = 'auto'.
        When t is given, optimal_t is None.

    Examples
    --------
    >>> import phate
    >>> import matplotlib.pyplot as plt
    >>> tree_data, tree_clusters = phate.tree.gen_dla(n_dim=100, n_branch=20,
    ...                                               branch_length=100)
    >>> tree_data.shape
    (2000, 100)
    >>> phate_operator = phate.PHATE(knn=5, decay=20, t=150)
    >>> tree_phate = phate_operator.fit_transform(tree_data)
    >>> tree_phate.shape
    (2000, 2)
    >>> phate.plot.scatter2d(tree_phate, c=tree_clusters)

    References
    ----------
    .. [1] Moon KR, van Dijk D, Zheng W, *et al.* (2017),
        *PHATE: A Dimensionality Reduction Method for Visualizing Trajectory
        Structures in High-Dimensional Biological Data*,
        `BioRxiv <http://biorxiv.org/content/early/2017/03/24/120378>`_.
    """

    def __init__(
        self,
        n_components=2,
        knn=5,
        decay=40,
        n_landmark=2000,
        t="auto",
        gamma=1,
        n_pca=100,
        knn_dist="euclidean",
        mds_dist="euclidean",
        mds="metric",
        n_jobs=1,
        random_state=None,
        verbose=1,
        potential_method=None,
        alpha_decay=None,
        njobs=None,
        k=None,
        a=None,
        **kwargs
    ):
        if k is not None:
            knn = k
        if a is not None:
            decay = a
        self.n_components = n_components
        self.t = t
        self.mds = mds
        self.mds_dist = mds_dist
        self.kwargs = kwargs

        self._diff_potential = None
        self.embedding = None
        self.optimal_t = None

        if (alpha_decay is True and decay is None) or (
            alpha_decay is False and decay is not None
        ):
            warnings.warn(
                "alpha_decay is deprecated. Use `decay=None`"
                " to disable alpha decay in future.",
                FutureWarning,
            )
            if not alpha_decay:
                decay = None

        if njobs is not None:
            warnings.warn(
                "njobs is deprecated. Please use n_jobs in future.", FutureWarning
            )
            n_jobs = njobs
        n_jobs = n_jobs

        if potential_method is not None:
            if potential_method == "log":
                gamma = 1
            elif potential_method == "sqrt":
                gamma = 0
            else:
                raise ValueError(
                    "potential_method {} not recognized. Please "
                    "use gamma between -1 and 1".format(potential_method)
                )
            warnings.warn(
                "potential_method is deprecated. "
                "Setting gamma to {} to achieve"
                " {} transformation.".format(gamma, potential_method),
                FutureWarning,
            )
        elif gamma > 0.99 and gamma < 1:
            warnings.warn(
                "0.99 < gamma < 1 is numerically unstable. " "Setting gamma to 0.99",
                RuntimeWarning,
            )
            gamma = 0.99
        self.gamma = gamma

        self._check_params()
        super().__init__(
            n_pca=n_pca,
            n_landmark=n_landmark,
            random_state=random_state,
            knn=knn,
            decay=decay,
            distance=knn_dist,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs
        )

    @property
    def diff_op(self):
        """The diffusion operator calculated from the data
        """
        if self.graph is not None:
            if isinstance(self.graph, graphtools.graphs.LandmarkGraph):
                diff_op = self.graph.landmark_op
            else:
                diff_op = self.graph.diff_op
            if sparse.issparse(diff_op):
                diff_op = diff_op.toarray()
            return diff_op
        else:
            raise NotFittedError(
                "This PHATE instance is not fitted yet. Call "
                "'fit' with appropriate arguments before "
                "using this method."
            )

    @property
    def diff_potential(self):
        """Interpolates the PHATE potential to one entry per cell

        This is equivalent to calculating infinite-dimensional PHATE,
        or running PHATE without the MDS step.

        Returns
        -------
        diff_potential : ndarray, shape=[n_samples, min(n_landmark, n_samples)]
        """
        diff_potential = self._calculate_potential()
        if isinstance(self.graph, graphtools.graphs.LandmarkGraph):
            diff_potential = self.graph.interpolate(diff_potential)
        return diff_potential

    def _check_params(self):
        """Check PHATE parameters

        This allows us to fail early - otherwise certain unacceptable
        parameter choices, such as mds='mmds', would only fail after
        minutes of runtime.

        Raises
        ------
        ValueError : unacceptable choice of parameters
        """
        graphtools.utils.check_positive(n_components=self.n_components)
        graphtools.utils.check_int(n_components=self.n_components)
        graphtools.utils.check_between(-1, 1, gamma=self.gamma)
        graphtools.utils.check_if_not(
            "auto",
            graphtools.utils.check_positive,
            graphtools.utils.check_int,
            t=self.t,
        )
        if not callable(self.mds_dist):
            graphtools.utils.check_in(
                [
                    "euclidean",
                    "cosine",
                    "correlation",
                    "braycurtis",
                    "canberra",
                    "chebyshev",
                    "cityblock",
                    "dice",
                    "hamming",
                    "jaccard",
                    "kulsinski",
                    "mahalanobis",
                    "matching",
                    "minkowski",
                    "rogerstanimoto",
                    "russellrao",
                    "seuclidean",
                    "sokalmichener",
                    "sokalsneath",
                    "sqeuclidean",
                    "yule",
                ],
                mds_dist=self.mds_dist,
            )
        graphtools.utils.check_in(["classic", "metric", "nonmetric"], mds=self.mds)

    def _reset_graph(self):
        self._reset_potential()

    def _reset_potential(self):
        self._diff_potential = None
        self._reset_embedding()

    def _reset_embedding(self):
        self.embedding = None

    def set_params(self, **params):
        """Set the parameters on this estimator.

        Any parameters not given as named arguments will be left at their
        current value.

        Parameters
        ----------

        n_components : int, optional, default: 2
            number of dimensions in which the data will be embedded

        knn : int, optional, default: 5
            number of nearest neighbors on which to build kernel

        decay : int, optional, default: 40
            sets decay rate of kernel tails.
            If None, alpha decaying kernel is not used

        n_landmark : int, optional, default: 2000
            number of landmarks to use in fast PHATE

        t : int, optional, default: 'auto'
            power to which the diffusion operator is powered.
            This sets the level of diffusion. If 'auto', t is selected
            according to the knee point in the Von Neumann Entropy of
            the diffusion operator

        gamma : float, optional, default: 1
            Informational distance constant between -1 and 1.
            `gamma=1` gives the PHATE log potential, `gamma=0` gives
            a square root potential.

        n_pca : int, optional, default: 100
            Number of principal components to use for calculating
            neighborhoods. For extremely large datasets, using
            n_pca < 20 allows neighborhoods to be calculated in
            roughly log(n_samples) time.

        knn_dist : string, optional, default: 'euclidean'
            recommended values: 'euclidean', 'cosine', 'precomputed'
            Any metric from `scipy.spatial.distance` can be used
            distance metric for building kNN graph. Custom distance
            functions of form `f(x, y) = d` are also accepted. If 'precomputed',
            `data` should be an n_samples x n_samples distance or
            affinity matrix. Distance matrices are assumed to have zeros
            down the diagonal, while affinity matrices are assumed to have
            non-zero values down the diagonal. This is detected automatically
            using `data[0,0]`. You can override this detection with
            `knn_dist='precomputed_distance'` or `knn_dist='precomputed_affinity'`.

        mds_dist : string, optional, default: 'euclidean'
            recommended values: 'euclidean' and 'cosine'
            Any metric from `scipy.spatial.distance` can be used
            distance metric for MDS

        mds : string, optional, default: 'metric'
            choose from ['classic', 'metric', 'nonmetric'].
            Selects which MDS algorithm is used for dimensionality reduction

        n_jobs : integer, optional, default: 1
            The number of jobs to use for the computation.
            If -1 all CPUs are used. If 1 is given, no parallel computing code
            is used at all, which is useful for debugging.
            For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
            n_jobs = -2, all CPUs but one are used

        random_state : integer or numpy.RandomState, optional, default: None
            The generator used to initialize SMACOF (metric, nonmetric) MDS
            If an integer is given, it fixes the seed
            Defaults to the global `numpy` random number generator

        verbose : `int` or `boolean`, optional (default: 1)
            If `True` or `> 0`, print status messages

        k : Deprecated for `knn`

        a : Deprecated for `decay`

        Examples
        --------
        >>> import phate
        >>> import matplotlib.pyplot as plt
        >>> tree_data, tree_clusters = phate.tree.gen_dla(n_dim=50, n_branch=5,
        ...                                               branch_length=50)
        >>> tree_data.shape
        (250, 50)
        >>> phate_operator = phate.PHATE(k=5, a=20, t=150)
        >>> tree_phate = phate_operator.fit_transform(tree_data)
        >>> tree_phate.shape
        (250, 2)
        >>> phate_operator.set_params(n_components=10)
        PHATE(a=20, alpha_decay=None, k=5, knn_dist='euclidean', mds='metric',
           mds_dist='euclidean', n_components=10, n_jobs=1, n_landmark=2000,
           n_pca=100, njobs=None, potential_method='log', random_state=None, t=150,
           verbose=1)
        >>> tree_phate = phate_operator.transform()
        >>> tree_phate.shape
        (250, 10)
        >>> # plt.scatter(tree_phate[:,0], tree_phate[:,1], c=tree_clusters)
        >>> # plt.show()

        Returns
        -------
        self
        """
        reset_kernel = False
        reset_potential = False
        reset_embedding = False

        # mds parameters
        if "n_components" in params and params["n_components"] != self.n_components:
            self.n_components = params["n_components"]
            reset_embedding = True
            del params["n_components"]
        if "mds" in params and params["mds"] != self.mds:
            self.mds = params["mds"]
            reset_embedding = True
            del params["mds"]
        if "mds_dist" in params and params["mds_dist"] != self.mds_dist:
            self.mds_dist = params["mds_dist"]
            reset_embedding = True
            del params["mds_dist"]

        # diff potential parameters
        if "t" in params and params["t"] != self.t:
            self.t = params["t"]
            reset_potential = True
            del params["t"]
        if "potential_method" in params:
            if params["potential_method"] == "log":
                params["gamma"] = 1
            elif params["potential_method"] == "sqrt":
                params["gamma"] = 0
            else:
                raise ValueError(
                    "potential_method {} not recognized. Please "
                    "use gamma between -1 and 1".format(params["potential_method"])
                )
            warnings.warn(
                "potential_method is deprecated. Setting gamma to {} to "
                "achieve {} transformation.".format(
                    params["gamma"], params["potential_method"]
                ),
                FutureWarning,
            )
            del params["potential_method"]
        if "gamma" in params and params["gamma"] != self.gamma:
            self.gamma = params["gamma"]
            reset_potential = True
            del params["gamma"]

        # kernel parameters
        if "k" in params:
            params["knn"] = params["k"]
        if "a" in params:
            params["decay"] = params["a"]
        if "knn_dist" in params:
            params["distance"] = params["knn_dist"]

        if reset_kernel:
            # can't reset the graph kernel without making a new graph
            self._reset_graph()
        if reset_potential:
            self._reset_potential()
        if reset_embedding:
            self._reset_embedding()

        self._check_params()

        super().set_params(**params)
        return self

    def reset_mds(self, **kwargs):
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
        warnings.warn(
            "PHATE.reset_mds is deprecated. " "Please use PHATE.set_params in future.",
            FutureWarning,
        )
        self.set_params(**kwargs)

    def reset_potential(self, **kwargs):
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
        warnings.warn(
            "PHATE.reset_potential is deprecated. "
            "Please use PHATE.set_params in future.",
            FutureWarning,
        )
        self.set_params(**kwargs)

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
        super().fit(X)
        # landmark op doesn't build unless forced
        self.diff_op
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
            raise NotFittedError(
                "This PHATE instance is not fitted yet. Call "
                "'fit' with appropriate arguments before "
                "using this method."
            )
        elif X is not None and not utils.matrix_is_equivalent(X, self.X):
            # fit to external data
            warnings.warn(
                "Pre-fit PHATE should not be used to transform a "
                "new data matrix. Please fit PHATE to the new"
                " data by running 'fit' with the new data.",
                RuntimeWarning,
            )
            if (
                isinstance(self.graph, graphtools.graphs.TraditionalGraph)
                and self.graph.precomputed is not None
            ):
                raise ValueError(
                    "Cannot transform additional data using a "
                    "precomputed distance matrix."
                )
            else:
                if self.embedding is None:
                    self.transform()
                transitions = self.graph.extend_to_data(X)
                return self.graph.interpolate(self.embedding, transitions)
        else:
            diff_potential = self._calculate_potential(
                t_max=t_max, plot_optimal_t=plot_optimal_t, ax=ax
            )
            if self.embedding is None:
                with _logger.task("{} MDS".format(self.mds)):
                    self.embedding = mds.embed_MDS(
                        diff_potential,
                        ndim=self.n_components,
                        how=self.mds,
                        distance_metric=self.mds_dist,
                        n_jobs=self.n_jobs,
                        seed=self.random_state,
                        verbose=max(self.verbose - 1, 0),
                    )
            if isinstance(self.graph, graphtools.graphs.LandmarkGraph):
                _logger.debug("Extending to original data...")
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
        with _logger.task("PHATE"):
            self.fit(X)
            embedding = self.transform(**kwargs)
        return embedding

    def _calculate_potential(self, t=None, t_max=100, plot_optimal_t=False, ax=None):
        """Calculates the diffusion potential

        Parameters
        ----------

        t : int
            power to which the diffusion operator is powered
            sets the level of diffusion

        t_max : int, default: 100
            Maximum value of `t` to test

        plot_optimal_t : boolean, default: False
            If true, plots the Von Neumann Entropy and knee point

        ax : matplotlib.Axes, default: None
            If plot=True and ax is not None, plots the VNE on the given axis
            Otherwise, creates a new axis and displays the plot

        Returns
        -------

        diff_potential : array-like, shape=[n_samples, n_samples]
            The diffusion potential fit on the input data
        """
        if t is None:
            t = self.t
        if self._diff_potential is None:
            if t == "auto":
                t = self._find_optimal_t(t_max=t_max, plot=plot_optimal_t, ax=ax)
            else:
                t = self.t
            with _logger.task("diffusion potential"):
                # diffused diffusion operator
                diff_op_t = np.linalg.matrix_power(self.diff_op, t)
                if self.gamma == 1:
                    # handling small values
                    diff_op_t = diff_op_t + 1e-7
                    self._diff_potential = -1 * np.log(diff_op_t)
                elif self.gamma == -1:
                    self._diff_potential = diff_op_t
                else:
                    c = (1 - self.gamma) / 2
                    self._diff_potential = ((diff_op_t) ** c) / c
        elif plot_optimal_t:
            self._find_optimal_t(t_max=t_max, plot=plot_optimal_t, ax=ax)

        return self._diff_potential

    def _von_neumann_entropy(self, t_max=100):
        """Calculate Von Neumann Entropy

        Determines the Von Neumann entropy of the diffusion affinities
        at varying levels of `t`. The user should select a value of `t`
        around the "knee" of the entropy curve.

        We require that 'fit' stores the value of `PHATE.diff_op`
        in order to calculate the Von Neumann entropy.

        Parameters
        ----------
        t_max : int, default: 100
            Maximum value of `t` to test

        Returns
        -------
        entropy : array, shape=[t_max]
            The entropy of the diffusion affinities for each value of `t`
        """
        t = np.arange(t_max)
        return t, vne.compute_von_neumann_entropy(self.diff_op, t_max=t_max)

    def _find_optimal_t(self, t_max=100, plot=False, ax=None):
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
        with _logger.task("optimal t"):
            t, h = self._von_neumann_entropy(t_max=t_max)
            t_opt = vne.find_knee_point(y=h, x=t)
            _logger.task("Automatically selected t = {}".format(t_opt))

        if plot:
            if ax is None:
                fig, ax = plt.subplots()
                show = True
            else:
                show = False
            ax.plot(t, h)
            ax.scatter(t_opt, h[t == t_opt], marker="*", c="k", s=50)
            ax.set_xlabel("t")
            ax.set_ylabel("Von Neumann Entropy")
            ax.set_title("Optimal t = {}".format(t_opt))
            if show:
                plt.show()

        self.optimal_t = t_opt

        return t_opt
