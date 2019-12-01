# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from __future__ import print_function, division
from sklearn import manifold
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from deprecated import deprecated

import tasklogger
import scprep
import s_gd2

_logger = tasklogger.get_tasklogger("graphtools")


# Fast classical MDS using random svd
@deprecated("1.5.0", reason="Use phate.mds.classic instead")
def cmdscale_fast(D, ndim):
    return classic(D=D, n_components=ndim)


def classic(D, n_components=2, random_state=None):
    """Fast CMDS using random SVD

    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances

    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`

    random_state : int, RandomState or None, optional (default: None)
        numpy random state

    Returns
    -------
    Y : array-like, embedded data [n_sample, ndim]
    """
    _logger.debug(
        "Performing classic MDS on {} of shape {}...".format(type(D).__name__, D.shape)
    )
    D = D ** 2
    D = D - D.mean(axis=0)[None, :]
    D = D - D.mean(axis=1)[:, None]
    pca = PCA(
        n_components=n_components, svd_solver="randomized", random_state=random_state
    )
    Y = pca.fit_transform(D)
    return Y


@scprep.utils._with_pkg(pkg="s_gd2", min_version="1.3")
def sgd(D, n_components=2, random_state=None, init=None):
    """Metric MDS using stochastic gradient descent

    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances

    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`

    random_state : int or None, optional (default: None)
        numpy random state

    init : array-like or None
        Initialization algorithm or state to use for MMDS

    Returns
    -------
    Y : array-like, embedded data [n_sample, ndim]
    """
    if not n_components == 2:
        raise NotImplementedError
    _logger.debug("Performing SGD MDS on " "{} of shape {}...".format(type(D), D.shape))
    N = D.shape[0]
    D = squareform(D)
    # Metric MDS from s_gd2
    Y = s_gd2.mds_direct(N, D, init=init, random_seed=random_state)
    return Y


def smacof(
    D,
    n_components=2,
    metric=True,
    init=None,
    random_state=None,
    verbose=0,
    max_iter=3000,
    eps=1e-6,
    n_jobs=1,
):
    """Metric and non-metric MDS using SMACOF

    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances

    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`
        
    metric : bool, optional (default: True)
        Use metric MDS. If False, uses non-metric MDS

    init : array-like or None, optional (default: None)
        Initialization state

    random_state : int, RandomState or None, optional (default: None)
        numpy random state

    verbose : int or bool, optional (default: 0)
        verbosity

    max_iter : int, optional (default: 3000)
        maximum iterations

    eps : float, optional (default: 1e-6)
        stopping criterion

    Returns
    -------
    Y : array-like, shape=[n_samples, n_components]
        embedded data
    """
    _logger.debug(
        "Performing non-metric MDS on " "{} of shape {}...".format(type(D), D.shape)
    )
    # Metric MDS from sklearn
    Y, _ = manifold.smacof(
        D,
        n_components=n_components,
        metric=metric,
        max_iter=max_iter,
        eps=eps,
        random_state=random_state,
        n_jobs=n_jobs,
        n_init=1,
        init=init,
        verbose=verbose,
    )
    return Y


def embed_MDS(
    X, ndim=2, how="metric", distance_metric="euclidean", n_jobs=1, seed=None, verbose=0
):
    """Performs classic, metric, and non-metric MDS

    Metric MDS is initialized using classic MDS,
    non-metric MDS is initialized using metric MDS.

    Parameters
    ----------
    X: ndarray [n_samples, n_features]
        2 dimensional input data array with n_samples

    n_dim : int, optional, default: 2
        number of dimensions in which the data will be embedded

    how : string, optional, default: 'classic'
        choose from ['classic', 'metric', 'nonmetric']
        which MDS algorithm is used for dimensionality reduction

    distance_metric : string, optional, default: 'euclidean'
        choose from ['cosine', 'euclidean']
        distance metric for MDS

    n_jobs : integer, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used

    seed: integer or numpy.RandomState, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global numpy random number generator

    Returns
    -------
    Y : ndarray [n_samples, n_dim]
        low dimensional embedding of X using MDS
    """

    if how not in ["classic", "metric", "nonmetric"]:
        raise ValueError(
            "Allowable 'how' values for MDS: 'classic', "
            "'metric', or 'nonmetric'. "
            "'{}' was passed.".format(how)
        )

    # MDS embeddings, each gives a different output.
    X_dist = squareform(pdist(X, distance_metric))

    # initialize all by CMDS
    Y = classic(X_dist, n_components=ndim, random_state=seed)
    if how == "classic":
        return Y

    # metric is next fastest
    try:
        # use sgd2 if it is available
        Y = sgd(X_dist, n_components=ndim, random_state=seed, init=Y)
    except NotImplementedError:
        # sgd2 currently only supports n_components==2
        pass
    Y = smacof(X_dist, n_components=ndim, random_state=seed, init=Y, metric=True)
    if how == "metric":
        return Y
    # nonmetric is slowest
    Y = smacof(X_dist, n_components=ndim, random_state=seed, init=Y, metric=False)
    return Y
