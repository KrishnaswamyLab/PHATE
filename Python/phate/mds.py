# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from __future__ import print_function, division
from sklearn.manifold import smacof
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import tasklogger
# Fast classical MDS using random svd


def cmdscale_fast(D, ndim):
    """Fast CMDS using random SVD

    Parameters
    ----------
    D : array-like, input data [n_samples, n_dimensions]

    ndim : int, number of dimensions in which to embed `D`

    Returns
    -------
    Y : array-like, embedded data [n_sample, ndim]
    """
    tasklogger.log_debug("Performing classic MDS on {} of shape {}...".format(
        type(D).__name__, D.shape))
    D = D**2
    D = D - D.mean(axis=0)[None, :]
    D = D - D.mean(axis=1)[:, None]
    pca = PCA(n_components=ndim, svd_solver='randomized')
    Y = pca.fit_transform(D)
    return Y


def embed_MDS(X, ndim=2, how='metric', distance_metric='euclidean',
              n_jobs=1, seed=None, verbose=0):
    """Performs classic, metric, and non-metric MDS

    Metric MDS is initialized using classic MDS,
    non-metric MDS is initialized using metric MDS.

    Parameters
    ----------
    X: ndarray [n_samples, n_samples]
        2 dimensional input data array with n_samples
        embed_MDS does not check for matrix squareness,
        but this is necessary for PHATE

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
    if how not in ['classic', 'metric', 'nonmetric']:
        raise ValueError("Allowable 'how' values for MDS: 'classic', "
                         "'metric', or 'nonmetric'. "
                         "'{}' was passed.".format(how))

    # MDS embeddings, each gives a different output.
    X_dist = squareform(pdist(X, distance_metric))

    # initialize all by CMDS
    Y = cmdscale_fast(X_dist, ndim)
    if how in ['metric', 'nonmetric']:
        tasklogger.log_debug("Performing metric MDS on "
                             "{} of shape {}...".format(type(X_dist),
                                                        X_dist.shape))
        # Metric MDS from sklearn
        Y, _ = smacof(X_dist, n_components=ndim, metric=True, max_iter=3000,
                      eps=1e-6, random_state=seed, n_jobs=n_jobs,
                      n_init=1, init=Y, verbose=verbose)
    if how == 'nonmetric':
        tasklogger.log_debug(
            "Performing non-metric MDS on "
            "{} of shape {}...".format(type(X_dist),
                                       X_dist.shape))
        # Nonmetric MDS from sklearn using metric MDS as an initialization
        Y, _ = smacof(X_dist, n_components=ndim, metric=True, max_iter=3000,
                      eps=1e-6, random_state=seed, n_jobs=n_jobs,
                      n_init=1, init=Y, verbose=verbose)
    return Y
