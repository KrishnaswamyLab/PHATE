import numpy as np
from sklearn import metrics
import numbers


def auc(X, E, k=0.1):
    """Area Under the Precision Recall Curve

    Precision and recall for dimensionality reduction as defined in
    Lui et al, 2018 [1]_

    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
        Input data
    E : array-like, shape=[n_samples, n_dimensions]
        Embedding
    k : int or float, optional (default: 0.1)
        Number of neighbors to use for precision-recall calculation.
        If a float, this is treated as a fraction of the dataset.

    Returns
    -------
    auc : float
        Area under the curve between 0 and 1

    References
    ----------

    .. [1] Lui KYC, Ding GW, Huang R and McCann RJ (2018),
        *Dimensionality Reduction has Quantifiable Imperfections:
        Two Geometric Bounds*,
        `Advances in Neural Information Processing Systems 31
        <http://papers.nips.cc/paper/8065-dimensionality-reduction-has-quantifiable-imperfections-two-geometric-bounds.pdf>`_.
    """
    # find the radius that gives you an average of k% neighbors for each point
    if not isinstance(k, numbers.Integral):
        k = int(k * X.shape[0])

    # pairwise distances
    X_dist = metrics.pairwise_distances(X, X)
    min_ = X_dist[X_dist > 0].min()
    max_ = X_dist.max()
    for radius in np.linspace(min_, max_, 1000):
        k_radius = (X_dist < radius).sum(axis=1).mean()
        if k_radius > k:
            break

    # create a mask for neighbors in original space
    is_neighbor_X = (X_dist < radius).flatten()
    # distances in embedding space
    E_dist = metrics.pairwise_distances(E, E).flatten()
    # auc, using distances
    auc = metrics.roc_auc_score(~is_neighbor_X, E_dist)
    return auc


def precision_recall(X, E, k=0.1):
    """Dimensionality reduction precision and recall

    Precision and recall for dimensionality reduction as defined in
    Lui et al, 2018 [1]_

    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
        Input data
    E : array-like, shape=[n_samples, n_dimensions]
        Embedding
    k : int or float, optional (default: 0.1)
        Number of neighbors to use for precision-recall calculation.
        If a float, this is treated as a fraction of the dataset.

    Returns
    -------
    auc : float
        Area under the curve between 0 and 1

    References
    ----------

    .. [1] Lui KYC, Ding GW, Huang R and McCann RJ (2018),
        *Dimensionality Reduction has Quantifiable Imperfections:
        Two Geometric Bounds*,
        `Advances in Neural Information Processing Systems 31
        <http://papers.nips.cc/paper/8065-dimensionality-reduction-has-quantifiable-imperfections-two-geometric-bounds.pdf>`_.
    """
    # find the radius that gives you an average of k% neighbors for each point
    if not isinstance(k, numbers.Integral):
        k = int(k * X.shape[0])

    # radius in E to get k neighbors
    X_dist = metrics.pairwise_distances(X, X)
    min_ = X_dist[X_dist > 0].min()
    max_ = X_dist.max()
    for X_radius in np.linspace(min_, max_, 1000):
        k_radius = (X_dist < X_radius).sum(axis=1).mean()
        if k_radius > k:
            break

    # radius in E to get k neighbors
    E_dist = metrics.pairwise_distances(E, E)
    min_ = E_dist[E_dist > 0].min()
    max_ = E_dist.max()
    for E_radius in np.linspace(min_, max_, 1000):
        k_radius = (E_dist < E_radius).sum(axis=1).mean()
        if k_radius > k:
            break

    is_neighbor_X = X_dist < X_radius
    is_neighbor_E = E_dist < E_radius
    both_neighbors = is_neighbor_X & is_neighbor_E

    precision = both_neighbors.sum(1) / is_neighbor_E.sum(1)
    recall = both_neighbors.sum(1) / is_neighbor_X.sum(1)

    return precision, recall
