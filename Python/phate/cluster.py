from sklearn import cluster, exceptions, metrics
import warnings
import numpy as np
from .phate import PHATE


def silhouette_score(phate_op, n_clusters, random_state=None, **kwargs):
    """Compute the Silhouette score on KMeans on the PHATE potential

    Parameters
    ----------
    phate_op : phate.PHATE
        Fitted PHATE operator
    n_clusters : int
        Number of clusters.
    random_state : int or None, optional (default: None)
        Random seed for k-means

    Returns
    -------
    score : float
    """
    cluster_labels = kmeans(
        phate_op, n_clusters=n_clusters, random_state=random_state, **kwargs
    )
    return metrics.silhouette_score(phate_op.diff_potential, cluster_labels)


def kmeans(
    phate_op, n_clusters="auto", max_clusters=10, random_state=None, k=None, **kwargs
):
    """KMeans on the PHATE potential

    Clustering on the PHATE operator as introduced in Moon et al.
    This is similar to spectral clustering.

    Parameters
    ----------
    phate_op : phate.PHATE
        Fitted PHATE operator
    n_clusters : int, optional (default: 'auto')
        Number of clusters.
        If 'auto', uses the Silhouette score to determine the optimal number of clusters
    max_clusters : int, optional (default: 10)
        Maximum number of clusters to test if using the Silhouette score.
    random_state : int or None, optional (default: None)
        Random seed for k-means
    k : deprecated for `n_clusters`
    kwargs : additional arguments for `sklearn.cluster.KMeans`

    Returns
    -------
    clusters : np.ndarray
        Integer array of cluster assignments
    """
    if k is not None:
        warnings.warn(
            "k is deprecated. Please use n_clusters in future.", FutureWarning
        )
        n_clusters = k
    if not isinstance(phate_op, PHATE):
        raise TypeError(
            "Expected phate_op to be of type PHATE. Got {}".format(phate_op)
        )
    if phate_op.graph is not None:
        if n_clusters == "auto":
            n_clusters = np.arange(2, max_clusters)
            silhouette_scores = [
                silhouette_score(phate_op, k, random_state=random_state, **kwargs)
                for k in n_clusters
            ]
            n_clusters = n_clusters[np.argmax(silhouette_scores)]
        return cluster.KMeans(
            n_clusters, random_state=random_state, **kwargs
        ).fit_predict(phate_op.diff_potential)
    else:
        raise exceptions.NotFittedError(
            "This PHATE instance is not fitted yet. Call "
            "'fit' with appropriate arguments before "
            "using this method."
        )
