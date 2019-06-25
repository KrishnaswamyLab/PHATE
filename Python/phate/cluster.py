from sklearn import cluster, exceptions
import warnings


def kmeans(phate_op, n_clusters=8, random_state=None, k=None):
    """KMeans on the PHATE potential

    Clustering on the PHATE operator as introduced in Moon et al.
    This is similar to spectral clustering.

    Parameters
    ----------

    phate_op : phate.PHATE
        Fitted PHATE operator
    n_clusters : int, optional (default: 8)
        Number of clusters
    random_state : int or None, optional (default: None)
        Random seed for k-means
    k : deprecated for `n_clusters`

    Returns
    -------
    clusters : np.ndarray
        Integer array of cluster assignments
    """
    if k is not None:
        warnings.warn(
            "k is deprecated. Please use n_clusters in future.",
            FutureWarning)
        n_clusters = k
    if phate_op.graph is not None:
        diff_potential = phate_op.diff_potential
        return cluster.KMeans(n_clusters, random_state=random_state).fit_predict(diff_potential)
    else:
        raise exceptions.NotFittedError(
            "This PHATE instance is not fitted yet. Call "
            "'fit' with appropriate arguments before "
            "using this method.")
