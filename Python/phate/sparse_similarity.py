"""Sparse similarity computation with batched top-k selection.

Computes pairwise similarities in batches to avoid materializing a dense
N x N matrix. For each batch of rows, the full N-column similarity is
computed, then only the top-k entries per row are retained. The result is
a scipy.sparse.csr_matrix.

Memory: O(batch_size * N) for the batch similarity matrix + O(N * k) for
the sparse result. For N=20000, batch_size=256, float64: ~40 MB peak.
"""

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, issparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from scipy import sparse
import tasklogger

_logger = tasklogger.get_tasklogger("graphtools")


def compute_sparse_similarity(
    X,
    k=10,
    metric="euclidean",
    decay=40,
    batch_size=256,
    verbose=1,
):
    """Compute a sparse similarity matrix using batched top-k selection.

    For each batch of rows, pairwise distances/similarities are computed
    against ALL columns. Only the top-k entries per row are retained.
    Distances are converted to affinities via an alpha-decaying kernel:
    ``affinity = exp(-distance / decay)``. All other entries are zero.

    For distance metrics: keeps the k smallest distances, applies kernel.
    For similarity metrics: keeps the k largest values.

    Parameters
    ----------
    X : ndarray, shape=[n_samples, n_features]
        Input data matrix.
    k : int
        Number of top entries to retain per row.
    metric : str
        Distance/similarity metric for ``scipy.spatial.distance.cdist``.
    decay : float or None
        Alpha decay parameter for kernel: ``exp(-d / decay)``.
        If None, distances are stored as-is.
    batch_size : int
        Number of rows to process per batch. Controls peak memory.
    verbose : int
        Verbosity level.

    Returns
    -------
    S : scipy.sparse.csr_matrix, shape=[n_samples, n_samples]
        Sparse similarity matrix. Symmetrized: (S + S.T) / 2.
    """
    n_samples, n_features = X.shape
    k = min(k, n_samples)

    # Determine whether metric is a similarity (keep largest)
    similarity_metrics = {"cosine", "correlation"}
    is_similarity = metric in similarity_metrics

    # Accumulate COO entries: (row, col, value)
    all_rows = []
    all_cols = []
    all_vals = []

    kernel_str = f"exp(-d/{decay})" if decay else "raw"
    _logger.log_info(
        f"Computing sparse {k}-NN similarity on {n_samples} samples "
        f"with metric='{metric}', kernel={kernel_str}, "
        f"batch_size={batch_size}..."
    )

    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch = X[batch_start:batch_end]  # (batch_sz, n_features)

        # Compute distances/similarities for this batch against all samples
        pair = cdist(batch, X, metric=metric)  # (batch_sz, n_samples)

        for i in range(pair.shape[0]):
            global_i = batch_start + i
            row = pair[i]

            if is_similarity:
                # Keep k largest similarity values
                if k < n_samples:
                    idx = np.argpartition(-row, k)[:k]
                else:
                    idx = np.arange(n_samples)
                top_vals = row[idx].astype(np.float64, copy=True)
            else:
                # Keep k smallest distances, then apply kernel
                if k < n_samples:
                    idx = np.argpartition(row, k)[:k]
                else:
                    idx = np.arange(n_samples)
                top_vals = row[idx].astype(np.float64, copy=True)
                # Alpha-decaying kernel: exp(-d / decay), unselected = 0
                if decay is not None:
                    np.exp(-top_vals / decay, out=top_vals)

            all_rows.extend([global_i] * len(idx))
            all_cols.extend(idx)
            all_vals.extend(top_vals)

        if verbose > 0 and (batch_end % (batch_size * 10) == 0 or batch_end == n_samples):
            _logger.log_info(f"  Processed {batch_end}/{n_samples} samples...")

    n_entries = len(all_rows)
    _logger.log_info(f"Building sparse matrix with {n_entries} entries...")

    S_coo = coo_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=(n_samples, n_samples),
    )
    S = S_coo.tocsr()

    # Symmetrize: average with transpose for undirected graph
    S = (S + S.T) * 0.5

    _logger.log_info(
        f"Result: {S.nnz} non-zeros ({100 * S.nnz / (n_samples * n_samples):.3f}% dense)"
    )

    return S


def compute_sparse_diffusion_operator(S):
    """Row-normalize a sparse similarity matrix into a diffusion operator.

    The diffusion operator P is the row-stochastic transition probability
    matrix: P = D^{-1} S, where D is the diagonal degree matrix.

    This is a sparse-aware implementation that avoids densifying.

    Parameters
    ----------
    S : scipy.sparse.csr_matrix
        Sparse similarity/affinity matrix.

    Returns
    -------
    P : scipy.sparse.csr_matrix
        Row-normalized diffusion operator.
    """
    if not issparse(S):
        return _dense_row_normalize(S)

    # Compute row sums (degrees)
    row_sums = np.array(S.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0  # avoid division by zero

    # D^{-1} S: scale each row by its inverse degree
    inv_degrees = 1.0 / row_sums
    D_inv = sparse.diags(inv_degrees)
    P = D_inv @ S

    return P.tocsr()


def _dense_row_normalize(S):
    """Fallback row normalization for dense input."""
    row_sums = S.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    return S / row_sums[:, np.newaxis]


def check_connectivity(S):
    """Check if the graph represented by sparse matrix S is connected.

    Parameters
    ----------
    S : scipy.sparse.csr_matrix or ndarray
        Adjacency/similarity matrix.

    Returns
    -------
    n_components : int
        Number of connected components.
    labels : ndarray
        Component label for each node.
    """
    if issparse(S):
        n_components, labels = connected_components(
            S, directed=False, return_labels=True
        )
    else:
        from scipy.sparse.csgraph import connected_components as cc

        n_components, labels = cc(S, directed=False, return_labels=True)
    return n_components, labels


def find_minimal_k(
    X,
    k_max=1000,
    k_min=1,
    metric="euclidean",
    decay=40,
    batch_size=256,
    verbose=1,
):
    """Binary search for the minimal k that maintains graph connectivity.

    Starts by verifying that k_max produces a connected graph. Then
    binary-searches between k_min and k_max for the smallest k that
    still yields a single connected component.

    Parameters
    ----------
    X : ndarray, shape=[n_samples, n_features]
        Input data matrix.
    k_max : int
        Upper bound for k (must be large enough for connectivity).
    k_min : int
        Lower bound for k.
    metric : str
        Distance/similarity metric.
    decay : float or None
        Alpha decay parameter for kernel.
    batch_size : int
        Batch size for sparse similarity computation.
    verbose : int
        Verbosity level.

    Returns
    -------
    k_opt : int
        Minimal k achieving connectivity.
    S : scipy.sparse.csr_matrix
        Sparse similarity matrix at k_opt.
    """
    n_samples = X.shape[0]
    k_max = min(k_max, n_samples - 1)
    k_min = max(k_min, 1)

    _logger.log_info(
        f"Finding minimal k for connectivity on {n_samples} samples "
        f"(searching [{k_min}, {k_max}])..."
    )

    # Verify k_max works
    S_max = compute_sparse_similarity(
        X, k=k_max, metric=metric, decay=decay,
        batch_size=batch_size, verbose=max(verbose - 1, 0)
    )
    n_comp, _ = check_connectivity(S_max)
    if n_comp > 1:
        raise ValueError(
            f"Graph is disconnected even with k=k_max={k_max} "
            f"({n_comp} components). Increase k_max."
        )
    _logger.log_info(f"k_max={k_max} is connected ✓")

    # Binary search
    lo, hi = k_min, k_max
    best_S = S_max

    while lo < hi:
        mid = (lo + hi) // 2
        _logger.log_debug(f"Testing k={mid} [{lo}, {hi}]...")
        S_mid = compute_sparse_similarity(
            X, k=mid, metric=metric, decay=decay,
            batch_size=batch_size, verbose=0
        )
        n_comp, _ = check_connectivity(S_mid)

        if n_comp == 1:
            hi = mid
            best_S = S_mid
            _logger.log_debug(f"  k={mid} connected, searching lower half")
        else:
            lo = mid + 1
            _logger.log_debug(f"  k={mid} disconnected ({n_comp} components), "
                              f"searching upper half")

    _logger.log_info(f"Minimal k for connectivity: {lo}")
    return lo, best_S
