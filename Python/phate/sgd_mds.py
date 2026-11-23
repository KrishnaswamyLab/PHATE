# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

"""Metric MDS by stochastic gradient descent, following s_gd2.

Uses the per-pair update of Zheng, Pawar & Goodman (2019), "Graph Drawing by Stochastic Gradient Descent"
(the algorithm behind the s_gd2 package that PHATE 1.x used), with uniform weights (standard metric MDS):
each visited pair (i, j) moves both points along the line between them by
``mu * (|y_i - y_j| - d_ij) / 2``, with ``mu = min(eta, 1)``, and ``eta`` decays exponentially from 1 to
``eps`` (0.001) over the passes. Pairs are updated one at a time in effect: each batch is a matching (no point
appears twice), so a vectorised batch equals sequential updates.

To keep the cost low, only the first ``full_passes`` passes visit every pair (O(n^2) each). The remaining
passes visit each point's ``n_neighbors`` nearest pairs (by target distance) plus a few random perfect
matchings (O(n * (n_neighbors + log n)) each).

The early full passes take large steps over every pair, which places tight, well-separated groups as a
whole. Without them, small summed steps can tear such a group apart early on and never rejoin it; with
landmarks, PHATE then draws the group's samples as a straight line between the pieces.
"""

from __future__ import print_function, division
import warnings

import numpy as np
import tasklogger

_logger = tasklogger.get_tasklogger("graphtools")


def _get_rng(random_state):
    if isinstance(random_state, np.random.RandomState):
        return np.random.default_rng(random_state.randint(np.iinfo(np.int32).max))
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _round_robin(n):
    """All pairs of n points as n-1 perfect matchings (circle method); n is padded to even with index -1."""
    m = n + (n % 2)
    half = m // 2
    base = np.arange(1, m)
    rounds = np.empty((m - 1, 2, half), dtype=np.int64)
    for r in range(m - 1):
        arr = np.concatenate(([0], np.roll(base, r)))
        rounds[r, 0], rounds[r, 1] = arr[:half], arr[::-1][:half]
    rounds[rounds == n] = -1  # padding index for odd n
    return rounds


def _neighbor_matchings(D, k):
    """Each point's k nearest pairs (by target distance), split into matchings by greedy edge colouring."""
    n = D.shape[0]
    if k <= 0 or n < 2:
        return []
    k = min(k, n - 1)
    Dk = np.array(D, dtype=np.float64, copy=True)
    np.fill_diagonal(Dk, np.inf)
    nb = np.argpartition(Dk, k - 1, axis=1)[:, :k]
    a = np.repeat(np.arange(n), k)
    b = nb.ravel()
    edges = np.unique(np.c_[np.minimum(a, b), np.maximum(a, b)], axis=0)
    used = [set() for _ in range(n)]
    colour = np.empty(len(edges), dtype=np.int64)
    for e, (u, v) in enumerate(edges):
        c = 0
        while c in used[u] or c in used[v]:
            c += 1
        colour[e] = c
        used[u].add(c)
        used[v].add(c)
    return [edges[colour == c] for c in range(colour.max() + 1)]


def _update(Y, D, a, b, mu):
    """s_gd2 step for pairs (a, b) that share no point."""
    d = D[a, b]
    diff = Y[a] - Y[b]
    mag = np.maximum(np.sqrt((diff * diff).sum(axis=1)), 1e-12)
    step = (mu * (mag - d) / (2.0 * mag))[:, None] * diff
    Y[a] -= step
    Y[b] += step


def sgd_mds(
    D,
    n_components=2,
    learning_rate=None,
    n_iter=30,
    init=None,
    random_state=None,
    verbose=0,
    pairs_per_iter=None,
    full_passes=2,
    n_neighbors=10,
    eps=0.001,
):
    """Metric MDS by stochastic gradient descent (s_gd2 update, uniform weights)

    Parameters
    ----------
    D : array-like, shape (n_samples, n_samples)
        Distance matrix
    n_components : int, default=2
        Number of dimensions for the output embedding
    learning_rate : None
        Unused; kept for backwards compatibility. The step size follows the s_gd2 schedule (1 to `eps`).
    n_iter : int, default=30
        Number of passes. The step size decays from 1 to `eps` over the passes.
    init : array-like, shape (n_samples, n_components), optional
        Initial embedding (e.g., from classical MDS). Random if None.
    random_state : int, RandomState, Generator or None, optional
        Random state for reproducibility
    verbose : int, default=0
        Verbosity level
    pairs_per_iter : int, optional
        Random pairs per cheap pass (rounded to whole random perfect matchings).
        If None, uses about n * log(n) pairs (2 * log(n) matchings).
    full_passes : int, default=2
        Number of initial passes that visit every pair (O(n^2) each).
    n_neighbors : int, default=10
        Nearest pairs per point visited in every cheap pass.
    eps : float, default=0.001
        Final step size.

    Returns
    -------
    Y : array-like, shape (n_samples, n_components)
        Embedded coordinates
    """
    if learning_rate is not None:
        warnings.warn(
            "`learning_rate` is no longer used by sgd_mds: the step size follows the s_gd2 schedule.",
            DeprecationWarning,
        )
    rng = _get_rng(random_state)
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if init is None:
        Y = rng.standard_normal((n, n_components))
    else:
        Y = np.array(init, dtype=np.float64, copy=True)
    if n < 2:
        return Y

    # match the scale of the initial layout to the target distances
    iu = np.triu_indices(n, 1)
    d_init = np.sqrt(((Y[iu[0]] - Y[iu[1]]) ** 2).sum(axis=1))
    if np.median(d_init) > 0 and np.median(D[iu]) > 0:
        Y *= np.median(D[iu]) / np.median(d_init)

    n_iter = max(int(n_iter), 1)
    etas = np.exp(-np.log(1.0 / eps) / max(n_iter - 1, 1) * np.arange(n_iter))
    half = n // 2
    if pairs_per_iter is None:
        n_random = max(1, int(round(2 * np.log(n))))
    else:
        n_random = max(1, int(np.ceil(pairs_per_iter / max(half, 1))))
    full_passes = min(int(full_passes), n_iter)
    rounds = _round_robin(n) if full_passes > 0 else None
    neighbors = _neighbor_matchings(D, n_neighbors) if full_passes < n_iter else []

    if verbose > 0:
        _logger.log_debug(
            "SGD-MDS: {} passes ({} over all pairs), {} neighbour matchings and {} random "
            "matchings per cheap pass".format(
                n_iter, full_passes, len(neighbors), n_random
            )
        )

    for it, eta in enumerate(etas):
        mu = min(eta, 1.0)
        if it < full_passes:
            perm = np.append(rng.permutation(n), -1)  # index -1 maps padding to -1
            for r in rng.permutation(len(rounds)):
                a, b = rounds[r]
                ok = (a >= 0) & (b >= 0)
                _update(Y, D, perm[a[ok]], perm[b[ok]], mu)
        else:
            batches = [(e[:, 0], e[:, 1]) for e in neighbors]
            for _ in range(n_random):
                p = rng.permutation(n)
                batches.append((p[:half], p[half : 2 * half]))
            for idx in rng.permutation(len(batches)):
                _update(Y, D, batches[idx][0], batches[idx][1], mu)
    return Y


def sgd_mds_metric(
    D,
    n_components=2,
    init=None,
    random_state=None,
    verbose=0,
):
    """Metric MDS by SGD with PHATE's default settings

    30 passes; the first 2 visit every pair, the rest visit each point's 10 nearest pairs plus about
    n * log(n) random pairs.

    Parameters
    ----------
    D : array-like, shape (n_samples, n_samples)
        Distance matrix
    n_components : int, default=2
        Number of dimensions for the output embedding
    init : array-like, shape (n_samples, n_components), optional
        Initial embedding (e.g., from classical MDS)
    random_state : int or RandomState, optional
        Random state for reproducibility
    verbose : int, default=0
        Verbosity level (0=silent, 1=progress, 2=debug)

    Returns
    -------
    Y : array-like, shape (n_samples, n_components)
        Embedded coordinates
    """
    return sgd_mds(
        D=D,
        n_components=n_components,
        n_iter=30,
        init=init,
        random_state=random_state,
        verbose=verbose,
    )
