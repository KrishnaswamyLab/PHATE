# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from __future__ import print_function, division
import numpy as np
from scipy.linalg import svd

# Von Neumann Entropy


def compute_von_neumann_entropy(data, t_max=100):
    """
    Determines the Von Neumann entropy of data
    at varying matrix powers. The user should select a value of t
    around the "knee" of the entropy curve.

    Parameters
    ----------
    t_max : int, default: 100
        Maximum value of t to test

    Returns
    -------
    entropy : array, shape=[t_max]
        The entropy of the diffusion affinities for each value of t

    Examples
    --------
    >>> import numpy as np
    >>> import phate
    >>> X = np.eye(10)
    >>> X[0,0] = 5
    >>> X[3,2] = 4
    >>> h = phate.vne.compute_von_neumann_entropy(X)
    >>> phate.vne.find_knee_point(h)
    23

    """
    _, eigenvalues, _ = svd(data)
    entropy = []
    eigenvalues_t = np.copy(eigenvalues)
    for _ in range(t_max):
        prob = eigenvalues_t / np.sum(eigenvalues_t)
        prob = prob + np.finfo(float).eps
        entropy.append(-np.sum(prob * np.log(prob)))
        eigenvalues_t = eigenvalues_t * eigenvalues
    entropy = np.array(entropy)

    return np.array(entropy)


def find_knee_point(y, x=None):
    """
    Returns the x-location of a (single) knee of curve y=f(x)

    Parameters
    ----------

    y : array, shape=[n]
        data for which to find the knee point

    x : array, optional, shape=[n], default=np.arange(len(y))
        indices of the data points of y,
        if these are not in order and evenly spaced

    Returns
    -------
    knee_point : int
    The index (or x value) of the knee point on y

    Examples
    --------
    >>> import numpy as np
    >>> import phate
    >>> x = np.arange(20)
    >>> y = np.exp(-x/10)
    >>> phate.vne.find_knee_point(y,x)
    8

    """
    try:
        y.shape
    except AttributeError:
        y = np.array(y)

    if len(y) < 3:
        raise ValueError("Cannot find knee point on vector of length 3")
    elif len(y.shape) > 1:
        raise ValueError("y must be 1-dimensional")

    if x is None:
        x = np.arange(len(y))
    else:
        try:
            x.shape
        except AttributeError:
            x = np.array(x)
        if not x.shape == y.shape:
            raise ValueError("x and y must be the same shape")
        else:
            # ensure x is sorted float
            idx = np.argsort(x)
            x = x[idx]
            y = y[idx]

    n = np.arange(2, len(y) + 1).astype(np.float32)
    # figure out the m and b (in the y=mx+b sense) for the "left-of-knee"
    sigma_xy = np.cumsum(x * y)[1:]
    sigma_x = np.cumsum(x)[1:]
    sigma_y = np.cumsum(y)[1:]
    sigma_xx = np.cumsum(x * x)[1:]
    det = (n * sigma_xx - sigma_x * sigma_x)
    mfwd = (n * sigma_xy - sigma_x * sigma_y) / det
    bfwd = -(sigma_x * sigma_xy - sigma_xx * sigma_y) / det

    # figure out the m and b (in the y=mx+b sense) for the "right-of-knee"
    sigma_xy = np.cumsum(x[::-1] * y[::-1])[1:]
    sigma_x = np.cumsum(x[::-1])[1:]
    sigma_y = np.cumsum(y[::-1])[1:]
    sigma_xx = np.cumsum(x[::-1] * x[::-1])[1:]
    det = (n * sigma_xx - sigma_x * sigma_x)
    mbck = ((n * sigma_xy - sigma_x * sigma_y) / det)[::-1]
    bbck = (-(sigma_x * sigma_xy - sigma_xx * sigma_y) / det)[::-1]

    # figure out the sum of per-point errors for left- and right- of-knee fits
    error_curve = np.full_like(y, np.float('nan'))
    for breakpt in np.arange(1, len(y) - 1):
        delsfwd = (mfwd[breakpt - 1] * x[:breakpt + 1] +
                   bfwd[breakpt - 1]) - y[:breakpt + 1]
        delsbck = (mbck[breakpt - 1] * x[breakpt:] +
                   bbck[breakpt - 1]) - y[breakpt:]

        error_curve[breakpt] = np.sum(np.abs(delsfwd)) + \
            np.sum(np.abs(delsbck))

    # find location of the min of the error curve
    loc = np.argmin(error_curve[1:-1]) + 1
    knee_point = x[loc]
    return knee_point
