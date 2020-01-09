import numpy as np
from scipy import sparse


def _as_callable_sparse_operator(A):
    if callable(A):
        A_fun = A
    else:
        if not sparse.issparse(A):
            A = sparse.csr_matrix(A)
        A_fun = lambda x: A * x
    return A_fun


def chebyshev_estimator(A, n_probes=100, order=50, kind=1):
    """Compute Chebyshev moments via a stochastic estimator
    
    Computes a column vector of Chebyshev moments of the form c(k) = tr(T_k(A)) 
    for k = 0 to N-1. This routine does no scaling; the spectrum of A should 
    already lie in [-1,1]. The traces are computed via a stochastic estimator 
    with n_probes probes

    Parameters
    ----------
    A: array-like, shape=[n, n]
        Matrix or function apply matrix (for RHS matrix multiplication)
    n_probes: int or array-like, optional (default: 100)
        Number of probe vectors with which we compute moments,
        or an array of probes of shape [n_probes, n]
    order: int, optinal (default: 50)
        Number of moments to compute
    kind: int, optional (default: 1)
        first or second kind Chebyshev functions

    Returns
    -------
    chebyshev_mean: array-like, shape=[order, 1]
        estimated chebyshev moments
    chebyshev_std: array-like, shape=[order, 1]
        standard deviation of the moment estimator (std/sqrt(n_probes))
    """
    n = A.shape[0]

    if order < 2:
        order = 2

    # Set up random probe vectors (allowed to be passed in)
    if not isinstance(n_probes, int):
        probes = n_probes
        n_probes = probes.shape[1]
    else:
        probes = np.sign(np.random.randn(n, n_probes))

    # Estimate moments for each probe vector
    probes_chebyshev = chebyshev_moments(A, probes, order, kind)
    chebyshev_mean = np.mean(probes_chebyshev, 1)
    chebyshev_std = np.std(probes_chebyshev, 1, ddof=1) / np.sqrt(n_probes)

    chebyshev_mean = chebyshev_mean.reshape([order, -1])
    chebyshev_std = chebyshev_std.reshape([order, -1])
    return chebyshev_mean, chebyshev_std


def chebyshev_moments(A, V, order=50, kind=1):
    """Compute Chebyshev moments from probe vectors
    
    Computes a column vector of Chebyshev moments of the form c(k) = v'*T_k(A)*v 
    for k = 0 to N-1. This routine does no scaling; the spectrum of A should 
    already lie in [-1,1]

    Parameters
    ----------
    A: array-like, shape=[n, n]
        Matrix or function apply matrix (for RHS matrix multiplication)
    V: array-like, shape=[n_probes, n]
        Probe vectors with which we compute moments
    order: int, optinal (default: 50)
        Number of moments to compute
    kind: int, optional (default: 1)
        first or second kind Chebyshev functions

    Output
    ------
    cheby_moments : array-like, shape=[order, n_probes]
        Chebyshev moments w.r.t. each of the probe vectors
    """

    if order < 2:
        order = 2

    if not isinstance(V, np.ndarray):
        V = V.toarray()

    A = _as_callable_sparse_operator(A)

    n, p = V.shape
    cheby_moments = np.zeros((order, p))

    # Run three-term recurrence to compute moments
    TVp = V  # x
    TVk = kind * A(V)  # Ax
    cheby_moments[0] = np.sum(V * TVp, 0)  # xx
    cheby_moments[1] = np.sum(V * TVk, 0)  # xAx
    for i in range(2, order):
        TV = 2 * A(TVk) - TVp  # A*2T_1 - T_o
        TVp = TVk
        TVk = TV
        cheby_moments[i] = sum(V * TVk, 0)
    return cheby_moments


def chebyshev_cumulative_density(
    cheby_moments, n_points=60, jackson=False, sample_points=None, scale=1, intercept=0
):
    """Compute density of eigenvalues from Chebyshev moments

    Given a (filtered) set of first-kind Chebyshev moments, compute the integral
    of the density:

        $$int_0^s (2/pi)*sqrt(1-x^2)*( c(0)/2+sum_{n=1}^{N-1}c_nT_n(x) )$$

    Parameters
    ----------
    cheby_moments : array-like, shape=[order, 1]
        Vector of Chebyshev moments
    n_points : int, optional (default: 60)
        Number of points to compute along the integral
    sample_points : list-like or None, optional (default: None)
        Input sampling mesh in the original coordinates, shape=[n_points].
        If None, defaults to linear interpolation from -1 to 1.
    scale : float, optional (default: 1)
        Multiplier to apply to the default interval [-1, 1]
        to obtain the original coordinates
    intercept: float, optional (default: 0)
        Offset to add to the default interval [-1, 1]
        to obtain the original coordinates

    Returns
    -------
    sample_points : list-like, shape=[n_points]
        Input sampling mesh in the original coordinates
    density : list-like, shape=[n_points]
        Estimated cumulative density up to each sample_points point
    """
    # Parse arguments
    if sample_points is not None:
        sample_points_scaled = (sample_points - intercept) / scale
    else:
        sample_points_scaled = np.linspace(-1 + 1e-8, 1 - 1e-8, n_points)

    if sample_points is None:
        sample_points = scale * sample_points_scaled + intercept

    order = len(cheby_moments)

    # First compute jackson coeffs
    if jackson:
        jackson_coef = np.zeros(order)
        alpha = np.pi / (order + 2)
        for i in range(len(jackson_coef)):
            jackson_coef[i] = (1 / np.sin(alpha)) * (
                (1 - i / (order + 2)) * np.sin(alpha) * np.cos(i * alpha)
                + (1 / (order + 2)) * np.cos(alpha) * np.sin(i * alpha)
            )
    else:
        jackson_coef = np.ones(order)

    arccos_points = np.arccos(sample_points_scaled)
    density = jackson_coef[0] * cheby_moments[0] * (arccos_points - np.pi) / 2
    for idx in np.arange(1, order):
        density += (
            jackson_coef[idx] * cheby_moments[idx] * np.sin(idx * arccos_points) / idx
        )

    density *= -2 / np.pi

    return sample_points, density


def estimate_eigenvalue_density(
    A,
    n_points=60,
    order=50,
    eigmin=None,
    eigmax=None,
    jackson=False,
    symmetric=None,
    cumulative=False,
):
    """Compute density of eigenvalues of a square matrix
    
    Uses Chebyshev polynomials to approximate the spectral density (or density of states).
    
    Parameters
    ----------
    A : array-like, shape=[n, n]
        Input matrix
    n_points : int, optional (default: 60)
        Number of points to compute along the integral
    order: int, optinal (default: 50)
        Number of moments to compute
    eigmin : float or None, optional (default: None)
        Minimum eigenvalue of A.
        If None, it is approximated using the Lanczos method.
    eigmax : float or None, optional (default: None)
        Maximum eigenvalue of A.
        If None, it is approximated using the Lanczos method.
    jackson : bool, optional (default: False)
        If True, use Jackson-Chebyshev coefficients
    symmetric : bool, optional (default: None)
        Flag to mark whether or not A == A.T.
        If None, it is computed.
    cumulative : bool, optional (default: False)
        If True, return the cumulative density instead of the density

    Returns
    -------
    eigs : list-like, shape=[n_points]
        Eigenvalues
    density : list-like, shape=[n_points]
        Density at each value in eigs
    """
    A = sparse.csr_matrix(A)
    if symmetric is None:
        symmetric = np.allclose((A - A.T).data, 0)
    if symmetric:
        eigs = sparse.linalg.eigsh
        which = "A"
    else:
        eigs = sparse.linalg.eigs
        which = "R"
    if eigmin is None:
        eigmin = eigs(
            A,
            k=1,
            return_eigenvectors=False,
            which="S{}".format(which),
            tol=5e-3,
            ncv=min(order, 10),
        )[0]
        eigmin = 0.99 * eigmin
    if eigmax is None:
        eigmax = eigs(
            A,
            k=1,
            return_eigenvectors=False,
            which="L{}".format(which),
            tol=5e-3,
            ncv=min(order, 10),
        )[0]
        eigmax = 1.01 * eigmax
    # rescale to -1, 1
    A = (
        2
        * (A - (eigmin + 1 / 2 * (eigmax - eigmin)) * sparse.eye(A.shape[0]).tocsr())
        / (eigmax - eigmin)
    )
    cheby_moments = chebyshev_estimator(A, order=order)[0]
    eigs, density = chebyshev_cumulative_density(
        cheby_moments, n_points, jackson=jackson
    )
    eigs = (eigs + 1) / 2 * (eigmax - eigmin) + eigmin
    if not cumulative:
        density[1:] -= density[:-1].copy()
    return eigs, density
