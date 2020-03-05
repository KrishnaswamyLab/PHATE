import numpy as np
import phate
import graphtools
import scipy
import warnings
from parameterized import parameterized


@parameterized(
    [
        (n_points, symmetric, jackson, cumulative)
        for n_points in [50, 100]
        for symmetric in [True, False]
        for jackson in [True, False]
        for cumulative in [True, False]
    ]
)
def test_no_eigmin_eigmax(n_points, symmetric, jackson, cumulative):
    data = np.random.uniform(0, 1, (100, 100))
    G = graphtools.Graph(data)
    A = G.diff_aff if symmetric else G.diff_op
    eigs = np.real(scipy.linalg.eig(A.toarray(), left=False, right=False))
    if not symmetric:
        warnings.filterwarnings(
            "ignore", "Casting complex values to real discards the imaginary part"
        )
    sample_points, density_est = phate.eig.estimate_eigenvalue_density(
        A,
        n_points=n_points,
        cumulative=cumulative,
        jackson=jackson,
        symmetric=symmetric,
    )
    warnings.resetwarnings()
    density = np.array([np.sum(eigs <= s) for s in sample_points])
    if not cumulative:
        density[1:] -= density[:-1].copy()
    np.testing.assert_allclose(density, density_est, atol=4, rtol=0.01)
    assert np.mean(density - np.round(density_est) <= 1) > 0.5, np.mean(
        density - np.round(density_est) <= 1
    )
    assert np.mean(density - np.round(density_est) <= 2) > 0.95, np.mean(
        density - np.round(density_est) <= 2
    )


@parameterized(
    [
        (n_points, symmetric, jackson, cumulative)
        for n_points in [50, 100]
        for symmetric in [True, False]
        for jackson in [True, False]
        for cumulative in [True, False]
    ]
)
def test_eigmin_eigmax(n_points, symmetric, jackson, cumulative):
    data = np.random.uniform(0, 1, (100, 100))
    G = graphtools.Graph(data)
    A = G.diff_aff if symmetric else G.diff_op
    eigs = np.real(scipy.linalg.eig(A.toarray(), left=False, right=False))
    sample_points = np.linspace(np.min(eigs), np.max(eigs), n_points)
    density = np.array([np.sum(eigs <= s) for s in sample_points])
    if not cumulative:
        density[1:] -= density[:-1].copy()
    if not symmetric:
        warnings.filterwarnings(
            "ignore", "Casting complex values to real discards the imaginary part"
        )
    sample_points_est, density_est = phate.eig.estimate_eigenvalue_density(
        A,
        n_points=n_points,
        cumulative=cumulative,
        jackson=jackson,
        symmetric=symmetric,
        eigmin=np.min(eigs),
        eigmax=np.max(eigs),
    )
    warnings.resetwarnings()
    np.testing.assert_allclose(sample_points, sample_points_est, atol=1e-8)
    np.testing.assert_allclose(density, density_est, atol=3, rtol=0.01)
    assert np.mean(density - np.round(density_est) <= 1) > 0.5, np.mean(
        density - np.round(density_est) <= 1
    )
    assert np.mean(density - np.round(density_est) <= 2) > 0.95, np.mean(
        density - np.round(density_est) <= 2
    )


@parameterized(
    [
        (10, 1.6, False, 42),
        (50, 2.2, False, 42),
        (100, 61, False, 42),
        (10, 2.1, True, 42),
        (50, 1.7, True, 42),
        (100, 1.62, True, 42),
    ]
)
def test_order(
    order, atol, jackson, seed, n_points=100, cumulative=False, symmetric=True
):
    np.random.seed(seed)
    data = np.random.uniform(0, 1, (100, 100))
    G = graphtools.Graph(data)
    A = G.diff_aff if symmetric else G.diff_op
    eigs = np.real(scipy.linalg.eig(A.toarray(), left=False, right=False))
    if not symmetric:
        warnings.filterwarnings(
            "ignore", "Casting complex values to real discards the imaginary part"
        )
    sample_points, density_est = phate.eig.estimate_eigenvalue_density(
        A,
        n_points=n_points,
        cumulative=cumulative,
        jackson=jackson,
        symmetric=symmetric,
        order=order,
    )
    warnings.resetwarnings()
    density = np.array([np.sum(eigs <= s) for s in sample_points])
    if not cumulative:
        density[1:] -= density[:-1].copy()
    np.testing.assert_allclose(density, density_est, atol=atol, rtol=0.01)
    assert not np.allclose(density, density_est, atol=atol * 0.9, rtol=0.01)
