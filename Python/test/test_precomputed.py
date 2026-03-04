#!/usr/bin/env python
"""
Comprehensive test suite for PHATE precomputed distance/affinity inputs
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import phate
import pytest
from scipy.spatial.distance import pdist, squareform
from scipy import sparse


#####################################################
# Test fixtures
#####################################################


def create_random_precomputed_data(seed=42, n_samples=80, n_features=12):
    """Create random data and a valid precomputed Euclidean distance matrix"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    D = squareform(pdist(X, metric="euclidean"))
    return X, D


def create_precomputed_kwargs():
    """Common PHATE params used for precomputed regression tests"""
    return dict(knn=7, t=10, n_jobs=-1, verbose=False, random_state=42)


#####################################################
# Core behavior tests
#####################################################


def test_precomputed_distance_alias_random_matrix():
    """Regression test for the reported precomputed-distance error"""
    print("\n" + "=" * 70)
    print("TEST 1: precomputed alias with random valid distance matrix")
    print("=" * 70)

    _, D = create_random_precomputed_data()

    # Sanity checks for generated matrix
    assert np.allclose(D, D.T), "Distance matrix must be symmetric"
    assert np.allclose(np.diag(D), 0.0), "Distance matrix diagonal must be zero"
    print("✓ Generated valid precomputed distance matrix")

    phate_op = phate.PHATE(knn_dist="precomputed", **create_precomputed_kwargs())
    emb = phate_op.fit_transform(D)

    assert emb.shape == (D.shape[0], 2), f"Unexpected embedding shape: {emb.shape}"
    assert np.all(np.isfinite(emb)), "Embedding contains non-finite values"
    print("✓ Reported configuration runs successfully")

    print("✓ Test 1 PASSED\n")


def test_precomputed_alias_matches_explicit_distance():
    """'precomputed' should match explicit 'precomputed_distance' for distances"""
    print("=" * 70)
    print("TEST 2: precomputed alias matches precomputed_distance")
    print("=" * 70)

    _, D = create_random_precomputed_data(seed=123)
    kwargs = create_precomputed_kwargs()

    emb_alias = phate.PHATE(knn_dist="precomputed", **kwargs).fit_transform(D)
    emb_distance = phate.PHATE(knn_dist="precomputed_distance", **kwargs).fit_transform(
        D
    )

    assert np.allclose(
        emb_alias, emb_distance, atol=1e-10
    ), "Alias and explicit distance mode diverged"
    print("✓ Alias behavior is consistent with explicit precomputed_distance")

    print("✓ Test 2 PASSED\n")


def test_precomputed_alias_matches_explicit_affinity():
    """'precomputed' should match explicit 'precomputed_affinity' for affinities"""
    print("=" * 70)
    print("TEST 3: precomputed alias matches precomputed_affinity")
    print("=" * 70)

    X, _ = create_random_precomputed_data(seed=321)
    kwargs = create_precomputed_kwargs()

    # Build a valid affinity matrix from a fitted PHATE graph.
    base = phate.PHATE(knn_dist="euclidean", **kwargs)
    base.fit_transform(X)
    K = base.graph.kernel

    emb_alias = phate.PHATE(knn_dist="precomputed", **kwargs).fit_transform(K)
    emb_affinity = phate.PHATE(knn_dist="precomputed_affinity", **kwargs).fit_transform(
        K
    )

    assert np.allclose(
        emb_alias, emb_affinity, atol=1e-10
    ), "Alias and explicit affinity mode diverged"
    print("✓ Alias behavior is consistent with explicit precomputed_affinity")

    print("✓ Test 3 PASSED\n")


def test_precomputed_accepts_sparse_coo_distance():
    """Corner case: coo_matrix input should be handled in precomputed mode"""
    print("=" * 70)
    print("TEST 4: sparse COO precomputed distance input")
    print("=" * 70)

    _, D = create_random_precomputed_data(seed=999, n_samples=50, n_features=10)
    D_coo = sparse.coo_matrix(D)

    emb = phate.PHATE(knn_dist="precomputed", **create_precomputed_kwargs()).fit_transform(
        D_coo
    )
    assert emb.shape == (D.shape[0], 2)
    assert np.all(np.isfinite(emb))
    print("✓ Sparse COO precomputed distance matrix works")

    print("✓ Test 4 PASSED\n")


#####################################################
# Error handling tests
#####################################################


def test_precomputed_rejects_non_square_matrix():
    """Precomputed distance input must be square"""
    print("=" * 70)
    print("TEST 5: reject non-square precomputed matrix")
    print("=" * 70)

    _, D = create_random_precomputed_data(seed=2024, n_samples=40, n_features=8)
    D_non_square = D[:, :-1]

    with pytest.raises(ValueError, match="square matrix"):
        phate.PHATE(knn_dist="precomputed", **create_precomputed_kwargs()).fit_transform(
            D_non_square
        )
    print("✓ Non-square precomputed matrix is rejected")

    print("✓ Test 5 PASSED\n")


def test_precomputed_rejects_negative_distances():
    """Distance inputs with negative values should fail fast"""
    print("=" * 70)
    print("TEST 6: reject negative precomputed distances")
    print("=" * 70)

    _, D = create_random_precomputed_data(seed=2025, n_samples=40, n_features=8)
    D_negative = D.copy()
    D_negative[0, 1] = -1.0
    D_negative[1, 0] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        phate.PHATE(knn_dist="precomputed", **create_precomputed_kwargs()).fit_transform(
            D_negative
        )
    print("✓ Negative precomputed distances are rejected")

    print("✓ Test 6 PASSED\n")


def test_precomputed_cannot_transform_new_data():
    """Precomputed mode should not allow out-of-sample transform"""
    print("=" * 70)
    print("TEST 7: reject transform(X_new) after precomputed fit")
    print("=" * 70)

    rng = np.random.default_rng(7)
    X = rng.normal(size=(35, 6))
    D = squareform(pdist(X, metric="euclidean"))
    X_new = rng.normal(size=(5, 6))

    phate_op = phate.PHATE(knn_dist="precomputed", **create_precomputed_kwargs())
    phate_op.fit(D)

    with pytest.raises(ValueError, match="Cannot transform additional data"):
        phate_op.transform(X_new)
    print("✓ Out-of-sample transform is correctly blocked in precomputed mode")

    print("✓ Test 7 PASSED\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
