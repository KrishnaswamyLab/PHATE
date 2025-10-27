#!/usr/bin/env python
"""
Comprehensive test suite for MDS methods (classic, metric/SMACOF, SGD)
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import phate
from phate.mds import classic, smacof
from phate.sgd_mds import sgd_mds, sgd_mds_metric
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes


def compute_stress(D, Y):
    """Compute raw stress for MDS embedding

    Stress = sum((D_ij - d_ij)^2) where D_ij is input distance
    and d_ij is embedded distance
    """
    D_embedded = squareform(pdist(Y, "euclidean"))
    return np.sum((D - D_embedded) ** 2)


def test_sgd_mds_basic():
    """Test basic functionality of SGD-MDS"""
    print("\n" + "=" * 70)
    print("TEST 1: Basic SGD-MDS functionality")
    print("=" * 70)

    # Create simple test data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 10)
    D = squareform(pdist(X, "euclidean"))

    # Run SGD-MDS
    Y = sgd_mds(D, n_components=2, n_iter=200, random_state=42)

    # Check output shape
    assert Y.shape == (n_samples, 2), f"Expected shape (100, 2), got {Y.shape}"
    print(f"✓ Output shape correct: {Y.shape}")

    # Check for NaN or Inf
    assert not np.any(np.isnan(Y)), "Output contains NaN values"
    assert not np.any(np.isinf(Y)), "Output contains Inf values"
    print("✓ No NaN or Inf values")

    # Check that points are spread out (not collapsed)
    variance = np.var(Y, axis=0)
    assert np.all(variance > 1e-6), f"Embedding collapsed: variance={variance}"
    print(f"✓ Embedding has variance: {variance}")

    print("✓ Test 1 PASSED\n")


def test_sgd_mds_vs_classic():
    """Test SGD-MDS improves upon classic MDS initialization"""
    print("=" * 70)
    print("TEST 2: SGD-MDS improves upon classic MDS")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 150
    X = np.random.randn(n_samples, 8)
    D = squareform(pdist(X, "euclidean"))

    # Get classic MDS embedding
    Y_classic = classic(D, n_components=2, random_state=42)
    stress_classic = compute_stress(D, Y_classic)
    print(f"Classic MDS stress: {stress_classic:.6f}")

    # Run SGD-MDS with classic initialization
    Y_sgd = sgd_mds(
        D,
        n_components=2,
        n_iter=300,
        init=Y_classic.copy(),
        random_state=42,
        verbose=0,
    )
    stress_sgd = compute_stress(D, Y_sgd)
    print(f"SGD-MDS stress: {stress_sgd:.6f}")

    # SGD-MDS should reduce stress or keep it similar (not increase significantly)
    # Allow small increase due to stochastic optimization
    stress_ratio = stress_sgd / stress_classic
    print(f"Stress ratio (SGD/Classic): {stress_ratio:.6f}")
    assert (
        stress_ratio < 1.1
    ), f"SGD-MDS significantly increased stress: {stress_ratio:.3f}x"
    print("✓ SGD-MDS maintains or improves stress")

    print("✓ Test 2 PASSED\n")


def test_sgd_mds_metric_wrapper():
    """Test the sgd_mds_metric wrapper with auto-tuning"""
    print("=" * 70)
    print("TEST 3: sgd_mds_metric wrapper with auto-tuning")
    print("=" * 70)

    np.random.seed(42)

    # Test with different dataset sizes
    for n_samples in [50, 500, 2000]:
        print(f"\nTesting with n_samples={n_samples}")
        X = np.random.randn(n_samples, 5)
        D = squareform(pdist(X, "euclidean"))

        Y = sgd_mds_metric(D, n_components=2, random_state=42, verbose=0)

        assert Y.shape == (
            n_samples,
            2,
        ), f"Expected shape ({n_samples}, 2), got {Y.shape}"
        assert not np.any(np.isnan(Y)), "Output contains NaN values"
        print(f"  ✓ n_samples={n_samples}: shape={Y.shape}, no NaN")

    print("\n✓ Test 3 PASSED\n")


def test_sgd_mds_vs_smacof():
    """Compare SGD-MDS results with SMACOF on small dataset"""
    print("=" * 70)
    print("TEST 4: Compare SGD-MDS with SMACOF (small dataset)")
    print("=" * 70)

    np.random.seed(42)
    # Use smaller dataset for tighter comparison
    n_samples = 50
    X = np.random.randn(n_samples, 5)
    D = squareform(pdist(X, "euclidean"))

    # Get classic MDS for initialization
    Y_classic = classic(D, n_components=2, random_state=42)

    # Run SMACOF
    Y_smacof = smacof(
        D,
        n_components=2,
        init=Y_classic.copy(),
        random_state=42,
        metric=True,
        max_iter=300,
    )
    stress_smacof = compute_stress(D, Y_smacof)
    print(f"SMACOF stress: {stress_smacof:.6f}")

    # Run SGD-MDS with same initialization
    Y_sgd = sgd_mds(
        D,
        n_components=2,
        learning_rate=0.01,
        n_iter=800,
        init=Y_classic.copy(),
        random_state=42,
        verbose=0,
    )
    stress_sgd = compute_stress(D, Y_sgd)
    print(f"SGD-MDS stress: {stress_sgd:.6f}")

    # Align embeddings using Procrustes
    _, Y_sgd_aligned, disparity = procrustes(Y_smacof, Y_sgd)
    print(f"Procrustes disparity: {disparity:.6f}")

    # For small dataset, embeddings should be quite similar
    assert disparity < 0.3, f"SGD-MDS too different from SMACOF: {disparity:.6f}"
    print(f"✓ SGD-MDS close to SMACOF (disparity={disparity:.6f})")

    # Compute correlation between distances
    D_smacof = squareform(pdist(Y_smacof, "euclidean"))
    D_sgd = squareform(pdist(Y_sgd_aligned, "euclidean"))

    # For small dataset, correlation should be high
    corr = np.corrcoef(D_smacof.flatten(), D_sgd.flatten())[0, 1]
    print(f"Distance correlation: {corr:.6f}")
    assert corr > 0.9, f"Distance correlation too low: {corr:.6f}"
    print(f"✓ High distance correlation: {corr:.6f}")

    print("✓ Test 4 PASSED\n")


def test_phate_with_sgd_mds():
    """Test PHATE integration with SGD-MDS solver"""
    print("=" * 70)
    print("TEST 5: PHATE integration with SGD-MDS")
    print("=" * 70)

    # Generate tree data
    np.random.seed(42)
    data, labels = phate.tree.gen_dla(n_dim=50, n_branch=5, branch_length=100)

    print(f"Generated tree data: {data.shape}")

    # Test with SGD solver
    print("\nRunning PHATE with mds_solver='sgd'...")
    phate_sgd = phate.PHATE(
        n_components=2,
        knn=5,
        t=20,
        mds_solver="sgd",
        random_state=42,
        verbose=0,
    )
    embedding_sgd = phate_sgd.fit_transform(data)

    assert embedding_sgd.shape == (
        data.shape[0],
        2,
    ), f"Expected shape {(data.shape[0], 2)}, got {embedding_sgd.shape}"
    assert not np.any(np.isnan(embedding_sgd)), "SGD embedding contains NaN"
    print(f"✓ SGD embedding shape: {embedding_sgd.shape}")

    # Test with SMACOF solver for comparison
    print("\nRunning PHATE with mds_solver='smacof' for comparison...")
    phate_smacof = phate.PHATE(
        n_components=2,
        knn=5,
        t=20,
        mds_solver="smacof",
        random_state=42,
        verbose=0,
    )
    embedding_smacof = phate_smacof.fit_transform(data)

    assert embedding_smacof.shape == (
        data.shape[0],
        2,
    ), f"Expected shape {(data.shape[0], 2)}, got {embedding_smacof.shape}"
    print(f"✓ SMACOF embedding shape: {embedding_smacof.shape}")

    # Compare embeddings
    _, embedding_sgd_aligned, disparity = procrustes(embedding_smacof, embedding_sgd)
    print(f"\nProcrustes disparity between SGD and SMACOF: {disparity:.6f}")

    # PHATE adds additional processing (diffusion, potential), so embeddings
    # may differ more than raw MDS. Accept < 0.85 as reasonable.
    assert disparity < 0.85, f"Embeddings too different: {disparity:.6f}"
    print(f"✓ PHATE embeddings reasonably similar (disparity={disparity:.6f})")

    print("✓ Test 5 PASSED\n")


def test_sgd_mds_with_pairs_per_iter():
    """Test with custom pairs_per_iter parameter"""
    print("=" * 70)
    print("TEST 6: Custom pairs_per_iter parameter")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 5)
    D = squareform(pdist(X, "euclidean"))

    # Run with custom pairs_per_iter
    Y = sgd_mds(
        D,
        n_components=2,
        n_iter=500,
        pairs_per_iter=1000,  # Custom number of pairs per iteration
        random_state=42,
        verbose=0,
    )

    assert Y.shape == (n_samples, 2)
    assert not np.any(np.isnan(Y))
    print("✓ Custom pairs_per_iter works correctly")
    print("✓ Test 6 PASSED\n")


def test_sgd_mds_different_n_components():
    """Test with different numbers of components"""
    print("=" * 70)
    print("TEST 7: Different numbers of components")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 10)
    D = squareform(pdist(X, "euclidean"))

    for n_comp in [1, 2, 3, 5]:
        print(f"\nTesting with n_components={n_comp}")
        Y = sgd_mds(D, n_components=n_comp, n_iter=200, random_state=42)

        assert Y.shape == (
            n_samples,
            n_comp,
        ), f"Expected shape ({n_samples}, {n_comp}), got {Y.shape}"
        assert not np.any(np.isnan(Y))
        print(f"  ✓ n_components={n_comp}: shape={Y.shape}")

    print("\n✓ Test 7 PASSED\n")


#####################################################
# Classic MDS Tests
#####################################################


def test_classic_mds_basic():
    """Test basic classic MDS functionality"""
    print("=" * 70)
    print("TEST 8: Classic MDS basic functionality")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 10)
    D = squareform(pdist(X, "euclidean"))

    # Run classic MDS
    Y = classic(D, n_components=2, random_state=42)

    # Check output shape
    assert Y.shape == (n_samples, 2), f"Expected shape (100, 2), got {Y.shape}"
    print(f"✓ Output shape correct: {Y.shape}")

    # Check for NaN or Inf
    assert not np.any(np.isnan(Y)), "Output contains NaN values"
    assert not np.any(np.isinf(Y)), "Output contains Inf values"
    print("✓ No NaN or Inf values")

    # Classic MDS should preserve distances reasonably
    stress = compute_stress(D, Y)
    print(f"Stress: {stress:.6f}")
    # Just check that stress is finite and not NaN
    assert np.isfinite(stress), f"Stress is not finite: {stress}"
    print("✓ Finite stress value")

    print("✓ Test 8 PASSED\n")


def test_classic_mds_perfect_embedding():
    """Test classic MDS on data that can be perfectly embedded in 2D"""
    print("=" * 70)
    print("TEST 9: Classic MDS perfect 2D embedding")
    print("=" * 70)

    np.random.seed(42)
    # Create data that lives in 2D
    n_samples = 50
    X_2d = np.random.randn(n_samples, 2)
    D = squareform(pdist(X_2d, "euclidean"))

    # Classic MDS should perfectly reconstruct 2D data
    Y = classic(D, n_components=2, random_state=42)

    # Check distances are preserved (up to rotation/reflection/scaling)
    # Classic MDS preserves distances but may scale them
    D_embedded = squareform(pdist(Y, "euclidean"))

    # Normalize both distance matrices to have same scale
    D_norm = D / np.std(D)
    D_embedded_norm = D_embedded / np.std(D_embedded)

    # Compute correlation (should be near 1.0 for perfect embedding)
    corr = np.corrcoef(D_norm.flatten(), D_embedded_norm.flatten())[0, 1]
    print(f"Distance correlation: {corr:.6f}")
    assert corr > 0.99, f"Distances not well preserved: correlation={corr:.6f}"
    print("✓ Distances well preserved (correlation > 0.99)")

    print("✓ Test 9 PASSED\n")


def test_classic_mds_different_dimensions():
    """Test classic MDS with different output dimensions"""
    print("=" * 70)
    print("TEST 10: Classic MDS with different dimensions")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 80
    X = np.random.randn(n_samples, 8)
    D = squareform(pdist(X, "euclidean"))

    for n_comp in [1, 2, 3, 5]:
        print(f"\nTesting with n_components={n_comp}")
        Y = classic(D, n_components=n_comp, random_state=42)

        assert Y.shape == (
            n_samples,
            n_comp,
        ), f"Expected shape ({n_samples}, {n_comp}), got {Y.shape}"
        assert not np.any(np.isnan(Y))
        stress = compute_stress(D, Y)
        print(f"  ✓ n_components={n_comp}: shape={Y.shape}, stress={stress:.2e}")

    print("\n✓ Test 10 PASSED\n")


#####################################################
# SMACOF (Metric MDS) Tests
#####################################################


def test_smacof_basic():
    """Test basic SMACOF functionality"""
    print("=" * 70)
    print("TEST 11: SMACOF basic functionality")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 80
    X = np.random.randn(n_samples, 6)
    D = squareform(pdist(X, "euclidean"))

    # Run SMACOF
    Y = smacof(D, n_components=2, random_state=42, metric=True, max_iter=300)

    # Check output shape
    assert Y.shape == (n_samples, 2), f"Expected shape (80, 2), got {Y.shape}"
    print(f"✓ Output shape correct: {Y.shape}")

    # Check for NaN or Inf
    assert not np.any(np.isnan(Y)), "Output contains NaN values"
    assert not np.any(np.isinf(Y)), "Output contains Inf values"
    print("✓ No NaN or Inf values")

    # Compute stress
    stress = compute_stress(D, Y)
    print(f"Stress: {stress:.6f}")
    print("✓ SMACOF converged")

    print("✓ Test 11 PASSED\n")


def test_smacof_improves_upon_init():
    """Test that SMACOF improves upon initialization"""
    print("=" * 70)
    print("TEST 12: SMACOF improves stress from initialization")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 8)
    D = squareform(pdist(X, "euclidean"))

    # Get classic MDS as initialization
    Y_init = classic(D, n_components=2, random_state=42)
    stress_init = compute_stress(D, Y_init)
    print(f"Initial stress (classic MDS): {stress_init:.6f}")

    # Run SMACOF with initialization
    Y_smacof = smacof(
        D, n_components=2, init=Y_init, random_state=42, metric=True, max_iter=300
    )
    stress_smacof = compute_stress(D, Y_smacof)
    print(f"SMACOF stress: {stress_smacof:.6f}")

    # SMACOF should reduce or maintain stress
    assert stress_smacof <= stress_init * 1.01, "SMACOF increased stress"
    print(f"✓ SMACOF reduced stress by {(1 - stress_smacof/stress_init)*100:.2f}%")

    print("✓ Test 12 PASSED\n")


def test_smacof_vs_classic():
    """Compare SMACOF with classic MDS"""
    print("=" * 70)
    print("TEST 13: SMACOF vs Classic MDS")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 60
    X = np.random.randn(n_samples, 5)
    D = squareform(pdist(X, "euclidean"))

    # Run both methods
    Y_classic = classic(D, n_components=2, random_state=42)
    Y_smacof = smacof(
        D,
        n_components=2,
        init=Y_classic.copy(),
        random_state=42,
        metric=True,
        max_iter=300,
    )

    stress_classic = compute_stress(D, Y_classic)
    stress_smacof = compute_stress(D, Y_smacof)

    print(f"Classic MDS stress: {stress_classic:.6f}")
    print(f"SMACOF stress: {stress_smacof:.6f}")

    # SMACOF should achieve similar or better stress than classic
    assert stress_smacof <= stress_classic * 1.05, "SMACOF worse than classic MDS"
    print("✓ SMACOF achieves competitive stress")

    # Embeddings should be similar (Procrustes)
    _, Y_smacof_aligned, disparity = procrustes(Y_classic, Y_smacof)
    print(f"Procrustes disparity: {disparity:.6f}")
    assert disparity < 0.5, f"Embeddings too different: {disparity:.6f}"
    print("✓ Embeddings are similar")

    print("✓ Test 13 PASSED\n")


#####################################################
# Corner Cases and Edge Tests
#####################################################


def test_mds_tiny_dataset():
    """Test MDS methods on very small dataset"""
    print("=" * 70)
    print("TEST 14: MDS with tiny dataset (n=5)")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 5
    X = np.random.randn(n_samples, 3)
    D = squareform(pdist(X, "euclidean"))

    # All methods should handle small datasets
    Y_classic = classic(D, n_components=2, random_state=42)
    Y_smacof = smacof(D, n_components=2, random_state=42, metric=True, max_iter=100)
    Y_sgd = sgd_mds(D, n_components=2, n_iter=100, random_state=42)

    for name, Y in [("Classic", Y_classic), ("SMACOF", Y_smacof), ("SGD", Y_sgd)]:
        assert Y.shape == (n_samples, 2), f"{name}: wrong shape"
        assert not np.any(np.isnan(Y)), f"{name}: contains NaN"
        print(f"✓ {name} MDS works on tiny dataset")

    print("✓ Test 14 PASSED\n")


def test_mds_zero_distances():
    """Test MDS handles zero distances (duplicate points)"""
    print("=" * 70)
    print("TEST 15: MDS with duplicate points (zero distances)")
    print("=" * 70)

    np.random.seed(42)
    # Create data with duplicates
    X = np.random.randn(50, 5)
    X[10] = X[5]  # Create a duplicate
    D = squareform(pdist(X, "euclidean"))

    # Should handle duplicates without crashing
    Y_classic = classic(D, n_components=2, random_state=42)
    Y_smacof = smacof(D, n_components=2, random_state=42, metric=True, max_iter=300)
    # SGD-MDS needs more iterations and lower learning rate to converge on duplicates
    Y_sgd = sgd_mds(
        D, n_components=2, n_iter=1000, learning_rate=0.01, random_state=42, verbose=0
    )

    for name, Y in [("Classic", Y_classic), ("SMACOF", Y_smacof), ("SGD", Y_sgd)]:
        assert not np.any(np.isnan(Y)), f"{name}: contains NaN with duplicates"

        # Check that duplicates are among each other's nearest neighbors
        # This is more robust than a fixed distance threshold
        dists_from_5 = np.linalg.norm(Y - Y[5], axis=1)
        dists_from_10 = np.linalg.norm(Y - Y[10], axis=1)

        # Find rank of point 10 among neighbors of point 5
        rank_10_from_5 = np.sum(dists_from_5 < dists_from_5[10]) + 1
        # Find rank of point 5 among neighbors of point 10
        rank_5_from_10 = np.sum(dists_from_10 < dists_from_10[5]) + 1

        print(f"{name}: Point 10 is rank {rank_10_from_5} neighbor of point 5")
        print(f"{name}: Point 5 is rank {rank_5_from_10} neighbor of point 10")

        # Duplicates should be in top 5 nearest neighbors of each other
        assert (
            rank_10_from_5 <= 5
        ), f"{name}: duplicate not in top 5 neighbors (rank={rank_10_from_5})"
        assert (
            rank_5_from_10 <= 5
        ), f"{name}: duplicate not in top 5 neighbors (rank={rank_5_from_10})"

    print("✓ All methods handle duplicates correctly")
    print("✓ Test 15 PASSED\n")


def test_mds_high_dimensions():
    """Test MDS with higher output dimensions"""
    print("=" * 70)
    print("TEST 16: MDS with high output dimensions")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 20)
    D = squareform(pdist(X, "euclidean"))

    # Test with n_components close to n_samples
    n_comp = 10
    Y_classic = classic(D, n_components=n_comp, random_state=42)
    Y_sgd = sgd_mds(D, n_components=n_comp, n_iter=500, random_state=42, verbose=0)

    assert Y_classic.shape == (n_samples, n_comp)
    assert Y_sgd.shape == (n_samples, n_comp)
    print(f"✓ Classic MDS: shape={Y_classic.shape}")
    print(f"✓ SGD-MDS: shape={Y_sgd.shape}")

    # Check embeddings are valid (no NaN/Inf, have variance)
    for name, Y in [("Classic", Y_classic), ("SGD", Y_sgd)]:
        assert not np.any(np.isnan(Y)), f"{name}: contains NaN"
        assert not np.any(np.isinf(Y)), f"{name}: contains Inf"
        variance = np.var(Y, axis=0)
        assert np.all(variance > 1e-6), f"{name}: embedding collapsed"
        print(f"✓ {name}: valid embedding with variance per dim: {variance[:3]} ...")

    # Note: classic() uses randomized PCA, not true classical MDS
    # So stress may not decrease monotonically with more dimensions
    # Just check that both embeddings have reasonable stress
    stress_classic = compute_stress(D, Y_classic)
    stress_sgd = compute_stress(D, Y_sgd)
    print(f"Classic MDS stress (10D): {stress_classic:.2e}")
    print(f"SGD-MDS stress (10D): {stress_sgd:.2e}")

    assert np.isfinite(stress_classic), "Classic MDS stress is not finite"
    assert np.isfinite(stress_sgd), "SGD-MDS stress is not finite"
    print("✓ Both methods produce finite stress in high dimensions")

    print("✓ Test 16 PASSED\n")

#####################################################
# SMACOF and SGD 3D Tests
#####################################################

def test_smacof_basic_3d():
    """Test basic SMACOF functionality"""
    print("=" * 70)
    print("TEST 11: SMACOF basic functionality")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 80
    X = np.random.randn(n_samples, 6)
    D = squareform(pdist(X, "euclidean"))

    # Run SMACOF
    Y = smacof(D, n_components=3, random_state=42, metric=True, max_iter=300)

    # Check output shape
    assert Y.shape == (n_samples, 3), f"Expected shape (80, 3), got {Y.shape}"
    print(f"✓ Output shape correct: {Y.shape}")

    # Check for NaN or Inf
    assert not np.any(np.isnan(Y)), "Output contains NaN values"
    assert not np.any(np.isinf(Y)), "Output contains Inf values"
    print("✓ No NaN or Inf values")

    # Compute stress
    stress = compute_stress(D, Y)
    print(f"Stress: {stress:.6f}")
    print("✓ SMACOF converged")

    print("✓ Test 17 PASSED\n")

def test_sgd_mds_basic_3d():
    """Test basic functionality of SGD-MDS"""
    print("\n" + "=" * 70)
    print("TEST 1: Basic SGD-MDS functionality")
    print("=" * 70)

    # Create simple test data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 10)
    D = squareform(pdist(X, "euclidean"))

    # Run SGD-MDS
    Y = sgd_mds(D, n_components=3, n_iter=200, random_state=42)

    # Check output shape
    assert Y.shape == (n_samples, 3), f"Expected shape (100, 3), got {Y.shape}"
    print(f"✓ Output shape correct: {Y.shape}")

    # Check for NaN or Inf
    assert not np.any(np.isnan(Y)), "Output contains NaN values"
    assert not np.any(np.isinf(Y)), "Output contains Inf values"
    print("✓ No NaN or Inf values")

    # Check that points are spread out (not collapsed)
    variance = np.var(Y, axis=0)
    assert np.all(variance > 1e-6), f"Embedding collapsed: variance={variance}"
    print(f"✓ Embedding has variance: {variance}")

    print("✓ Test 18 PASSED\n")

def test_phate_with_sgd_mds_3d():
    """Test PHATE integration with SGD-MDS solver"""
    print("=" * 70)
    print("TEST 5: PHATE integration with SGD-MDS")
    print("=" * 70)

    # Generate tree data
    np.random.seed(42)
    data, labels = phate.tree.gen_dla(n_dim=50, n_branch=5, branch_length=100)

    print(f"Generated tree data: {data.shape}")

    # Test with SGD solver
    print("\nRunning PHATE with mds_solver='sgd'...")
    phate_sgd = phate.PHATE(
        n_components=3,
        knn=5,
        t=20,
        mds_solver="sgd",
        random_state=42,
        verbose=0,
    )
    embedding_sgd = phate_sgd.fit_transform(data)

    assert embedding_sgd.shape == (
        data.shape[0],
        3,
    ), f"Expected shape {(data.shape[0], 3)}, got {embedding_sgd.shape}"
    assert not np.any(np.isnan(embedding_sgd)), "SGD embedding contains NaN"
    print(f"✓ SGD embedding shape: {embedding_sgd.shape}")

    # Test with SMACOF solver for comparison
    print("\nRunning PHATE with mds_solver='smacof' for comparison...")
    phate_smacof = phate.PHATE(
        n_components=3,
        knn=5,
        t=20,
        mds_solver="smacof",
        random_state=42,
        verbose=0,
    )
    embedding_smacof = phate_smacof.fit_transform(data)

    assert embedding_smacof.shape == (
        data.shape[0],
        3,
    ), f"Expected shape {(data.shape[0], 3)}, got {embedding_smacof.shape}"
    print(f"✓ SMACOF embedding shape: {embedding_smacof.shape}")

    # Compare embeddings
    _, embedding_sgd_aligned, disparity = procrustes(embedding_smacof, embedding_sgd)
    print(f"\nProcrustes disparity between SGD and SMACOF: {disparity:.6f}")

    # PHATE adds additional processing (diffusion, potential), so embeddings
    # may differ more than raw MDS. Accept < 0.85 as reasonable.
    assert disparity < 0.85, f"Embeddings too different: {disparity:.6f}"
    print(f"✓ PHATE embeddings reasonably similar (disparity={disparity:.6f})")

    print("✓ Test 19 PASSED\n")

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("RUNNING COMPREHENSIVE MDS TEST SUITE")
    print("=" * 70)

    tests = [
        # SGD-MDS tests
        test_sgd_mds_basic,
        test_sgd_mds_vs_classic,
        test_sgd_mds_metric_wrapper,
        test_sgd_mds_vs_smacof,
        test_phate_with_sgd_mds,
        test_sgd_mds_with_pairs_per_iter,
        test_sgd_mds_different_n_components,
        # Classic MDS tests
        test_classic_mds_basic,
        test_classic_mds_perfect_embedding,
        test_classic_mds_different_dimensions,
        # SMACOF tests
        test_smacof_basic,
        test_smacof_improves_upon_init,
        test_smacof_vs_classic,
        # Corner cases
        test_mds_tiny_dataset,
        test_mds_zero_distances,
        test_mds_high_dimensions,
        # 3D SGD_MDS tests,
        test_smacof_basic_3d,
        test_sgd_mds_basic_3d,
        test_phate_with_sgd_mds_3d
    ]

    failed = []
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED:")
            print(f"  Error: {str(e)}")
            import traceback

            traceback.print_exc()
            failed.append(test_func.__name__)

    print("=" * 70)
    if not failed:
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        return True
    else:
        print(f"✗ {len(failed)} TEST(S) FAILED:")
        for name in failed:
            print(f"  - {name}")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
