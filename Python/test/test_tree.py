#!/usr/bin/env python
"""
Comprehensive test suite for phate.tree module
Tests DLA tree generation for synthetic test data
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import phate
import phate.tree as tree
import pytest


#####################################################
# Tests for gen_dla()
#####################################################


def test_gen_dla_basic():
    """Test basic gen_dla functionality with default parameters"""
    print("\n" + "=" * 70)
    print("TEST 1: gen_dla() basic functionality")
    print("=" * 70)

    # Generate with default parameters
    M, C = tree.gen_dla()

    # Check return types
    assert isinstance(M, np.ndarray), f"Expected M to be ndarray, got {type(M)}"
    assert isinstance(C, np.ndarray), f"Expected C to be ndarray, got {type(C)}"
    print(f"✓ Returns numpy arrays")

    # Check shapes with defaults: n_branch=20, branch_length=100
    expected_n_points = 20 * 100  # n_branch * branch_length
    assert M.shape[0] == expected_n_points, \
        f"Expected {expected_n_points} points, got {M.shape[0]}"
    print(f"✓ Correct number of points: {M.shape[0]}")

    # Default n_dim=100
    assert M.shape[1] == 100, f"Expected 100 dimensions, got {M.shape[1]}"
    print(f"✓ Correct dimensionality: {M.shape[1]}")

    # Cluster labels should match data
    assert C.shape[0] == M.shape[0], \
        f"Mismatched shapes: M has {M.shape[0]} points, C has {C.shape[0]} labels"
    print(f"✓ Cluster labels match data points")

    # Cluster labels should be integers from 0 to n_branch-1
    assert np.issubdtype(C.dtype, np.integer), f"Expected integer labels, got {C.dtype}"
    assert np.min(C) == 0, f"Expected min label 0, got {np.min(C)}"
    assert np.max(C) == 19, f"Expected max label 19 (n_branch-1), got {np.max(C)}"
    print(f"✓ Cluster labels in correct range [0, 19]")

    # All values should be finite
    assert np.all(np.isfinite(M)), "M contains non-finite values"
    print(f"✓ All data values are finite")

    print("✓ Test 1 PASSED\n")


def test_gen_dla_custom_parameters():
    """Test gen_dla with custom parameters"""
    print("=" * 70)
    print("TEST 2: gen_dla() with custom parameters")
    print("=" * 70)

    # Test with custom n_dim
    M, C = tree.gen_dla(n_dim=50, n_branch=3, branch_length=20, seed=42)

    assert M.shape == (60, 50), f"Expected shape (60, 50), got {M.shape}"
    assert C.shape == (60,), f"Expected C shape (60,), got {C.shape}"
    assert len(np.unique(C)) == 3, f"Expected 3 branches, got {len(np.unique(C))}"
    print(f"✓ n_dim=50, n_branch=3, branch_length=20 works correctly")

    # Test with different branch_length
    M, C = tree.gen_dla(n_dim=30, n_branch=5, branch_length=50, seed=42)

    assert M.shape == (250, 30), f"Expected shape (250, 30), got {M.shape}"
    assert C.shape == (250,), f"Expected C shape (250,), got {C.shape}"
    assert len(np.unique(C)) == 5, f"Expected 5 branches, got {len(np.unique(C))}"
    print(f"✓ n_dim=30, n_branch=5, branch_length=50 works correctly")

    print("✓ Test 2 PASSED\n")


def test_gen_dla_single_branch():
    """Test gen_dla with single branch (n_branch=1)"""
    print("=" * 70)
    print("TEST 3: gen_dla() with single branch")
    print("=" * 70)

    M, C = tree.gen_dla(n_dim=10, n_branch=1, branch_length=50, seed=42)

    assert M.shape == (50, 10), f"Expected shape (50, 10), got {M.shape}"
    assert C.shape == (50,), f"Expected C shape (50,), got {C.shape}"

    # With single branch, all labels should be 0
    assert np.all(C == 0), f"Expected all labels to be 0, got {np.unique(C)}"
    print(f"✓ Single branch: all labels are 0")

    assert np.all(np.isfinite(M)), "M contains non-finite values"
    print(f"✓ Data is finite")

    print("✓ Test 3 PASSED\n")


def test_gen_dla_reproducibility():
    """Test gen_dla reproducibility with same seed"""
    print("=" * 70)
    print("TEST 4: gen_dla() reproducibility with seed")
    print("=" * 70)

    # Generate twice with same seed
    M1, C1 = tree.gen_dla(n_dim=20, n_branch=3, branch_length=30, seed=42)
    M2, C2 = tree.gen_dla(n_dim=20, n_branch=3, branch_length=30, seed=42)

    # Should be identical
    assert np.array_equal(M1, M2), "Same seed should produce identical data"
    assert np.array_equal(C1, C2), "Same seed should produce identical labels"
    print(f"✓ Same seed produces identical results")

    # Different seed should produce different results
    M3, C3 = tree.gen_dla(n_dim=20, n_branch=3, branch_length=30, seed=999)

    # Should be different (very unlikely to be identical by chance)
    assert not np.array_equal(M1, M3), "Different seeds should produce different data"
    print(f"✓ Different seed produces different results")

    # But should have same shape and label structure
    assert M1.shape == M3.shape, "Should have same shape"
    assert C1.shape == C3.shape, "Should have same label shape"
    print(f"✓ Different seeds maintain consistent structure")

    print("✓ Test 4 PASSED\n")


def test_gen_dla_rand_multiplier():
    """Test gen_dla with different rand_multiplier values"""
    print("=" * 70)
    print("TEST 5: gen_dla() with different rand_multiplier")
    print("=" * 70)

    # Generate with different rand_multiplier values
    M1, C1 = tree.gen_dla(n_dim=10, n_branch=2, branch_length=20,
                          rand_multiplier=1, seed=42)
    M2, C2 = tree.gen_dla(n_dim=10, n_branch=2, branch_length=20,
                          rand_multiplier=5, seed=42)

    # Higher rand_multiplier should generally give larger spread
    spread1 = np.std(M1)
    spread2 = np.std(M2)

    print(f"rand_multiplier=1: std={spread1:.4f}")
    print(f"rand_multiplier=5: std={spread2:.4f}")

    # Larger multiplier should give larger spread (in most cases)
    assert spread2 > spread1, \
        f"Expected larger spread with higher rand_multiplier, got {spread1:.4f} vs {spread2:.4f}"
    print(f"✓ Higher rand_multiplier gives larger spread")

    print("✓ Test 5 PASSED\n")


def test_gen_dla_sigma():
    """Test gen_dla with different sigma (noise) values"""
    print("=" * 70)
    print("TEST 6: gen_dla() with different sigma (noise)")
    print("=" * 70)

    # Generate with no noise
    M1, C1 = tree.gen_dla(n_dim=10, n_branch=2, branch_length=20,
                          sigma=0, seed=42)

    # Generate with noise
    M2, C2 = tree.gen_dla(n_dim=10, n_branch=2, branch_length=20,
                          sigma=10, seed=42)

    # Should have same shape
    assert M1.shape == M2.shape
    print(f"✓ Same shape with different sigma")

    # Should be different due to noise
    assert not np.array_equal(M1, M2), "Different sigma should give different results"
    print(f"✓ Different sigma produces different results")

    # Both should be finite
    assert np.all(np.isfinite(M1)), "sigma=0 data should be finite"
    assert np.all(np.isfinite(M2)), "sigma=10 data should be finite"
    print(f"✓ All data finite for both sigma values")

    print("✓ Test 6 PASSED\n")


def test_gen_dla_cluster_labels():
    """Test gen_dla cluster label structure"""
    print("=" * 70)
    print("TEST 7: gen_dla() cluster label structure")
    print("=" * 70)

    n_branch = 4
    branch_length = 25
    M, C = tree.gen_dla(n_dim=10, n_branch=n_branch,
                       branch_length=branch_length, seed=42)

    # Each branch should have exactly branch_length points
    for i in range(n_branch):
        n_points_in_branch = np.sum(C == i)
        assert n_points_in_branch == branch_length, \
            f"Branch {i}: expected {branch_length} points, got {n_points_in_branch}"
        print(f"✓ Branch {i}: {n_points_in_branch} points")

    # Labels should be sequential
    # First branch_length points have label 0, next branch_length have label 1, etc.
    for i in range(n_branch):
        start_idx = i * branch_length
        end_idx = (i + 1) * branch_length
        branch_labels = C[start_idx:end_idx]
        assert np.all(branch_labels == i), \
            f"Branch {i}: labels not all {i} in positions [{start_idx}, {end_idx})"

    print(f"✓ Labels are correctly sequential")

    print("✓ Test 7 PASSED\n")


def test_gen_dla_various_dimensions():
    """Test gen_dla with various dimensionalities"""
    print("=" * 70)
    print("TEST 8: gen_dla() with various dimensionalities")
    print("=" * 70)

    for n_dim in [2, 5, 10, 50, 100, 200]:
        M, C = tree.gen_dla(n_dim=n_dim, n_branch=2, branch_length=20, seed=42)

        assert M.shape[1] == n_dim, f"Expected {n_dim} dimensions, got {M.shape[1]}"
        assert M.shape[0] == 40, f"Expected 40 points, got {M.shape[0]}"
        assert np.all(np.isfinite(M)), f"Non-finite values with n_dim={n_dim}"
        print(f"✓ n_dim={n_dim}: shape={M.shape}, all finite")

    print("✓ Test 8 PASSED\n")


def test_gen_dla_large_dataset():
    """Test gen_dla with larger dataset"""
    print("=" * 70)
    print("TEST 9: gen_dla() with large dataset")
    print("=" * 70)

    # Generate larger dataset
    M, C = tree.gen_dla(n_dim=100, n_branch=50, branch_length=200, seed=42)

    expected_n_points = 50 * 200  # 10,000 points
    assert M.shape == (expected_n_points, 100), \
        f"Expected shape ({expected_n_points}, 100), got {M.shape}"
    print(f"✓ Large dataset shape: {M.shape}")

    assert C.shape == (expected_n_points,), \
        f"Expected {expected_n_points} labels, got {C.shape[0]}"
    print(f"✓ Correct number of labels: {C.shape[0]}")

    assert len(np.unique(C)) == 50, f"Expected 50 unique labels, got {len(np.unique(C))}"
    print(f"✓ Correct number of branches: {len(np.unique(C))}")

    assert np.all(np.isfinite(M)), "Large dataset contains non-finite values"
    print(f"✓ All values finite")

    print("✓ Test 9 PASSED\n")


def test_gen_dla_data_structure():
    """Test that gen_dla produces tree-like structure"""
    print("=" * 70)
    print("TEST 10: gen_dla() produces branching structure")
    print("=" * 70)

    n_branch = 3
    branch_length = 50
    M, C = tree.gen_dla(n_dim=20, n_branch=n_branch,
                       branch_length=branch_length, seed=42)

    # Each branch should form a path (cumulative sum of random steps)
    # Points within a branch should be relatively close to each other
    # compared to points in different branches (on average)

    for i in range(n_branch):
        # Get points in this branch
        branch_points = M[C == i]

        # Check that branch forms a continuous path
        # (consecutive points should be close)
        consecutive_dists = []
        for j in range(len(branch_points) - 1):
            dist = np.linalg.norm(branch_points[j+1] - branch_points[j])
            consecutive_dists.append(dist)

        mean_consecutive_dist = np.mean(consecutive_dists)
        print(f"Branch {i}: mean consecutive distance = {mean_consecutive_dist:.4f}")

        # Consecutive distances should be relatively small (local structure)
        # This is a sanity check that the tree structure makes sense
        assert mean_consecutive_dist < 100, \
            f"Branch {i}: consecutive points too far apart ({mean_consecutive_dist:.4f})"

    print(f"✓ All branches show local continuity")

    print("✓ Test 10 PASSED\n")


def test_gen_dla_minimal_parameters():
    """Test gen_dla with minimal/extreme parameters"""
    print("=" * 70)
    print("TEST 11: gen_dla() with minimal parameters")
    print("=" * 70)

    # Very small dataset
    M, C = tree.gen_dla(n_dim=2, n_branch=1, branch_length=5, seed=42)

    assert M.shape == (5, 2), f"Expected shape (5, 2), got {M.shape}"
    assert C.shape == (5,), f"Expected C shape (5,), got {C.shape}"
    assert np.all(np.isfinite(M)), "Minimal dataset contains non-finite values"
    print(f"✓ Minimal dataset (n_dim=2, n_branch=1, branch_length=5) works")

    # Very small dimensionality with multiple branches
    M, C = tree.gen_dla(n_dim=1, n_branch=3, branch_length=10, seed=42)

    assert M.shape == (30, 1), f"Expected shape (30, 1), got {M.shape}"
    assert np.all(np.isfinite(M)), "1D dataset contains non-finite values"
    print(f"✓ 1D dataset (n_dim=1) works")

    print("✓ Test 11 PASSED\n")


#####################################################
# Integration test
#####################################################


def test_gen_dla_with_phate():
    """Test that gen_dla output works with PHATE"""
    print("=" * 70)
    print("TEST 12: gen_dla() output works with PHATE")
    print("=" * 70)

    # Generate tree data
    M, C = tree.gen_dla(n_dim=50, n_branch=4, branch_length=100, seed=42)

    # Should work with PHATE
    phate_op = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    Y = phate_op.fit_transform(M)

    # Check PHATE output
    assert Y.shape == (400, 2), f"Expected PHATE output (400, 2), got {Y.shape}"
    assert np.all(np.isfinite(Y)), "PHATE embedding contains non-finite values"
    print(f"✓ gen_dla() output works with PHATE")
    print(f"  Input shape: {M.shape}, Output shape: {Y.shape}")

    print("✓ Test 12 PASSED\n")


#####################################################
# Run all tests
#####################################################


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
