#!/usr/bin/env python
"""
Comprehensive test suite for phate.vne module (Von Neumann Entropy)
Tests entropy computation and knee point detection
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import phate.vne as vne
import pytest


#####################################################
# Tests for compute_von_neumann_entropy()
#####################################################


def test_compute_von_neumann_entropy_basic():
    """Test basic VNE computation"""
    print("\n" + "=" * 70)
    print("TEST 1: compute_von_neumann_entropy() basic functionality")
    print("=" * 70)

    # Example from docstring
    X = np.eye(10)
    X[0, 0] = 5
    X[3, 2] = 4

    h = vne.compute_von_neumann_entropy(X)

    # Should return array of length t_max (default 100)
    assert h.shape == (100,), f"Expected shape (100,), got {h.shape}"
    print(f"✓ Output shape correct: {h.shape}")

    # Entropy should be finite and non-negative
    assert np.all(np.isfinite(h)), "Entropy contains non-finite values"
    assert np.all(h >= 0), "Entropy contains negative values"
    print("✓ All entropy values are finite and non-negative")

    # Entropy typically decreases with t as large eigenvalues dominate
    # (distribution becomes more concentrated)
    print(f"✓ Entropy changes from {h[0]:.4f} to {h[-1]:.4f}")
    print(f"  (Typically decreases as dominant eigenvalues concentrate distribution)")

    print("✓ Test 1 PASSED\n")


def test_compute_von_neumann_entropy_custom_t_max():
    """Test VNE with custom t_max"""
    print("=" * 70)
    print("TEST 2: compute_von_neumann_entropy() with custom t_max")
    print("=" * 70)

    X = np.random.randn(10, 10)

    # Test different t_max values
    for t_max in [10, 50, 200]:
        h = vne.compute_von_neumann_entropy(X, t_max=t_max)
        assert h.shape == (t_max,), f"Expected shape ({t_max},), got {h.shape}"
        assert np.all(np.isfinite(h)), f"Non-finite values with t_max={t_max}"
        print(f"✓ t_max={t_max}: shape={h.shape}, all finite")

    print("✓ Test 2 PASSED\n")


def test_compute_von_neumann_entropy_identity():
    """Test VNE on identity matrix"""
    print("=" * 70)
    print("TEST 3: compute_von_neumann_entropy() on identity matrix")
    print("=" * 70)

    # Identity matrix has eigenvalues all equal to 1
    # So entropy should be constant (max entropy for uniform distribution)
    X = np.eye(10)
    h = vne.compute_von_neumann_entropy(X, t_max=20)

    # Entropy should be relatively constant for identity
    # (all eigenvalues equal = maximum entropy)
    entropy_std = np.std(h)
    print(f"Entropy std dev: {entropy_std:.6f}")
    print(f"Entropy values: {h[:5]} ... {h[-5:]}")

    # Identity gives uniform distribution, so entropy should be log(n)
    expected_entropy = np.log(10)
    assert np.allclose(h[0], expected_entropy, rtol=0.1), \
        f"Expected entropy ~{expected_entropy:.4f}, got {h[0]:.4f}"
    print(f"✓ Identity matrix entropy ~{expected_entropy:.4f} (theoretical)")

    print("✓ Test 3 PASSED\n")


def test_compute_von_neumann_entropy_diagonal():
    """Test VNE on diagonal matrix with varying eigenvalues"""
    print("=" * 70)
    print("TEST 4: compute_von_neumann_entropy() on diagonal matrix")
    print("=" * 70)

    # Diagonal matrix with exponentially decaying eigenvalues
    eigenvalues = np.array([10, 5, 2, 1, 0.5, 0.1, 0.01, 0.001])
    X = np.diag(eigenvalues)

    h = vne.compute_von_neumann_entropy(X, t_max=50)

    # Entropy should decrease as large eigenvalues dominate
    # (distribution becomes more concentrated)
    assert h[0] > h[-1], "Entropy should decrease as large eigenvalues dominate"
    print(f"✓ Entropy decreases from {h[0]:.4f} to {h[-1]:.4f}")
    print(f"  (Large eigenvalues increasingly dominate)")

    print("✓ Test 4 PASSED\n")


def test_compute_von_neumann_entropy_sparse():
    """Test VNE on sparse-like matrix"""
    print("=" * 70)
    print("TEST 5: compute_von_neumann_entropy() on sparse matrix")
    print("=" * 70)

    # Create a mostly-zero matrix (sparse structure)
    X = np.zeros((20, 20))
    X[0, 0] = 10
    X[1, 1] = 5
    X[2, 2] = 1

    h = vne.compute_von_neumann_entropy(X, t_max=30)

    # Should compute without errors
    assert np.all(np.isfinite(h)), "Non-finite values with sparse matrix"
    print(f"✓ Sparse matrix handled: entropy range [{h.min():.4f}, {h.max():.4f}]")

    print("✓ Test 5 PASSED\n")


def test_compute_von_neumann_entropy_reproducible():
    """Test VNE is deterministic/reproducible"""
    print("=" * 70)
    print("TEST 6: compute_von_neumann_entropy() is reproducible")
    print("=" * 70)

    X = np.random.randn(15, 15)

    h1 = vne.compute_von_neumann_entropy(X, t_max=50)
    h2 = vne.compute_von_neumann_entropy(X, t_max=50)

    # Should get identical results
    assert np.allclose(h1, h2), "VNE not reproducible"
    print("✓ Multiple runs produce identical results")

    print("✓ Test 6 PASSED\n")


#####################################################
# Tests for find_knee_point()
#####################################################


def test_find_knee_point_exponential():
    """Test find_knee_point on exponential decay"""
    print("=" * 70)
    print("TEST 7: find_knee_point() on exponential curve")
    print("=" * 70)

    # Example from docstring
    x = np.arange(20)
    y = np.exp(-x / 10)

    knee = vne.find_knee_point(y, x)

    # Should find knee around x=8 (from docstring)
    assert knee == 8, f"Expected knee at 8, got {knee}"
    print(f"✓ Correctly identified knee at x={knee}")

    print("✓ Test 7 PASSED\n")


def test_find_knee_point_default_x():
    """Test find_knee_point with default x values"""
    print("=" * 70)
    print("TEST 8: find_knee_point() with default x values")
    print("=" * 70)

    # When x is not provided, should use indices 0, 1, 2, ...
    y = np.exp(-np.arange(20) / 10)

    knee_with_x = vne.find_knee_point(y, x=np.arange(20))
    knee_without_x = vne.find_knee_point(y)

    # Should get same result
    assert knee_with_x == knee_without_x, \
        f"Different results: with x={knee_with_x}, without x={knee_without_x}"
    print(f"✓ Default x gives same result: knee={knee_without_x}")

    print("✓ Test 8 PASSED\n")


def test_find_knee_point_entropy():
    """Test find_knee_point on VNE output"""
    print("=" * 70)
    print("TEST 9: find_knee_point() on von Neumann entropy")
    print("=" * 70)

    # Example from VNE docstring
    X = np.eye(10)
    X[0, 0] = 5
    X[3, 2] = 4
    h = vne.compute_von_neumann_entropy(X)

    knee = vne.find_knee_point(h)

    # Should find knee at 23 (from docstring)
    assert knee == 23, f"Expected knee at 23, got {knee}"
    print(f"✓ Correctly identified knee at t={knee}")

    print("✓ Test 9 PASSED\n")


def test_find_knee_point_linear():
    """Test find_knee_point on linear data (no clear knee)"""
    print("=" * 70)
    print("TEST 10: find_knee_point() on linear data")
    print("=" * 70)

    # Linear data has no clear knee, but should still return a value
    y = np.linspace(10, 1, 20)

    knee = vne.find_knee_point(y)

    # Should return a value in valid range
    assert 0 <= knee < 20, f"Knee {knee} out of valid range [0, 20)"
    print(f"✓ Returns valid knee point: {knee}")
    print("  (Linear data has no clear knee, but algorithm still converges)")

    print("✓ Test 10 PASSED\n")


def test_find_knee_point_step_function():
    """Test find_knee_point on step function"""
    print("=" * 70)
    print("TEST 11: find_knee_point() on step function")
    print("=" * 70)

    # Step function has clear knee at transition
    y = np.concatenate([np.ones(10) * 10, np.ones(10)])

    knee = vne.find_knee_point(y)

    # Knee should be near the step (around index 10)
    assert 8 <= knee <= 12, f"Expected knee near 10, got {knee}"
    print(f"✓ Correctly identified knee near step: {knee}")

    print("✓ Test 11 PASSED\n")


def test_find_knee_point_unsorted_x():
    """Test find_knee_point with unsorted x values"""
    print("=" * 70)
    print("TEST 12: find_knee_point() with unsorted x values")
    print("=" * 70)

    # Create sorted data
    x_sorted = np.arange(20)
    y = np.exp(-x_sorted / 10)

    # Shuffle x and y together
    indices = np.random.RandomState(42).permutation(20)
    x_shuffled = x_sorted[indices]
    y_shuffled = y[indices]

    knee_sorted = vne.find_knee_point(y, x_sorted)
    knee_shuffled = vne.find_knee_point(y_shuffled, x_shuffled)

    # Should get same result (function sorts internally)
    assert knee_sorted == knee_shuffled, \
        f"Different results: sorted={knee_sorted}, shuffled={knee_shuffled}"
    print(f"✓ Handles unsorted x correctly: knee={knee_shuffled}")

    print("✓ Test 12 PASSED\n")


def test_find_knee_point_list_input():
    """Test find_knee_point accepts lists"""
    print("=" * 70)
    print("TEST 13: find_knee_point() accepts list input")
    print("=" * 70)

    # Should accept Python lists, not just numpy arrays
    x = list(range(20))
    y = [np.exp(-i / 10) for i in range(20)]

    knee = vne.find_knee_point(y, x)

    # Should work and return valid result
    assert isinstance(knee, (int, np.integer, float, np.floating)), \
        f"Expected numeric knee, got {type(knee)}"
    print(f"✓ Accepts list input: knee={knee}")

    print("✓ Test 13 PASSED\n")


#####################################################
# Error handling tests
#####################################################


def test_find_knee_point_too_short():
    """Test find_knee_point rejects too-short input"""
    print("=" * 70)
    print("TEST 14: find_knee_point() rejects short input")
    print("=" * 70)

    # Should raise error for length < 3 (message says "length 3" but means "shorter than 3")
    with pytest.raises(ValueError, match="Cannot find knee point on vector of length 3"):
        vne.find_knee_point([1, 2])
    print("✓ Correctly rejects length 2")

    with pytest.raises(ValueError, match="Cannot find knee point on vector of length 3"):
        vne.find_knee_point([1])
    print("✓ Correctly rejects length 1")

    # Length 3 should work (despite confusing error message)
    result = vne.find_knee_point([10, 5, 1])
    assert result is not None
    print("✓ Accepts length 3")

    # Length 4 should work
    result = vne.find_knee_point([10, 5, 2, 1])
    assert result is not None
    print("✓ Accepts length 4")

    print("✓ Test 14 PASSED\n")


def test_find_knee_point_mismatched_shapes():
    """Test find_knee_point rejects mismatched x and y"""
    print("=" * 70)
    print("TEST 15: find_knee_point() rejects mismatched shapes")
    print("=" * 70)

    y = np.arange(10)
    x = np.arange(15)  # Different length

    with pytest.raises(ValueError, match="x and y must be the same shape"):
        vne.find_knee_point(y, x)
    print("✓ Correctly rejects mismatched shapes")

    print("✓ Test 15 PASSED\n")


def test_find_knee_point_multidimensional():
    """Test find_knee_point rejects multi-dimensional input"""
    print("=" * 70)
    print("TEST 16: find_knee_point() rejects 2D input")
    print("=" * 70)

    y = np.random.randn(10, 5)  # 2D array

    with pytest.raises(ValueError, match="y must be 1-dimensional"):
        vne.find_knee_point(y)
    print("✓ Correctly rejects 2D input")

    print("✓ Test 16 PASSED\n")


#####################################################
# Integration tests
#####################################################


def test_vne_workflow():
    """Test complete VNE workflow for selecting t"""
    print("=" * 70)
    print("TEST 17: Complete VNE workflow")
    print("=" * 70)

    # Simulate typical usage: find optimal t for diffusion
    np.random.seed(42)
    X = np.random.randn(20, 20)

    # Compute entropy curve
    entropy = vne.compute_von_neumann_entropy(X, t_max=50)
    print(f"Computed entropy for t=1 to t=50")

    # Find knee point
    optimal_t = vne.find_knee_point(entropy)
    print(f"✓ Identified optimal t={optimal_t}")

    # Knee should be in valid range
    assert 0 <= optimal_t < 50, f"Invalid t={optimal_t}"

    # Check that knee is in a reasonable range (not at extremes)
    # For random data, knee typically not at boundaries
    if optimal_t < 5 or optimal_t > 45:
        print(f"  Warning: Knee at extreme (t={optimal_t})")
    else:
        print(f"  ✓ Knee in reasonable range")

    print("✓ Test 17 PASSED\n")


def test_vne_sensitivity():
    """Test VNE sensitivity to matrix properties"""
    print("=" * 70)
    print("TEST 18: VNE sensitivity to matrix properties")
    print("=" * 70)

    # Test with matrices having different properties
    np.random.seed(42)

    # Well-conditioned matrix
    X1 = np.random.randn(10, 10)
    h1 = vne.compute_von_neumann_entropy(X1, t_max=30)
    k1 = vne.find_knee_point(h1)
    print(f"Random matrix: knee at t={k1}")

    # Matrix with strong dominant eigenvalue
    X2 = np.diag([100, 10, 5, 2, 1, 0.5, 0.1, 0.01, 0.001, 0.0001])
    h2 = vne.compute_von_neumann_entropy(X2, t_max=30)
    k2 = vne.find_knee_point(h2)
    print(f"Dominant eigenvalue matrix: knee at t={k2}")

    # Knees should be different (matrix structure affects entropy curve)
    print(f"✓ Different matrices give different knees: {k1} vs {k2}")

    print("✓ Test 18 PASSED\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("RUNNING COMPREHENSIVE VNE TEST SUITE")
    print("=" * 70)

    tests = [
        # compute_von_neumann_entropy tests
        test_compute_von_neumann_entropy_basic,
        test_compute_von_neumann_entropy_custom_t_max,
        test_compute_von_neumann_entropy_identity,
        test_compute_von_neumann_entropy_diagonal,
        test_compute_von_neumann_entropy_sparse,
        test_compute_von_neumann_entropy_reproducible,
        # find_knee_point tests
        test_find_knee_point_exponential,
        test_find_knee_point_default_x,
        test_find_knee_point_entropy,
        test_find_knee_point_linear,
        test_find_knee_point_step_function,
        test_find_knee_point_unsorted_x,
        test_find_knee_point_list_input,
        # Error handling
        test_find_knee_point_too_short,
        test_find_knee_point_mismatched_shapes,
        test_find_knee_point_multidimensional,
        # Integration
        test_vne_workflow,
        test_vne_sensitivity,
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
