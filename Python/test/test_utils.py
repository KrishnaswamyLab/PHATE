#!/usr/bin/env python
"""
Comprehensive test suite for phate.utils module
Tests validation and utility functions
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import phate.utils as utils
import pytest


#####################################################
# Tests for check_positive()
#####################################################


def test_check_positive_valid():
    """Test check_positive with valid positive numbers"""
    print("\n" + "=" * 70)
    print("TEST 1: check_positive() with valid inputs")
    print("=" * 70)

    # Should not raise for positive integers
    utils.check_positive(x=1, y=100, z=1000)
    print("✓ Positive integers pass")

    # Should not raise for positive floats
    utils.check_positive(a=0.1, b=1.5, c=999.99)
    print("✓ Positive floats pass")

    # Should not raise for very small positive numbers
    utils.check_positive(epsilon=1e-10, tiny=1e-100)
    print("✓ Very small positive numbers pass")

    print("✓ Test 1 PASSED\n")


def test_check_positive_zero():
    """Test check_positive rejects zero"""
    print("=" * 70)
    print("TEST 2: check_positive() rejects zero")
    print("=" * 70)

    with pytest.raises(ValueError, match="Expected x > 0, got 0"):
        utils.check_positive(x=0)
    print("✓ Correctly rejects zero")

    print("✓ Test 2 PASSED\n")


def test_check_positive_negative():
    """Test check_positive rejects negative numbers"""
    print("=" * 70)
    print("TEST 3: check_positive() rejects negative numbers")
    print("=" * 70)

    with pytest.raises(ValueError, match="Expected x > 0, got -1"):
        utils.check_positive(x=-1)
    print("✓ Correctly rejects negative integer")

    with pytest.raises(ValueError, match="Expected y > 0, got -0.5"):
        utils.check_positive(y=-0.5)
    print("✓ Correctly rejects negative float")

    print("✓ Test 3 PASSED\n")


def test_check_positive_non_numeric():
    """Test check_positive rejects non-numeric values"""
    print("=" * 70)
    print("TEST 4: check_positive() rejects non-numeric values")
    print("=" * 70)

    with pytest.raises(ValueError, match="Expected x > 0, got foo"):
        utils.check_positive(x="foo")
    print("✓ Correctly rejects string")

    with pytest.raises(ValueError):
        utils.check_positive(x=None)
    print("✓ Correctly rejects None")

    with pytest.raises(ValueError):
        utils.check_positive(x=[1, 2, 3])
    print("✓ Correctly rejects list")

    print("✓ Test 4 PASSED\n")


def test_check_positive_multiple_params():
    """Test check_positive with multiple parameters"""
    print("=" * 70)
    print("TEST 5: check_positive() with multiple parameters")
    print("=" * 70)

    # All valid - should pass
    utils.check_positive(a=1, b=2, c=3, d=4)
    print("✓ All valid parameters pass")

    # One invalid - should fail
    with pytest.raises(ValueError, match="Expected b > 0, got -1"):
        utils.check_positive(a=1, b=-1, c=3)
    print("✓ Correctly catches one invalid among many")

    print("✓ Test 5 PASSED\n")


#####################################################
# Tests for check_int()
#####################################################


def test_check_int_valid():
    """Test check_int with valid integers"""
    print("=" * 70)
    print("TEST 6: check_int() with valid integers")
    print("=" * 70)

    # Should pass for positive integers
    utils.check_int(x=1, y=100, z=1000)
    print("✓ Positive integers pass")

    # Should pass for zero
    utils.check_int(zero=0)
    print("✓ Zero passes")

    # Should pass for negative integers
    utils.check_int(neg=-5)
    print("✓ Negative integers pass")

    # Should pass for numpy integers
    utils.check_int(np_int=np.int32(42), np_int64=np.int64(100))
    print("✓ Numpy integers pass")

    print("✓ Test 6 PASSED\n")


def test_check_int_floats():
    """Test check_int rejects floats"""
    print("=" * 70)
    print("TEST 7: check_int() rejects floats")
    print("=" * 70)

    with pytest.raises(ValueError, match="Expected x integer, got 1.5"):
        utils.check_int(x=1.5)
    print("✓ Correctly rejects float")

    # Even integer-valued floats should be rejected
    with pytest.raises(ValueError, match="Expected x integer, got 1.0"):
        utils.check_int(x=1.0)
    print("✓ Correctly rejects integer-valued float")

    print("✓ Test 7 PASSED\n")


def test_check_int_non_numeric():
    """Test check_int rejects non-numeric values"""
    print("=" * 70)
    print("TEST 8: check_int() rejects non-numeric values")
    print("=" * 70)

    with pytest.raises(ValueError, match="Expected x integer, got foo"):
        utils.check_int(x="foo")
    print("✓ Correctly rejects string")

    with pytest.raises(ValueError):
        utils.check_int(x=None)
    print("✓ Correctly rejects None")

    print("✓ Test 8 PASSED\n")


#####################################################
# Tests for check_in()
#####################################################


def test_check_in_valid():
    """Test check_in with valid choices"""
    print("=" * 70)
    print("TEST 9: check_in() with valid choices")
    print("=" * 70)

    # String choices
    utils.check_in(["a", "b", "c"], x="a", y="b")
    print("✓ Valid string choices pass")

    # Numeric choices
    utils.check_in([1, 2, 3], val=2)
    print("✓ Valid numeric choices pass")

    # Mixed choices
    utils.check_in([1, "two", 3.0], a=1, b="two", c=3.0)
    print("✓ Valid mixed choices pass")

    print("✓ Test 9 PASSED\n")


def test_check_in_invalid():
    """Test check_in rejects invalid choices"""
    print("=" * 70)
    print("TEST 10: check_in() rejects invalid choices")
    print("=" * 70)

    with pytest.raises(ValueError, match="x value d not recognized. Choose from"):
        utils.check_in(["a", "b", "c"], x="d")
    print("✓ Correctly rejects invalid choice")

    with pytest.raises(ValueError, match="solver value invalid not recognized"):
        utils.check_in(["sgd", "smacof", "classic"], solver="invalid")
    print("✓ Correctly rejects invalid solver choice")

    print("✓ Test 10 PASSED\n")


def test_check_in_case_sensitive():
    """Test check_in is case sensitive"""
    print("=" * 70)
    print("TEST 11: check_in() is case sensitive")
    print("=" * 70)

    # Should pass for exact match
    utils.check_in(["sgd", "SGD"], method="sgd")
    print("✓ Exact match passes")

    # Should fail for different case
    with pytest.raises(ValueError, match="method value SGD not recognized"):
        utils.check_in(["sgd"], method="SGD")
    print("✓ Case mismatch correctly rejected")

    print("✓ Test 11 PASSED\n")


#####################################################
# Tests for check_between()
#####################################################


def test_check_between_valid():
    """Test check_between with valid ranges"""
    print("=" * 70)
    print("TEST 12: check_between() with valid ranges")
    print("=" * 70)

    # Value at boundaries (inclusive)
    utils.check_between(0, 1, alpha=0, beta=1)
    print("✓ Boundary values pass (inclusive)")

    # Value in middle of range
    utils.check_between(0, 10, x=5, y=7.5)
    print("✓ Mid-range values pass")

    # Negative ranges
    utils.check_between(-10, -5, temp=-7)
    print("✓ Negative ranges work")

    print("✓ Test 12 PASSED\n")


def test_check_between_out_of_range():
    """Test check_between rejects out-of-range values"""
    print("=" * 70)
    print("TEST 13: check_between() rejects out-of-range values")
    print("=" * 70)

    # Below minimum
    with pytest.raises(ValueError, match="Expected x between 0 and 1, got -0.1"):
        utils.check_between(0, 1, x=-0.1)
    print("✓ Correctly rejects below minimum")

    # Above maximum
    with pytest.raises(ValueError, match="Expected x between 0 and 1, got 1.1"):
        utils.check_between(0, 1, x=1.1)
    print("✓ Correctly rejects above maximum")

    print("✓ Test 13 PASSED\n")


def test_check_between_multiple():
    """Test check_between with multiple parameters"""
    print("=" * 70)
    print("TEST 14: check_between() with multiple parameters")
    print("=" * 70)

    # All valid
    utils.check_between(0, 100, a=10, b=50, c=99)
    print("✓ Multiple valid parameters pass")

    # One invalid
    with pytest.raises(ValueError, match="Expected b between 0 and 100, got 101"):
        utils.check_between(0, 100, a=10, b=101, c=50)
    print("✓ Correctly catches one invalid among many")

    print("✓ Test 14 PASSED\n")


#####################################################
# Tests for check_if_not()
#####################################################


def test_check_if_not_skip():
    """Test check_if_not skips checks when value matches"""
    print("=" * 70)
    print("TEST 15: check_if_not() skips checks for matching values")
    print("=" * 70)

    # Should NOT raise even though None is not positive (because decay matches None)
    utils.check_if_not(None, utils.check_positive, decay=None)
    print("✓ Correctly skips check when value is None (using 'is')")

    # Should NOT raise even though "auto" is not an int
    utils.check_if_not("auto", utils.check_int, clusters="auto")
    print("✓ Correctly skips check when value equals 'auto'")

    # Multiple parameters, some match - only non-matching should be checked
    utils.check_if_not(None, utils.check_positive, param1=5, param2=None, param3=10)
    print("✓ Correctly skips check for matching parameters (param2=None) but checks others")

    print("✓ Test 15 PASSED\n")


def test_check_if_not_run_checks():
    """Test check_if_not runs checks when value doesn't match"""
    print("=" * 70)
    print("TEST 16: check_if_not() runs checks for non-matching values")
    print("=" * 70)

    # Should raise because 0 is not positive and doesn't match None
    with pytest.raises(ValueError, match="Expected decay > 0, got 0"):
        utils.check_if_not(None, utils.check_positive, decay=0)
    print("✓ Correctly runs check when value doesn't match")

    # Should raise because 1.5 is not an integer
    with pytest.raises(ValueError, match="Expected t integer, got 1.5"):
        utils.check_if_not("auto", utils.check_int, t=1.5)
    print("✓ Correctly runs check for non-auto value")

    print("✓ Test 16 PASSED\n")


def test_check_if_not_multiple_checks():
    """Test check_if_not with multiple check functions"""
    print("=" * 70)
    print("TEST 17: check_if_not() with multiple check functions")
    print("=" * 70)

    # Should pass both checks (positive and integer)
    utils.check_if_not("auto", utils.check_positive, utils.check_int, t=5)
    print("✓ Passes multiple checks for valid value")

    # Should fail first check (not positive)
    with pytest.raises(ValueError, match="Expected t > 0"):
        utils.check_if_not("auto", utils.check_positive, utils.check_int, t=-5)
    print("✓ Fails when first check fails")

    # Should fail second check (not int)
    with pytest.raises(ValueError, match="Expected t integer"):
        utils.check_if_not("auto", utils.check_positive, utils.check_int, t=1.5)
    print("✓ Fails when second check fails")

    print("✓ Test 17 PASSED\n")


#####################################################
# Tests for matrix_is_equivalent()
#####################################################


def test_matrix_is_equivalent_identical():
    """Test matrix_is_equivalent with identical matrices"""
    print("=" * 70)
    print("TEST 18: matrix_is_equivalent() with identical matrices")
    print("=" * 70)

    X = np.array([[1, 2], [3, 4]])

    # Same object (identity check)
    assert utils.matrix_is_equivalent(X, X)
    print("✓ Same object returns True")

    # Equal numpy arrays
    Y = np.array([[1, 2], [3, 4]])
    assert utils.matrix_is_equivalent(X, Y)
    print("✓ Equal numpy arrays return True")

    print("✓ Test 18 PASSED\n")


def test_matrix_is_equivalent_different():
    """Test matrix_is_equivalent with different matrices"""
    print("=" * 70)
    print("TEST 19: matrix_is_equivalent() with different matrices")
    print("=" * 70)

    X = np.array([[1, 2], [3, 4]])

    # Different values
    Y = np.array([[5, 6], [7, 8]])
    assert not utils.matrix_is_equivalent(X, Y)
    print("✓ Different values return False")

    # Different shapes
    Z = np.array([[1, 2, 3], [4, 5, 6]])
    assert not utils.matrix_is_equivalent(X, Z)
    print("✓ Different shapes return False")

    print("✓ Test 19 PASSED\n")


def test_matrix_is_equivalent_sparse():
    """Test matrix_is_equivalent with sparse matrices"""
    print("=" * 70)
    print("TEST 20: matrix_is_equivalent() with sparse matrices")
    print("=" * 70)

    from scipy import sparse

    # Create sparse matrices
    X_sparse = sparse.csr_matrix([[1, 0], [0, 2]])
    Y_sparse = sparse.csr_matrix([[1, 0], [0, 2]])
    Z_sparse = sparse.csr_matrix([[1, 1], [0, 2]])

    # Same sparse matrices
    assert utils.matrix_is_equivalent(X_sparse, Y_sparse)
    print("✓ Equal sparse matrices return True")

    # Different sparse matrices
    assert not utils.matrix_is_equivalent(X_sparse, Z_sparse)
    print("✓ Different sparse matrices return False")

    print("✓ Test 20 PASSED\n")


def test_matrix_is_equivalent_mixed_types():
    """Test matrix_is_equivalent with mixed types"""
    print("=" * 70)
    print("TEST 21: matrix_is_equivalent() with mixed types")
    print("=" * 70)

    from scipy import sparse

    X_dense = np.array([[1, 0], [0, 2]])
    X_sparse = sparse.csr_matrix([[1, 0], [0, 2]])

    # Dense vs sparse (different types)
    assert not utils.matrix_is_equivalent(X_dense, X_sparse)
    print("✓ Dense vs sparse returns False")

    print("✓ Test 21 PASSED\n")


#####################################################
# Tests for in_ipynb()
#####################################################


def test_in_ipynb_not_notebook():
    """Test in_ipynb returns False outside notebook"""
    print("=" * 70)
    print("TEST 22: in_ipynb() returns False outside Jupyter")
    print("=" * 70)

    # When running in pytest, we're not in a notebook
    result = utils.in_ipynb()
    assert result is False
    print("✓ Correctly returns False in pytest environment")

    print("✓ Test 22 PASSED\n")


# Note: Testing in_ipynb() returning True would require actually running in a notebook
# which is not feasible in a standard pytest environment


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("RUNNING COMPREHENSIVE UTILS TEST SUITE")
    print("=" * 70)

    tests = [
        # check_positive tests
        test_check_positive_valid,
        test_check_positive_zero,
        test_check_positive_negative,
        test_check_positive_non_numeric,
        test_check_positive_multiple_params,
        # check_int tests
        test_check_int_valid,
        test_check_int_floats,
        test_check_int_non_numeric,
        # check_in tests
        test_check_in_valid,
        test_check_in_invalid,
        test_check_in_case_sensitive,
        # check_between tests
        test_check_between_valid,
        test_check_between_out_of_range,
        test_check_between_multiple,
        # check_if_not tests
        test_check_if_not_skip,
        test_check_if_not_run_checks,
        test_check_if_not_multiple_checks,
        # matrix_is_equivalent tests
        test_matrix_is_equivalent_identical,
        test_matrix_is_equivalent_different,
        test_matrix_is_equivalent_sparse,
        test_matrix_is_equivalent_mixed_types,
        # in_ipynb tests
        test_in_ipynb_not_notebook,
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
