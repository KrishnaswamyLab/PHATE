#!/usr/bin/env python
"""
Comprehensive test suite for phate.cluster module
Tests KMeans clustering on PHATE embeddings
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import phate
import phate.cluster as cluster
from sklearn import exceptions
import pytest
import warnings


#####################################################
# Test fixtures - create common test data
#####################################################


def create_simple_phate_op():
    """Create a simple fitted PHATE operator for testing"""
    # Generate simple tree data
    tree_data, tree_clusters = phate.tree.gen_dla(n_branch=3, seed=42)
    phate_op = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    phate_op.fit_transform(tree_data)
    return phate_op, tree_data, tree_clusters


def create_unfitted_phate_op():
    """Create an unfitted PHATE operator"""
    return phate.PHATE(knn=5, t=10, verbose=False)


#####################################################
# Tests for kmeans()
#####################################################


def test_kmeans_basic():
    """Test basic kmeans clustering with fixed n_clusters"""
    print("\n" + "=" * 70)
    print("TEST 1: kmeans() with fixed n_clusters")
    print("=" * 70)

    phate_op, tree_data, _ = create_simple_phate_op()

    # Test with n_clusters=3
    clusters = cluster.kmeans(phate_op, n_clusters=3)

    # Should return integer array
    assert np.issubdtype(clusters.dtype, np.signedinteger), \
        f"Expected integer dtype, got {clusters.dtype}"
    print(f"✓ Returns integer dtype: {clusters.dtype}")

    # Should have correct shape
    assert len(clusters.shape) == 1, f"Expected 1D array, got shape {clusters.shape}"
    assert len(clusters) == tree_data.shape[0], \
        f"Expected {tree_data.shape[0]} labels, got {len(clusters)}"
    print(f"✓ Correct shape: {clusters.shape}")

    # Should have exactly 3 clusters
    assert len(np.unique(clusters)) == 3, \
        f"Expected 3 clusters, got {len(np.unique(clusters))}"
    print(f"✓ Correct number of clusters: {len(np.unique(clusters))}")

    # Cluster labels should start from 0
    assert np.min(clusters) == 0, f"Expected min label 0, got {np.min(clusters)}"
    assert np.max(clusters) == 2, f"Expected max label 2, got {np.max(clusters)}"
    print(f"✓ Cluster labels in range [0, 2]")

    print("✓ Test 1 PASSED\n")


def test_kmeans_auto():
    """Test kmeans with n_clusters='auto'"""
    print("=" * 70)
    print("TEST 2: kmeans() with n_clusters='auto'")
    print("=" * 70)

    phate_op, tree_data, _ = create_simple_phate_op()

    # Test with auto cluster selection
    clusters = cluster.kmeans(phate_op, n_clusters="auto")

    # Should return integer array
    assert np.issubdtype(clusters.dtype, np.signedinteger)
    print(f"✓ Returns integer dtype: {clusters.dtype}")

    # Should have at least 2 clusters (minimum for auto mode)
    n_clusters = len(np.unique(clusters))
    assert n_clusters >= 2, f"Expected at least 2 clusters, got {n_clusters}"
    print(f"✓ Auto-selected {n_clusters} clusters")

    # Should have correct shape
    assert len(clusters) == tree_data.shape[0]
    print(f"✓ Correct shape: {clusters.shape}")

    print("✓ Test 2 PASSED\n")


def test_kmeans_different_n_clusters():
    """Test kmeans with various n_clusters values"""
    print("=" * 70)
    print("TEST 3: kmeans() with different n_clusters")
    print("=" * 70)

    phate_op, tree_data, _ = create_simple_phate_op()

    for k in [2, 3, 5, 7]:
        clusters = cluster.kmeans(phate_op, n_clusters=k)
        n_unique = len(np.unique(clusters))
        assert n_unique == k, f"Expected {k} clusters, got {n_unique}"
        assert len(clusters) == tree_data.shape[0]
        print(f"✓ n_clusters={k}: found {n_unique} clusters")

    print("✓ Test 3 PASSED\n")


def test_kmeans_max_clusters():
    """Test kmeans auto mode with different max_clusters"""
    print("=" * 70)
    print("TEST 4: kmeans() auto mode with max_clusters parameter")
    print("=" * 70)

    phate_op, _, _ = create_simple_phate_op()

    # Test different max_clusters values
    for max_k in [5, 8, 12]:
        clusters = cluster.kmeans(phate_op, n_clusters="auto", max_clusters=max_k)
        n_clusters = len(np.unique(clusters))

        # Should respect max_clusters (search from 2 to max_k)
        assert 2 <= n_clusters < max_k, \
            f"Expected clusters in [2, {max_k}), got {n_clusters}"
        print(f"✓ max_clusters={max_k}: selected {n_clusters} clusters")

    print("✓ Test 4 PASSED\n")


def test_kmeans_random_state():
    """Test kmeans reproducibility with random_state"""
    print("=" * 70)
    print("TEST 5: kmeans() reproducibility with random_state")
    print("=" * 70)

    phate_op, _, _ = create_simple_phate_op()

    # Run twice with same random_state
    clusters1 = cluster.kmeans(phate_op, n_clusters=4, random_state=42)
    clusters2 = cluster.kmeans(phate_op, n_clusters=4, random_state=42)

    # Should be identical
    assert np.array_equal(clusters1, clusters2), \
        "Same random_state should give identical results"
    print("✓ Same random_state gives identical results")

    # Run with different random_state
    clusters3 = cluster.kmeans(phate_op, n_clusters=4, random_state=123)

    # May be different (though not guaranteed)
    # At minimum, should have same properties
    assert len(clusters3) == len(clusters1)
    assert len(np.unique(clusters3)) == 4
    print("✓ Different random_state runs successfully")

    print("✓ Test 5 PASSED\n")


def test_kmeans_deprecated_k_parameter():
    """Test deprecated k parameter shows warning"""
    print("=" * 70)
    print("TEST 6: kmeans() deprecated k parameter")
    print("=" * 70)

    phate_op, _, _ = create_simple_phate_op()

    # Should show FutureWarning when using k
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        clusters = cluster.kmeans(phate_op, k=3)

        # Check warning was raised
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
        assert issubclass(w[0].category, FutureWarning), \
            f"Expected FutureWarning, got {w[0].category}"
        assert "k is deprecated" in str(w[0].message).lower(), \
            f"Unexpected warning message: {w[0].message}"
        print(f"✓ FutureWarning raised: {w[0].message}")

    # Should still work correctly
    assert len(np.unique(clusters)) == 3
    print("✓ k parameter still functions correctly")

    print("✓ Test 6 PASSED\n")


#####################################################
# Tests for silhouette_score()
#####################################################


def test_silhouette_score_basic():
    """Test silhouette_score function"""
    print("=" * 70)
    print("TEST 7: silhouette_score() basic functionality")
    print("=" * 70)

    phate_op, _, _ = create_simple_phate_op()

    # Compute silhouette score
    score = cluster.silhouette_score(phate_op, n_clusters=3)

    # Should return a float
    assert isinstance(score, (float, np.floating)), \
        f"Expected float score, got {type(score)}"
    print(f"✓ Returns float: {score}")

    # Silhouette score should be in [-1, 1]
    assert -1 <= score <= 1, f"Silhouette score {score} not in [-1, 1]"
    print(f"✓ Score in valid range: {score:.4f}")

    print("✓ Test 7 PASSED\n")


def test_silhouette_score_different_k():
    """Test silhouette_score with different n_clusters"""
    print("=" * 70)
    print("TEST 8: silhouette_score() with different n_clusters")
    print("=" * 70)

    phate_op, _, _ = create_simple_phate_op()

    scores = []
    for k in [2, 3, 4, 5]:
        score = cluster.silhouette_score(phate_op, n_clusters=k)
        assert -1 <= score <= 1
        scores.append(score)
        print(f"✓ k={k}: silhouette score = {score:.4f}")

    # All scores should be valid
    assert len(scores) == 4
    print("✓ All silhouette scores computed successfully")

    print("✓ Test 8 PASSED\n")


def test_silhouette_score_random_state():
    """Test silhouette_score reproducibility"""
    print("=" * 70)
    print("TEST 9: silhouette_score() reproducibility")
    print("=" * 70)

    phate_op, _, _ = create_simple_phate_op()

    # Same random_state should give same score
    score1 = cluster.silhouette_score(phate_op, n_clusters=3, random_state=42)
    score2 = cluster.silhouette_score(phate_op, n_clusters=3, random_state=42)

    assert np.isclose(score1, score2), \
        f"Expected same scores, got {score1} and {score2}"
    print(f"✓ Reproducible with random_state: {score1:.4f}")

    print("✓ Test 9 PASSED\n")


#####################################################
# Error handling tests
#####################################################


def test_kmeans_invalid_phate_op():
    """Test kmeans rejects non-PHATE input"""
    print("=" * 70)
    print("TEST 10: kmeans() rejects non-PHATE input")
    print("=" * 70)

    # Should raise TypeError for non-PHATE input
    with pytest.raises(TypeError, match="Expected phate_op to be of type PHATE"):
        cluster.kmeans(1, n_clusters=3)
    print("✓ Rejects integer input")

    with pytest.raises(TypeError, match="Expected phate_op to be of type PHATE"):
        cluster.kmeans("not a phate op", n_clusters=3)
    print("✓ Rejects string input")

    with pytest.raises(TypeError, match="Expected phate_op to be of type PHATE"):
        cluster.kmeans(np.array([1, 2, 3]), n_clusters=3)
    print("✓ Rejects array input")

    print("✓ Test 10 PASSED\n")


def test_kmeans_unfitted_phate():
    """Test kmeans rejects unfitted PHATE operator"""
    print("=" * 70)
    print("TEST 11: kmeans() rejects unfitted PHATE operator")
    print("=" * 70)

    phate_op = create_unfitted_phate_op()

    # Should raise NotFittedError
    with pytest.raises(exceptions.NotFittedError,
                      match="This PHATE instance is not fitted yet"):
        cluster.kmeans(phate_op, n_clusters=3)
    print("✓ Correctly raises NotFittedError for unfitted operator")

    print("✓ Test 11 PASSED\n")


def test_silhouette_score_unfitted_phate():
    """Test silhouette_score rejects unfitted PHATE operator"""
    print("=" * 70)
    print("TEST 12: silhouette_score() rejects unfitted PHATE")
    print("=" * 70)

    phate_op = create_unfitted_phate_op()

    # Should raise error (via kmeans call)
    with pytest.raises(exceptions.NotFittedError):
        cluster.silhouette_score(phate_op, n_clusters=3)
    print("✓ Correctly raises NotFittedError")

    print("✓ Test 12 PASSED\n")


#####################################################
# Integration tests
#####################################################


def test_kmeans_auto_uses_silhouette():
    """Test that auto mode actually uses silhouette scores"""
    print("=" * 70)
    print("TEST 13: kmeans() auto mode uses silhouette scoring")
    print("=" * 70)

    phate_op, _, _ = create_simple_phate_op()

    # Get auto-selected clustering
    clusters_auto = cluster.kmeans(phate_op, n_clusters="auto",
                                   max_clusters=8, random_state=42)
    n_auto = len(np.unique(clusters_auto))

    print(f"Auto-selected: {n_auto} clusters")

    # Compute silhouette scores manually for comparison
    silhouette_scores = []
    for k in range(2, 8):
        score = cluster.silhouette_score(phate_op, n_clusters=k, random_state=42)
        silhouette_scores.append(score)
        print(f"  k={k}: silhouette={score:.4f}")

    # The auto-selected k should correspond to max silhouette
    best_k = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2

    assert n_auto == best_k, \
        f"Auto-selected {n_auto} but best silhouette at k={best_k}"
    print(f"✓ Auto mode correctly selected k={best_k} (max silhouette)")

    print("✓ Test 13 PASSED\n")


def test_clustering_stability():
    """Test that clustering is stable with same parameters"""
    print("=" * 70)
    print("TEST 14: Clustering stability")
    print("=" * 70)

    # Create same data twice
    tree_data1, _ = phate.tree.gen_dla(n_branch=3, seed=42)
    tree_data2, _ = phate.tree.gen_dla(n_branch=3, seed=42)

    # Fit PHATE with same parameters
    phate_op1 = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    phate_op1.fit(tree_data1)

    phate_op2 = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    phate_op2.fit(tree_data2)

    # Cluster with same parameters
    clusters1 = cluster.kmeans(phate_op1, n_clusters=3, random_state=42)
    clusters2 = cluster.kmeans(phate_op2, n_clusters=3, random_state=42)

    # Should get identical results
    assert np.array_equal(clusters1, clusters2), \
        "Same data and parameters should give identical clustering"
    print("✓ Clustering is reproducible with same data and parameters")

    print("✓ Test 14 PASSED\n")


def test_kmeans_with_sklearn_kwargs():
    """Test that additional sklearn kwargs are passed through"""
    print("=" * 70)
    print("TEST 15: kmeans() passes kwargs to sklearn.KMeans")
    print("=" * 70)

    phate_op, _, _ = create_simple_phate_op()

    # Test with different sklearn parameters
    # n_init parameter controls number of initializations
    clusters1 = cluster.kmeans(phate_op, n_clusters=3, random_state=42, n_init=10)
    clusters2 = cluster.kmeans(phate_op, n_clusters=3, random_state=42, n_init=1)

    # Should both complete without error
    assert len(clusters1) == len(clusters2)
    print("✓ Successfully passes n_init parameter")

    # Test with max_iter parameter
    clusters3 = cluster.kmeans(phate_op, n_clusters=3, random_state=42, max_iter=100)
    assert len(clusters3) == len(clusters1)
    print("✓ Successfully passes max_iter parameter")

    print("✓ Test 15 PASSED\n")


#####################################################
# Run all tests
#####################################################


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
