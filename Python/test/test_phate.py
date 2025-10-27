#!/usr/bin/env python
"""
Comprehensive test suite for main PHATE class
Tests the complete PHATE dimensionality reduction pipeline
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import phate
import graphtools
import pytest
from scipy.spatial.distance import pdist, squareform

# Optional dependencies
try:
    import pygsp

    PYGSP_AVAILABLE = True
except ImportError:
    PYGSP_AVAILABLE = False

try:
    import anndata

    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False


#####################################################
# Test fixtures
#####################################################


def create_test_data(seed=42):
    """Create simple test data"""
    tree_data, tree_clusters = phate.tree.gen_dla(
        n_dim=50, n_branch=3, branch_length=100, seed=seed
    )
    return tree_data, tree_clusters


#####################################################
# Basic functionality tests
#####################################################


def test_phate_basic_workflow():
    """Test basic PHATE fit_transform workflow"""
    print("\n" + "=" * 70)
    print("TEST 1: Basic PHATE workflow")
    print("=" * 70)

    data, _ = create_test_data()

    # Create and fit PHATE
    phate_op = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    embedding = phate_op.fit_transform(data)

    # Check output shape (default n_components=2)
    assert embedding.shape == (
        data.shape[0],
        2,
    ), f"Expected shape ({data.shape[0]}, 2), got {embedding.shape}"
    print(f"✓ Correct output shape: {embedding.shape}")

    # Check output is finite
    assert np.all(np.isfinite(embedding)), "Embedding contains non-finite values"
    print(f"✓ All embedding values are finite")

    # Check that phate_op is fitted
    assert phate_op.graph is not None, "Graph not created after fit"
    assert phate_op.diff_potential is not None, "Diffusion potential not computed"
    print(f"✓ PHATE operator fitted successfully")

    print("✓ Test 1 PASSED\n")


def test_phate_str_repr():
    """Test __str__ and __repr__ methods"""
    print("=" * 70)
    print("TEST 2: PHATE __str__ and __repr__ methods")
    print("=" * 70)

    phate_op = phate.PHATE(knn=15, t=100, verbose=False)

    # Test __str__
    str_output = str(phate_op)
    assert isinstance(
        str_output, str
    ), f"__str__ should return str, got {type(str_output)}"
    assert len(str_output) > 0, "__str__ returned empty string"
    print(f"✓ __str__() returns non-empty string")

    # Test __repr__
    repr_output = repr(phate_op)
    assert isinstance(
        repr_output, str
    ), f"__repr__ should return str, got {type(repr_output)}"
    assert len(repr_output) > 0, "__repr__ returned empty string"
    print(f"✓ __repr__() returns non-empty string")

    print("✓ Test 2 PASSED\n")


def test_phate_fit_vs_fit_transform():
    """Test fit() vs fit_transform()"""
    print("=" * 70)
    print("TEST 3: fit() vs fit_transform()")
    print("=" * 70)

    data, _ = create_test_data()

    # Test fit_transform
    phate_op1 = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    embedding1 = phate_op1.fit_transform(data)

    # Test fit + transform
    phate_op2 = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    phate_op2.fit(data)
    embedding2 = phate_op2.transform()

    # Should give same results
    assert np.allclose(
        embedding1, embedding2, atol=1e-10
    ), "fit_transform and fit+transform give different results"
    print(f"✓ fit_transform() and fit()+transform() give identical results")

    print("✓ Test 3 PASSED\n")


#####################################################
# Parameter tests
#####################################################


def test_phate_n_components():
    """Test different n_components values"""
    print("=" * 70)
    print("TEST 4: n_components parameter")
    print("=" * 70)

    data, _ = create_test_data()

    for n_comp in [1, 2, 3, 5, 10]:
        phate_op = phate.PHATE(
            n_components=n_comp, knn=5, t=10, verbose=False, random_state=42
        )
        embedding = phate_op.fit_transform(data)

        assert embedding.shape == (
            data.shape[0],
            n_comp,
        ), f"Expected shape ({data.shape[0]}, {n_comp}), got {embedding.shape}"
        assert np.all(
            np.isfinite(embedding)
        ), f"Non-finite values with n_components={n_comp}"
        print(f"✓ n_components={n_comp}: shape={embedding.shape}")

    print("✓ Test 4 PASSED\n")


def test_phate_knn_parameters():
    """Test knn and knn_max parameters"""
    print("=" * 70)
    print("TEST 5: knn and knn_max parameters")
    print("=" * 70)

    data, _ = create_test_data()

    # Test different knn values
    for knn in [3, 5, 10, 20]:
        phate_op = phate.PHATE(knn=knn, t=10, verbose=False, random_state=42)
        phate_op.fit(data)

        assert (
            phate_op.graph.knn == knn
        ), f"Expected knn={knn}, got {phate_op.graph.knn}"
        print(f"✓ knn={knn} set correctly")

    # Test knn_max
    phate_op = phate.PHATE(knn=5, knn_max=15, t=10, verbose=False, random_state=42)
    phate_op.fit(data)

    assert phate_op.graph.knn == 5, f"Expected knn=5, got {phate_op.graph.knn}"
    assert (
        phate_op.graph.knn_max == 15
    ), f"Expected knn_max=15, got {phate_op.graph.knn_max}"
    print(f"✓ knn_max parameter set correctly")

    print("✓ Test 5 PASSED\n")


def test_phate_decay_parameter():
    """Test decay parameter"""
    print("=" * 70)
    print("TEST 6: decay parameter")
    print("=" * 70)

    data, _ = create_test_data()

    for decay in [5, 10, 20, 40]:
        phate_op = phate.PHATE(knn=5, decay=decay, t=10, verbose=False, random_state=42)
        phate_op.fit(data)

        assert (
            phate_op.graph.decay == decay
        ), f"Expected decay={decay}, got {phate_op.graph.decay}"
        print(f"✓ decay={decay} set correctly")

    print("✓ Test 6 PASSED\n")


def test_phate_t_parameter():
    """Test t (diffusion time) parameter"""
    print("=" * 70)
    print("TEST 7: t (diffusion time) parameter")
    print("=" * 70)

    data, _ = create_test_data()

    # Test with fixed t values
    for t in [5, 10, 20, 50]:
        phate_op = phate.PHATE(knn=5, t=t, verbose=False, random_state=42)
        embedding = phate_op.fit_transform(data)

        assert np.all(np.isfinite(embedding)), f"Non-finite values with t={t}"
        print(f"✓ t={t} works correctly")

    # Test with t='auto'
    phate_op = phate.PHATE(knn=5, t="auto", verbose=False, random_state=42)
    embedding = phate_op.fit_transform(data)

    assert np.all(np.isfinite(embedding)), "Non-finite values with t='auto'"
    print(f"✓ t='auto' works correctly")

    print("✓ Test 7 PASSED\n")


def test_phate_mds_methods():
    """Test different MDS methods"""
    print("=" * 70)
    print("TEST 8: Different MDS methods")
    print("=" * 70)

    data, _ = create_test_data()

    for mds_method in ["classic", "metric", "nonmetric"]:
        phate_op = phate.PHATE(
            knn=5, t=10, mds=mds_method, verbose=False, random_state=42
        )
        embedding = phate_op.fit_transform(data)

        assert embedding.shape == (
            data.shape[0],
            2,
        ), f"Unexpected shape with mds={mds_method}"
        assert np.all(
            np.isfinite(embedding)
        ), f"Non-finite values with mds={mds_method}"
        print(f"✓ mds='{mds_method}' works correctly")

    print("✓ Test 8 PASSED\n")


def test_phate_set_params():
    """Test set_params() method"""
    print("=" * 70)
    print("TEST 9: set_params() method")
    print("=" * 70)

    data, _ = create_test_data()

    # Fit with classic MDS
    phate_op = phate.PHATE(knn=5, t=10, mds="classic", verbose=False, random_state=42)
    embedding1 = phate_op.fit_transform(data)

    # Change to metric MDS without refitting
    phate_op.set_params(mds="metric")
    embedding2 = phate_op.transform()

    # Should produce different embeddings
    assert not np.allclose(
        embedding1, embedding2
    ), "Different MDS methods should give different embeddings"
    print(f"✓ set_params() changes embedding")

    # Both should be valid
    assert np.all(np.isfinite(embedding1))
    assert np.all(np.isfinite(embedding2))
    print(f"✓ Both embeddings are finite")

    print("✓ Test 9 PASSED\n")


def test_phate_gamma_parameter():
    """Test gamma parameter (for metric MDS)"""
    print("=" * 70)
    print("TEST 10: gamma parameter")
    print("=" * 70)

    data, _ = create_test_data()

    # gamma=1 (log transform)
    phate_op1 = phate.PHATE(
        knn=5, t=10, mds="metric", gamma=1, verbose=False, random_state=42
    )
    embedding1 = phate_op1.fit_transform(data)

    # gamma=0 (sqrt transform)
    phate_op2 = phate.PHATE(
        knn=5, t=10, mds="metric", gamma=0, verbose=False, random_state=42
    )
    embedding2 = phate_op2.fit_transform(data)

    # Different gamma should give different embeddings
    assert not np.allclose(
        embedding1, embedding2
    ), "Different gamma values should give different embeddings"
    print(f"✓ gamma=1 and gamma=0 give different embeddings")

    # Both should be valid
    assert np.all(np.isfinite(embedding1))
    assert np.all(np.isfinite(embedding2))
    print(f"✓ Both embeddings are finite")

    print("✓ Test 10 PASSED\n")


#####################################################
# Distance metric tests
#####################################################


def test_phate_distance_metrics():
    """Test different distance metrics"""
    print("=" * 70)
    print("TEST 11: Different distance metrics")
    print("=" * 70)

    data, _ = create_test_data()

    for metric in ["euclidean", "cosine", "cityblock"]:
        phate_op = phate.PHATE(
            knn=5,
            t=10,
            knn_dist=metric,
            mds_dist=metric,
            verbose=False,
            random_state=42,
        )
        embedding = phate_op.fit_transform(data)

        assert embedding.shape == (data.shape[0], 2)
        assert np.all(np.isfinite(embedding)), f"Non-finite values with metric={metric}"
        print(f"✓ metric='{metric}' works correctly")

    print("✓ Test 11 PASSED\n")


def test_phate_precomputed_distance():
    """Test precomputed distance matrix"""
    print("=" * 70)
    print("TEST 12: Precomputed distance matrix")
    print("=" * 70)

    data, _ = create_test_data()

    # Compute distance matrix
    D = squareform(pdist(data, "euclidean"))

    # Fit with raw data
    phate_op1 = phate.PHATE(
        knn=5, t=10, knn_dist="euclidean", verbose=False, random_state=42
    )
    embedding1 = phate_op1.fit_transform(data)

    # Fit with precomputed distance
    phate_op2 = phate.PHATE(
        knn=5, t=10, knn_dist="precomputed_distance", verbose=False, random_state=42
    )
    embedding2 = phate_op2.fit_transform(D)

    # Should give similar results (may have small numerical differences)
    assert np.allclose(
        embedding1, embedding2, atol=1e-3
    ), "Precomputed distance gives different results"
    print(f"✓ Precomputed distance matrix works correctly")

    print("✓ Test 12 PASSED\n")


def test_phate_precomputed_affinity():
    """Test precomputed affinity matrix"""
    print("=" * 70)
    print("TEST 13: Precomputed affinity matrix")
    print("=" * 70)

    data, _ = create_test_data()

    # Fit with raw data first to get affinity matrix
    phate_op1 = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    embedding1 = phate_op1.fit_transform(data)
    K = phate_op1.graph.kernel

    # Fit with precomputed affinity
    phate_op2 = phate.PHATE(
        knn=5, t=10, knn_dist="precomputed_affinity", verbose=False, random_state=42
    )
    embedding2 = phate_op2.fit_transform(K)

    # Should give nearly identical results
    assert np.allclose(
        embedding1, embedding2, atol=1e-3
    ), "Precomputed affinity gives different results"
    print(f"✓ Precomputed affinity matrix works correctly")

    print("✓ Test 13 PASSED\n")


#####################################################
# Input type tests
#####################################################


def test_phate_graphtools_input():
    """Test PHATE with graphtools.Graph input"""
    print("=" * 70)
    print("TEST 14: graphtools.Graph input")
    print("=" * 70)

    data, _ = create_test_data()

    # Create PHATE operator and fit with data
    phate_op = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    phate_op.fit(data)

    # Get the graph
    graph = phate_op.graph

    # Fit new PHATE operator with graph
    phate_op2 = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    phate_op2.fit(graph)
    embedding = phate_op2.transform()

    assert embedding.shape == (data.shape[0], 2)
    assert np.all(np.isfinite(embedding))
    print(f"✓ graphtools.Graph input works correctly")

    print("✓ Test 14 PASSED\n")


@pytest.mark.skipif(not PYGSP_AVAILABLE, reason="pygsp not installed")
def test_phate_pygsp_input():
    """Test PHATE with PyGSP graph input"""
    print("=" * 70)
    print("TEST 15: PyGSP graph input")
    print("=" * 70)

    data, _ = create_test_data()

    # Create PHATE operator and fit
    phate_op = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    phate_op.fit(data)

    # Create PyGSP graph from graphtools graph
    G_graphtools = graphtools.Graph(
        phate_op.graph.kernel, precomputed="affinity", use_pygsp=True, verbose=False
    )
    G_pygsp = pygsp.graphs.Graph(G_graphtools.W)

    # Fit PHATE with PyGSP graph
    phate_op2 = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    phate_op2.fit(G_pygsp)
    embedding = phate_op2.transform()

    assert embedding.shape == (data.shape[0], 2)
    assert np.all(np.isfinite(embedding))
    print(f"✓ PyGSP graph input works correctly")

    print("✓ Test 15 PASSED\n")


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")
def test_phate_anndata_input():
    """Test PHATE with AnnData input"""
    print("=" * 70)
    print("TEST 16: AnnData input")
    print("=" * 70)

    data, _ = create_test_data()

    # Create AnnData object
    adata = anndata.AnnData(data)

    # Fit PHATE with AnnData
    phate_op = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    phate_op.fit(adata)
    embedding = phate_op.transform()

    assert embedding.shape == (data.shape[0], 2)
    assert np.all(np.isfinite(embedding))
    print(f"✓ AnnData input works correctly")

    print("✓ Test 16 PASSED\n")


#####################################################
# Reproducibility tests
#####################################################


def test_phate_reproducibility():
    """Test PHATE reproducibility with random_state"""
    print("=" * 70)
    print("TEST 17: Reproducibility with random_state")
    print("=" * 70)

    data, _ = create_test_data()

    # Run twice with same random_state
    phate_op1 = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    embedding1 = phate_op1.fit_transform(data)

    phate_op2 = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    embedding2 = phate_op2.fit_transform(data)

    # Should be identical
    assert np.allclose(
        embedding1, embedding2, atol=1e-10
    ), "Same random_state should give identical results"
    print(f"✓ Same random_state gives identical results")

    print("✓ Test 17 PASSED\n")


def test_phate_different_random_state():
    """Test PHATE with different random_state"""
    print("=" * 70)
    print("TEST 18: Different random_state values")
    print("=" * 70)

    data, _ = create_test_data()

    # Run with different random_state
    phate_op1 = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    embedding1 = phate_op1.fit_transform(data)

    phate_op2 = phate.PHATE(knn=5, t=10, verbose=False, random_state=999)
    embedding2 = phate_op2.fit_transform(data)

    # May be different (depending on MDS method), but should have same shape
    assert embedding1.shape == embedding2.shape
    print(f"✓ Different random_state runs successfully")

    # Both should be valid
    assert np.all(np.isfinite(embedding1))
    assert np.all(np.isfinite(embedding2))
    print(f"✓ Both embeddings are finite")

    print("✓ Test 18 PASSED\n")


#####################################################
# n_jobs parameter test
#####################################################


def test_phate_n_jobs():
    """Test n_jobs parameter"""
    print("=" * 70)
    print("TEST 19: n_jobs parameter")
    print("=" * 70)

    data, _ = create_test_data()

    for n_jobs in [1, -1, -2]:
        phate_op = phate.PHATE(
            knn=5, t=10, n_jobs=n_jobs, verbose=False, random_state=42
        )
        phate_op.fit(data)

        assert (
            phate_op.graph.n_jobs == n_jobs
        ), f"Expected n_jobs={n_jobs}, got {phate_op.graph.n_jobs}"
        print(f"✓ n_jobs={n_jobs} set correctly")

    print("✓ Test 19 PASSED\n")


#####################################################
# Verbose parameter test
#####################################################


def test_phate_verbose():
    """Test verbose parameter"""
    print("=" * 70)
    print("TEST 20: verbose parameter")
    print("=" * 70)

    data, _ = create_test_data()

    # verbose=False should work silently
    phate_op = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    phate_op.fit(data)
    assert phate_op.graph.verbose == 0 or phate_op.graph.verbose == False
    print(f"✓ verbose=False works correctly")

    # verbose=True should also work (just with output)
    phate_op = phate.PHATE(knn=5, t=10, verbose=True, random_state=42)
    phate_op.fit(data)
    # Just check it doesn't crash
    print(f"✓ verbose=True works correctly")

    print("✓ Test 20 PASSED\n")


#####################################################
# Run all tests
#####################################################


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
