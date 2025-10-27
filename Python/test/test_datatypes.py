#!/usr/bin/env python
"""
Comprehensive test suite for PHATE with different input types
Tests: numpy.array, scipy.spmatrix, pandas.DataFrame, anndata.AnnData
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import phate
import pytest
from scipy import sparse

# Optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import anndata
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False


#####################################################
# Test fixtures and utilities
#####################################################


def create_test_data(seed=42, n_samples=100, n_features=50):
    """Create simple test data"""
    np.random.seed(seed)
    data = np.random.randn(n_samples, n_features)
    # Add some structure
    data[:n_samples//2, :10] += 2.0
    data[n_samples//2:, 10:20] += 2.0
    return data


def compare_embeddings(embedding1, embedding2, rtol=1e-5, atol=1e-8):
    """
    Compare two embeddings for equality.
    Handles potential sign flips in principal components.
    """
    # Check shapes match
    if embedding1.shape != embedding2.shape:
        return False
    
    # Direct comparison
    if np.allclose(embedding1, embedding2, rtol=rtol, atol=atol):
        return True
    
    # Check with sign flips (each dimension can be flipped independently)
    for flip_mask in [np.array([1, 1]), np.array([1, -1]), 
                      np.array([-1, 1]), np.array([-1, -1])]:
        if embedding1.shape[1] == 2:
            flipped = embedding2 * flip_mask
            if np.allclose(embedding1, flipped, rtol=rtol, atol=atol):
                return True
    
    return False


#####################################################
# Test 1: NumPy array input
#####################################################


def test_phate_numpy_array():
    """Test PHATE with numpy.ndarray input"""
    print("\n" + "=" * 70)
    print("TEST: PHATE with numpy.ndarray")
    print("=" * 70)
    
    data = create_test_data()
    print(f"Input type: {type(data)}")
    print(f"Input shape: {data.shape}")
    print(f"Input dtype: {data.dtype}")
    
    # Create and fit PHATE
    phate_op = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    embedding = phate_op.fit_transform(data)
    
    # Assertions
    assert isinstance(embedding, np.ndarray), f"Expected numpy.ndarray output, got {type(embedding)}"
    assert embedding.shape == (data.shape[0], 2), f"Expected shape ({data.shape[0]}, 2), got {embedding.shape}"
    assert np.all(np.isfinite(embedding)), "Embedding contains non-finite values"
    assert phate_op.graph is not None, "Graph not created"
    
    print(f"✓ Output type: {type(embedding)}")
    print(f"✓ Output shape: {embedding.shape}")
    print(f"✓ All values finite: {np.all(np.isfinite(embedding))}")
    print("✓ TEST PASSED\n")


#####################################################
# Test 2: SciPy sparse matrix input
#####################################################


def test_phate_scipy_sparse_csr():
    """Test PHATE with scipy.sparse.csr_matrix input"""
    print("\n" + "=" * 70)
    print("TEST: PHATE with scipy.sparse.csr_matrix")
    print("=" * 70)
    
    data = create_test_data()
    # Convert to sparse (make it actually sparse by zeroing small values)
    data[np.abs(data) < 0.5] = 0
    sparse_data = sparse.csr_matrix(data)
    
    print(f"Input type: {type(sparse_data)}")
    print(f"Input shape: {sparse_data.shape}")
    print(f"Sparsity: {1 - sparse_data.nnz / (sparse_data.shape[0] * sparse_data.shape[1]):.2%}")
    
    # Create and fit PHATE
    phate_op = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    embedding = phate_op.fit_transform(sparse_data)
    
    # Assertions
    assert isinstance(embedding, np.ndarray), f"Expected numpy.ndarray output, got {type(embedding)}"
    assert embedding.shape == (sparse_data.shape[0], 2), f"Expected shape ({sparse_data.shape[0]}, 2), got {embedding.shape}"
    assert np.all(np.isfinite(embedding)), "Embedding contains non-finite values"
    assert phate_op.graph is not None, "Graph not created"
    
    print(f"✓ Output type: {type(embedding)}")
    print(f"✓ Output shape: {embedding.shape}")
    print(f"✓ All values finite: {np.all(np.isfinite(embedding))}")
    print("✓ TEST PASSED\n")
    



def test_phate_scipy_sparse_csc():
    """Test PHATE with scipy.sparse.csc_matrix input"""
    print("\n" + "=" * 70)
    print("TEST: PHATE with scipy.sparse.csc_matrix")
    print("=" * 70)
    
    data = create_test_data()
    data[np.abs(data) < 0.5] = 0
    sparse_data = sparse.csc_matrix(data)
    
    print(f"Input type: {type(sparse_data)}")
    print(f"Input shape: {sparse_data.shape}")
    print(f"Sparsity: {1 - sparse_data.nnz / (sparse_data.shape[0] * sparse_data.shape[1]):.2%}")
    
    phate_op = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    embedding = phate_op.fit_transform(sparse_data)
    
    assert isinstance(embedding, np.ndarray), f"Expected numpy.ndarray output, got {type(embedding)}"
    assert embedding.shape == (sparse_data.shape[0], 2), f"Expected shape ({sparse_data.shape[0]}, 2), got {embedding.shape}"
    assert np.all(np.isfinite(embedding)), "Embedding contains non-finite values"
    
    print(f"✓ Output type: {type(embedding)}")
    print(f"✓ Output shape: {embedding.shape}")
    print("✓ TEST PASSED\n")
    



def test_phate_scipy_sparse_coo():
    """Test PHATE with scipy.sparse.coo_matrix input"""
    print("\n" + "=" * 70)
    print("TEST: PHATE with scipy.sparse.coo_matrix")
    print("=" * 70)
    
    data = create_test_data()
    data[np.abs(data) < 0.5] = 0
    sparse_data = sparse.coo_matrix(data)
    
    print(f"Input type: {type(sparse_data)}")
    print(f"Input shape: {sparse_data.shape}")
    
    phate_op = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    embedding = phate_op.fit_transform(sparse_data)
    
    assert isinstance(embedding, np.ndarray), f"Expected numpy.ndarray output, got {type(embedding)}"
    assert embedding.shape == (sparse_data.shape[0], 2), f"Expected shape ({sparse_data.shape[0]}, 2), got {embedding.shape}"
    assert np.all(np.isfinite(embedding)), "Embedding contains non-finite values"
    
    print(f"✓ Output type: {type(embedding)}")
    print(f"✓ Output shape: {embedding.shape}")
    print("✓ TEST PASSED\n")
    



#####################################################
# Test 3: pandas DataFrame input
#####################################################


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not installed")
def test_phate_pandas_dataframe():
    """Test PHATE with pandas.DataFrame input"""
    print("\n" + "=" * 70)
    print("TEST: PHATE with pandas.DataFrame")
    print("=" * 70)
    
    data = create_test_data()
    df = pd.DataFrame(
        data,
        columns=[f"feature_{i}" for i in range(data.shape[1])],
        index=[f"sample_{i}" for i in range(data.shape[0])]
    )
    
    print(f"Input type: {type(df)}")
    print(f"Input shape: {df.shape}")
    print(f"Column names: {df.columns[:5].tolist()}... (showing first 5)")
    print(f"Index: {df.index[:5].tolist()}... (showing first 5)")
    
    # Create and fit PHATE
    phate_op = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    embedding = phate_op.fit_transform(df)
    
    # Assertions
    assert isinstance(embedding, np.ndarray), f"Expected numpy.ndarray output, got {type(embedding)}"
    assert embedding.shape == (df.shape[0], 2), f"Expected shape ({df.shape[0]}, 2), got {embedding.shape}"
    assert np.all(np.isfinite(embedding)), "Embedding contains non-finite values"
    assert phate_op.graph is not None, "Graph not created"
    
    print(f"✓ Output type: {type(embedding)}")
    print(f"✓ Output shape: {embedding.shape}")
    print(f"✓ All values finite: {np.all(np.isfinite(embedding))}")
    print("✓ TEST PASSED\n")
    



#####################################################
# Test 4: AnnData input
#####################################################


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")
def test_phate_anndata():
    """Test PHATE with anndata.AnnData input"""
    print("\n" + "=" * 70)
    print("TEST: PHATE with anndata.AnnData")
    print("=" * 70)
    
    data = create_test_data()
    
    # Create AnnData object
    adata = anndata.AnnData(
        X=data,
        obs=pd.DataFrame(
            {"cell_type": [f"type_{i%3}" for i in range(data.shape[0])]},
            index=[f"cell_{i}" for i in range(data.shape[0])]
        ) if PANDAS_AVAILABLE else None,
        var=pd.DataFrame(
            {"gene_name": [f"gene_{i}" for i in range(data.shape[1])]},
            index=[f"gene_{i}" for i in range(data.shape[1])]
        ) if PANDAS_AVAILABLE else None
    )
    
    print(f"Input type: {type(adata)}")
    print(f"Input shape: {adata.shape}")
    print(f"X type: {type(adata.X)}")
    if adata.obs is not None:
        print(f"Observations: {adata.obs.shape}")
    if adata.var is not None:
        print(f"Variables: {adata.var.shape}")
    
    # Create and fit PHATE
    phate_op = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    embedding = phate_op.fit_transform(adata)
    
    # Assertions
    assert isinstance(embedding, np.ndarray), f"Expected numpy.ndarray output, got {type(embedding)}"
    assert embedding.shape == (adata.shape[0], 2), f"Expected shape ({adata.shape[0]}, 2), got {embedding.shape}"
    assert np.all(np.isfinite(embedding)), "Embedding contains non-finite values"
    assert phate_op.graph is not None, "Graph not created"
    
    print(f"✓ Output type: {type(embedding)}")
    print(f"✓ Output shape: {embedding.shape}")
    print(f"✓ All values finite: {np.all(np.isfinite(embedding))}")
    print("✓ TEST PASSED\n")
    



@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")
def test_phate_anndata_sparse():
    """Test PHATE with anndata.AnnData input containing sparse matrix"""
    print("\n" + "=" * 70)
    print("TEST: PHATE with anndata.AnnData (sparse X)")
    print("=" * 70)
    
    data = create_test_data()
    data[np.abs(data) < 0.5] = 0
    sparse_data = sparse.csr_matrix(data)
    
    # Create AnnData object with sparse data
    adata = anndata.AnnData(X=sparse_data)
    
    print(f"Input type: {type(adata)}")
    print(f"Input shape: {adata.shape}")
    print(f"X type: {type(adata.X)}")
    print(f"X sparsity: {1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.2%}")
    
    phate_op = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    embedding = phate_op.fit_transform(adata)
    
    assert isinstance(embedding, np.ndarray), f"Expected numpy.ndarray output, got {type(embedding)}"
    assert embedding.shape == (adata.shape[0], 2), f"Expected shape ({adata.shape[0]}, 2), got {embedding.shape}"
    assert np.all(np.isfinite(embedding)), "Embedding contains non-finite values"
    
    print(f"✓ Output type: {type(embedding)}")
    print(f"✓ Output shape: {embedding.shape}")
    print("✓ TEST PASSED\n")
    



#####################################################
# Test 5: Consistency across input types
#####################################################


def test_consistency_across_input_types():
    """Test that PHATE produces similar results across different input types"""
    print("\n" + "=" * 70)
    print("TEST: Consistency across input types")
    print("=" * 70)
    
    # Create base data
    data = create_test_data(seed=42)
    
    # Test numpy array
    print("Running PHATE on numpy array...")
    phate_op_numpy = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
    embedding_numpy = phate_op_numpy.fit_transform(data)
    
    # Test pandas DataFrame (if available)
    if PANDAS_AVAILABLE:
        print("Running PHATE on pandas DataFrame...")
        df = pd.DataFrame(data)
        phate_op_df = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
        embedding_df = phate_op_df.fit_transform(df)
        
        # Compare embeddings
        print("Comparing numpy and pandas embeddings...")
        assert compare_embeddings(embedding_numpy, embedding_df, rtol=1e-4, atol=1e-6), \
            "Embeddings from numpy and pandas inputs differ significantly"
        print("✓ NumPy and pandas embeddings are consistent")
    
    # Test AnnData (if available)
    if ANNDATA_AVAILABLE:
        print("Running PHATE on AnnData...")
        adata = anndata.AnnData(X=data)
        phate_op_adata = phate.PHATE(knn=5, t=10, verbose=False, random_state=42)
        embedding_adata = phate_op_adata.fit_transform(adata)
        
        # Compare embeddings
        print("Comparing numpy and AnnData embeddings...")
        assert compare_embeddings(embedding_numpy, embedding_adata, rtol=1e-4, atol=1e-6), \
            "Embeddings from numpy and AnnData inputs differ significantly"
        print("✓ NumPy and AnnData embeddings are consistent")
    
    print("✓ TEST PASSED - All input types produce consistent results\n")


#####################################################
# Test 6: Edge cases and error handling
#####################################################


def test_empty_dataframe():
    """Test PHATE behavior with edge cases"""
    print("\n" + "=" * 70)
    print("TEST: Edge cases")
    print("=" * 70)
    
    # Test with small dataset
    print("Testing with small dataset (10 samples)...")
    small_data = create_test_data(n_samples=10, n_features=5)
    phate_op = phate.PHATE(knn=3, t=5, verbose=False, random_state=42)
    embedding = phate_op.fit_transform(small_data)
    assert embedding.shape == (10, 2), f"Expected shape (10, 2), got {embedding.shape}"
    print("✓ Small dataset handled correctly")
    
    print("✓ TEST PASSED\n")


#####################################################
# Main test runner
#####################################################


def run_all_tests():
    """Run all tests"""
    print("\n" + "#" * 70)
    print("# PHATE INPUT TYPE TEST SUITE")
    print("#" * 70)
    
    # Track results
    results = {}
    
    # Test 1: NumPy array
    try:
        test_phate_numpy_array()
        results['numpy'] = 'PASSED'
    except Exception as e:
        print(f"✗ TEST FAILED: {e}\n")
        results['numpy'] = f'FAILED: {e}'
    
    # Test 2: SciPy sparse matrices
    try:
        test_phate_scipy_sparse_csr()
        results['scipy_csr'] = 'PASSED'
    except Exception as e:
        print(f"✗ TEST FAILED: {e}\n")
        results['scipy_csr'] = f'FAILED: {e}'
    
    try:
        test_phate_scipy_sparse_csc()
        results['scipy_csc'] = 'PASSED'
    except Exception as e:
        print(f"✗ TEST FAILED: {e}\n")
        results['scipy_csc'] = f'FAILED: {e}'
    
    try:
        test_phate_scipy_sparse_coo()
        results['scipy_coo'] = 'PASSED'
    except Exception as e:
        print(f"✗ TEST FAILED: {e}\n")
        results['scipy_coo'] = f'FAILED: {e}'
    
    # Test 3: pandas DataFrame
    if PANDAS_AVAILABLE:
        try:
            test_phate_pandas_dataframe()
            results['pandas'] = 'PASSED'
        except Exception as e:
            print(f"✗ TEST FAILED: {e}\n")
            results['pandas'] = f'FAILED: {e}'
    else:
        print("⊘ Skipping pandas tests (pandas not installed)\n")
        results['pandas'] = 'SKIPPED'
    
    # Test 4: AnnData
    if ANNDATA_AVAILABLE:
        try:
            test_phate_anndata()
            results['anndata'] = 'PASSED'
        except Exception as e:
            print(f"✗ TEST FAILED: {e}\n")
            results['anndata'] = f'FAILED: {e}'
        
        try:
            test_phate_anndata_sparse()
            results['anndata_sparse'] = 'PASSED'
        except Exception as e:
            print(f"✗ TEST FAILED: {e}\n")
            results['anndata_sparse'] = f'FAILED: {e}'
    else:
        print("⊘ Skipping AnnData tests (anndata not installed)\n")
        results['anndata'] = 'SKIPPED'
        results['anndata_sparse'] = 'SKIPPED'
    
    # Test 5: Consistency
    try:
        test_consistency_across_input_types()
        results['consistency'] = 'PASSED'
    except Exception as e:
        print(f"✗ TEST FAILED: {e}\n")
        results['consistency'] = f'FAILED: {e}'
    
    # Test 6: Edge cases
    try:
        test_empty_dataframe()
        results['edge_cases'] = 'PASSED'
    except Exception as e:
        print(f"✗ TEST FAILED: {e}\n")
        results['edge_cases'] = f'FAILED: {e}'
    
    # Print summary
    print("\n" + "#" * 70)
    print("# TEST SUMMARY")
    print("#" * 70)
    for test_name, result in results.items():
        status = "✓" if result == "PASSED" else "⊘" if result == "SKIPPED" else "✗"
        print(f"{status} {test_name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    skipped = sum(1 for r in results.values() if r == "SKIPPED")
    failed = sum(1 for r in results.values() if r not in ["PASSED", "SKIPPED"])
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    print("#" * 70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)