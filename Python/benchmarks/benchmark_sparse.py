#!/usr/bin/env python
"""Benchmark sparse vs original PHATE on increasing dataset sizes.

Measures peak memory, runtime, and embedding quality (Procrustes distance).
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import phate
from phate import sparse_similarity


def peak_memory_mb():
    """Return peak RSS in MB (macOS/Linux)."""
    import resource

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes, Linux returns KB
    if sys.platform == "darwin":
        return maxrss / (1024 * 1024)
    return maxrss / 1024


def gen_data(n_samples, n_features=100, seed=42):
    """Generate synthetic tree data."""
    return phate.tree.gen_dla(
        n_dim=n_features,
        n_branch=max(1, n_samples // 100),
        branch_length=100,
        seed=seed,
    )


def procrustes_distance(X, Y):
    """Normalized Procrustes disparity: 0 = perfect match, 1 = worst."""
    from scipy.spatial import procrustes

    _, _, disparity = procrustes(X, Y)
    # Normalize by the sum of variances
    norm = np.var(X) * X.shape[0] * X.shape[1] + np.var(Y) * Y.shape[0] * Y.shape[1]
    return disparity / (norm + 1e-10)


def benchmark_phate(data, sparse_k=None, **kwargs):
    """Run PHATE and return timing + memory + embedding."""
    mem_before = peak_memory_mb()
    t0 = time.time()

    phate_op = phate.PHATE(
        knn=5,
        t=20,
        sparse_k=sparse_k,
        sparse_metric="euclidean",
        sparse_batch_size=256,
        verbose=False,
        random_state=42,
        **kwargs,
    )
    embedding = phate_op.fit_transform(data)

    elapsed = time.time() - t0
    mem_after = peak_memory_mb()
    mem_delta = mem_after - mem_before

    return embedding, elapsed, mem_delta, phate_op


def benchmark_sizes():
    """Run benchmarks at increasing sizes."""
    sizes = [500, 1000, 2000, 3000, 5000]
    results = []

    print("=" * 80)
    print("PHATE Sparse Benchmark")
    print("=" * 80)
    print(f"{'N':>6} {'Mode':>12} {'Time(s)':>10} {'Mem(MB)':>10} {'Sparsity%':>12} {'Quality':>10}")
    print("-" * 62)

    for n in sizes:
        data, clusters = gen_data(n, n_features=100)
        print(f"\n--- N={n} ---")

        # Original PHATE (skip for larger sizes if memory constrained)
        if n <= 2000:
            try:
                emb_orig, t_orig, m_orig, _ = benchmark_phate(data, sparse_k=None)
                print(
                    f"{n:>6} {'original':>12} {t_orig:>10.2f} {m_orig:>10.1f} "
                    f"{'N/A':>12} {'-':>10}"
                )
            except Exception as e:
                print(f"{n:>6} {'original':>12} {'FAILED':>10}: {e}")
                emb_orig, t_orig, m_orig = None, None, None
        else:
            emb_orig = None

        # Sparse PHATE
        try:
            emb_sp, t_sp, m_sp, op_sp = benchmark_phate(data, sparse_k=10)
            diff_op = op_sp.diff_op
            if hasattr(diff_op, "nnz"):
                sparsity = 100 * diff_op.nnz / (n * n)
            else:
                sparsity = 0.0

            # Quality vs original (if available)
            if emb_orig is not None:
                # Align via Procrustes before comparing
                quality = procrustes_distance(emb_orig, emb_sp)
                quality_str = f"{quality:.4f}"
            else:
                quality_str = "N/A"

            print(
                f"{n:>6} {'sparse':>12} {t_sp:>10.2f} {m_sp:>10.1f} "
                f"{sparsity:>10.2f}% {quality_str:>10}"
            )
            results.append((n, t_sp, m_sp, sparsity, quality_str))
        except Exception as e:
            print(f"{n:>6} {'sparse':>12} {'FAILED':>10}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    return results


def benchmark_minimal_k():
    """Benchmark binary search for minimal k."""
    print("\n" + "=" * 80)
    print("Minimal k Connectivity Search")
    print("=" * 80)

    n = 500
    data, _ = gen_data(n, n_features=100)
    print(f"Dataset: N={n}, features=100")

    # Time the minimal-k search
    t0 = time.time()
    k_opt, S = sparse_similarity.find_minimal_k(
        data, k_max=50, metric="euclidean", batch_size=256, verbose=1
    )
    elapsed = time.time() - t0

    n_comp, _ = sparse_similarity.check_connectivity(S)
    print(f"\nMinimal k for connectivity: {k_opt}")
    print(f"Components at k_opt: {n_comp}")
    print(f"Search time: {elapsed:.2f}s")
    print(f"Sparsity: {100 * S.nnz / (n * n):.3f}% ({S.nnz} non-zeros)")

    # Test that k_opt-1 is disconnected
    if k_opt > 1:
        S_prev = sparse_similarity.compute_sparse_similarity(
            data, k=k_opt - 1, metric="euclidean", batch_size=256, verbose=0
        )
        n_comp_prev, _ = sparse_similarity.check_connectivity(S_prev)
        print(f"Components at k_opt-1={k_opt - 1}: {n_comp_prev}")

    return k_opt


if __name__ == "__main__":
    print("PHATE Sparse Similarity Benchmarks\n")
    benchmark_sizes()
    benchmark_minimal_k()
