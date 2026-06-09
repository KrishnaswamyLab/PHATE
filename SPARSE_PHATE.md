# Sparse PHATE

Batched top-k similarity computation with connectivity guarantee. Avoids
materializing the dense N×N distance matrix — the primary memory
bottleneck in standard PHATE.

## Problem

Standard PHATE densifies the diffusion operator via `.toarray()` at
`phate.py:322`, creating an N×N dense float64 matrix. For N=50,000 this
is 20 GB. The MDS step additionally computes a dense N×N distance matrix
via `squareform(pdist())`.

## Solution

Compute similarities in batches, retain only top-k entries per sample,
keep the diffusion operator as `scipy.sparse.csr_matrix` throughout the
pipeline, and densify only at the final MDS step.

```
Traditional:  data → graphtools.Graph (kNN + kernel) → .toarray() [DENSE]
              → np.linalg.matrix_power(dense)
              → VNE via dense SVD
              → MDS on dense N×N matrix

Sparse:       data → batched cdist(batch, X) → keep top-k → exp(-d/decay) [SPARSE]
              → D⁻¹S row-normalization [SPARSE]
              → sparse ** t [SPARSE until >30% fill]
              → VNE via scipy.sparse.linalg.eigsh
              → MDS on dense (final step only)
```

## Files

| File | Status | Description |
|------|--------|-------------|
| `Python/phate/sparse_similarity.py` | new | Batched top-k similarity, connectivity check, binary search |
| `Python/benchmarks/benchmark_sparse.py` | new | Memory/time/quality comparison script |
| `Python/phate/phate.py` | modified | `sparse_k`, `sparse_metric`, `sparse_batch_size` params; sparse fit/transform path |
| `Python/phate/vne.py` | modified | `compute_von_neumann_entropy_sparse()` via `eigsh` |
| `Python/phate/mds.py` | modified | Handle sparse CSR input to `embed_MDS()` |
| `Python/doc/source/api.rst` | modified | Sparse similarity API docs |
| `CLAUDE.md` | new | Repo guide for Claude Code |

## Mechanism

### `compute_sparse_similarity(X, k, metric, decay, batch_size)`

1. For each batch of `batch_size` rows: compute `cdist(batch, X, metric)` →
   a `(batch_size × N)` dense block (small, e.g. 256×20K = 40 MB)
2. For each row in the batch: select the `k` smallest distances (or `k`
   largest similarities for cosine/correlation)
3. Apply alpha-decaying kernel: `exp(-distance / decay)` to the selected
   entries only. All other entries = 0 (kernel of infinite distance = 0)
4. Accumulate COO triples, build CSR, symmetrize: `(S + S.T) / 2`

Peak memory: O(batch_size × N) for the pairwise block + O(N × k) for the
sparse result. For N=20,000, batch_size=256, float64: ~40 MB peak.

### `compute_sparse_diffusion_operator(S)`

Row-normalize to a Markov transition matrix: `P = D⁻¹S` where
`D_ii = Σⱼ S_ij`. All row sums = 1.0. Uses sparse diags + matmul,
no densification.

### `find_minimal_k(X, k_max, k_min, metric, decay)`

Binary search for the smallest `k` that yields a single connected
component. Each iteration builds a sparse similarity matrix and runs
`scipy.sparse.csgraph.connected_components` (O(N×k)).

## Benchmarks

MacBook Pro M-series, 24 GB RAM. Synthetic tree data (100-dim features).

| N | Mode | Time | Memory | Sparsity | Quality |
|---|------|------|--------|----------|---------|
| 500 | original | 1.25s | 64 MB | — | — |
| 500 | **sparse** | **0.60s** | **2.7 MB** | 2.76% | 0.0032 |
| 1000 | original | 1.68s | 43 MB | — | — |
| 1000 | **sparse** | **1.44s** | **8.0 MB** | 1.39% | 0.0026 |
| 2000 | original | 10.33s | 174 MB | — | — |
| 2000 | **sparse** | **7.67s** | **31 MB** | 0.69% | 0.0028 |
| 3000 | **sparse** | 23.9s | 408 MB | 0.47% | — |
| 5000 | **sparse** | 114.8s | 723 MB | 0.28% | — |

Quality: Procrustes normalized disparity vs original. Lower is better
(0 = identical). All ≤ 0.003 — near-identical embeddings.

Minimal k for connectivity (N=500): k=3, found in 0.06s.
At k=2: 75 disconnected components.

## Usage

```python
import phate

# Sparse path with explicit k
phate_op = phate.PHATE(
    sparse_k=100,                # top-k entries per sample
    sparse_metric="correlation", # "euclidean", "cosine", "correlation"
    t=20,
    verbose=False,
    random_state=42,
)
embedding = phate_op.fit_transform(data)

# Auto-find minimal k that guarantees connectivity
from phate.sparse_similarity import find_minimal_k

k_opt, S = find_minimal_k(
    data,
    k_max=1000,
    metric="correlation",
    decay=40,
)
print(f"Minimal k for connectivity: {k_opt}")

# Use the found k
phate_op = phate.PHATE(
    sparse_k=k_opt,
    sparse_metric="correlation",
    t="auto",
)
embedding = phate_op.fit_transform(data)
```

### Choosing k

- **distance metrics** (euclidean, cityblock): smaller k = sparser, faster. Too
  small → disconnected graph. Use `find_minimal_k()` to auto-select.
- **similarity metrics** (correlation, cosine): k is the number of most-similar
  entries kept per sample. The kernel formula does not apply to these
  metrics (they are already in [0,1] range).
- Rule of thumb: start with `k = max(10, int(sqrt(N)))`, then binary-search
  down to the minimal connected k.

## Scaling

Memory ratio sparse/original converges to **~1/3** — sparse uses ~3× less
peak memory regardless of N because it eliminates 2 of 3 dense N×N matrices.

| Phase | Original | Sparse |
|-------|----------|--------|
| Graph / similarity | N×N dense (`.toarray()`) | N×k sparse CSR |
| Diffusion power | dense `matrix_power` | sparse `**t` |
| MDS distance | N×N dense | N×N dense |

Both need an N×N dense matrix at the final MDS step (a shared floor).
Original needs 2 additional dense N×N matrices before that.

| N | Original Peak | Sparse Peak | Ratio | Savings |
|---|---------------|-------------|-------|---------|
| 2,000 | 96 MB | 36 MB | 0.38 | 62% |
| 5,000 | 600 MB | 211 MB | 0.35 | 65% |
| 10,000 | 2.4 GB | 822 MB | 0.34 | 66% |
| 20,000 | 9.6 GB | 3.2 GB | 0.34 | 66% |
| 50,000 | 60 GB | 20 GB | 0.33 | 67% |

The sparse similarity storage alone scales as **O(k/N)** — at N=10,000,
k=10 it's 1.2 MB vs 800 MB for a dense matrix (0.15%).

Quality metric: Procrustes normalized disparity. 0 = identical embeddings,
1 = maximally different. Rigidly aligns (rotates/reflects) the sparse
embedding to the original, then computes residual sum-of-squares divided
by total variance. Values ≤ 0.003 mean near-identical manifold structure.

## Reproduce the comparison plot

```bash
cd Python
python -c "
import numpy as np, phate, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data, clusters = phate.tree.gen_dla(n_dim=100, n_branch=10, branch_length=100, seed=42)

# Sparse PHATE
phate_op = phate.PHATE(sparse_k=50, sparse_metric='euclidean', t=30,
                       mds_solver='smacof', verbose=False, random_state=42)
emb_s = phate_op.fit_transform(data)

# Original PHATE
phate_op2 = phate.PHATE(knn=5, t=30, mds_solver='smacof',
                        verbose=False, random_state=42)
emb_o = phate_op2.fit_transform(data)

# Plot side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.scatter(emb_s[:,0], emb_s[:,1], c=clusters, s=3, cmap='tab10')
ax1.set_title('Sparse PHATE (k=50, smacof)')
ax2.scatter(emb_o[:,0], emb_o[:,1], c=clusters, s=3, cmap='tab10')
ax2.set_title('Original PHATE (knn=5, smacof)')
for ax in (ax1, ax2):
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig('sparse_vs_original.png', dpi=120, bbox_inches='tight')
print('Saved: sparse_vs_original.png')
"
```

## Installation

```bash
cd Python
uv pip install -e ".[test]"
pytest -v                           # 118 tests

# Wheel
python -m build --wheel             # → dist/phate-2.0.0-py3-none-any.whl
```
