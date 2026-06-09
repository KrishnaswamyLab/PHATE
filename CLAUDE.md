# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding) is a dimensionality reduction / manifold learning tool for high-dimensional biomedical data. The repo is a monorepo with Python, MATLAB, and R implementations. The active Python package lives in `Python/`.

## Commands

All commands run from the `Python/` directory:

```bash
# Install in editable mode with test dependencies (uses uv)
uv pip install -e ".[test]"

# Run all tests
pytest -v

# Run a single test file
pytest test/test_phate.py -v

# Run a single test function
pytest test/test_phate.py::test_phate_basic_workflow -v

# Run tests with coverage
pytest --cov=phate --cov-report=term -v

# Lint (CI uses these)
black . --check --diff
flake8 phate

# Build documentation
cd doc && make html

# Build distribution
python -m build --sdist --wheel

# Run sparse PHATE benchmarks
python benchmarks/benchmark_sparse.py
```

## Architecture

**PHATE pipeline** (`phate/phate.py`): The main class is a scikit-learn `BaseEstimator`. The pipeline runs in `fit_transform()`:

*Traditional path (default):*
1. Build a kNN graph with alpha-decaying kernel via `graphtools.Graph`
2. Compute the diffusion operator from the graph (densified via `.toarray()`)
3. Power the diffusion operator to time `t` (auto-selected via Von Neumann Entropy knee point, or fixed)
4. Apply potential transform: log (`gamma=1`), sqrt (`gamma=0`), or interpolated (`-1 < gamma < 1`)
5. Embed with MDS (classic, metric, or nonmetric)

*Sparse path (when `sparse_k` is set):*
1. Build sparse similarity matrix via batched top-k computation (`phate/sparse_similarity.py`) — avoids materializing N×N dense matrices
2. Row-normalize into sparse diffusion operator (kept as `scipy.sparse.csr_matrix` throughout)
3. Power with sparse matrix exponentiation (`**t`); densify only if >30% fill
4. Apply potential transform element-wise on sparse data
5. Embed with MDS (densifies at this final step)

**Sparse similarity** (`phate/sparse_similarity.py`): Memory-efficient similarity computation that processes data in batches. For each batch of rows, computes pairwise metric against ALL columns, then retains only the top-k entries per row (smallest for distance metrics, largest for similarity metrics like cosine/correlation). The result is symmetrized: `(S + S.T) / 2`. Includes:
- `compute_sparse_similarity()` — batched construction with batch_size controlling peak memory (~batch_size × N)
- `compute_sparse_diffusion_operator()` — sparse row-normalization
- `check_connectivity()` — uses `scipy.sparse.csgraph.connected_components`
- `find_minimal_k()` — binary search for minimal k achieving single connected component

**MDS** (`phate/mds.py`): Routes to classic MDS (randomized SVD PCA), SMACOF (sklearn), or SGD. The SGD solver (`phate/sgd_mds.py`) is 7-10x faster than SMACOF with nearly identical output quality. `embed_MDS()` is the central dispatch function. Handles sparse input by densifying at entry point.

**Von Neumann Entropy** (`phate/vne.py`): Computes VNE of the diffusion operator across t values and finds the knee point to auto-select optimal diffusion time `t`. Sparse variant uses `scipy.sparse.linalg.eigsh` for approximate eigenvalue decomposition.

**Supporting modules:**
- `phate/tree.py` — Generates synthetic fractal tree data via Diffusion-Limited Aggregation (used in tests and tutorials)
- `phate/plot.py` — `scatter2d`/`scatter3d` convenience functions accepting PHATE operators, numpy arrays, or AnnData objects
- `phate/cluster.py` — KMeans clustering and silhouette scoring on the PHATE diffusion potential
- `phate/utils.py` — Parameter validation helpers (`check_positive`, `check_int`, `check_in`, `check_between`, `matrix_is_equivalent`)
- `phate/version.py` — Single-source version string (`__version__ = "2.0.0"`)

## Key dependencies

`graphtools` (>=2.1.0) is the most critical external dependency — it builds the kNN graph and diffusion operator. The code has version-gated features (`random_landmarking`, `is_connected` check) that require graphtools >= 2.1.0. `anndata` and `pygsp` are optional. When using the sparse path (`sparse_k`), graphtools is not required for graph construction.

## CI

GitHub Actions (`.github/workflows/run_tests.yml`) runs linting (black + flake8) on Python 3.12 and tests on Python 3.9–3.13 with coverage via Coveralls. Docs are built on ReadTheDocs per `.readthedocs.yml`, sourcing from `Python/doc/`.
