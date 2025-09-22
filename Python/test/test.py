#!/usr/bin/env python
# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

# Generating random fractal tree via DLA
from __future__ import print_function, division, absolute_import
import matplotlib

matplotlib.use("Agg")  # noqa

import os
import phate
import graphtools
import pygsp
import anndata
import numpy as np
from scipy.spatial.distance import pdist, squareform

import re

import pytest

import warnings

def test_simple():
    tree_data, tree_clusters = phate.tree.gen_dla(n_branch=3)
    phate_operator = phate.PHATE(knn=15, t=100, verbose=False)
    assert isinstance(phate_operator.__str__(), str)
    assert isinstance(phate_operator.__repr__(), str)
    tree_phate = phate_operator.fit_transform(tree_data)
    assert tree_phate.shape == (tree_data.shape[0], 2)
    clusters = phate.cluster.kmeans(phate_operator, n_clusters="auto")
    assert np.issubdtype(clusters.dtype, np.signedinteger)
    assert len(np.unique(clusters)) >= 2
    assert len(clusters.shape) == 1
    assert len(clusters) == tree_data.shape[0]
    clusters = phate.cluster.kmeans(phate_operator, n_clusters=3)
    assert np.issubdtype(clusters.dtype, np.signedinteger)
    assert len(np.unique(clusters)) == 3
    assert len(clusters.shape) == 1
    assert len(clusters) == tree_data.shape[0]
    phate_operator.fit(phate_operator.graph)
    G = graphtools.Graph(
        phate_operator.graph.kernel,
        precomputed="affinity",
        use_pygsp=True,
        verbose=False,
    )
    phate_operator.fit(G)
    G = pygsp.graphs.Graph(G.W)
    phate_operator.fit(G)
    phate_operator.fit(anndata.AnnData(tree_data))
    with pytest.raises(
        TypeError, match="Expected phate_op to be of type PHATE. Got 1"
    ):
        phate.cluster.kmeans(1)


def test_vne():
    X = np.eye(10)
    X[0, 0] = 5
    X[3, 2] = 4
    h = phate.vne.compute_von_neumann_entropy(X)
    assert phate.vne.find_knee_point(h) == 23
    x = np.arange(20)
    y = np.exp(-x / 10)
    assert phate.vne.find_knee_point(y, x) == 8


def test_tree():
    # generate DLA tree
    M, C = phate.tree.gen_dla(
        n_dim=50, n_branch=4, branch_length=50, rand_multiplier=2, seed=37, sigma=4
    )

    # instantiate phate_operator
    phate_operator = phate.PHATE(
        n_components=2,
        decay=10,
        knn=5,
        knn_max=15,
        t=30,
        mds="classic",
        knn_dist="euclidean",
        mds_dist="euclidean",
        n_jobs=-2,
        n_landmark=None,
        verbose=False,
    )
    phate_operator.fit(M)
    assert phate_operator.graph.knn == 5
    assert phate_operator.graph.knn_max == 15
    assert phate_operator.graph.decay == 10
    assert phate_operator.graph.n_jobs == -2
    assert phate_operator.graph.verbose == 0

    # run phate with classic MDS
    print("DLA tree, classic MDS")
    Y_cmds = phate_operator.fit_transform(M)
    assert Y_cmds.shape == (M.shape[0], 2)

    # run phate with metric MDS
    # change the MDS embedding without recalculating diffusion potential
    phate_operator.set_params(mds="metric")
    print("DLA tree, metric MDS (log)")
    Y_mmds = phate_operator.fit_transform(M)
    assert Y_mmds.shape == (M.shape[0], 2)

    # run phate with nonmetric MDS
    phate_operator.set_params(gamma=0)
    print("DLA tree, metric MDS (sqrt)")
    Y_sqrt = phate_operator.fit_transform(M)
    assert Y_sqrt.shape == (M.shape[0], 2)

    D = squareform(pdist(M))
    K = phate_operator.graph.kernel
    phate_operator.set_params(knn_dist="precomputed", random_state=42, verbose=False)
    phate_precomputed_D = phate_operator.fit_transform(D)
    phate_precomputed_K = phate_operator.fit_transform(K)

    phate_operator.set_params(knn_dist="precomputed_distance")
    phate_precomputed_distance = phate_operator.fit_transform(D)

    phate_operator.set_params(knn_dist="precomputed_affinity")
    phate_precomputed_affinity = phate_operator.fit_transform(K)

    np.testing.assert_allclose(
        phate_precomputed_K, phate_precomputed_affinity, atol=5e-4
    )
    np.testing.assert_allclose(
        phate_precomputed_D, phate_precomputed_distance, atol=5e-4
    )
    return 0

if __name__ == "__main__":
    pytest.main([__file__])