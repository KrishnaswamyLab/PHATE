#!/usr/bin/env python
# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

# Generating random fractal tree via DLA
from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use("Agg")  # noqa
import scprep
import nose2
import phate
import graphtools
import pygsp
import anndata
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.testing import assert_warns_message


def test_simple():
    tree_data, tree_clusters = phate.tree.gen_dla(n_branch=3)
    phate_operator = phate.PHATE(k=15, t=100)
    tree_phate = phate_operator.fit_transform(tree_data)
    assert tree_phate.shape == (tree_data.shape[0], 2)
    clusters = phate.cluster.kmeans(phate_operator, k=3)
    assert np.issubdtype(clusters.dtype, int)
    assert len(clusters.shape) == 1
    assert len(clusters) == tree_data.shape[0]
    phate_operator.fit(phate_operator.graph)
    G = graphtools.Graph(phate_operator.graph.kernel,
                         precomputed='affinity', use_pygsp=True)
    phate_operator.fit(G)
    G = pygsp.graphs.Graph(G.W)
    phate_operator.fit(G)
    phate_operator.fit(anndata.AnnData(tree_data))


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
    M, C = phate.tree.gen_dla(n_dim=50, n_branch=4, branch_length=50,
                              rand_multiplier=2, seed=37, sigma=4)

    # instantiate phate_operator
    phate_operator = phate.PHATE(n_components=2, a=10, k=5, t=30, mds='classic',
                                 knn_dist='euclidean', mds_dist='euclidean',
                                 n_jobs=-2, n_landmark=None)

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
    phate_operator.set_params(knn_dist="precomputed", random_state=42)
    phate_precomputed_D = phate_operator.fit_transform(D)
    phate_precomputed_K = phate_operator.fit_transform(K)

    phate_operator.set_params(knn_dist="precomputed_distance")
    phate_precomputed_distance = phate_operator.fit_transform(D)

    phate_operator.set_params(knn_dist="precomputed_affinity")
    phate_precomputed_affinity = phate_operator.fit_transform(K)

    np.testing.assert_allclose(
        phate_precomputed_K, phate_precomputed_affinity, atol=5e-4)
    np.testing.assert_allclose(
        phate_precomputed_D, phate_precomputed_distance, atol=5e-4)
    return 0


def test_bmmsc():
    clusters = scprep.io.load_csv("../data/MAP.csv",
                                  gene_names=['clusters'])
    bmmsc = scprep.io.load_csv("../data/BMMC_myeloid.csv.gz")

    C = clusters['clusters']  # using cluster labels from original publication

    # library_size_normalize performs L1 normalization on each cell
    bmmsc_norm = scprep.normalize.library_size_normalize(bmmsc)
    bmmsc_norm = scprep.transform.sqrt(bmmsc_norm)
    phate_fast_operator = phate.PHATE(
        n_components=2, t='auto', a=200, k=10, mds='metric', mds_dist='euclidean',
        n_landmark=1000)

    print("BMMSC, fast PHATE")
    Y_mmds_fast = phate_fast_operator.fit_transform(bmmsc_norm, t_max=100)
    assert Y_mmds_fast.shape == (bmmsc_norm.shape[0], 2)
    return 0


def test_plot():
    tree_data, tree_clusters = phate.tree.gen_dla()
    # scatter
    assert_warns_message(FutureWarning,
                         "`phate.plot.scatter` is deprecated. "
                         "Use `scprep.plot.scatter` instead.",
                         phate.plot.scatter, tree_data[:, 0], tree_data[:, 1],
                         c=tree_clusters, discrete=True)
    # scatter2d
    assert_warns_message(FutureWarning,
                         "`phate.plot.scatter2d` is deprecated. "
                         "Use `scprep.plot.scatter2d` instead.",
                         phate.plot.scatter2d, tree_data,
                         c=tree_clusters, discrete=True)
    # scatter3d
    assert_warns_message(FutureWarning,
                         "`phate.plot.scatter3d` is deprecated. "
                         "Use `scprep.plot.scatter3d` instead.",
                         phate.plot.scatter3d, tree_data,
                         c=tree_clusters, discrete=False)
    # rotate_scatter3d
    assert_warns_message(FutureWarning,
                         "`phate.plot.rotate_scatter3d` is deprecated. "
                         "Use `scprep.plot.rotate_scatter3d` instead.",
                         phate.plot.rotate_scatter3d, tree_data,
                         c=tree_clusters, discrete=False)


if __name__ == "__main__":
    exit(nose2.run())
