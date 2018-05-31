#!/usr/bin/env python
# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

# Generating random fractal tree via DLA
from __future__ import print_function, division, absolute_import
import doctest
import nose2
import phate
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def test_docs():
    doctest.testmod()


def test_simple():
    tree_data, tree_clusters = phate.tree.gen_dla()
    phate_operator = phate.PHATE(k=15, t=100)
    tree_phate = phate_operator.fit_transform(tree_data)


def test_tree():
    # generate DLA tree
    M, C = phate.tree.gen_dla(n_dim=100, n_branch=10, branch_length=300,
                              rand_multiplier=2, seed=37, sigma=4)

    # instantiate phate_operator
    phate_operator = phate.PHATE(n_components=2, a=10, k=5, t=30, mds='classic',
                                 knn_dist='euclidean', mds_dist='euclidean',
                                 njobs=-2, n_landmark=None)

    # run phate with classic MDS
    print("DLA tree, classic MDS")
    Y_cmds = phate_operator.fit_transform(M)

    # run phate with metric MDS
    # change the MDS embedding without recalculating diffusion potential
    phate_operator.reset_mds(mds="metric")
    print("DLA tree, metric MDS (log)")
    Y_mmds = phate_operator.fit_transform(M)

    # run phate with nonmetric MDS
    phate_operator.reset_potential(potential_method="sqrt")
    print("DLA tree, metric MDS (sqrt)")
    Y_sqrt = phate_operator.fit_transform(M)

    phate_fast_operator = phate.PHATE(
        n_components=2, a=10, t=90, k=5, mds='classic', mds_dist='euclidean',
        n_landmark=1000)
    # run phate with classic MDS
    print("DLA tree, fast classic MDS")
    Y_cmds_fast = phate_fast_operator.fit_transform(M)

    # run phate with metric MDS
    # change the MDS embedding without recalculating diffusion potential
    phate_fast_operator.reset_mds(mds="metric")
    print("DLA tree, fast metric MDS (log)")
    Y_mmds_fast = phate_fast_operator.fit_transform(M)

    # run phate with nonmetric MDS
    phate_fast_operator.reset_potential(potential_method="sqrt")
    print("DLA tree, fast metric MDS (sqrt)")
    Y_sqrt_fast = phate_fast_operator.fit_transform(M)
    return 0


def test_bmmsc():
    clusters = pd.read_csv("../data/MAP.csv", header=None)
    clusters.columns = pd.Index(['wells', 'clusters'])
    bmmsc = pd.read_csv("../data/BMMC_myeloid.csv.gz", index_col=0)

    C = clusters['clusters']  # using cluster labels from original publication

    # library_size_normalize performs L1 normalization on each cell
    bmmsc_norm = phate.preprocessing.library_size_normalize(bmmsc)
    phate_operator = phate.PHATE(
        n_components=2, t='auto', a=200, k=10, mds='metric', mds_dist='euclidean',
        n_landmark=None)
    phate_fast_operator = phate.PHATE(
        n_components=2, t='auto', a=200, k=10, mds='metric', mds_dist='euclidean',
        n_landmark=1000)

    print("BMMSC, exact PHATE")
    Y_mmds = phate_operator.fit_transform(bmmsc_norm, t_max=100)
    print("BMMSC, fast PHATE")
    return 0


if __name__ == "__main__":
    exit(nose2.run())
