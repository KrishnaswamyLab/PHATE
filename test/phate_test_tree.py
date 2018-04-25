#!/usr/bin/env python3

# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from __future__ import print_function
import phate
import matplotlib.pyplot as plt
import sklearn.manifold
import sys


def main(argv=None):
    # generate DLA tree
    print("Running PHATE test on DLA tree...\n")
    M, C = phate.tree.gen_dla(n_dim=100, n_branch=20, branch_length=100,
                              n_drop=0, rand_multiplier=2, seed=37, sigma=4)

    # instantiate phate_operator
    phate_operator = phate.PHATE(n_components=2, a=10, k=5, t=30, mds='classic',
                                 knn_dist='euclidean', mds_dist='euclidean', njobs=-2)

    # run phate with classic MDS
    Y_cmds = phate_operator.fit_transform(M)

    # run phate with metric MDS
    # change the MDS embedding without recalculating diffusion potential
    phate_operator.reset_mds(mds="metric")
    Y_mmds = phate_operator.fit_transform(M)

    # run phate with nonmetric MDS
    phate_operator.reset_mds(mds="nonmetric")
    Y_nmmds = phate_operator.fit_transform(M)

    pca = phate.preprocessing.pca_reduce(M, n_components=2)
    tsne = sklearn.manifold.TSNE().fit_transform(M)

    f, axes = plt.subplots(2, 3)

    f.set_size_inches(12, 8)

    plt.setp(axes, xticks=[], xticklabels=[],
             yticks=[])

    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    # plotting PCA
    ax1.scatter(pca[:, 0], pca[:, 1], s=10, c=C)
    ax1.set_xlabel("PC 1")
    ax1.set_ylabel("PC 2")
    ax1.set_title("PCA")

    # plotting tSNE
    ax2.scatter(tsne[:, 0], tsne[:, 1], s=10, c=C)
    ax2.set_xlabel("tSNE 1")
    ax2.set_ylabel("tSNE 2")
    ax2.set_title("tSNE")

    # plotting PHATE - classic MDS
    ax4.scatter(Y_cmds[:, 0], Y_cmds[:, 1], s=10, c=C)
    ax4.set_xlabel("phate 1")
    ax4.set_ylabel("phate 2")
    ax4.set_title("PHATE embedding of DLA fractal tree\nClassic MDS")

    # plotting PHATE - metric MDS
    ax5.scatter(Y_mmds[:, 0], Y_mmds[:, 1], s=10, c=C)
    ax5.set_xlabel("phate 1")
    ax5.set_title("PHATE embedding of DLA fractal tree\nMetric MDS")

    # plotting PHATE - nonmetric MDS
    ax6.scatter(Y_nmmds[:, 0], Y_nmmds[:, 1], s=10, c=C)
    ax6.set_xlabel("phate 1")
    ax6.set_title("PHATE embedding of DLA fractal tree\nNonmetric MDS")

    plt.tight_layout()
    plt.savefig("./phate_tree_test.png", dpi=300)
    print("output saved in './phate_tree_test.png'")

if __name__ == '__main__':
    sys.exit(main())
