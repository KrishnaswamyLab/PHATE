# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

# Generating random fractal tree via DLA
from __future__ import print_function, division
import numpy as np
from scipy.io import loadmat

# random tree via diffusion limited aggregation


def gen_dla(n_dim=100, n_branch=20, branch_length=100,
            rand_multiplier=2, seed=37, sigma=4):
    np.random.seed(seed)
    M = np.cumsum(-1 + rand_multiplier *
                  np.random.rand(branch_length, n_dim), 0)
    for i in range(n_branch - 1):
        ind = np.random.randint(branch_length)
        new_branch = np.cumsum(-1 + rand_multiplier *
                               np.random.rand(branch_length, n_dim), 0)
        M = np.concatenate([M, new_branch + M[ind, :]])

    noise = np.random.normal(0, sigma, M.shape)
    M = M + noise

    # returns the group labels for each point to make it easier to visualize
    # embeddings
    C = np.array([i // branch_length for i in range(n_branch * branch_length)])

    return M, C


def artificial_tree():
    tree = loadmat("../../data/TreeData.mat")
    return tree['M'], tree['C']
