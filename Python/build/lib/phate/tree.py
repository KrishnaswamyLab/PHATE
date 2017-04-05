### Generating random fractal tree via DLA
import numpy as np

def gen_dla(n_dim = 100, n_branch = 20, branch_length = 100,n_drop = 0, rand_multiplier = 2, seed=37, sigma = 4):

    np.random.seed(37)
    M = np.cumsum(-1 + rand_multiplier*np.random.rand(branch_length,n_dim),0)
    for i in range(n_branch-1):
        ind = np.random.randint(branch_length)
        new_branch = np.cumsum(-1 + rand_multiplier*np.random.rand(branch_length,n_dim),0)
        M = np.concatenate([M,new_branch+M[ind,:]])

    noise = np.random.normal(0, sigma,M.shape)
    M = M + noise

    C = np.array([i//n_branch for i in range(n_branch*branch_length)]) #returns the group labels for each point to make it easier to visualize embeddings

    return M, C
