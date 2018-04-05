import phate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# generate DLA tree
M, C = phate.tree.gen_dla(n_dim=100, n_branch=10, branch_length=300,
                          rand_multiplier=2, seed=37, sigma=4)

# instantiate phate_operator
phate_operator = phate.PHATE(n_components=2, a=10, k=5, t=30, mds='classic',
                             knn_dist='euclidean', mds_dist='euclidean', njobs=-2,
                             n_landmark=None)

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
    n_components=2, a=10, t=30, k=5, mds='classic', mds_dist='euclidean',
    alpha_decay=True, n_landmark=1000)
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

f, axes = plt.subplots(2, 3)

(ax1, ax2, ax3) = axes[0]
f.set_size_inches(12, 8)
plt.setp((ax1, ax2, ax3), xticks=[], xticklabels=[], yticks=[])

# plotting PHATE - classic MDS
ax1.scatter(Y_cmds[:, 0], Y_cmds[:, 1], s=10, c=C)
ax1.set_xlabel("phate 1")
ax1.set_ylabel("phate 2")
ax1.set_title("PHATE embedding of DLA fractal tree\nClassic MDS")

# plotting PHATE - metric MDS
ax2.scatter(Y_mmds[:, 0], Y_mmds[:, 1], s=10, c=C)
ax2.set_xlabel("phate 1")
ax2.set_title("PHATE embedding of DLA fractal tree\nMetric MDS, log")

# plotting PHATE - nonmetric MDS
ax3.scatter(Y_sqrt[:, 0], Y_sqrt[:, 1], s=10, c=C)
ax3.set_xlabel("phate 1")
ax3.set_title("PHATE embedding of DLA fractal tree\nMetric MDS, sqrt")

(ax1, ax2, ax3) = axes[1]
f.set_size_inches(12, 8)
plt.setp((ax1, ax2, ax3), xticks=[], xticklabels=[], yticks=[])

# plotting PHATE - classic MDS
ax1.scatter(Y_cmds_fast[:, 0], Y_cmds_fast[:, 1], s=10, c=C)
ax1.set_xlabel("phate 1")
ax1.set_ylabel("phate 2")
ax1.set_title("PHATE embedding of DLA fractal tree\nFast Classic MDS")

# plotting PHATE - metric MDS
ax2.scatter(Y_mmds_fast[:, 0], Y_mmds_fast[:, 1], s=10, c=C)
ax2.set_xlabel("phate 1")
ax2.set_title("PHATE embedding of DLA fractal tree\nFast Metric MDS, log")

# plotting PHATE - nonmetric MDS
ax3.scatter(Y_sqrt_fast[:, 0], Y_sqrt_fast[:, 1], s=10, c=C)
ax3.set_xlabel("phate 1")
ax3.set_title("PHATE embedding of DLA fractal tree\nFast Metric MDS, sqrt")

plt.tight_layout()
plt.savefig("python_tree.png", dpi=100)

clusters = pd.read_csv("../../data/MAP.csv", header=None)
clusters.columns = pd.Index(['wells', 'clusters'])
bmmsc = pd.read_csv("../../data/BMMC_myeloid.csv.gz", index_col=0)

C = clusters['clusters']  # using cluster labels from original publication

# library_size_normalize performs L1 normalization on each cell
bmmsc_norm = phate.preprocessing.library_size_normalize(bmmsc)
bmmsc_reduced = phate.preprocessing.pca_reduce(bmmsc_norm, n_components=20)
phate_operator = phate.PHATE(
    n_components=2, t=24, a=200, k=10, mds='metric', mds_dist='euclidean',
    n_landmark=None)
print("BMMSC, exact PHATE")
Y_mmds = phate_operator.fit_transform(bmmsc_reduced)
phate_fast_operator = phate.PHATE(
    n_components=2, t=24, k=10, mds='metric', mds_dist='euclidean',
    n_landmark=1000)
print("BMMSC, fast PHATE")
Y_mmds_fast = phate_fast_operator.fit_transform(bmmsc_reduced)

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.setp((ax1, ax2), xticklabels=[], yticklabels=[])


# plotting PHATE - classic MDS
ax1.scatter(Y_mmds[:, 0], Y_mmds[:, 1], s=1, c=C, cmap="Paired")
ax1.set_xlabel("phate 1")
ax1.set_ylabel("phate 2")
ax1.set_title("PHATE embedding of 2730 BMMSCs\nExact PHATE")

ax2.scatter(Y_mmds_fast[:, 0], Y_mmds_fast[:, 1],
            s=1, c=C, cmap="Paired")
ax2.set_xlabel("phate 1")
ax2.set_ylabel("phate 2")
ax2.set_title("PHATE embedding of 2730 BMMSCs\nFast PHATE")

plt.gcf().set_size_inches(8, 8)
plt.tight_layout()
plt.savefig("python_bmmsc.png", dpi=100)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey='all')
ax1.plot(phate_operator.von_neumann_entropy(t_max=100))
ax1.set_xlabel("t")
ax1.set_ylabel("Von Neumann Entropy")
ax1.set_title("Optimal t selection on 2730 BMMSCs\nExact PHATE")
ax2.plot(np.arange(50) * 2, phate_fast_operator.von_neumann_entropy(t_max=100))
ax2.set_xlabel("t")
ax2.set_ylabel("Von Neumann Entropy")
ax2.set_title("Optimal t selection on 2730 BMMSCs\nFast PHATE")
plt.tight_layout()
plt.savefig("python_bmmsc_optimal_t.png", dpi=100)
