library(phater)
library(reticulate)
library(ggplot2)
pyphate <- reticulate::import("phate")

# generate DLA tree

tree <- pyphate$tree$gen_dla(n_dim=100, n_branch=10, branch_length=300, rand_multiplier=2, seed=37, sigma=4)
M <- tree[[1]]
C <- tree[[2]]

# run phate with classic MDS
print("DLA tree, classic MDS")
Y_cmds <- phate(M, ndim=2, alpha=10, k=5, t=30, mds.method='classic',
                knn.dist.method='euclidean', mds.dist.method='euclidean',
                n.landmark=NA)

# run phate with metric MDS
phate_operator.reset_mds(mds="metric")
print("DLA tree, metric MDS (log)")
Y_mmds <- phate(M, ndim=2, alpha=10, k=5, t=30, mds.method='metric',
                knn.dist.method='euclidean', mds.dist.method='euclidean',
                n.landmark=NA, diff.op.t=Y_cmds$diff.op.t)

# run phate with nonmetric MDS
print("DLA tree, metric MDS (sqrt)")
Y_sqrt <- phate(M, ndim=2, alpha=10, k=5, t=30, mds.method='metric',
                knn.dist.method='euclidean', mds.dist.method='euclidean',
                n.landmark=NA, potential.method='sqrt', diff.op=Y_cmds$diff.op)

# run phate with classic MDS
print("DLA tree, classic MDS")
Y_cmds_fast <- phate(M, ndim=2, alpha=10, k=5, t=30, mds.method='classic',
                knn.dist.method='euclidean', mds.dist.method='euclidean',
                n.landmark=1000)

# run phate with metric MDS
print("DLA tree, metric MDS (log)")
Y_mmds_fast <- phate(M, ndim=2, alpha=10, k=5, t=30, mds.method='metric',
                knn.dist.method='euclidean', mds.dist.method='euclidean',
                n.landmark=1000, diff.op.t=Y_cmds$diff.op.t)

# run phate with nonmetric MDS
print("DLA tree, metric MDS (sqrt)")
Y_sqrt_fast <- phate(M, ndim=2, alpha=10, k=5, t=30, mds.method='metric',
                knn.dist.method='euclidean', mds.dist.method='euclidean',
                n.landmark=1000, potential.method='sqrt', diff.op=Y_cmds$diff.op)

fig, axes = plt.subplots(2, 3)

(ax1, ax2, ax3) = axes[0]
fig.set_size_inches(12, 8)
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
plt.savefig("python{}_tree.png".format(python_version), dpi=100)

clusters = pd.read_csv("../../data/MAP.csv", header=None)
clusters.columns = pd.Index(['wells', 'clusters'])
bmmsc = pd.read_csv("../../data/BMMC_myeloid.csv.gz", index_col=0)

C = clusters['clusters']  # using cluster labels from original publication

# library_size_normalize performs L1 normalization on each cell
bmmsc_norm = phate.preprocessing.library_size_normalize(bmmsc)
bmmsc_reduced = phate.preprocessing.pca_reduce(bmmsc_norm, n_components=20)
phate_operator = phate.PHATE(
  n_components=2, t='auto', a=200, k=10, mds='metric', mds_dist='euclidean',
  n_landmark=None)
phate_fast_operator = phate.PHATE(
  n_components=2, t='auto', k=10, mds='metric', mds_dist='euclidean',
  n_landmark=1000)


fig, (ax1, ax2) = plt.subplots(1, 2, sharey='all')
print("BMMSC, exact PHATE")
Y_mmds = phate_operator.fit_transform(bmmsc_reduced, t_max=100,
                                      plot_optimal_t=True, ax=ax1)
ax1.set_title("Optimal t selection on 2730 BMMSCs\n"
              "Exact PHATE, t = {}".format(phate_operator.t))
print("BMMSC, fast PHATE")
Y_mmds_fast = phate_fast_operator.fit_transform(bmmsc_reduced, t_max=100,
                                                plot_optimal_t=True, ax=ax2)
ax2.set_title("Optimal t selection on 2730 BMMSCs\n"
              "Fast PHATE, t = {}".format(phate_fast_operator.t))
plt.tight_layout()
plt.savefig("python{}_bmmsc_optimal_t.png".format(python_version), dpi=100)


fig, (ax1, ax2) = plt.subplots(1, 2)
plt.setp((ax1, ax2), xticklabels=[], yticklabels=[])

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
plt.savefig("python{}_bmmsc.png".format(python_version), dpi=100)
