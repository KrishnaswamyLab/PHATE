library(phateR)
library(ggplot2)
library(gridGraphics)
library(cowplot)

# generate DLA tree

data(tree.data)
M <- tree.data$data
C <- tree.data$branches

# run phate with classic MDS
print("DLA tree, classic MDS")
Y_cmds <- phate(M, ndim=2, alpha=10, k=5, t=30, mds.method='classic',
                knn.dist.method='euclidean', mds.dist.method='euclidean',
                n.landmark=NA)

# run phate with metric MDS
print("DLA tree, metric MDS (log)")
Y_mmds <- phate(M, ndim=2, alpha=10, k=5, t=30, mds.method='metric',
                knn.dist.method='euclidean', mds.dist.method='euclidean',
                n.landmark=NA, init=Y_cmds)

# run phate with nonmetric MDS
print("DLA tree, metric MDS (sqrt)")
Y_sqrt <- phate(M, ndim=2, alpha=10, k=5, t=30, mds.method='metric',
                knn.dist.method='euclidean', mds.dist.method='euclidean',
                n.landmark=NA, potential.method='sqrt', init=Y_mmds)

# run phate with classic MDS
print("DLA tree, fast classic MDS")
Y_cmds_fast <- phate(M, ndim=2, alpha=10, k=5, t=90, mds.method='classic',
                knn.dist.method='euclidean', mds.dist.method='euclidean',
                n.landmark=1000)

# run phate with metric MDS
print("DLA tree, fast metric MDS (log)")
Y_mmds_fast <- phate(M, ndim=2, alpha=10, k=5, t=90, mds.method='metric',
                knn.dist.method='euclidean', mds.dist.method='euclidean',
                n.landmark=1000, init=Y_cmds_fast)

# run phate with nonmetric MDS
print("DLA tree, fast metric MDS (sqrt)")
Y_sqrt_fast <- phate(M, ndim=2, alpha=10, k=5, t=90, mds.method='metric',
                knn.dist.method='euclidean', mds.dist.method='euclidean',
                n.landmark=1000, potential.method='sqrt', init=Y_mmds_fast)

p <- plot_grid(ggplot(Y_cmds) + 
            geom_point(aes(PHATE1, PHATE2, color=C), show.legend=FALSE) + 
            labs(title="PHATE embedding of DLA fractal tree", 
                 subtitle="Classic MDS"),
          ggplot(Y_mmds) + 
            geom_point(aes(PHATE1, PHATE2, color=C), show.legend=FALSE) + 
            labs(title="PHATE embedding of DLA fractal tree", 
                 subtitle="Metric MDS, log"),
          ggplot(Y_sqrt) + 
            geom_point(aes(PHATE1, PHATE2, color=C), show.legend=FALSE) + 
            labs(title="PHATE embedding of DLA fractal tree", 
                 subtitle="Metric MDS, sqrt"),
          ggplot(Y_cmds_fast) + 
            geom_point(aes(PHATE1, PHATE2, color=C), show.legend=FALSE) + 
            labs(title="PHATE embedding of DLA fractal tree", 
                 subtitle="Fast classic MDS"),
          ggplot(Y_mmds_fast) + 
            geom_point(aes(PHATE1, PHATE2, color=C), show.legend=FALSE) + 
            labs(title="PHATE embedding of DLA fractal tree", 
                 subtitle="Fast metric MDS, log"),
          ggplot(Y_sqrt_fast) + 
            geom_point(aes(PHATE1, PHATE2, color=C), show.legend=FALSE) + 
            labs(title="PHATE embedding of DLA fractal tree", 
                 subtitle="Fast metric MDS, sqrt"),
          ncol=3)
save_plot("R_tree.png", p, base_height = 6, base_width=12)

clusters <- read.csv("../data/MAP.csv", row.names=NULL, header=FALSE,
                     col.names=c('wells', 'clusters'))
bmmsc <- read.csv("../data/BMMC_myeloid.csv.gz", row.names=NULL)
bmmsc <- bmmsc[,2:ncol(bmmsc)]

C <- as.factor(clusters$clusters)  # using cluster labels from original publication

# library.size.normalize performs L1 normalization on each cell
bmmsc_norm <- library.size.normalize(bmmsc)

png("tmp.png")
dev.control(displaylist="enable") 
print("BMMSC, exact PHATE")
Y_mmds <- phate(bmmsc_norm, ndim=2, t='auto', a=200, k=10, 
                mds.method='metric', mds.dist.method='euclidean',
                n.landmark=NA, plot.optimal.t = TRUE)
title(main="\n\nExact PHATE on 2730 BMMSCs")
t_exact_p <- recordPlot()
dev.off()
png("tmp.png")
dev.control(displaylist="enable")  
print("BMMSC, fast PHATE")
Y_mmds_fast <- phate(bmmsc_norm, ndim=2, t='auto', a=NA, k=10, 
                     mds.method='metric', mds.dist.method='euclidean',
                     n.landmark=1000, plot.optimal.t = TRUE)
title(main="\n\nFast PHATE on 2730 BMMSCs")
t_fast_p <- recordPlot()
dev.off()
p <- plot_grid(t_exact_p, t_fast_p, ncol=2)
save_plot("R_bmmsc_optimal_t.png", p, base_height=6, base_width=8)
file.remove("tmp.png")

p <- plot_grid(ggplot(Y_mmds) + 
                 geom_point(aes(PHATE1, PHATE2, color=C), show.legend=FALSE) + 
                 labs(title="PHATE embedding of 2730 BMMSCs", 
                      subtitle="Exact PHATE"),
               ggplot(Y_mmds_fast) + 
                 geom_point(aes(PHATE1, PHATE2, color=C), show.legend=FALSE) + 
                 labs(title="PHATE embedding of 2730 BMMSCs", 
                      subtitle="Fast PHATE"),
               ncol=2)
save_plot("R_bmmsc.png", p, base_height=8, base_width=16)
