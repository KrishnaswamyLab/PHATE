%% init
rseed = 8;

%% generate tree data
n_samp = 3000; % number of points
n_dim = 100; % number of dimensions
n_branch = 10; % number of branches
sigma = 5; % noise
rng(rseed)
[M, C] = dla_tree(n_samp, n_dim, n_branch, sigma);

%% PCA
Y_PCA = svdpca(M, 2, 'random');

%% plot PCA
figure;
scatter(Y_PCA(:,1), Y_PCA(:,2), 10, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PCA1'
ylabel 'PCA2'

%% PHATE 2D
Y_PHATE_2D = phate(M, 't', 20, 'gamma', 0);

%% plot PHATE 2D
figure;
scatter(Y_PHATE_2D(:,1), Y_PHATE_2D(:,2), 10, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'

%% PHATE 3D
Y_PHATE_3D = phate(M, 'ndim', 3, 't', 20);

%% plot PHATE 3D
figure;
scatter3(Y_PHATE_3D(:,1), Y_PHATE_3D(:,2), Y_PHATE_3D(:,3), 10, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
zlabel 'PHATE3'

%% tSNE
tic;
Y_tSNE = tsne(M,'Theta',0.5,'Verbose',2, 'perplexity', 20);
toc

%% plot tSNE
figure;
scatter(Y_tSNE(:,1), Y_tSNE(:,2), 10, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'tSNE1'
ylabel 'tSNE2'

%% plot combined
figure;

subplot(2,2,1);
scatter(Y_PCA(:,1), Y_PCA(:,2), 10, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PCA1'
ylabel 'PCA2'
title 'PCA'

subplot(2,2,2);
scatter(Y_tSNE(:,1), Y_tSNE(:,2), 10, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'tSNE1'
ylabel 'tSNE2'
title 'tSNE'

subplot(2,2,3);
scatter(Y_PHATE_2D(:,1), Y_PHATE_2D(:,2), 10, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
title 'PHATE'

subplot(2,2,4);
scatter3(Y_PHATE_3D(:,1), Y_PHATE_3D(:,2), Y_PHATE_3D(:,3), 10, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
zlabel 'PHATE3'
title 'PHATE'


