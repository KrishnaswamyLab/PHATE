%% init
rseed = 8;

%% Load tree data generated in GenerateTree.m. See paper for details in how this data is generated
load TreeData.mat

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
tic;
Y_PHATE_2D = phate(M, 'pot_method', 'sqrt', 't', 44);
toc

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
tic;
Y_PHATE_3D = phate(M, 'pot_method', 'sqrt', 'ndim', 3, 't', 44);
toc

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
view([-15 20]);

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
view([-15 20]);

