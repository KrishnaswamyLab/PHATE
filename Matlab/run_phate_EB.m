%% init
rseed = 7;

%% load embryoid body data
load('../data/EBdata.mat');
data = full(data);

%% lib size norm global
libsize = sum(data,2);
data = bsxfun(@rdivide, data, libsize) * median(libsize);

%% sqrt transform
data = sqrt(data);

%% PCA
tic;
Y_PCA = svdpca(data, 2, 'random');
toc

%% plot PCA 2D
figure;
scatter(Y_PCA(:,1), Y_PCA(:,2), 3, cells, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PCA1'
ylabel 'PCA2'
title 'PCA'
h = colorbar;
set(h,'xtick',1:5);
ylabel(h, 'time');

%% PHATE 2D
Y_PHATE_2D = phate(data, 't', 20);

%% plot PHATE 2D
figure;
scatter(Y_PHATE_2D(:,1), Y_PHATE_2D(:,2), 3, cells, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
title 'PHATE'
h = colorbar;
set(h,'xtick',1:5);
ylabel(h, 'time');

%% PHATE 3D
Y_PHATE_3D = phate(data, 't', 20, 'ndim', 3);

%% plot PHATE 3D
figure;
scatter3(Y_PHATE_3D(:,1), Y_PHATE_3D(:,2), Y_PHATE_3D(:,3), 3, cells, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
zlabel 'PHATE3'
title 'PHATE'
h = colorbar;
set(h,'xtick',1:5);
ylabel(h, 'time');
view([-170 15]);

%% tSNE -- slow!!!
% tic;
% Y_tSNE = tsne(data,'Theta',0.5,'NumPCAComponents',100,'Verbose',2, 'perplexity', 20);
% toc

%% plot combined
figure;

subplot(2,2,1);
scatter(Y_PCA(:,1), Y_PCA(:,2), 3, cells, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PCA1'
ylabel 'PCA2'
title 'PCA'
h = colorbar;
set(h,'xtick',1:5);
ylabel(h, 'time');

% subplot(2,2,2);
% scatter(Y_tSNE(:,1), Y_tSNE(:,2), 3, cells, 'filled');
% colormap(jet)
% set(gca,'xticklabel',[]);
% set(gca,'yticklabel',[]);
% axis tight
% xlabel 'tSNE1'
% ylabel 'tSNE2'
% title 'tSNE'
% h = colorbar;
% set(h,'xtick',1:5);
% ylabel(h, 'time');

subplot(2,2,3);
scatter(Y_PHATE_2D(:,1), Y_PHATE_2D(:,2), 3, cells, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
title 'PHATE'
h = colorbar;
set(h,'xtick',1:5);
ylabel(h, 'time');

subplot(2,2,4);
scatter3(Y_PHATE_3D(:,1), Y_PHATE_3D(:,2), Y_PHATE_3D(:,3), 3, cells, 'filled');
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
zlabel 'PHATE3'
title 'PHATE'
view([-170 15]);
h = colorbar;
set(h,'xtick',1:5);
ylabel(h, 'time');

