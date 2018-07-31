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







%% 1D PHATE per t
t_vec = 1:64;
pc = svdpca(data, 100, 'random');
Y_mat = nan(length(t_vec), size(pc,1));
for I=1:length(t_vec)
    I
    t_vec(I)
    Y_PHATE_1D = phate(pc, 't', t_vec(I), 'ndim', 1, 'npca', [], 'a', [], 'k', 15, 'mds_method', 'cmds');
    if I>1
        [~,Y_PHATE_1D] = procrustes(Y_mat(1,:)', Y_PHATE_1D);
    end
    Y_mat(I,:) = Y_PHATE_1D;
end

%% plot
figure;
imagesc(Y_mat);

%% get 2D hist
nbins = 100;
edges = linspace(min(Y_mat(:)), max(Y_mat(:)), nbins+1);
N_mat = nan(length(t_vec), nbins);
for I=1:size(Y_mat,1)
    N = histcounts(Y_mat(I,:), edges);
    N_mat(I,:) = N; % ./ max(N);
end

%% plot
figure;
imagesc(N_mat);

%%
figure;
hold on;
for I=1:size(Y_mat,1)
    scatter(Y_mat(I,:), repmat(I,1,size(Y_mat,2)) + randn(1,size(Y_mat,2))*0.25, 1, 'k', 'filled');
end
axis tight
set(gca,'xtick',[]);
set(gca,'ytick',1:length(t_vec));
set(gca,'yticklabel',t_vec);
xlabel '1D PHATE'
ylabel 't'


%% 1D phate cell X pca heatmap
pc = svdpca(data, 100, 'random');
Y_PHATE_1D = phate(pc, 't', 17, 'ndim', 1, 'npca', [], 'a', [], 'k', 15);
[~,idx_1d] = sort(Y_PHATE_1D);
Y_PHATE_10D = phate(pc, 't', 17, 'ndim', 10, 'npca', [], 'a', [], 'k', 15);

%% plot
figure;
M = Y_PHATE_10D(idx,:);
M = zscore(M,[],2);
imagesc(M);

%% high D phate
Y_PHATE_6D = phate(pc, 't', 17, 'ndim', 20, 'npca', [], 'a', [], 'k', 15, 'mds_method', 'cmds');

%% plot
figure;
M = Y_PHATE_6D;
M = zscore(M,[],2);
%[~,idx_rows] = sort(M(:,1));
D_cols = pdist(M');
Z_cols = linkage(D_cols,'average');
idx_cols = optimalleaforder(Z_cols,D_cols);
imagesc(M(idx_1d,idx_cols));


