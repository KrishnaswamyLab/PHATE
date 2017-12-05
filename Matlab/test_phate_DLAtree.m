% Test the embedding on the diffusion limited aggregation tree
% branching data.
%
% Authors: Kevin Moon, David van Dijk
% Created: March 2017

%% Generate the data
n_dim = 100;
n_branch = 20;
n_steps = 100;
n_drop = 0;
seed = 37;
rng(seed);

%% Generate random fractal tree via DLA
M = cumsum(-1 + 2*(rand(n_steps,n_dim)),1);
for I=1:n_branch-1
    ind = randsample(size(M,1), 1);
    M2 = cumsum(-1 + 2*(rand(n_steps,n_dim)),1);
    M = [M; repmat(M(ind,:),n_steps,1) + M2];
end
C = repmat(1:n_branch,n_steps,1);
C = C(:); % Color coding of branches
fprintf(1,'%u data points by %u features\n',size(M,1),size(M,2));

%% Add noise
sigma = 4;
M = M + normrnd(0, sigma, size(M,1), size(M,2));

%% PCA
Y_pca = svdpca(M, 2, 'random');

%% Plot the embedding colored by branch
figure;
scatter(Y_pca(:,1),Y_pca(:,2),5,C,'filled')
axis tight
xlabel('PCA 1')
ylabel('PCA 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PCA'

%% Run PHATE using CMDS (faster but less accurate)
t = 30;
a = 13;
k = 5;
pca_method = 'random'; % fast PCA
mds_method = 'cmds_fast'; % fast CMDS using random PCA
distfun_mds = 'euclidean';
ndim = 2;
Y_cmds = phate(M,'t',t,'k',k,'a',a,'pca_method',pca_method,'mds_method',mds_method,'ndim',ndim,'distfun_mds',distfun_mds);

%% Plot the embedding colored by branch
figure;
scatter(Y_cmds(:,1),Y_cmds(:,2),5,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PHATE using CMDS'

%% Run PHATE using NMMDS (slower but more accurate)
t = 30;
a = 13;
k = 5;
pca_method = 'random'; % fast PCA
mds_method = 'nmmds';
distfun_mds = 'euclidean';
ndim = 2;
Y_nmmds = phate(M,'t',t,'k',k,'a',a,'pca_method',pca_method,'mds_method',mds_method,'ndim',ndim,'distfun_mds',distfun_mds);

%% Plot the embedding colored by branch
figure;
scatter(Y_nmmds(:,1),Y_nmmds(:,2),5,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PHATE using NMMDS'

%% Run PHATE using NMMDS and 
t = 30; 
a = 13;
k = 5;
pca_method = 'random'; % fast PCA
mds_method = 'nmmds';
distfun_mds = 'cosine'; % <--- different mds distance
ndim = 2;
Y_nmmds2 = phate(M,'t',t,'k',k,'a',a,'pca_method',pca_method,'mds_method',mds_method,'ndim',ndim,'distfun_mds',distfun_mds);

%% Plot the embedding colored by branch
figure;
scatter(Y_nmmds2(:,1),Y_nmmds2(:,2),5,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PHATE using NMMDS with Cosine MDS distance'

%% plot all embeddings combined
figure;

subplot(2,2,1);
scatter(Y_pca(:,1),Y_pca(:,2),5,C,'filled')
axis tight
xlabel('PCA 1')
ylabel('PCA 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PCA'

subplot(2,2,2);
scatter(Y_cmds(:,1),Y_cmds(:,2),5,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PHATE using CMDS'

subplot(2,2,3);
scatter(Y_nmmds(:,1),Y_nmmds(:,2),5,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PHATE using NMMDS'

subplot(2,2,4);
scatter(Y_nmmds2(:,1),Y_nmmds2(:,2),5,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PHATE using NMMDS with X*Xt'

