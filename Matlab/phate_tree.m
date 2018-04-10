%% init
out_base = '~/Dropbox/PHATE/figures/fast_phate_runtime/april10/';
mkdir(out_base)
rseed = 7;

%% generate tree data
n_samp = 3000; % number of points
n_dim = 100; % number of dimensions
n_branch = 10; % number of branches
sigma = 5; % noise
rng(rseed)
[M, C] = dla_tree(n_samp, n_dim, n_branch, sigma);

%% PHATE 2D
tic;
Y_PHATE_2D = phate(M, 'pot_method', 'sqrt', 't', 45);
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
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'PHATE_2D_tree.tiff']);
%close

%% PHATE 3D
tic;
Y_PHATE_3D = phate(M, 'pot_method', 'sqrt', 'ndim', 3, 't', 45);
toc

%% plot PHATE 2D
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
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'PHATE_3D_tree.tiff']);
%close

