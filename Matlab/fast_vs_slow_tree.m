%% init
n_samp = 5000;
n_dim = 100;
n_branch = 25;
sigma = 5;
rng(7);
out_base = '~/Dropbox/PHATE/figures/fast_phate_runtime/april3/';
mkdir(out_base)

%% tree
rng(7)
[M, C] = dla_tree(n_samp, n_dim, n_branch, sigma);

%% fast PHATE
rng(7)
tic;
Y_mmds_fast = phate_fast(M, 'k', 10, 'ndim', 2, 't', [], 'npca', 100, 'nsvd', 100, ...
    'ncluster', 100, 'pot_method', 'sqrt', 't_max', 100);
toc

%% slow PHATE
tic;
Y_slow = phate(M, 't', 100, 'npca', 100, 'k', 10, 'a', 100, 'ndim', 2, 'mds_method', 'mmds', ...
    'pot_method', 'sqrt');
toc

%% plot PHATE 2D
figure;
scatter(Y_mmds_fast(:,1), Y_mmds_fast(:,2), 5, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
title 'PHATE fast'
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'PHATE_fast_tree.tiff']);
%close

%% slow vs fast PHATE
figure;
subplot(1,2,1)
scatter(Y_slow(:,1), Y_slow(:,2), 10, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
title 'PHATE original (3 min)'
subplot(1,2,2)
scatter(Y_mmds_fast(:,1), Y_mmds_fast(:,2), 10, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
title 'PHATE fast (10 sec)'
set(gcf,'paperposition',[0 0 16 6]);
print('-dtiff',[out_base 'PHATE_tree_slow_vs_fast.tiff']);
close

