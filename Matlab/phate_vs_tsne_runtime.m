%% init
n_dim = 100;
n_branch = 40;
sigma = 5;
rng(7);
out_base = '~/Dropbox/PHATE/figures/fast_phate_runtime/';
mkdir(out_base)

%% compute runtime
N_vec = [1000 2000 5000 10000 20000 50000];
runtime_vec_phate = nan(size(N_vec));
%runtime_vec_phate_slow = nan(size(N_vec));
runtime_vec_tsne_05 = nan(size(N_vec));
runtime_vec_tsne_08 = nan(size(N_vec));
for I=1:length(N_vec)
    I
    n_samp = N_vec(I);
    [M, C] = dla_tree(n_samp, n_dim, n_branch, sigma);
    % fast PHATE
    tic;
    Y_phate = phate_fast(M, 'k', 10, 'ndim', 2, 't', 48, 'npca', 100, 'nsvd', 100, ...
        'ncluster', 500, 'pot_method', 'sqrt');
    runtime_vec_phate(I) = toc
%     tic;
%     Y_slow = phate(M, 't', 100, 'npca', 100, 'k', 10, 'a', 100, 'ndim', 2, 'mds_method', 'mmds', ...
%     'pot_method', 'sqrt');
%     runtime_vec_phate_slow(I) = toc
    tSNE theta=0.5
    tic;
    Y_tsne05 = tsne(M,'Algorithm','barneshut','Perplexity',20,'Theta',0.5,'NumPCAComponents',100);
    runtime_vec_tsne_05(I) = toc
    % tSNE theta=0.8
    tic;
    Y_tsne08 = tsne(M,'Algorithm','barneshut','Perplexity',20,'Theta',0.8,'NumPCAComponents',100);
    runtime_vec_tsne_08(I) = toc
end

%% plot
figure;
hold all
x = 1:length(N_vec);
plot(x, runtime_vec_tsne_05, '*-', 'displayname', 'tSNE (\theta=0.5)', 'linewidth', 2, 'markersize', 5);
plot(x, runtime_vec_tsne_08, '*-', 'displayname', 'tSNE (\theta=0.8)', 'linewidth', 2, 'markersize', 5);
% plot(x, runtime_vec_phate_slow, '*-', 'displayname', 'PHATE (MMDS, original)', 'linewidth', 2, 'markersize', 5);
plot(x, runtime_vec_phate, '*-', 'displayname', 'PHATE (MMDS)', 'linewidth', 2, 'markersize', 5);
xlabel 'Number of cells'
ylabel 'Runtime (seconds)'
set(gca,'xtick',1:length(N_vec));
set(gca,'xticklabel',N_vec);
%set(gca,'yscale','log')
legend('location','NW');
legend boxoff
%axis tight
set(gcf,'paperposition',[0 0 4 3]);
print('-dtiff',[out_base 'PHATE_tSNE_runtime.tiff']);
%close


