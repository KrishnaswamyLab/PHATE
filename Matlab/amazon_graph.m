%% data
file = '~/Downloads/com-amazon.ungraph.txt';
M = dlmread(file,'\t',4);

%% operator
n = max(M(:));
A = sparse(M(:,1), M(:,2), 1, n, n);
P = spdiags (1./sum (A,2), 0, size(A,1), size(A,1)) * A;

%% fast PHATE
rng(7)
tic;
Y_mmds_fast = phate_fast([], 'operator', P, 'k', 10, 'ndim', 2, 't', [], 'npca', 100, 'nsvd', 100, ...
    'ncluster', 1000, 'pot_method', 'sqrt');
toc

%% plot PHATE 2D
figure;
scatter(Y_mmds_fast(:,1), Y_mmds_fast(:,2), 3, 'k', 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
% title 'PHATE fast (33 sec., 17k cells)'
% set(gcf,'paperposition',[0 0 8 6]);
% print('-dtiff',[out_base 'PHATE_fast_EB.tiff']);
% %close