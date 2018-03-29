%% generate random fractal tree via DLA
rng(17)
n_samp = 10000;
n_dim = 100;
n_branch = 40;
sigma = 5;
n_steps = round(n_samp/n_branch);
n_samp = n_branch * n_steps;
M = cumsum(-1 + 2*(rand(n_steps,n_dim)),1);
for I=1:n_branch-1
    ind = randsample(size(M,1), 1);
    M2 = cumsum(-1 + 2*(rand(n_steps,n_dim)),1);
    M = [M; repmat(M(ind,:),n_steps,1) + M2];
end
C = repmat(1:n_branch,n_steps,1);
C = C(:);
M = M + normrnd(0,sigma,size(M,1),size(M,2));

%% fast PHATE
tic;
Y_fast = phate_fast(M, 'k', 10, 'ndim', 2, 't', 48, 'npca', 100, 'nsvd', 100, ...
    'ncluster', 1000, 'pot_method', 'sqrt');
toc

%% plot PHATE 2D
figure;
scatter(Y_fast(:,1), Y_fast(:,2), 5, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
title 'PHATE fast, sqrt'

%% slow PHATE
tic
Y_slow = phate(M, 't', 100, 'npca', 100, 'k', 10, 'a', 100, 'ndim', 2, 'mds_method', 'mmds', ...
    'pot_method', 'sqrt');
toc;

%% plot PHATE 2D
figure;
scatter(Y_slow(:,1), Y_slow(:,2), 5, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
title 'PHATE slow, sqrt'




