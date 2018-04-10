function [M, C] = dla_tree(n_samp, n_dim, n_branch, sigma)
% [M, C] = dla_tree(n_samp, n_dim, n_branch, sigma)

n_steps = round(n_samp/n_branch);
M = cumsum(-1 + 2*(rand(n_steps,n_dim)),1);
for I=1:n_branch-1
    ind = randsample(size(M,1), 1);
    M2 = cumsum(-1 + 2*(rand(n_steps,n_dim)),1);
    M = [M; repmat(M(ind,:),n_steps,1) + M2];
end
C = repmat(1:n_branch,n_steps,1);
C = C(:);
M = M + normrnd(0,sigma,size(M,1),size(M,2));