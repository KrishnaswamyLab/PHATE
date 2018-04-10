function [pc,U,S] = svdpca_sparse(X, k, method)

if ~exist('method','var')
    method = 'svd';
end

switch method
    case 'svd'
        disp 'PCA using SVD'
        [U,S,~] = svds(X', k);
        pc = X * U;
    case 'random'
        disp 'PCA using random SVD'
        [U,S,~] = randPCA(X', k);
        pc = X * U;
        S = diag(S);
    case 'none'
        disp 'No PCA performed'
        pc = X;
end

