function Y = randmds(X, ndim)
% Y = randmds(X, ndim)
%    CMDS using random SVD

X = bsxfun(@minus, X, mean(X,1));
X = bsxfun(@minus, X, mean(X,2));
[U,~,~] = randPCA(X', ndim);
Y = X * U;
