function Y = randmds(X, ndim)
% Y = randmds(X, ndim)
%    CMDS using random SVD

X = X.^2; % 
X = bsxfun(@minus, X, mean(X,1));
X = bsxfun(@minus, X, mean(X,2));
[Y,~,~] = randPCA(X', ndim);

