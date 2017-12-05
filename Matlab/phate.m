function [Y, DiffOp, DiffOp_t] = phate(data, varargin)
% Runs PHATE on input data. data must have cells on the rows and genes on the columns
%
% Authors: Kevin Moon, David van Dijk
% Created: March 2017

% OUTPUT
%       Y = the PHATE embedding
%       DiffOp = the diffusion operator which can be used as optional input with another run
%       DiffOp_t = DiffOp^t

% INPUT
%       data = data matrix. Must have cells on the rows and genes on the columns
% varargin:
%   't' (default = 20)
%       Diffusion time scale
%   'k' (default = 5)
%       k for the adaptive kernel bandwidth
%   'a' (default = 10)
%       The alpha parameter in the exponent of the kernel function. Determines the kernel decay rate
%   'ndim' (default = 2)
%       The number of desired PHATE dimensions in the output Y. 2 or 3 is best for visualization. A higher number can be used for
%       running other analyses on the PHATE dimensions.
%   'pca_method' (default = 'random')
%       The desired method for implementing pca for preprocessing the data. Options include 'svd', 'random', and 'none' (no pca)
%   'npca' (default = 100)
%       The number of PCA components for preprocessing the data
%   'mds_method' (default = 'cmds_fast')
%       Method for implementing MDS. Choices are 'cmds' (built-in matlab function), 'cmds_fast' (uses fast PCA), and 'nmmds'
%   'distfun' (default = 'euclidean')
%       The desired distance function for calculating pairwise distances on the data.
%   'distfun_mds' (default = 'euclidean')
%       The desired distance function for MDS. Choices are 'euclidean',
%       'cosine'.
%   'DiffOp' (default = [])
%       If the diffusion operator has been computed on a prior run with the desired parameters, then this option can be used to
%       directly input the diffusion operator to save on computational time.
%   'DiffOp_t' (default = [])
%       Same as for 'DiffOp', if the powered diffusion operator has been computed on a prior run with the desired parameters,
%       then this option can be used to directly input the diffusion operator to save on computational time.

% set up default parameters
k = 5;
a = 10;
npca = 100;
t = 20;
ndim = 2;
mds_method = 'cmds_fast';
distfun = 'euclidean';
distfun_mds = 'euclidean';
pca_method = 'random';
DiffOp = [];
DiffOp_t = [];

% get input parameters
for i=1:length(varargin)
    % adaptive k-nn bandwidth
    if(strcmp(varargin{i},'k'))
       k =  lower(varargin{i+1});
    end
    % alpha parameter for kernel decay rate
    if(strcmp(varargin{i},'a'))
       a =  lower(varargin{i+1});
    end
    % diffusion time
    if(strcmp(varargin{i},'t'))
       t =  lower(varargin{i+1});
    end
    % Number of pca components
    if(strcmp(varargin{i},'npca'))
       npca =  lower(varargin{i+1});
    end
    % Number of dimensions for the PHATE embedding
    if(strcmp(varargin{i},'ndim'))
       ndim =  lower(varargin{i+1});
    end
    % Method for MDS
    if(strcmp(varargin{i},'mds_method'))
       mds_method =  lower(varargin{i+1});
    end
    % Distance function for the inputs
    if(strcmp(varargin{i},'distfun'))
       distfun =  lower(varargin{i+1});
    end
    % distfun for MDS
    if(strcmp(varargin{i},'distfun_mds'))
       distfun_mds =  lower(varargin{i+1});
    end
    % Method for PCA
    if(strcmp(varargin{i},'pca_method'))
       pca_method =  lower(varargin{i+1});
    end
    % Use precomputed diffusion operator?
    if(strcmp(varargin{i},'DiffOp'))
       DiffOp = lower(varargin{i+1});
    end
    % Use precomputed powered diffusion operator?
    if(strcmp(varargin{i},'DiffOp_t'))
       DiffOp_t = lower(varargin{i+1});
    end
end

disp '======= PHATE ======='

% Check to see if precomputed DiffOp or DiffOp_t are given

if(isempty(DiffOp)&isempty(DiffOp_t))
    M = svdpca(data, npca, pca_method);

    disp 'computing distances'
    PDX = squareform(pdist(M, distfun));
    [~, knnDST] = knnsearch(M,M,'K',k+1,'dist',distfun);

    disp 'computing kernel and operator'
    epsilon = knnDST(:,k+1); % bandwidth(x) = distance to k-th neighbor of x
    PDX = bsxfun(@rdivide,PDX,epsilon); % autotuning d(x,:) using epsilon(x)
    GsKer = exp(-PDX.^a); % not really Gaussian kernel
    GsKer=GsKer+GsKer'; % symmetrization
    DiffDeg = diag(1./sum(GsKer,2)); % Inverse of degrees
    DiffOp = DiffDeg*GsKer; % row stochastic

    % Clear a bit of space for memory
    clear GsKer PDX DiffDeg
end

% Check to see if pre computed DiffOp_t is given
if(isempty(DiffOp_t))
    disp 'diffusing operator'
    DiffOp_t = DiffOp^t;
end

X = DiffOp_t;

disp 'potential recovery'
X(X<=eps)=eps;
X = -log(X);
disp(['MDS distfun: ' distfun_mds])
if strcmp(distfun_mds, 'euclidean')
    X = svdpca(X, npca, pca_method); % to make pdist faster
end
X = squareform(pdist(X, distfun_mds));

switch mds_method
    % CMDS using fast pca
    case 'cmds_fast'
        disp 'Fast CMDS'
        Y = randmds(X, ndim);
    % built-in MATLAB version of CMDS
    case 'cmds'
        disp 'CMDS'
        Y = cmdscale(X, ndim);
    % built-in MATLAB version of NMMDS
    case 'nmmds'
        disp 'NMMDS'
        opt = statset('display', 'iter');
        Y_start = randmds(X, ndim);
        Y = mdscale(X, ndim, 'options', opt, 'start', Y_start);
    % built-in MATLAB version of MMDS
    case 'mmds'
        disp 'MMDS'
        opt=statset('display','iter');
        Y_start = randmds(X,ndim);
        Y = mdscale(X,ndim,'options',opt,'start',Y_start,'Criterion','metricstress');
end

disp 'done.'
