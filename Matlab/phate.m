function [Y, P, K] = phate(data, varargin)
% phate  Run PHATE for visualizing noisy non-linear data in lower dimensions
%   Y = phate(data) runs PHATE on data (rows: samples, columns: features)
%   with default parameter settings and returns a 2 dimensional embedding.
%
%   If data is sparse PCA without mean centering will be done to maintain
%   low memory footprint. If data is dense then normal PCA (with mean
%   centering) is done.
%
%   Y = phate(data, 'PARAM1',val1, 'PARAM2',val2, ...) allows you to
%   specify optional parameter name/value pairs that control further details
%   of PHATE.  Parameters are:
%
%   'ndim' - number of (output) embedding dimensions. Common values are 2
%   or 3. Defaults to 2.
%
%   'k' - number of nearest neighbors for bandwidth of adaptive alpha
%   decaying kernel or, when a=[], number of nearest neighbors of the knn
%   graph. For the unweighted kernel we recommend k to be a bit larger,
%   e.g. 10 or 15. Defaults to 5.
%
%   'a' - alpha of alpha decaying kernel. when a=[] knn (unweighted) kernel
%   is used. Defaults to 40.
%
%   't' - number of diffusion steps. Defaults to [] wich autmatically picks
%   the optimal t.
%
%   't_max' - maximum t for finding optimal t. if t = [] optimal t will be
%   computed by computing Von Neumann Entropy for each t <= t_max and then
%   picking the kneepoint. Defaults to 100.
%
%   'npca' - number of pca components for computing distances. Defaults to
%   100.
%
%   'mds_method' - method of multidimensional scaling. Choices are:
%
%       'mmds' - metric MDS (default)
%       'cmds' - classical MDS
%       'nmmds' - non-metric MDS
%
%   'distfun' - distance function. Default is 'euclidean'.
%
%   'distfun_mds' - distance function for MDS. Default is 'euclidean'.
%
%   'pot_method' - method of computing the PHATE potential dstance. Choices
%   are:
%
%       'log' - -log(P + eps). (default)
%
%       'sqrt' - sqrt(P). (not default but often produces superior
%       embeddings)
%
%       'gamma' - 2/(1-\gamma)*P^((1-\gamma)/2)
%
%   'gamma' - gamma value for gamma potential method. Value between -1 and
%   1. -1 is diffusion distance. 1 is log potential. 0 is sqrt. Smaller
%   gamma is a more locally sensitive embedding whereas larger gamma is a
%   more globally sensitive embedding. Defaults to 0.5.
%
%   'pot_eps' - epsilon value added to diffusion operator prior to
%   computing potential. Only used for 'pot_method' is 'log', i.e.:
%   -log(P + pot_eps). Defaults to 1e-7.
%
%   'n_landmarks' - number of landmarks for fast and scalable PHATE. [] or
%   n_landmarks = npoints does no landmarking, which is slower. More
%   landmarks is more accurate but comes at the cost of speed and memory.
%   Defaults to 2000.
%
%   'nsvd' - number of singular vectors for spectral clustering (for
%   computing landmarks). Defaults to 100.
%
%   'kernel' - user supplied kernel. If not given ([]) kernel is
%   computed from the supplied data. Supplied kernel should be a square
%   (samples by samples) symmetric affinity matrix. If kernel is
%   supplied input data can be empty ([]). Defaults to [].

npca = 100;
k = 5;
nsvd = 100;
n_landmarks = 2000;
ndim = 2;
t = [];
mds_method = 'mmds';
distfun = 'euclidean';
distfun_mds = 'euclidean';
pot_method = 'log';
K = [];
a = 40;
Pnm = [];
t_max = 100;
pot_eps = 1e-7;
gamma = 0.5;

% get input parameters
for i=1:length(varargin)
    % k for knn adaptive sigma
    if(strcmp(varargin{i},'k'))
       k = lower(varargin{i+1});
    end
    % a (alpha) for alpha decaying kernel
    if(strcmp(varargin{i},'a'))
       a = lower(varargin{i+1});
    end
    % diffusion time
    if(strcmp(varargin{i},'t'))
       t = lower(varargin{i+1});
    end
    % t_max for VNE
    if(strcmp(varargin{i},'t_max'))
       t_max = lower(varargin{i+1});
    end
    % Number of pca components
    if(strcmp(varargin{i},'npca'))
       npca = lower(varargin{i+1});
    end
    % Number of dimensions for the PHATE embedding
    if(strcmp(varargin{i},'ndim'))
       ndim = lower(varargin{i+1});
    end
    % Method for MDS
    if(strcmp(varargin{i},'mds_method'))
       mds_method =  varargin{i+1};
    end
    % Distance function for the inputs
    if(strcmp(varargin{i},'distfun'))
       distfun = lower(varargin{i+1});
    end
    % distfun for MDS
    if(strcmp(varargin{i},'distfun_mds'))
       distfun_mds =  lower(varargin{i+1});
    end
    % nsvd for spectral clustering
    if(strcmp(varargin{i},'nsvd'))
       nsvd = lower(varargin{i+1});
    end
    % n_landmarks for spectral clustering
    if(strcmp(varargin{i},'n_landmarks'))
       n_landmarks = lower(varargin{i+1});
    end
    % potential method: log, sqrt, gamma
    if(strcmp(varargin{i},'pot_method'))
       pot_method = lower(varargin{i+1});
    end
    % kernel
    if(strcmp(varargin{i},'kernel'))
       K = lower(varargin{i+1});
    end
    % kernel
    if(strcmp(varargin{i},'gamma'))
       gamma = lower(varargin{i+1});
    end
    % pot_eps
    if(strcmp(varargin{i},'pot_eps'))
       pot_eps = lower(varargin{i+1});
    end
end

if isempty(a) && k <=5
    disp '======================================================================='
    disp 'Make sure k is not too small when using an unweighted knn kernel (a=[])'
    disp(['Currently k = ' numstr(k) ', which may be too small']);
    disp '======================================================================='
end

tt_pca = 0;
tt_kernel = 0;
tt_svd = 0;
tt_kmeans = 0;
tt_lo = 0;
tt_mmds = 0;
tt_nmmds = 0;

if isempty(K)
    if ~isempty(npca) && size(data,2) > npca && size(data,1) > npca
        % PCA
        disp 'Doing PCA'
        tic;
        if issparse(data)
            disp 'Data is sparse, doing SVD instead of PCA (no mean centering)'
            pc = svdpca_sparse(data, npca, 'random');
        else
            pc = svdpca(data, npca, 'random');
        end
        tt_pca = toc;
        disp(['PCA took ' num2str(tt_pca) ' seconds']);
    else
        pc = data;
    end
    % kernel
    tic;
    if isempty(a)
        disp 'using unweighted knn kernel'
        K = compute_kernel_sparse(pc, 'k', k, 'distfun', distfun);
    else
        disp 'using alpha decaying kernel'
        K = compute_alpha_kernel_sparse(pc, 'k', k, 'a', a, 'distfun', distfun);
    end
    tt_kernel = toc;
    disp(['Computing kernel took ' num2str(tt_kernel) ' seconds']);
else
    disp 'Using supplied kernel'
end

disp 'Make kernel row stochastic'
P = bsxfun(@rdivide, K, sum(K,2));

if ~isempty(n_landmarks) && n_landmarks < size(K,1)
    % spectral cluster for landmarks
    disp 'Spectral clustering for landmarks'
    tic;
    [U,S,~] = randPCA(P, nsvd);
    tt_svd = toc;
    disp(['svd took ' num2str(tt_svd) ' seconds']);
    tic;
    IDX = kmeans(U*S, n_landmarks);
    tt_kmeans = toc;
    disp(['kmeans took ' num2str(tt_kmeans) ' seconds']);
    
    % create landmark operators
    disp 'Computing landmark operators'
    tic;
    n = size(K,1);
    m = max(IDX);
    Pnm = nan(n,m);
    for I=1:m
        Pnm(:,I) = sum(K(:,IDX==I),2);
    end
    Pmn = Pnm';
    Pmn = bsxfun(@rdivide, Pmn, sum(Pmn,2));
    Pnm = bsxfun(@rdivide, Pnm, sum(Pnm,2));
    tt_lo = toc;
    disp(['Computing landmark operators took ' num2str(tt_lo) ' seconds']);
    
    % Pmm
    Pmm = Pmn * Pnm;
else
    disp 'Running PHATE without landmarking'
    Pmm = bsxfun(@rdivide, K, sum(K,2));
end

% VNE
if isempty(t)
    disp 'Finding optimal t using VNE'
    t = vne_optimal_t(Pmm, t_max);
end

% diffuse
disp 'Diffusing operator'
tic;
P_t = Pmm^t;
tt_diff = toc;
disp(['Diffusion took ' num2str(tt_diff) ' seconds']);

% potential distances
tic;
disp 'Computing potential distances'
switch pot_method
    case 'log'
        disp 'using -log(P) potential distance'
        Pot = -log(P_t + pot_eps);
    case 'sqrt'
        disp 'using sqrt(P) potential distance'
        Pot = sqrt(P_t);
    case 'gamma'
        disp 'Pot = 2/(1-\gamma)*P^((1-\gamma)/2)'
        disp(['gamma = ' num2str(gamma)]);
        gamma = min(gamma, 0.95);
        Pot = 2/(1-gamma)*P_t.^((1-gamma)/2);
    otherwise
        error 'potential method unknown'
end
PDX = squareform(pdist(Pot, distfun_mds));
tt_pdx = toc;
disp(['Computing potential distance took ' num2str(tt_pdx) ' seconds']);

% CMDS
disp 'Doing classical MDS'
tic;
Y = randmds(PDX, ndim);
tt_cmds = toc;
disp(['CMDS took ' num2str(tt_cmds) ' seconds']);

% MMDS
if strcmpi(mds_method, 'mmds')
    tic;
    disp 'Doing metric MDS:'
    opt = statset('display','iter');
    Y = mdscale(PDX,ndim,'options',opt,'start',Y,'Criterion','metricstress');
    tt_mmds = toc;
    disp(['MMDS took ' num2str(tt_mmds) ' seconds']);
end

% NMMDS
if strcmpi(mds_method, 'nmmds')
    tic;
    disp 'Doing non-metric MDS:'
    opt = statset('display','iter');
    Y = mdscale(PDX,ndim,'options',opt,'start',Y,'Criterion','stress');
    tt_nmmds = toc;
    disp(['NMMDS took ' num2str(tt_nmmds) ' seconds']);
end

if ~isempty(Pnm)
    % out of sample extension from landmarks to all points
    disp 'Out of sample extension from landmarks to all points'
    Y = Pnm * Y;
end

disp 'Done.'

tt_total = tt_pca + tt_kernel + tt_svd + tt_kmeans + tt_lo + tt_diff + ...
    tt_pdx + tt_cmds + tt_mmds + tt_nmmds;

disp(['Total time ' num2str(tt_total) ' seconds']);

end






