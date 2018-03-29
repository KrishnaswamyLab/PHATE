function Y = phate_fast(data, varargin)

npca = 100;
k = 10;
nsvd = 100;
ncluster = 1000;
ndim = 2;
t = 12;
mds_method = 'mmds';
distfun = 'euclidean';
distfun_mds = 'euclidean';
pot_method = 'log';

% get input parameters
for i=1:length(varargin)
    % k for knn
    if(strcmp(varargin{i},'k'))
       k = lower(varargin{i+1});
    end
    % diffusion time
    if(strcmp(varargin{i},'t'))
       t = lower(varargin{i+1});
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
    % ncluster for spectral clustering
    if(strcmp(varargin{i},'ncluster'))
       ncluster = lower(varargin{i+1});
    end
    % potential method: log, sqrt
    if(strcmp(varargin{i},'pot_method'))
       pot_method = lower(varargin{i+1});
    end
end

% PCA
disp 'Doing PCA'
pc = svdpca(data, npca, 'random');

% diffusion operator
P = compute_operator_fast(pc, 'k', k, 'distfun', distfun);

% spectral cluster for landmarks
disp 'Spectral clustering for landmarks'
[U,S,~] = randPCA(P, nsvd);
IDX = kmeans(U*S, ncluster);

% create landmark operators
disp 'Computing landmark operators'
n = size(P,1);
m = max(IDX);
Pnm = nan(n,m);
Pmn = nan(m,n);
for I=1:m
    Pnm(:,I) = sum(P(:,IDX==I),2);
    Pmn(I,:) = sum(P(IDX==I,:),1);
end
Pmn = bsxfun(@rdivide, Pmn, sum(Pmn,2));

% diffuse
disp 'Diffusing landmark operators'
P_t = (Pmn * Pnm)^t;

% potential distances
disp 'Computing potential distances'
switch pot_method
    case 'log'
        P_t(P_t<=eps) = eps;
        Pot = -log(P_t);
    case 'sqrt'
        Pot = sqrt(P_t);
    otherwise
        disp 'potential method unknown'
end
PDX = squareform(pdist(Pot, distfun_mds));

% CMDS
disp 'Doing classical MDS'
Y = randmds(PDX, ndim);

% MMDS
if strcmpi(mds_method, 'mmds')
    disp 'Doing metric MDS:'
    opt = statset('display','iter');
    Y = mdscale(PDX,ndim,'options',opt,'start',Y,'Criterion','metricstress');
end

% out of sample extension from landmarks to all points
disp 'Out of sample extension from landmakrs to all points'
Y = Pnm * Y;

disp 'Done.'

end






