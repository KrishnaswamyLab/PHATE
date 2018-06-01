function K = compute_alpha_kernel_sparse(data, varargin)
% K = computer_alpha_kernel_sparse(data, varargin)
% Computes sparse alpha-decay kernel
% varargin: 
%   'npca' (default = [], no PCA)
%       Perform fast random PCA before computing distances
%   'k' (default = 5)
%       k for the knn distances for the locally adaptive bandwidth
%   'a' (default = 10)
%       The alpha exponent in the alpha-decaying kernel
%   'distfun' (default = 'euclidean')
%       Input distance function
k = 5;
a = 10;
npca = [];
distfun = 'euclidean';
% get the input parameters
if ~isempty(varargin)
    for j = 1:length(varargin)
        % k nearest neighbora
        if strcmp(varargin{j}, 'k')
            k = varargin{j+1};
        end
        % alpha
        if strcmp(varargin{j}, 'a')
            a = varargin{j+1};
        end
        % npca to project data
        if strcmp(varargin{j}, 'npca')
            npca = varargin{j+1};
        end
        % distfun
        if strcmp(varargin{j}, 'distfun')
            distfun = varargin{j+1};
        end
    end
end

th = 1e-4;

k_knn = k * 20;

bth=(-log(th))^(1/a);

disp 'Computing alpha decay kernel:'

N = size(data, 1); % number of cells

if ~isempty(npca)
    disp '   PCA'
    data_centr = bsxfun(@minus, data, mean(data,1));
    [U,~,~] = randPCA(data_centr', npca); % fast random svd
    %[U,~,~] = svds(data', npca);
    data_pc = data_centr * U; % PCA project
else
    data_pc = data;
end

disp(['Number of samples = ' num2str(N)])

% Initial knn search and set the radius
disp(['First iteration: k = ' num2str(k_knn)])
[idx, kdist]=knnsearch(data_pc,data_pc,'k',k_knn,'Distance',distfun);
epsilon=kdist(:,k+1);

% Find the points that have large enough distance to be below the kernel
% threshold
below_thresh=kdist(:,end)>=bth*epsilon;

idx_thresh=find(below_thresh);

if ~isempty(idx_thresh) 
    K=exp(-(kdist(idx_thresh,:)./epsilon(idx_thresh)).^a);
    K(K<=th)=0;
    K=K(:);
    i = repmat(idx_thresh',1,size(idx,2));
    i = i(:);
    idx_temp=idx(idx_thresh,:);
    j = idx_temp(:);
end

disp(['Number of samples below the threshold from 1st iter: ' num2str(length(idx_thresh))])

% Loop increasing k by factor of 20 until we cover 90% of the data
while length(idx_thresh)<.9*N
    k_knn=min(20*k_knn,N);
    data_pc2=data_pc(~below_thresh,:);
    epsilon2=epsilon(~below_thresh);
    disp(['Next iteration: k= ' num2str(k_knn)])
    [idx2, kdist2]=knnsearch(data_pc,data_pc2,'k',k_knn,'Distance',distfun);
    
%     Find the points that have large enough distance
    below_thresh2=kdist2(:,end)>=bth*epsilon2;
    idx_thresh2=find(below_thresh2);
    
    if ~isempty(idx_thresh2)
        K2=exp(-(kdist2(idx_thresh2,:)./epsilon2(idx_thresh2)).^a);
        K2(K2<=th)=0;
        idx_notthresh=find(~below_thresh);
        i2=repmat(idx_notthresh(idx_thresh2)',1,size(idx2,2));
        i2=i2(:);
        idx_temp=idx2(idx_thresh2,:);
        j2=idx_temp(:);
        
        i=[i; i2];
        j=[j; j2];
        K=[K; K2(:)];
%         Add the newly thresholded points to the old ones
        below_thresh(idx_notthresh(idx_thresh2))=1;
        idx_thresh=find(below_thresh);
    end
    disp(['Number of samples below the threshold from the next iter: ' num2str(length(idx_thresh))])
end

% Radius search for the rest
if length(idx_thresh)<N
    disp(['Using radius based search for the rest'])
    data_pc2=data_pc(~below_thresh,:);
    epsilon2=epsilon(~below_thresh);
    [idx2, kdist2]=rangesearch(data_pc,data_pc2,bth*max(epsilon2),'Distance',distfun);
    idx_notthresh=find(~below_thresh);
    for m=1:length(idx2)
        i=[i; idx_notthresh(m)*ones(length(idx2{m}),1)];
        j=[j; idx2{m}'];
        K2=exp(-(kdist2{m}./epsilon2(m)).^a);
        K2(K2<=th)=0;
        K=[K; K2(:)];
    end
    
end

% Build the kernel
K = sparse(i, j, K);

disp '   Symmetrize affinities'
K = K + K';
disp '   Done computing kernel'

