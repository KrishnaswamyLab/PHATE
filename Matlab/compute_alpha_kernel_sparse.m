function K = compute_alpha_kernel_sparse(data, varargin)

k = 11;
is_done = false;
a = 10;
th = 1e-4;
epsilon = [];
npca = [];
distfun = 'euclidean';
% get the input parameters
if ~isempty(varargin)
    for j = 1:length(varargin)
        % k nearest neighbora
        if strcmp(varargin{j}, 'k')
            k = varargin{j+1};
        end
        % threshold
        if strcmp(varargin{j}, 'th')
            th = varargin{j+1};
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

k = k + 1;
k_knn = k * 2;

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

while (~is_done)
    k_knn
    [idx, dist] = knnsearch(data_pc,data_pc,'k',k_knn,'Distance',distfun);
    if isempty(epsilon)
        epsilon = dist(:,k);
    end
    dist = bsxfun(@rdivide,dist,epsilon);
    K = exp(-dist.^a);
    K_max = max(K(:,end))
    if K_max <= th
        is_done = true;
        disp 'done'
    end
    if k_knn >= N
        is_done = true;
        disp 'done'
    end
    k_knn = k_knn * 2;
end

disp '   set < th to zero'
K(K<th) = 0;

i = repmat((1:N)',1,size(idx,2));
i = i(:);
j = idx(:);
K = sparse(i, j, K(:));

disp '   Symmetrize affinities'
K = K + K';

disp '   Done computing kernel'

