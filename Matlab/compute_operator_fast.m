function W = compute_operator_fast(data, varargin)
% W = compute_operator(data, varargin)
%   computes diffusion operator W
% varargin:
%   'npca' (default = 20)
%       perform fast random PCA before computing distances
%   'k' (default = 10)
%       k of kNN graph
%   'epsilon' (default = 1)
%       kernel bandwith, if epsilon = 0 kernel will be uniform, i.e.
%       unweighted kNN graph (ka will be ignored)

% set up default parameters
k = 10;
npca = [];
distfun = 'euclidean';
% get the input parameters
if ~isempty(varargin)
    for j = 1:length(varargin)
        % k nearest neighbor
        if strcmp(varargin{j}, 'k')
            k = varargin{j+1};
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

disp 'Computing operator:'

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

disp '   Computing distances'
idx = knnsearch(data_pc, data_pc, 'k', k, 'Distance', distfun);

i = repmat((1:N)',1,size(idx,2));
i = i(:);
j = idx(:);
W = sparse(i, j, ones(size(j))); % unweighted kNN graph

disp '   Symmetrize distances'
W = W + W';

disp '   Markov normalization'
W = bsxfun(@rdivide, W, sum(W,2)); % Markov normalization

disp '   Done computing operator'
