function [W,Wns] = compute_kernel_sparse(data, varargin)
% W = compute_kernel_sparse(data, varargin)
%   computes kernel W
% varargin:
%   'npca' (default = [], no PCA)
%       perform fast random PCA before computing distances
%   'k' (default = 5)
%       k of kNN graph

% set up default parameters
k = 5;
npca = [];
distfun = 'euclidean';

% get the input parameters
if ~isempty(varargin)
    for j = 1:length(varargin)
        % k nearest neighbora
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

disp 'Computing kernel:'

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
[idx, ~] = knnsearch(data_pc, data_pc, 'k', k+1, 'Distance', distfun);

i = repmat((1:N)',1,size(idx,2));
i = i(:);
j = idx(:);
Wns = sparse(i, j, ones(size(j)));

disp '   Symmetrize affinities'
W = Wns + Wns';

disp '   Done computing kernel'

