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
%k_knn = 10;
%a = [];
npca = [];
distfun = 'euclidean';
% epsilon = 1;
% get the input parameters
if ~isempty(varargin)
    for j = 1:length(varargin)
        % k nearest neighbor for adaptive sigma
        if strcmp(varargin{j}, 'k')
            k = varargin{j+1};
        end
%         % a for alpha decay
%         if strcmp(varargin{j}, 'a')
%             a = varargin{j+1};
%         end
%         % k nearest neighbor
%         if strcmp(varargin{j}, 'k_knn')
%             k_knn = varargin{j+1};
%         end
        % epsilon
%         if strcmp(varargin{j}, 'epsilon')
%             epsilon = varargin{j+1};
%         end
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
[idx, ~] = knnsearch(data_pc, data_pc, 'k', k, 'Distance', distfun);

% if epsilon > 0 && ~isempty(a)
%     disp '   Adapting sigma'
%     dist = bsxfun(@rdivide, dist, dist(:,k));
% end

i = repmat((1:N)',1,size(idx,2));
i = i(:);
j = idx(:);
% if epsilon > 0 && ~isempty(a)
%     s = dist(:);
%     W = sparse(i, j, s);
% else
    W = sparse(i, j, ones(size(j))); % unweighted kNN graph
% end

disp '   Symmetrize distances'
W = W + W';

% if epsilon > 0 && ~isempty(a)
%     disp '   Computing kernel'
%     [i,j,s] = find(W);
%     i = [i; (1:N)'];
%     j = [j; (1:N)'];
%     s = [s./(epsilon^a); zeros(N,1)];
%     s = exp(-s);
%     W = sparse(i,j,s);
% end

disp '   Markov normalization'
W = bsxfun(@rdivide, W, sum(W,2)); % Markov normalization

disp '   Done computing operator'
