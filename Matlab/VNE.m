function [H, t_vec, DiffOp] = VNE(data, varargin)
% Calculates and plots the Von Neumann Entropy (VNE) for choosing the diffusion time 't'. A
% good choice of 't' is a value in the relatively flat region after the knee in the VNE plot
%
% Authors: Kevin Moon, David van Dijk
% Created: March 2017

% OUTPUT
%   H = VNE
%   t_vec = range of t values
%   Diffusion operator that can be input into the phate code to save on
%   computational time

% INPUT
%   data = data matrix. Must have cells on the rows and genes on the
%   columns
% varargin:
%   't_vec' (default = 1:150)
%       Diffusion time scales for computing VNE
%   'k' (default = 5)
%       k for the adaptive kernel bandwidth
%   'a' (default = 10)
%       The alpha parameter in the exponent of the kernel function. Determines the kernel decay rate
%   'pca_method' (default = 'random')
%       The desired method for implementing pca for preprocessing the data. Options include 'svd', 'random', and 'none' (no pca) 
%   'npca' (default = 100)
%       The number of PCA components for preprocessing the data
%   'distfun' (default = 'euclidean')
%       The desired distance function for calculating pairwise distances on the data.
%   'plot' (default = 'true')
%       Toggle whether to plot the VNE or not


% set up default parameters
t_vec=1:150;
k = 5;
a = 10;
npca = 100;
distfun = 'euclidean';
pca_method = 'random';
plot_on = 'true';

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
    % diffusion time vector
    if(strcmp(varargin{i},'t_vec'))
       t_vec =  lower(varargin{i+1});
    end
    % Number of pca components
    if(strcmp(varargin{i},'npca'))
       npca =  lower(varargin{i+1});
    end
    % Distance function for the inputs
    if(strcmp(varargin{i},'distfun'))
       distfun =  lower(varargin{i+1});
    end
    % Method for PCA
    if(strcmp(varargin{i},'pca_method'))
       pca_method =  lower(varargin{i+1});
    end
%     Plot the VNE?
    if(strcmp(varargin{i},'plot'))
        plot_on = lower(varargin{i+1});
    end
end

M = svdpca(data, npca, pca_method);

disp 'computing distances'
PDX = squareform(pdist(M, distfun));
[~, knnDST] = knnsearch(M,M,'K',k+1,'dist',distfun);

disp 'computing kernel and operator'
epsilon = knnDST(:,k+1); % bandwidth(x) = distance to k-th neighbor of x
PDX = bsxfun(@rdivide,PDX,epsilon); % autotuning d(x,:) using epsilon(x)
GsKer = exp(-PDX.^a); % not really Gaussian kernel
GsKer=GsKer+GsKer'; % Symmetrization
DiffDeg = diag(sum(GsKer,2)); % degrees

DiffAff = DiffDeg^(-1/2)*GsKer*DiffDeg^(-1/2); % symmetric conjugate affinities
DiffAff = (DiffAff + DiffAff')/2; % clean up numerical inaccuracies to maintain symmetry

% Clear a bit of space for memory
clear GsKer PDX DiffDeg

X=DiffAff;

% Find the eigenvalues
disp 'Finding the eigenvalues'
[~,S] = svd(X); 
S=diag(S);

disp 'Computing VNE'
H = nan(size(t_vec)); 
for I=1:length(t_vec) 
    t = t_vec(I); 
    S_t=S.^t;
    P = S_t/sum(S_t);
    P=P(P>0);
    H(I) = -sum(P .* log(P)); % Entropy of eigenvalues
end

if(plot_on)
% Plot the entropy; choose a t in the flatter range after the 'knee' for generally best results
    disp 'VNE plotted'
    figure;
    plot(t_vec,H,'*-')
    xlabel('t')
    ylabel('VNE')
end
