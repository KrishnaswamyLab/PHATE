%% Generate data

n_dim = 40;
n_branch = 10;
n_steps = 100;
n_density=40;

end1=30;
end2=40;
end3=70;
end4=80;
trajsize=4;

% Some initialization
trajectory=[linspace(0,end1,n_steps)' linspace(0,end2,n_steps)' linspace(0,end3,n_steps)' linspace(0,end4,n_steps)'];
trajectory2=[linspace(-end1,end1,n_steps)' linspace(-end2,end2,n_steps)' linspace(-end3,end3,n_steps)' linspace(-end4,end4,n_steps)'];

begin=10;
trajectory3=[linspace(begin,end1,n_steps)' linspace(begin,end2,n_steps)' linspace(begin,end3,n_steps)' linspace(begin,end4,n_steps)'];
constant=[end1*ones(n_steps,1) end2*ones(n_steps,1) end3*ones(n_steps,1) end4*ones(n_steps,1)];
constant2=[end1*ones(n_density,1) end2*ones(n_density,1) end3*ones(n_density,1) end4*ones(n_density,1)];

% Branch 1

M=[-constant2 zeros(n_density,n_dim-trajsize);
    trajectory2 zeros(n_steps,n_dim-trajsize);
    constant2 zeros(n_density,n_dim-trajsize)];
    

% Split into 2 branches
branch1=[constant trajectory zeros(n_steps,n_dim-2*trajsize);
    constant2 constant2 zeros(n_density,n_dim-2*trajsize)];
branch2=[constant zeros(n_steps,trajsize) -trajectory3 zeros(n_steps,n_dim-3*trajsize);
    constant2 zeros(n_density,trajsize) -constant2 zeros(n_density,n_dim-3*trajsize)];

M=[M;branch1; branch2];

% Top branch into 3 branches

branch1=[repmat(constant,1,2) zeros(n_steps,trajsize) trajectory3 zeros(n_steps,n_dim-4*trajsize);
    repmat(constant2,1,2) zeros(n_density,trajsize) constant2 zeros(n_density,n_dim-4*trajsize)];
branch2=[repmat(constant,1,2) zeros(n_steps,2*trajsize) -1.85*trajectory zeros(n_steps,n_dim-5*trajsize);
    repmat(constant2,1,2) zeros(n_density,2*trajsize) -1.85*constant2 zeros(n_density,n_dim-5*trajsize)];
branch3=[repmat(constant,1,2) zeros(n_steps,2*trajsize) 1.85*trajectory zeros(n_steps,n_dim-5*trajsize);
    repmat(constant2,1,2) zeros(n_density,2*trajsize) 1.85*constant2 zeros(n_density,n_dim-5*trajsize)];

M=[M; branch1; branch2; branch3];

% Split one of those top branches into 2
branch1=[repmat(constant,1,2) zeros(n_steps,trajsize) constant zeros(n_steps,2*trajsize) -2*trajectory zeros(n_steps,n_dim-7*trajsize);
    repmat(constant2,1,2) zeros(n_density,trajsize) constant2 zeros(n_density,2*trajsize) -2*constant2 zeros(n_density,n_dim-7*trajsize)];
branch2=[repmat(constant,1,2) zeros(n_steps,trajsize) constant zeros(n_steps,3*trajsize) 2*trajectory zeros(n_steps,n_dim-8*trajsize);
    repmat(constant2,1,2) zeros(n_density,trajsize) constant2 zeros(n_density,3*trajsize) 2*constant2 zeros(n_density,n_dim-8*trajsize)];

M=[M; branch1; branch2];

% Bottom branch

branch1=[constant zeros(n_steps,trajsize) -constant zeros(n_steps,5*trajsize) 2*trajectory zeros(n_steps,n_dim-9*trajsize);
    constant2 zeros(n_density,trajsize) -constant2 zeros(n_density,5*trajsize) 2*constant2 zeros(n_density,n_dim-9*trajsize)];
branch2=[constant zeros(n_steps,trajsize) -constant zeros(n_steps,6*trajsize) -2*trajectory zeros(n_steps,n_dim-10*trajsize);
    constant2 zeros(n_density,trajsize) -constant2 zeros(n_density,6*trajsize) -2*constant2 zeros(n_density,n_dim-10*trajsize)];


M=[M; branch1; branch2];

% % add branches to one of the other middle branches
% 


M=[M zeros(n_steps*n_branch+n_density*(n_branch+1),n_dim/2)];


C=ones(2*n_density+n_steps,1);
for m=2:n_branch

    C=[C; m*ones(n_steps+n_density,1)];
    
end

sigma = 7;
M = M + normrnd(0, sigma, size(M,1), size(M,2));

[p,pc]=pca(M,'numComponents',size(M,2)-5);
M=pc*p';