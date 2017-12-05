% Load tree data generated in GenerateTree.m. See paper for details in how this data is generated

load TreeData.mat

% Run PHATE
a=20;
k=15;
t=375;

Y = phate(M,'a',a,'k',k,'t',t,'mds_method','mmds','ndim',2,'pca_method','none');

% Display the embedding colored by branches
scatter(Y(:,1),Y(:,2),5,C,'filled')
