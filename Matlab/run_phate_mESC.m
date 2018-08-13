%% init
rseed = 7;

%% load mESC data Klein et al. 2015 (doi:10.1016/j.cell.2015.04.044)
file_name = '../data/GSM1599499_ES_d7_LIFminus.csv';
fid = fopen(file_name);
line1 = strsplit(fgetl(fid),',');
ncol = length(line1); % get number of cols
fclose(fid);
fid = fopen(file_name);
format = ['%s' repmat('%f',1,ncol-1)];
file_data = textscan(fid, format, 'delimiter', ',');
fclose(fid);
genes = file_data{1};
data = cell2mat(file_data(2:end));
M = data';
size(M)

%% library size normalization
libsize  = sum(M,2);
M = bsxfun(@rdivide, M, libsize) * median(libsize);

%% log/sqrt transform (some data requires transform)
%M = log(M + 0.1); % 0.1 pseudocount
%M = sqrt(M);

%% PCA
Y_PCA = svdpca(M, 2, 'random');

%% Plot the embedding colored by gene
gene = 'Actb';
ind = ismember(genes, gene);
C = log(M(:,ind) + 0.1);
figure;
scatter(Y_PCA(:,1),Y_PCA(:,2),20,C,'filled')
axis tight
xlabel('PCA 1')
ylabel('PCA 2')
set(gca,'xtick',[])
set(gca,'ytick',[])
title 'PCA'
h = colorbar;
ylabel(h, gene);
drawnow

%% PHATE 2D
Y_PHATE_2D = phate(M);

%% Plot PHATE 2D
gene = 'Actb';
ind = ismember(genes, gene);
C = log(M(:,ind) + 0.1);
figure;
scatter(Y_PHATE_2D(:,1),Y_PHATE_2D(:,2),20,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
title 'PHATE 2D'
h = colorbar;
ylabel(h, gene);
drawnow

%% PHATE 3D
Y_PHATE_3D = phate(M, 'ndim', 3);

%% Plot PHATE 3D
gene = 'Actb';
ind = ismember(genes, gene);
C = log(M(:,ind) + 0.1);
figure;
scatter3(Y_PHATE_3D(:,1),Y_PHATE_3D(:,2),Y_PHATE_3D(:,3),20,C,'filled')
axis tight
xlabel('PHATE 1')
ylabel('PHATE 2')
ylabel('PHATE 3')
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
set(gca,'zticklabel',[])
title 'PHATE 3D'
h = colorbar;
ylabel(h, gene);
drawnow

%% tSNE
tic;
Y_tSNE = tsne(M,'Theta',0.5,'NumPCAComponents',100,'Verbose',2, 'perplexity', 20);
toc

%% Plot tSNE
gene = 'Actb';
ind = ismember(genes, gene);
C = log(M(:,ind) + 0.1);
figure;
scatter(Y_tSNE(:,1),Y_tSNE(:,2),20,C,'filled')
axis tight
xlabel('tSNE 1')
ylabel('tSNE 2')
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
title 'tSNE'
h = colorbar;
ylabel(h, gene);
drawnow


%% plot combined
gene = 'Actb';
ind = ismember(genes, gene);
C = log(M(:,ind) + 0.1);

figure;

subplot(2,2,1);
scatter(Y_PCA(:,1), Y_PCA(:,2), 10, C, 'filled');
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PCA1'
ylabel 'PCA2'
title 'PCA'

subplot(2,2,2);
scatter(Y_tSNE(:,1), Y_tSNE(:,2), 10, C, 'filled');
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'tSNE1'
ylabel 'tSNE2'
title 'tSNE'

subplot(2,2,3);
scatter(Y_PHATE_2D(:,1), Y_PHATE_2D(:,2), 10, C, 'filled');
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
title 'PHATE'

subplot(2,2,4);
scatter3(Y_PHATE_3D(:,1), Y_PHATE_3D(:,2), Y_PHATE_3D(:,3), 10, C, 'filled');
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
zlabel 'PHATE3'
title 'PHATE'
view([165 35]);


