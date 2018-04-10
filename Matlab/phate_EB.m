%% init
load('/Users/david/Downloads/EBPHATEDavid.mat')
load('/Users/david/Downloads/EBdata.mat')
data = full(data);

%% lib size norm global
libsize = sum(data,2);
data = bsxfun(@rdivide, data, libsize) * median(libsize);

%% sqrt transform
data = sqrt(data);

%% fast PHATE
rng(7)
tic;
Y_mmds_fast = phate_fast(data, 'k', 7, 'ndim', 2, 't', [], 'npca', 100, 'nsvd', 100, ...
    'ncluster', 1000, 'pot_method', 'log');
toc
[~,Y_mmds_fast] = procrustes(Ymet2, Y_mmds_fast);

%% plot PHATE 2D
figure;
scatter(Y_mmds_fast(:,1), Y_mmds_fast(:,2), 3, cells, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
title 'PHATE fast (33 sec., 17k cells)'
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'PHATE_fast_EB.tiff']);
%close

%% fast PHATE 3D
tic;
Y_mmds_fast_3D = phate_fast(data, 'k', 7, 'ndim', 3, 't', 24, 'npca', 100, 'nsvd', 100, ...
    'ncluster', 2000, 'pot_method', 'log');
toc

%% plot PHATE 3D
figure;
scatter3(Y_mmds_fast_3D(:,1), Y_mmds_fast_3D(:,2), Y_mmds_fast_3D(:,3), 3, cells, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
zlabel 'PHATE3'

%% slow vs fast PHATE
figure;
subplot(1,2,1)
scatter(Ymet2(:,1), Ymet2(:,2), 3, cells, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
title 'PHATE original (20 min)'
subplot(1,2,2)
scatter(Y_mmds_fast(:,1), Y_mmds_fast(:,2), 3, cells, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PHATE1'
ylabel 'PHATE2'
title 'PHATE fast (33 sec)'
set(gcf,'paperposition',[0 0 16 6]);
print('-dtiff',[out_base 'PHATE_EB_slow_vs_fast.tiff']);
close
