function t = vne_optimal_t(P, t_max)

% VNE
Smm = svds(P, length(P));
vne = nan(1,t_max);
for I=1:t_max
    Smm_t = Smm.^I + eps;
    p_vne = Smm_t ./ sum(Smm_t);
    vne(I) = -sum(p_vne .* log(p_vne));
end

t = knee_pt(vne, 1:t_max, true);

figure;
hold on;
plot(1:t_max, vne, '-');
xlim([1 t_max]);
xlabel 't'
ylabel 'VNE'

if ~isnan(t)
    display(['VNE optimal t = ' num2str(t)]);
else
    error('optimal t not found -- increase t_max')
end

plot([t t], [vne(t) vne(t)], '*k', 'markersize', 15, 'linewidth', 2);
title(['VNE optimal t = ' num2str(t)]);
xlabel 't'
ylabel 'Von Neumann Entropy'
drawnow;