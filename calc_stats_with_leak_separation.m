function S = calc_stats_with_leak_separation(H_True, H_Est, Nodes, Pipes, Sys, t, leak_start_h, leak_end_h, exclude_ranges, outlier_info)
% CALC_STATS_WITH_LEAK_SEPARATION - Calculate statistics with leak period separation
% Hard-coded time periods:
%   Normal: 0-6h and 18-24h
%   Leak: 12-18h (Case 3) or 12-24h (Case 4)
%   Excluded: 6-12h (outlier period for Case 3)

nN = height(Nodes);
nP = height(Pipes);
PT = H_True(:,1:nN) * Sys.c2 / 1e5;
PE = H_Est(:,1:nN) * Sys.c2 / 1e5;
MT = H_True(:,nN+1:end);
ME = H_Est(:,nN+1:end);

N = length(t);
t = t(:)';  % Ensure row vector

% Create time masks with hard-coded periods
% Normal period: 0-6h and 18-24h (no outliers, no leak)
normal_mask = ((t >= 0) & (t < 6)) | ((t >= 18) & (t <= 24));

% Leak period: use input parameters
leak_mask = (t >= leak_start_h) & (t <= leak_end_h);

% Exclude outlier period if specified
if nargin >= 9 && ~isempty(exclude_ranges)
    for i = 1:size(exclude_ranges, 1)
        t_start = exclude_ranges(i, 1);
        t_end = exclude_ranges(i, 2);
        exclude_idx = (t >= t_start) & (t <= t_end);
        normal_mask(exclude_idx) = false;
        leak_mask(exclude_idx) = false;
    end
end

% Full period mask (exclude outlier period)
full_mask = true(1, N);
if nargin >= 9 && ~isempty(exclude_ranges)
    for i = 1:size(exclude_ranges, 1)
        t_start = exclude_ranges(i, 1);
        t_end = exclude_ranges(i, 2);
        exclude_idx = (t >= t_start) & (t <= t_end);
        full_mask(exclude_idx) = false;
    end
end

% Convert to column for indexing
normal_mask = normal_mask(:);
leak_mask = leak_mask(:);
full_mask = full_mask(:);

% Full period statistics
PT_full = PT(full_mask, :);
PE_full = PE(full_mask, :);
MT_full = MT(full_mask, :);
ME_full = ME(full_mask, :);

err_p_full = PE_full - PT_full;
err_m_full = ME_full - MT_full;

S.P_RMSE = sqrt(mean(err_p_full(:).^2));
S.P_MAE = mean(abs(err_p_full(:)));
S.M_RMSE = sqrt(mean(err_m_full(:).^2));
S.M_MAE = mean(abs(err_m_full(:)));

% Normal period statistics (0-6h and 18-24h)
if sum(normal_mask) > 0
    PT_normal = PT(normal_mask, :);
    PE_normal = PE(normal_mask, :);
    MT_normal = MT(normal_mask, :);
    ME_normal = ME(normal_mask, :);
    
    err_p_normal = PE_normal - PT_normal;
    err_m_normal = ME_normal - MT_normal;
    
    S.P_RMSE_Normal = sqrt(mean(err_p_normal(:).^2));
    S.P_MAE_Normal = mean(abs(err_p_normal(:)));
    S.M_RMSE_Normal = sqrt(mean(err_m_normal(:).^2));
    S.M_MAE_Normal = mean(abs(err_m_normal(:)));
else
    S.P_RMSE_Normal = NaN;
    S.P_MAE_Normal = NaN;
    S.M_RMSE_Normal = NaN;
    S.M_MAE_Normal = NaN;
end

% Leak period statistics
if sum(leak_mask) > 0
    PT_leak = PT(leak_mask, :);
    PE_leak = PE(leak_mask, :);
    MT_leak = MT(leak_mask, :);
    ME_leak = ME(leak_mask, :);
    
    err_p_leak = PE_leak - PT_leak;
    err_m_leak = ME_leak - MT_leak;
    
    S.P_RMSE_Leak = sqrt(mean(err_p_leak(:).^2));
    S.P_MAE_Leak = mean(abs(err_p_leak(:)));
    S.M_RMSE_Leak = sqrt(mean(err_m_leak(:).^2));
    S.M_MAE_Leak = mean(abs(err_m_leak(:)));
else
    S.P_RMSE_Leak = NaN;
    S.P_MAE_Leak = NaN;
    S.M_RMSE_Leak = NaN;
    S.M_MAE_Leak = NaN;
end

% Per-pipe statistics (Pipe 1-10)
for pp = 1:min(10, nP)
    err_pp_full = ME_full(:, pp) - MT_full(:, pp);
    S.(sprintf('Pipe%d_RMSE', pp)) = sqrt(mean(err_pp_full.^2));
    S.(sprintf('Pipe%d_MAE', pp)) = mean(abs(err_pp_full));
    
    if sum(normal_mask) > 0
        ME_normal = ME(normal_mask, :);
        MT_normal = MT(normal_mask, :);
        err_pp_normal = ME_normal(:, pp) - MT_normal(:, pp);
        S.(sprintf('Pipe%d_RMSE_Normal', pp)) = sqrt(mean(err_pp_normal.^2));
        S.(sprintf('Pipe%d_MAE_Normal', pp)) = mean(abs(err_pp_normal));
    else
        S.(sprintf('Pipe%d_RMSE_Normal', pp)) = NaN;
        S.(sprintf('Pipe%d_MAE_Normal', pp)) = NaN;
    end
    
    if sum(leak_mask) > 0
        ME_leak = ME(leak_mask, :);
        MT_leak = MT(leak_mask, :);
        err_pp_leak = ME_leak(:, pp) - MT_leak(:, pp);
        S.(sprintf('Pipe%d_RMSE_Leak', pp)) = sqrt(mean(err_pp_leak.^2));
        S.(sprintf('Pipe%d_MAE_Leak', pp)) = mean(abs(err_pp_leak));
    else
        S.(sprintf('Pipe%d_RMSE_Leak', pp)) = NaN;
        S.(sprintf('Pipe%d_MAE_Leak', pp)) = NaN;
    end
end
% Fill NaN for missing pipes
for pp = (nP+1):10
    S.(sprintf('Pipe%d_RMSE', pp)) = NaN;
    S.(sprintf('Pipe%d_MAE', pp)) = NaN;
    S.(sprintf('Pipe%d_RMSE_Normal', pp)) = NaN;
    S.(sprintf('Pipe%d_MAE_Normal', pp)) = NaN;
    S.(sprintf('Pipe%d_RMSE_Leak', pp)) = NaN;
    S.(sprintf('Pipe%d_MAE_Leak', pp)) = NaN;
end

% Sample counts
S.N_Total = N;
S.N_Used_Full = sum(full_mask);
S.N_Used_Normal = sum(normal_mask);
S.N_Used_Leak = sum(leak_mask);

end