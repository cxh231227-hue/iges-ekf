function dse_main_four_methods(mismatch, ifplot)
% DSE_MAIN_FOUR_METHODS - Four-method EKF comparison test
%
% Output structure (all relative to THIS script's directory):
%   ./csv_results/small/    <- mismatch < 1
%   ./csv_results/normal/   <- mismatch == 1
%   ./csv_results/big/      <- mismatch > 1
%   ./figures/              <- all .fig files

if ifplot == 1
    rng(42);
end

basedir = fileparts(mfilename('fullpath'));

figdir        = fullfile(basedir, 'figures');
csvdir_small  = fullfile(basedir, 'csv_results', 'small');
csvdir_normal = fullfile(basedir, 'csv_results', 'normal');
csvdir_big    = fullfile(basedir, 'csv_results', 'big');

for d = {figdir, csvdir_small, csvdir_normal, csvdir_big}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

if mismatch < 1
    csvdir = csvdir_small;
elseif mismatch == 1
    csvdir = csvdir_normal;
else
    csvdir = csvdir_big;
end

% System parameters
Sys.c  = 340;
Sys.c2 = 340^2;
Sys.dt = 600;
Sys.Hours = 24;

% Load gas network
[Nodes, Pipes, Compressors, GTU] = dse_load_gaslib40( ...
    'GasLib-40-v1-20211130.net', 'GasLib-40-v1-20211130.scn');

% Load power system (IEEE 39-bus)
PowerSys = power_load_ieee39();

% Fill in actual GasLib-40 BaseLoad for each GTU node
% so gtu_coupling can scale correctly
for k = 1:length(PowerSys.gtu)
    gid = PowerSys.gtu(k).gas_node_id;
    PowerSys.gtu(k).base_load = Nodes.BaseLoad(gid);
end

% Generate power state trajectory and PMU measurements
[X_E_true, Z_E, ~] = power_gen_trajectory(PowerSys, Sys);

fprintf('\n========================================================\n');
fprintf('   Four-Method EKF Comparison Test (with AFEKF)\n');
fprintf('========================================================\n');
fprintf('Methods:\n');
fprintf('  1. Standard EKF (baseline)\n');
fprintf('  2. Chen Robust EKF (innovation covariance)\n');
fprintf('  3. EKF-LE (detector feedback)\n');
fprintf('  4. AFEKF (adaptive fading + robust statistics)\n');
fprintf('--------------------------------------------------------\n');
fprintf('Cases:\n');
fprintf('  Case 1: Clean data\n');
fprintf('  Case 2: Sparse outliers (6h-12h)\n');
fprintf('  Case 3: Outliers + Leak\n');
fprintf('========================================================\n\n');

% =========================================================================
% CASE 1: Clean data
% =========================================================================
fprintf('CASE 1: Clean data (no outliers, no leaks)\n');
fprintf('--------------------------------------------------------\n');

Leaks_empty = table([],[],[],[],[], ...
    'VariableNames',{'PipeID','StartTime_s','EndTime_s','LeakRate_kg_s','Position'});

[H_True_c1, Z_clean, t, ~, P_GTU] = ...
    dse_3_gen_data_leak(Nodes, Pipes, Compressors, Sys, Leaks_empty, GTU, PowerSys, X_E_true);

[H_Normal_c1, H_Chen_c1, H_Adaptive_c1, H_AFEKF_c1, Stats_c1, ~, Diag_c1] = ...
    run_four_methods(Z_clean, H_True_c1, Nodes, Pipes, Compressors, Sys, t, GTU, ...
                     true, [], [], mismatch, PowerSys, Z_E);

if ifplot == 1
    plot_case1_four(H_True_c1, Z_clean, H_Normal_c1, H_Chen_c1, H_Adaptive_c1, H_AFEKF_c1, ...
        Nodes, Pipes, Sys, t, GTU, figdir);
end
save_case_results_four('Case1_Clean', Stats_c1);

% =========================================================================
% CASE 2: Sparse outliers (6h-12h, 20 dB)
% =========================================================================
fprintf('\nCASE 2: Sparse outliers (6h-12h)\n');
fprintf('--------------------------------------------------------\n');

N  = length(t);
nN = height(Nodes);

outlier_config_c2 = struct();
outlier_config_c2.start_step        = round(N * 0.25);
outlier_config_c2.end_step          = round(N * 0.5);
outlier_config_c2.target_nodes      = [2, 3, 4, 5, 7, 8, 10];
outlier_config_c2.target_pipes      = [1, 2, 3, 4, 7, 8, 10];
outlier_config_c2.outliers_per_hour = 2;
outlier_config_c2.nN                = nN;

[Z_outlier_20, outlier_info_20] = inject_outlier_sparse(Z_clean, outlier_config_c2, Sys, 20);

[H_Normal_c2, H_Chen_c2, H_Adaptive_c2, H_AFEKF_c2, Stats_c2, ~, ~] = ...
    run_four_methods(Z_outlier_20, H_True_c1, Nodes, Pipes, Compressors, Sys, t, GTU, ...
                     true, [6, 12], outlier_info_20, mismatch, PowerSys, Z_E);

if ifplot == 1
    plot_case2_four(H_True_c1, Z_outlier_20, H_Normal_c2, H_Chen_c2, H_Adaptive_c2, H_AFEKF_c2, ...
        Nodes, Pipes, Sys, t, figdir, outlier_config_c2);
end
save_case2_results_four('Case2_Outlier_20dB', Stats_c2);

% =========================================================================
% CASE 3: Outliers + Leak
% =========================================================================
fprintf('\nCASE 3: Outliers + Leak\n');
fprintf('Timeline: 0-6h normal | 6h-12h outliers | 12h-16h leak | 12h-24h normal\n');
fprintf('--------------------------------------------------------\n');

Leaks_c3 = dse_load_leak('leak.xlsx');
if height(Leaks_c3) > 0
    fprintf('Leak config: %.1fh-%.1fh, rate %.1f kg/s\n', ...
        Leaks_c3.StartTime_s(1)/3600, Leaks_c3.EndTime_s(1)/3600, Leaks_c3.LeakRate_kg_s(1));
end

[H_True_c3, Z_leak_clean_c3, t, Leak_True_c3, P_GTU] = ...
    dse_3_gen_data_leak(Nodes, Pipes, Compressors, Sys, Leaks_c3, GTU, PowerSys, X_E_true);

outlier_config = struct();
outlier_config.start_step        = round(N * 0.25);
outlier_config.end_step          = round(N * 0.5);
outlier_config.target_nodes      = [2, 3, 4, 5, 7, 8, 10];
outlier_config.target_pipes      = [1, 2, 3, 4, 7, 8, 10];
outlier_config.outliers_per_hour = 2;
outlier_config.nN                = nN;

[Z_outlier_c3, outlier_info_c3] = inject_outlier_sparse(Z_leak_clean_c3, outlier_config, Sys, 20);

leak_start_h = Leaks_c3.StartTime_s(1) / 3600;
leak_end_h   = Leaks_c3.EndTime_s(1)   / 3600;

[H_Normal_c3, H_Chen_c3, H_Adaptive_c3, H_AFEKF_c3, Stats_c3, Leak_Est_c3, Diag_c3] = ...
    run_four_methods_with_leak_stats(Z_outlier_c3, H_True_c3, Nodes, Pipes, Compressors, Sys, t, GTU, ...
                                     true, leak_start_h, leak_end_h, [6, 12], outlier_info_c3, ...
                                     mismatch, PowerSys, Z_E);

if ifplot == 1
    plot_case3_four(H_True_c3, Z_outlier_c3, H_Normal_c3, H_Chen_c3, H_Adaptive_c3, H_AFEKF_c3, ...
        Nodes, Pipes, Sys, t, GTU, Leaks_c3, outlier_config, outlier_info_c3, figdir);
end
save_case3_results_four('Case3_Out_Leak', Stats_c3, Leak_True_c3, Leak_Est_c3);

% =========================================================================
% Summary
% =========================================================================
fprintf('\n========================================================\n');
fprintf('   Four-Method RMSE Summary\n');
fprintf('========================================================\n');
print_summary_table_four_methods(Stats_c1, Stats_c2, Stats_c3);

save_comprehensive_results_four_methods(Stats_c1, Stats_c2, Stats_c3, ...
    fullfile(basedir, 'Comprehensive_Results_4Methods.xlsx'));

export_results_to_csv(Stats_c1, Stats_c2, Stats_c3, csvdir);

fprintf('\n========================================================\n');
fprintf('   All cases complete.\n');
fprintf('========================================================\n');
fprintf('Figures: %s\n', figdir);
fprintf('CSV:     %s\n', csvdir);
fprintf('Excel:   %s\n', fullfile(basedir, 'Comprehensive_Results_4Methods.xlsx'));

end


% =========================================================================
% RUN FOUR METHODS
% =========================================================================
function [H_Normal, H_Chen, H_Adaptive, H_AFEKF, Stats, Leak_Est, Diag] = ...
    run_four_methods(Z, H_True, Nodes, Pipes, Compressors, Sys, t, GTU, ...
                     enable_detector, exclude_ranges, outlier_info, mismatch, PowerSys, Z_E)

nN = height(Nodes);

if nargin < 11
    outlier_info = [];
end

fprintf('  [1/4] Running Standard EKF...\n');
[H_Normal, ~, ~] = dse_normal_ekf(Z, Nodes, Pipes, Compressors, Sys, t, GTU, mismatch, PowerSys, Z_E);

fprintf('  [2/4] Running Chen Robust EKF...\n');
[H_Chen, ~, ~, ~, ~] = dse_chen_ekf(Z, Nodes, Pipes, Compressors, Sys, t, GTU, mismatch, PowerSys, Z_E);

if enable_detector
    fprintf('  [3/4] Running 3-layer Detector + EKF-LE...\n');
    [Det, Diag, ~] = dse_leak_detector(H_Normal, Z, Nodes, Pipes, Sys, t);
    [H_Adaptive, Leak_Est_Nodes, ~, ~, ~] = dse_4_estimator_leak(Z, Nodes, Pipes, Compressors, Sys, t, GTU, Det, mismatch, PowerSys, Z_E);
    Leak_Est = sum(Leak_Est_Nodes, 2);
else
    fprintf('  [3/4] Running EKF-LE (no detector)...\n');
    Det = false(length(t), 1);
    [H_Adaptive, Leak_Est_Nodes, ~, ~, ~] = dse_4_estimator_leak(Z, Nodes, Pipes, Compressors, Sys, t, GTU, Det, mismatch, PowerSys, Z_E);
    Leak_Est = sum(Leak_Est_Nodes, 2);
    Diag = struct();
    Diag.Outlier_mask = false(length(t), nN + height(Pipes));
    Diag.Leak_mask    = false(length(t), nN + height(Pipes));
end

fprintf('  [4/4] Running AFEKF (Adaptive Fading)...\n');
if enable_detector
    [H_AFEKF, ~, ~, ~, ~, ~] = AFUKF(Z, Nodes, Pipes, Compressors, Sys, t, GTU, Det, mismatch, PowerSys, Z_E);
else
    [H_AFEKF, ~, ~, ~, ~, ~] = AFUKF(Z, Nodes, Pipes, Compressors, Sys, t, GTU, [],  mismatch, PowerSys, Z_E);
end

Stats.Normal   = calc_stats_simple(H_True, H_Normal,   Nodes, Pipes, Sys, t, exclude_ranges, outlier_info);
Stats.Chen     = calc_stats_simple(H_True, H_Chen,     Nodes, Pipes, Sys, t, exclude_ranges, outlier_info);
Stats.Adaptive = calc_stats_simple(H_True, H_Adaptive, Nodes, Pipes, Sys, t, exclude_ranges, outlier_info);
Stats.AFEKF    = calc_stats_simple(H_True, H_AFEKF,    Nodes, Pipes, Sys, t, exclude_ranges, outlier_info);

fprintf('  Results (Pipe RMSE [kg/s]):\n');
for pp = [1,2,3,4,5,7,8,10]
    fn = sprintf('Pipe%d_RMSE', pp);
    fprintf('    Pipe%d: Standard=%.4f | Chen=%.4f | EKF-LE=%.4f | AFEKF=%.4f\n', pp, ...
        Stats.Normal.(fn), Stats.Chen.(fn), Stats.Adaptive.(fn), Stats.AFEKF.(fn));
end
end


% =========================================================================
% RUN FOUR METHODS WITH LEAK STATS
% =========================================================================
function [H_Normal, H_Chen, H_Adaptive, H_AFEKF, Stats, Leak_Est, Diag] = ...
    run_four_methods_with_leak_stats(Z, H_True, Nodes, Pipes, Compressors, Sys, t, GTU, ...
                                     enable_detector, leak_start_h, leak_end_h, ...
                                     exclude_ranges, outlier_info, mismatch, PowerSys, Z_E)

nN = height(Nodes);

fprintf('  [1/4] Running Standard EKF...\n');
[H_Normal, ~, ~] = dse_normal_ekf(Z, Nodes, Pipes, Compressors, Sys, t, GTU, mismatch, PowerSys, Z_E);

fprintf('  [2/4] Running Chen Robust EKF...\n');
[H_Chen, ~, ~, ~, ~] = dse_chen_ekf(Z, Nodes, Pipes, Compressors, Sys, t, GTU, mismatch, PowerSys, Z_E);

if enable_detector
    fprintf('  [3/4] Running 3-layer Detector + EKF-LE...\n');
    [Det, Diag, ~] = dse_leak_detector(H_Normal, Z, Nodes, Pipes, Sys, t);
    [H_Adaptive, Leak_Est_Nodes, ~, ~, ~] = dse_4_estimator_leak(Z, Nodes, Pipes, Compressors, Sys, t, GTU, Det, mismatch, PowerSys, Z_E);
    Leak_Est = sum(Leak_Est_Nodes, 2);
else
    fprintf('  [3/4] Running EKF-LE (no detector)...\n');
    Det = false(length(t), 1);
    [H_Adaptive, Leak_Est_Nodes, ~, ~, ~] = dse_4_estimator_leak(Z, Nodes, Pipes, Compressors, Sys, t, GTU, Det, mismatch, PowerSys, Z_E);
    Leak_Est = sum(Leak_Est_Nodes, 2);
    Diag = struct();
end

fprintf('  [4/4] Running AFEKF (Adaptive Fading)...\n');
if enable_detector
    [H_AFEKF, ~, ~, ~, ~, ~] = AFUKF(Z, Nodes, Pipes, Compressors, Sys, t, GTU, Det, mismatch, PowerSys, Z_E);
else
    [H_AFEKF, ~, ~, ~, ~, ~] = AFUKF(Z, Nodes, Pipes, Compressors, Sys, t, GTU, [],  mismatch, PowerSys, Z_E);
end

if ~isnan(leak_start_h)
    fprintf('  Computing RMSE (separating normal and leak periods)...\n');
    Stats.Normal   = calc_stats_with_leak_separation(H_True, H_Normal,   Nodes, Pipes, Sys, t, leak_start_h, leak_end_h, exclude_ranges, outlier_info);
    Stats.Chen     = calc_stats_with_leak_separation(H_True, H_Chen,     Nodes, Pipes, Sys, t, leak_start_h, leak_end_h, exclude_ranges, outlier_info);
    Stats.Adaptive = calc_stats_with_leak_separation(H_True, H_Adaptive, Nodes, Pipes, Sys, t, leak_start_h, leak_end_h, exclude_ranges, outlier_info);
    Stats.AFEKF    = calc_stats_with_leak_separation(H_True, H_AFEKF,    Nodes, Pipes, Sys, t, leak_start_h, leak_end_h, exclude_ranges, outlier_info);

    fprintf('  Results (Normal Period P_RMSE [Bar]):\n');
    fprintf('    Standard: %.4f | Chen: %.4f | EKF-LE: %.4f | AFEKF: %.4f\n', ...
        Stats.Normal.P_RMSE_Normal, Stats.Chen.P_RMSE_Normal, Stats.Adaptive.P_RMSE_Normal, Stats.AFEKF.P_RMSE_Normal);
    fprintf('  Results (Leak Period P_RMSE [Bar]):\n');
    fprintf('    Standard: %.4f | Chen: %.4f | EKF-LE: %.4f | AFEKF: %.4f\n', ...
        Stats.Normal.P_RMSE_Leak, Stats.Chen.P_RMSE_Leak, Stats.Adaptive.P_RMSE_Leak, Stats.AFEKF.P_RMSE_Leak);
else
    Stats.Normal   = calc_stats_simple(H_True, H_Normal,   Nodes, Pipes, Sys, t, exclude_ranges, outlier_info);
    Stats.Chen     = calc_stats_simple(H_True, H_Chen,     Nodes, Pipes, Sys, t, exclude_ranges, outlier_info);
    Stats.Adaptive = calc_stats_simple(H_True, H_Adaptive, Nodes, Pipes, Sys, t, exclude_ranges, outlier_info);
    Stats.AFEKF    = calc_stats_simple(H_True, H_AFEKF,    Nodes, Pipes, Sys, t, exclude_ranges, outlier_info);
end
end


% =========================================================================
% STATISTICS CALCULATION
% =========================================================================
function S = calc_stats_simple(H_True, H_Est, Nodes, Pipes, Sys, t, exclude_ranges, outlier_info)

nN = height(Nodes);
nP = height(Pipes);
PT = H_True(:,1:nN) * Sys.c2 / 1e5;
PE = H_Est(:,1:nN)  * Sys.c2 / 1e5;
MT = H_True(:,nN+1:end);
ME = H_Est(:,nN+1:end);

N    = length(t);
mask = true(N, 1);

if nargin >= 7 && ~isempty(exclude_ranges)
    for i = 1:size(exclude_ranges, 1)
        exclude_idx = (t >= exclude_ranges(i,1)) & (t <= exclude_ranges(i,2));
        mask(exclude_idx) = false;
    end
end

if nargin >= 8 && ~isempty(outlier_info) && isstruct(outlier_info)
    outlier_steps = [];
    if isfield(outlier_info, 'node_outliers') && ~isempty(outlier_info.node_outliers)
        outlier_steps = [outlier_steps; outlier_info.node_outliers(:,1)];
    end
    if isfield(outlier_info, 'pipe_outliers') && ~isempty(outlier_info.pipe_outliers)
        outlier_steps = [outlier_steps; outlier_info.pipe_outliers(:,1)];
    end
    outlier_steps = unique(outlier_steps);
    for i = 1:length(outlier_steps)
        s = outlier_steps(i);
        if s >= 1 && s <= N, mask(s) = false; end
    end
end

err_p = PE(mask,:) - PT(mask,:);
err_m = ME(mask,:) - MT(mask,:);

S.P_RMSE = sqrt(mean(err_p(:).^2));
S.P_MAE  = mean(abs(err_p(:)));
S.M_RMSE = sqrt(mean(err_m(:).^2));
S.M_MAE  = mean(abs(err_m(:)));

for pp = 1:min(10, nP)
    err_pp = ME(mask,pp) - MT(mask,pp);
    S.(sprintf('Pipe%d_RMSE', pp)) = sqrt(mean(err_pp.^2));
    S.(sprintf('Pipe%d_MAE',  pp)) = mean(abs(err_pp));
end
for pp = (nP+1):10
    S.(sprintf('Pipe%d_RMSE', pp)) = NaN;
    S.(sprintf('Pipe%d_MAE',  pp)) = NaN;
end
end


% =========================================================================
% PLOTTING FUNCTIONS
% =========================================================================
function plot_one(t, tr, zm, no, ch, ad, af, ylabel_str, title_str, fname, figdir)
fig = figure('Color','w','Position',[100 100 900 500]);
hold on; box on;
plot(t, tr, 'g-',  'LineWidth',2.5, 'DisplayName','true value');
plot(t, zm, 'k--', 'LineWidth',1,   'DisplayName','measurement');
plot(t, no, 'b:',  'LineWidth',2,   'DisplayName','Standard EKF');
plot(t, ch, 'm-.', 'LineWidth',1.8, 'DisplayName','Chen Robust');
plot(t, ad, 'r--', 'LineWidth',1.8, 'DisplayName','EKF-LE');
plot(t, af, 'c-',  'LineWidth',2.2, 'DisplayName','AFEKF');
xlabel('Time (hour)','FontSize',12);
ylabel(ylabel_str,'FontSize',12);
title(title_str,'FontSize',13,'FontWeight','bold');
legend('Location','best','FontSize',9);
grid on; xlim([0 24]);
savefig(fig, fullfile(figdir, fname));
end

function plot_one_shaded(t, tr, zm, no, ch, ad, af, ylabel_str, title_str, fname, figdir, shade_segs)
fig = figure('Color','w','Position',[100 100 900 500]);
hold on; box on;
y_lo = min([tr;no;ch;ad;af]) * 0.995;
y_hi = max([tr;no;ch;ad;af]) * 1.005;
for s = 1:length(shade_segs)
    seg = shade_segs{s};
    fill([seg{1}(1) seg{1}(2) seg{1}(2) seg{1}(1)], ...
         [y_lo y_lo y_hi y_hi], seg{2}, ...
         'EdgeColor','none','FaceAlpha',0.3,'DisplayName',seg{3});
end
plot(t, tr, 'g-',  'LineWidth',2.5, 'DisplayName','true value');
plot(t, zm, 'k--', 'LineWidth',1,   'DisplayName','measurement');
plot(t, no, 'b:',  'LineWidth',2,   'DisplayName','Standard EKF');
plot(t, ch, 'm-.', 'LineWidth',1.8, 'DisplayName','Chen Robust');
plot(t, ad, 'r--', 'LineWidth',1.8, 'DisplayName','EKF-LE');
plot(t, af, 'c-',  'LineWidth',2.2, 'DisplayName','AFEKF');
xlabel('Time (hour)','FontSize',12);
ylabel(ylabel_str,'FontSize',12);
title(title_str,'FontSize',13,'FontWeight','bold');
legend('Location','best','FontSize',9);
grid on; xlim([0 24]);
savefig(fig, fullfile(figdir, fname));
end

function plot_case1_four(H_True, Z, H_Normal, H_Chen, H_Adaptive, H_AFEKF, Nodes, Pipes, Sys, t, GTU, figdir)
c2 = Sys.c2; nN = height(Nodes);
p  = @(H,i) H(:,i)*c2/1e5;
fl = @(H,k) abs(H(:,nN+k));
plot_one(t, p(H_True,7),   p(Z,7),   p(H_Normal,7),   p(H_Chen,7),   p(H_Adaptive,7),   p(H_AFEKF,7),   'Pressure (bar)',        'Case 1: Pressure at Node 4 (sink\_4)',          'Case1_P_Node4.fig',   figdir);
plot_one(t, p(H_True,27),  p(Z,27),  p(H_Normal,27),  p(H_Chen,27),  p(H_Adaptive,27),  p(H_AFEKF,27),  'Pressure (bar)',        'Case 1: Pressure at Node 24 (sink\_24)',        'Case1_P_Node24.fig',  figdir);
plot_one(t, p(H_True,9),   p(Z,9),   p(H_Normal,9),   p(H_Chen,9),   p(H_Adaptive,9),   p(H_AFEKF,9),   'Pressure (bar)',        'Case 1: Pressure at Node 6 (sink\_6)',          'Case1_P_Node6.fig',   figdir);
plot_one(t, p(H_True,10),  p(Z,10),  p(H_Normal,10),  p(H_Chen,10),  p(H_Adaptive,10),  p(H_AFEKF,10),  'Pressure (bar)',        'Case 1: Pressure at Node 7 (sink\_7)',          'Case1_P_Node7.fig',   figdir);
plot_one(t, fl(H_True,13), fl(Z,13), fl(H_Normal,13), fl(H_Chen,13), fl(H_Adaptive,13), fl(H_AFEKF,13), 'Mass flow rate (kg/s)', 'Case 1: Flow at Pipe 13 (sink\_6→sink\_7)',     'Case1_F_Pipe13.fig',  figdir);
fprintf('  Saved 5 Case1 figures to %s\n', figdir);
end

function plot_case2_four(H_True, Z, H_Normal, H_Chen, H_Adaptive, H_AFEKF, Nodes, Pipes, Sys, t, figdir, outlier_config_c2)
c2 = Sys.c2; nN = height(Nodes);
p  = @(H,i) H(:,i)*c2/1e5;
fl = @(H,k) abs(H(:,nN+k));
t_out_s = outlier_config_c2.start_step * Sys.dt / 3600;
t_out_e = outlier_config_c2.end_step   * Sys.dt / 3600;
segs = {{[t_out_s t_out_e], [1 0.9 0.9], 'Outlier period'}};
plot_one_shaded(t, p(H_True,7),   p(Z,7),   p(H_Normal,7),   p(H_Chen,7),   p(H_Adaptive,7),   p(H_AFEKF,7),   'Pressure (bar)',        'Case 2: Pressure at Node 4 (sink\_4)',          'Case2_P_Node4.fig',  figdir, segs);
plot_one_shaded(t, p(H_True,27),  p(Z,27),  p(H_Normal,27),  p(H_Chen,27),  p(H_Adaptive,27),  p(H_AFEKF,27),  'Pressure (bar)',        'Case 2: Pressure at Node 24 (sink\_24)',        'Case2_P_Node24.fig', figdir, segs);
plot_one_shaded(t, p(H_True,9),   p(Z,9),   p(H_Normal,9),   p(H_Chen,9),   p(H_Adaptive,9),   p(H_AFEKF,9),   'Pressure (bar)',        'Case 2: Pressure at Node 6 (sink\_6)',          'Case2_P_Node6.fig',  figdir, segs);
plot_one_shaded(t, p(H_True,10),  p(Z,10),  p(H_Normal,10),  p(H_Chen,10),  p(H_Adaptive,10),  p(H_AFEKF,10),  'Pressure (bar)',        'Case 2: Pressure at Node 7 (sink\_7)',          'Case2_P_Node7.fig',  figdir, segs);
plot_one_shaded(t, fl(H_True,13), fl(Z,13), fl(H_Normal,13), fl(H_Chen,13), fl(H_Adaptive,13), fl(H_AFEKF,13), 'Mass flow rate (kg/s)', 'Case 2: Flow at Pipe 13 (sink\_6→sink\_7)',     'Case2_F_Pipe13.fig', figdir, segs);
fprintf('  Saved 5 Case2 figures to %s\n', figdir);
end

function plot_case3_four(H_True, Z, H_Normal, H_Chen, H_Adaptive, H_AFEKF, Nodes, Pipes, Sys, t, GTU, Leaks, outlier_config, outlier_info, figdir)
c2 = Sys.c2; nN = height(Nodes);
p  = @(H,i) H(:,i)*c2/1e5;
fl = @(H,k) abs(H(:,nN+k));
t_out_s = outlier_config.start_step * Sys.dt / 3600;
t_out_e = outlier_config.end_step   * Sys.dt / 3600;
if height(Leaks) > 0
    t_lk_s = Leaks.StartTime_s(1)/3600;
    t_lk_e = Leaks.EndTime_s(1)/3600;
    if isinf(t_lk_e), t_lk_e = 24; end
else
    t_lk_s = NaN; t_lk_e = NaN;
end
segs = {{[t_out_s t_out_e], [1 0.9 0.9], 'Outlier period'}};
if ~isnan(t_lk_s)
    segs{end+1} = {[t_lk_s t_lk_e], [0.9 0.9 1], 'Leak period'};
end
plot_one_shaded(t, p(H_True,7),   p(Z,7),   p(H_Normal,7),   p(H_Chen,7),   p(H_Adaptive,7),   p(H_AFEKF,7),   'Pressure (bar)',        'Case 3: Pressure at Node 4 (sink\_4)',          'Case3_P_Node4.fig',  figdir, segs);
plot_one_shaded(t, p(H_True,27),  p(Z,27),  p(H_Normal,27),  p(H_Chen,27),  p(H_Adaptive,27),  p(H_AFEKF,27),  'Pressure (bar)',        'Case 3: Pressure at Node 24 (sink\_24)',        'Case3_P_Node24.fig', figdir, segs);
plot_one_shaded(t, p(H_True,9),   p(Z,9),   p(H_Normal,9),   p(H_Chen,9),   p(H_Adaptive,9),   p(H_AFEKF,9),   'Pressure (bar)',        'Case 3: Pressure at Node 6 (sink\_6)',          'Case3_P_Node6.fig',  figdir, segs);
plot_one_shaded(t, p(H_True,10),  p(Z,10),  p(H_Normal,10),  p(H_Chen,10),  p(H_Adaptive,10),  p(H_AFEKF,10),  'Pressure (bar)',        'Case 3: Pressure at Node 7 (sink\_7)',          'Case3_P_Node7.fig',  figdir, segs);
plot_one_shaded(t, fl(H_True,13), fl(Z,13), fl(H_Normal,13), fl(H_Chen,13), fl(H_Adaptive,13), fl(H_AFEKF,13), 'Mass flow rate (kg/s)', 'Case 3: Flow at Pipe 13 (sink\_6→sink\_7)',     'Case3_F_Pipe13.fig', figdir, segs);
fprintf('  Saved 5 Case3 figures to %s\n', figdir);
end


% =========================================================================
% RESULTS SAVING
% =========================================================================
function save_case_results_four(case_name, Stats)
fprintf('\n%s Results:\n', case_name);
for pp = [1,2,3,4,5,7,8,10]
    fn = sprintf('Pipe%d_RMSE', pp);
    fprintf('  %s [kg/s]: Std=%.4f | Chen=%.4f | EKF-LE=%.4f | AFEKF=%.4f\n', fn, ...
        Stats.Normal.(fn), Stats.Chen.(fn), Stats.Adaptive.(fn), Stats.AFEKF.(fn));
end
fprintf('  M_RMSE [kg/s]: Std=%.4f | Chen=%.4f | EKF-LE=%.4f | AFEKF=%.4f\n', ...
    Stats.Normal.M_RMSE, Stats.Chen.M_RMSE, Stats.Adaptive.M_RMSE, Stats.AFEKF.M_RMSE);
end

function save_case2_results_four(case_name, Stats)
fprintf('\n%s Results:\n', case_name);
for pp = [1,2,3,4,5,7,8,10]
    fn = sprintf('Pipe%d_RMSE', pp);
    fprintf('  20dB %s [kg/s]: Std=%.4f | Chen=%.4f | EKF-LE=%.4f | AFEKF=%.4f\n', fn, ...
        Stats.Normal.(fn), Stats.Chen.(fn), Stats.Adaptive.(fn), Stats.AFEKF.(fn));
end
end

function save_case3_results_four(case_name, Stats, Leak_True, Leak_Est)
fprintf('\n%s Results:\n', case_name);
fprintf('  Full Pipe1_RMSE [kg/s]: Std=%.4f | Chen=%.4f | EKF-LE=%.4f | AFEKF=%.4f\n', ...
    Stats.Normal.Pipe1_RMSE, Stats.Chen.Pipe1_RMSE, Stats.Adaptive.Pipe1_RMSE, Stats.AFEKF.Pipe1_RMSE);
if isfield(Stats.Normal, 'P_RMSE_Normal')
    fprintf('  Normal Pipe1_RMSE: Std=%.4f | Chen=%.4f | EKF-LE=%.4f | AFEKF=%.4f\n', ...
        Stats.Normal.Pipe1_RMSE_Normal, Stats.Chen.Pipe1_RMSE_Normal, Stats.Adaptive.Pipe1_RMSE_Normal, Stats.AFEKF.Pipe1_RMSE_Normal);
    fprintf('  Leak Pipe1_RMSE:   Std=%.4f | Chen=%.4f | EKF-LE=%.4f | AFEKF=%.4f\n', ...
        Stats.Normal.Pipe1_RMSE_Leak, Stats.Chen.Pipe1_RMSE_Leak, Stats.Adaptive.Pipe1_RMSE_Leak, Stats.AFEKF.Pipe1_RMSE_Leak);
end
end


% =========================================================================
% PRINT SUMMARY TABLE
% =========================================================================
function print_summary_table_four_methods(Stats_c1, Stats_c2, Stats_c3)
cases      = {'Case 1 (Clean)', 'Case 2 (20dB)'};
stats_list = {Stats_c1, Stats_c2};

for pp = [1,2,3,4,5,7,8,10]
    fn = sprintf('Pipe%d_RMSE', pp);
    fprintf('\nPipe %d RMSE Summary [kg/s] (Case 1 & 2):\n', pp);
    fprintf('%-25s | %-10s | %-10s | %-10s | %-10s\n', 'Case','Standard','Chen','EKF-LE','AFEKF');
    fprintf('%s\n', repmat('-',1,85));
    for ci = 1:2
        S = stats_list{ci};
        fprintf('%-25s | %.4f     | %.4f     | %.4f     | %.4f\n', cases{ci}, ...
            S.Normal.(fn), S.Chen.(fn), S.Adaptive.(fn), S.AFEKF.(fn));
    end
end

for pp = [1, 5]
    fn = sprintf('Pipe%d_RMSE', pp);
    fprintf('\nPipe %d RMSE Summary [kg/s] (Case 3):\n', pp);
    fprintf('%-25s | %-10s | %-10s | %-10s | %-10s\n', 'Case','Standard','Chen','EKF-LE','AFEKF');
    fprintf('%s\n', repmat('-',1,85));
    fprintf('%-25s | %.4f     | %.4f     | %.4f     | %.4f\n', 'Case 3 (Outlier+Leak)', ...
        Stats_c3.Normal.(fn), Stats_c3.Chen.(fn), Stats_c3.Adaptive.(fn), Stats_c3.AFEKF.(fn));
end

fprintf('\n--- Case 3 Normal vs Leak Period ---\n');
if isfield(Stats_c3.Normal, 'P_RMSE_Normal')
    fprintf('%-20s | %-10s | %-10s | %-10s | %-10s\n', 'Period','Standard','Chen','EKF-LE','AFEKF');
    fprintf('%s\n', repmat('-',1,80));
    metrics = {'P_RMSE_Normal','P_RMSE_Leak','M_RMSE_Normal','M_RMSE_Leak', ...
               'Pipe1_RMSE_Normal','Pipe1_RMSE_Leak','Pipe5_RMSE_Normal','Pipe5_RMSE_Leak'};
    for mi = 1:length(metrics)
        m = metrics{mi};
        fprintf('%-20s | %.4f     | %.4f     | %.4f     | %.4f\n', m, ...
            Stats_c3.Normal.(m), Stats_c3.Chen.(m), Stats_c3.Adaptive.(m), Stats_c3.AFEKF.(m));
    end
end
end


% =========================================================================
% SAVE TO EXCEL
% =========================================================================
function save_comprehensive_results_four_methods(Stats_c1, Stats_c2, Stats_c3, filename)
cases_c12  = {'Case1_Clean'; 'Case2_20dB'};
stats_c12  = {Stats_c1, Stats_c2};
all_stats  = {Stats_c1, Stats_c2, Stats_c3};
methods      = {'Normal','Chen','Adaptive','AFEKF'};
method_names = {'Standard_EKF','Chen_Robust','EKF_LE','AFEKF'};

for pp = [1,2,3,4,5,7,8,10]
    fn = sprintf('Pipe%d_RMSE', pp);
    T  = table();
    T.Case = cases_c12;
    for mi = 1:4
        vals = zeros(2,1);
        for ci = 1:2, vals(ci) = stats_c12{ci}.(methods{mi}).(fn); end
        T.(method_names{mi}) = vals;
    end
    T.AFEKF_vs_Std_pct = (T.Standard_EKF - T.AFEKF) ./ T.Standard_EKF * 100;
    writetable(T, filename, 'Sheet', sprintf('C12_Pipe%d_RMSE', pp));
end

for pp = [1, 5]
    fn = sprintf('Pipe%d_RMSE', pp);
    T  = table();
    T.Case = {'Case3_Out_Leak'};
    for mi = 1:4
        T.(method_names{mi}) = Stats_c3.(methods{mi}).(fn);
    end
    T.AFEKF_vs_Std_pct = (T.Standard_EKF - T.AFEKF) ./ T.Standard_EKF * 100;
    writetable(T, filename, 'Sheet', sprintf('C3_Pipe%d_RMSE', pp));
end

T3      = table();
T3.Case = {'Case1_Clean'; 'Case2_20dB'; 'Case3_Out_Leak'};
for mi = 1:4
    vals = zeros(3,1);
    for ci = 1:3, vals(ci) = all_stats{ci}.(methods{mi}).M_RMSE; end
    T3.(method_names{mi}) = vals;
end
writetable(T3, filename, 'Sheet', 'Flow_RMSE');

if isfield(Stats_c3.Normal, 'P_RMSE_Normal')
    T_c3        = table();
    T_c3.Metric = {'P_RMSE_Normal';'P_RMSE_Leak';'M_RMSE_Normal';'M_RMSE_Leak'; ...
                   'Pipe1_RMSE_Normal';'Pipe1_RMSE_Leak';'Pipe5_RMSE_Normal';'Pipe5_RMSE_Leak'};
    for mi = 1:4
        vals = zeros(8,1);
        for fi = 1:8, vals(fi) = Stats_c3.(methods{mi}).(T_c3.Metric{fi}); end
        T_c3.(method_names{mi}) = vals;
    end
    writetable(T_c3, filename, 'Sheet', 'Case3_Normal_vs_Leak');
end

metric_names = {};
std_vals = []; chen_vals = []; ekfle_vals = []; afekf_vals = [];
for pp = [1,2,3,4,5,7,8,10]
    fn = sprintf('Pipe%d_RMSE', pp);
    metric_names{end+1} = sprintf('Avg_C12_Pipe%d_RMSE', pp);
    std_vals(end+1)   = mean(cellfun(@(s) s.Normal.(fn),   stats_c12));
    chen_vals(end+1)  = mean(cellfun(@(s) s.Chen.(fn),     stats_c12));
    ekfle_vals(end+1) = mean(cellfun(@(s) s.Adaptive.(fn), stats_c12));
    afekf_vals(end+1) = mean(cellfun(@(s) s.AFEKF.(fn),    stats_c12));
end
for pp = [1, 5]
    fn = sprintf('Pipe%d_RMSE', pp);
    metric_names{end+1} = sprintf('C3_Pipe%d_RMSE', pp);
    std_vals(end+1)   = Stats_c3.Normal.(fn);
    chen_vals(end+1)  = Stats_c3.Chen.(fn);
    ekfle_vals(end+1) = Stats_c3.Adaptive.(fn);
    afekf_vals(end+1) = Stats_c3.AFEKF.(fn);
end
metric_names{end+1} = 'Avg_Flow_RMSE';
std_vals(end+1)   = mean(cellfun(@(s) s.Normal.M_RMSE,   all_stats));
chen_vals(end+1)  = mean(cellfun(@(s) s.Chen.M_RMSE,     all_stats));
ekfle_vals(end+1) = mean(cellfun(@(s) s.Adaptive.M_RMSE, all_stats));
afekf_vals(end+1) = mean(cellfun(@(s) s.AFEKF.M_RMSE,    all_stats));

T_sum         = table();
T_sum.Metric  = metric_names';
T_sum.Standard_EKF = std_vals';
T_sum.Chen_Robust  = chen_vals';
T_sum.EKF_LE       = ekfle_vals';
T_sum.AFEKF        = afekf_vals';
T_sum.AFEKF_Improvement_vs_Std_pct = (T_sum.Standard_EKF - T_sum.AFEKF) ./ T_sum.Standard_EKF * 100;
writetable(T_sum, filename, 'Sheet', 'Summary');

fprintf('Excel results saved: %s\n', filename);
end
