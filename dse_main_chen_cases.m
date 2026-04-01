function dse_main_chen_cases(mismatch)
% DSE_MAIN_CHEN_CASES - Replicate Chen (2022) metrics for all four EKF methods
%
% Case a: White Gaussian noises
% All four methods use full power-gas coupling (PowerSys + Z_E).
% Outputs eps1 and eps2 in Chen Table I/II/III format.
% Results saved to chen_copy/ folder.

if nargin < 1, mismatch = 1; end

rng(42);

basedir = fileparts(mfilename('fullpath'));
chendir = fullfile(basedir, 'chen_copy');
if ~exist(chendir, 'dir'), mkdir(chendir); end

Sys.c = 340; Sys.c2 = 340^2; Sys.dt = 600; Sys.Hours = 24;

[Nodes, Pipes, Compressors, GTU] = dse_load_gaslib40( ...
    'GasLib-40-v1-20211130.net', 'GasLib-40-v1-20211130.scn');
PowerSys = power_load_ieee39();
[X_E_true, Z_E, t] = power_gen_trajectory(PowerSys, Sys);

Leaks_empty = table([],[],[],[],[], ...
    'VariableNames',{'PipeID','StartTime_s','EndTime_s','LeakRate_kg_s','Position'});
[H_True, Z_clean, ~, ~, ~] = dse_3_gen_data_leak(Nodes, Pipes, Compressors, Sys, ...
    Leaks_empty, GTU, PowerSys, X_E_true);

fprintf('\n========================================================\n');
fprintf('   Chen (2022) Case a: White Gaussian Noises\n');
fprintf('   Four-method comparison with power-gas coupling\n');
fprintf('========================================================\n');

fprintf('  [1/4] Running Standard EKF...\n');
[H_Normal, ~, XE_Normal] = dse_normal_ekf(Z_clean, Nodes, Pipes, Compressors, Sys, t, GTU, mismatch, PowerSys, Z_E);

fprintf('  [2/4] Running Chen Robust EKF...\n');
[H_Chen, ~, ~, ~, XE_Chen] = dse_chen_ekf(Z_clean, Nodes, Pipes, Compressors, Sys, t, GTU, mismatch, PowerSys, Z_E);

fprintf('  [3/4] Running EKF-LE...\n');
Det = false(length(t), 1);
[H_Adaptive, ~, ~, ~, XE_Adaptive] = dse_4_estimator_leak(Z_clean, Nodes, Pipes, Compressors, Sys, t, GTU, Det, mismatch, PowerSys, Z_E);

fprintf('  [4/4] Running AFUKF...\n');
[H_AFEKF, ~, ~, ~, ~, XE_AFEKF] = AFUKF(Z_clean, Nodes, Pipes, Compressors, Sys, t, GTU, [], mismatch, PowerSys, Z_E);

% Compute metrics
M_Normal   = calc_chen_metrics(H_True, H_Normal,   Z_clean, X_E_true, XE_Normal,   Sys, Nodes, Pipes, Z_E, PowerSys);
M_Chen     = calc_chen_metrics(H_True, H_Chen,     Z_clean, X_E_true, XE_Chen,     Sys, Nodes, Pipes, Z_E, PowerSys);
M_Adaptive = calc_chen_metrics(H_True, H_Adaptive, Z_clean, X_E_true, XE_Adaptive, Sys, Nodes, Pipes, Z_E, PowerSys);
M_AFEKF    = calc_chen_metrics(H_True, H_AFEKF,    Z_clean, X_E_true, XE_AFEKF,    Sys, Nodes, Pipes, Z_E, PowerSys);

% Print tables
print_tables(M_Normal, M_Chen, M_Adaptive, M_AFEKF, Nodes, Pipes);

% Export Excel
export_excel(M_Normal, M_Chen, M_Adaptive, M_AFEKF, ...
    fullfile(chendir, 'Chen_Case_a.xlsx'), Nodes, Pipes);

fprintf('\nDone. Results in: %s\n', chendir);
end


% =========================================================================
function print_tables(Mn, Mc, Ma, Mf, Nodes, Pipes)

nN = height(Nodes);
chen_nodes = Mn.chen_nodes;
chen_buses = Mn.chen_buses;

% Table I: filter coefficients (eps1) for gas nodes
fprintf('\n--- Table I: Filter coefficients eps1 (pressure x1e-4, mass flow) ---\n');
fprintf('  %-5s | %-17s %-8s | %-17s %-8s | %-17s %-8s | %-17s %-8s\n', ...
    'Node', 'Std_P(x1e-4)', 'Std_M', 'Chen_P(x1e-4)', 'Chen_M', 'LE_P(x1e-4)', 'LE_M', 'AF_P(x1e-4)', 'AF_M');
fprintf('  %s\n', repmat('-',1,80));
for i = chen_nodes
    if i > nN, continue; end
    pk = find(Pipes.From==i | Pipes.To==i, 1);
    if isempty(pk), pk_val = [NaN NaN NaN NaN]; else
        pk_val = [Mn.pipe_eps1(pk), Mc.pipe_eps1(pk), Ma.pipe_eps1(pk), Mf.pipe_eps1(pk)];
    end
    fprintf('  %-5d | %-17.4f %-8.4f | %-17.4f %-8.4f | %-17.4f %-8.4f | %-17.4f %-8.4f\n', i, ...
        Mn.node_eps1(i)*1e4, pk_val(1), ...
        Mc.node_eps1(i)*1e4, pk_val(2), ...
        Ma.node_eps1(i)*1e4, pk_val(3), ...
        Mf.node_eps1(i)*1e4, pk_val(4));
end

% Table II: total variance eps2 for gas pressure (x1e-6 bar^2)
fprintf('\n--- Table II: Total variance eps2 for pressure (x1e-6 bar^2) ---\n');
fprintf('  %-5s | %-12s | %-12s | %-12s | %-12s\n', ...
    'Node', 'Standard', 'Chen', 'EKF-LE', 'AFUKF');
fprintf('  %s\n', repmat('-',1,65));
for i = chen_nodes
    if i > nN, continue; end
    fprintf('  %-5d | %-12.4f | %-12.4f | %-12.4f | %-12.4f\n', i, ...
        Mn.node_eps2(i)*1e6, Mc.node_eps2(i)*1e6, ...
        Ma.node_eps2(i)*1e6, Mf.node_eps2(i)*1e6);
end

% Table III: total variance eps2 for mass flow (kg/s)^2
fprintf('\n--- Table III: Total variance eps2 for mass flow (kg/s)^2 ---\n');
fprintf('  %-12s | %-10s | %-10s | %-10s | %-10s\n', ...
    'Pipe(i->j)', 'Standard', 'Chen', 'EKF-LE', 'AFUKF');
fprintf('  %s\n', repmat('-',1,60));
nP = height(Pipes);
for k = 1:nP
    i = Pipes.From(k); j = Pipes.To(k);
    fprintf('  %-5d->%-5d | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n', i, j, ...
        Mn.pipe_eps2(k), Mc.pipe_eps2(k), Ma.pipe_eps2(k), Mf.pipe_eps2(k));
end

% Power bus filter coefficients
fprintf('\n--- Power bus filter coefficients eps1 (e and f components) ---\n');
fprintf('  %-5s | %-8s %-8s | %-8s %-8s | %-8s %-8s | %-8s %-8s\n', ...
    'Bus', 'Std_e', 'Std_f', 'Chen_e', 'Chen_f', 'LE_e', 'LE_f', 'AF_e', 'AF_f');
fprintf('  %s\n', repmat('-',1,80));
for b = chen_buses
    if b > 39, continue; end
    fprintf('  %-5d | %-8.4f %-8.4f | %-8.4f %-8.4f | %-8.4f %-8.4f | %-8.4f %-8.4f\n', b, ...
        Mn.bus_eps1_e(b), Mn.bus_eps1_f(b), ...
        Mc.bus_eps1_e(b), Mc.bus_eps1_f(b), ...
        Ma.bus_eps1_e(b), Ma.bus_eps1_f(b), ...
        Mf.bus_eps1_e(b), Mf.bus_eps1_f(b));
end
end


% =========================================================================
function export_excel(Mn, Mc, Ma, Mf, filename, Nodes, Pipes)
nN = height(Nodes);
nP = height(Pipes);
cn = Mn.chen_nodes; cn = cn(cn<=nN);
cb = Mn.chen_buses; cb = cb(cb<=39);

% Table I: eps1 pressure
T1 = table(cn', Mn.node_eps1(cn)*1e4, Mc.node_eps1(cn)*1e4, ...
    Ma.node_eps1(cn)*1e4, Mf.node_eps1(cn)*1e4, ...
    'VariableNames',{'Node','Standard_x1e4','Chen_x1e4','EKF_LE_x1e4','AFUKF_x1e4'});
writetable(T1, filename, 'Sheet','Table1_eps1_Pressure');

% Table II: eps2 pressure
T2 = table(cn', Mn.node_eps2(cn)*1e6, Mc.node_eps2(cn)*1e6, ...
    Ma.node_eps2(cn)*1e6, Mf.node_eps2(cn)*1e6, ...
    'VariableNames',{'Node','Standard_x1e4','Chen_x1e4','EKF_LE_x1e4','AFUKF_x1e4'});
writetable(T2, filename, 'Sheet','Table2_eps2_Pressure');

% Table III: eps2 mass flow
T3 = table((1:nP)', Pipes.From(1:nP), Pipes.To(1:nP), ...
    Mn.pipe_eps2, Mc.pipe_eps2, Ma.pipe_eps2, Mf.pipe_eps2, ...
    'VariableNames',{'PipeIdx','From','To','Standard','Chen','EKF_LE','AFUKF'});
writetable(T3, filename, 'Sheet','Table3_eps2_MassFlow');

% Power bus eps1
T4 = table(cb', ...
    Mn.bus_eps1_e(cb), Mn.bus_eps1_f(cb), ...
    Mc.bus_eps1_e(cb), Mc.bus_eps1_f(cb), ...
    Ma.bus_eps1_e(cb), Ma.bus_eps1_f(cb), ...
    Mf.bus_eps1_e(cb), Mf.bus_eps1_f(cb), ...
    'VariableNames',{'Bus','Std_e','Std_f','Chen_e','Chen_f','LE_e','LE_f','AF_e','AF_f'});
writetable(T4, filename, 'Sheet','Table4_eps1_Voltage');

fprintf('  Excel saved: %s\n', filename);
end
