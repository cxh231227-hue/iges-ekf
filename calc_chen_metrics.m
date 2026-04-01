function Metrics = calc_chen_metrics(H_True, H_Est, Z_gas, X_E_true, X_E_est, Sys, Nodes, Pipes, Z_E, PowerSys)
% CALC_CHEN_METRICS - Compute Chen (2022) Table I/II/III metrics
%
% eps1 = filter coefficient  (Eq 36): sum(est-true)^2 / sum(meas-true)^2
% eps2 = total variance      (Eq 37): sum(est-true)^2 / S
%
% Inputs:
%   H_True    - true gas states [N x nS]
%   H_Est     - estimated gas states [N x nS]
%   Z_gas     - gas measurements [N x nS]
%   X_E_true  - true power states [N x 2*nB]
%   X_E_est   - estimated power states [N x 2*nB]
%   Sys       - system parameters
%   Nodes     - node table
%   Pipes     - pipe table
%   Z_E       - PMU measurements [N x nZ_E] (needed for correct voltage eps1)
%   PowerSys  - power system struct (needed for pmu_buses mapping)

nN = height(Nodes);
nP = height(Pipes);
S  = size(H_True, 1);

% Skip initial transient period (first 3 hours = 18 steps at dt=600s)
% Transient is caused by warm-up to steady-state mismatch
skip = 18;
idx  = (skip+1):S;   % index range used for metric calculation
S_eff = length(idx); % effective sample count

chen_nodes = [4, 5, 8, 10, 11, 15, 19, 20, 22, 24, 26, 29, 30, 31, 34];
chen_buses = [2, 4, 6, 9, 12, 15, 18, 21, 22, 25, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39];

% ---- Gas pressure metrics ----
P_True = H_True(:, 1:nN) * Sys.c2 / 1e5;   % bar
P_Est  = H_Est(:,  1:nN) * Sys.c2 / 1e5;
P_Meas = Z_gas(:,  1:nN) * Sys.c2 / 1e5;

node_eps1 = zeros(nN, 1);
node_eps2 = zeros(nN, 1);
for i = 1:nN
    num   = sum((P_Est(idx,i)  - P_True(idx,i)).^2);
    denom = sum((P_Meas(idx,i) - P_True(idx,i)).^2);
    node_eps1(i) = num / max(denom, 1e-12);
    node_eps2(i) = num / S_eff;
end

% ---- Gas mass flow metrics ----
M_True = H_True(:, nN+1:end);
M_Est  = H_Est(:,  nN+1:end);
M_Meas = Z_gas(:,  nN+1:end);

pipe_eps1 = zeros(nP, 1);
pipe_eps2 = zeros(nP, 1);
for k = 1:nP
    num   = sum((M_Est(idx,k)  - M_True(idx,k)).^2);
    denom = sum((M_Meas(idx,k) - M_True(idx,k)).^2);
    pipe_eps1(k) = num / max(denom, 1e-12);
    pipe_eps2(k) = num / S_eff;
end

% ---- Power state metrics ----
has_power = nargin >= 5 && ~isempty(X_E_true) && ~isempty(X_E_est);
has_ze    = nargin >= 9 && ~isempty(Z_E);
has_psys  = nargin >= 10 && ~isempty(PowerSys);
nB = 39;

bus_eps1_e = zeros(nB, 1);
bus_eps1_f = zeros(nB, 1);
bus_eps2_e = zeros(nB, 1);
bus_eps2_f = zeros(nB, 1);

if has_power
    for b = 1:nB
        e_true = X_E_true(:, 2*b-1);
        f_true = X_E_true(:, 2*b);
        e_est  = X_E_est(:,  2*b-1);
        f_est  = X_E_est(:,  2*b);

        num_e = sum((e_est(idx) - e_true(idx)).^2);
        num_f = sum((f_est(idx) - f_true(idx)).^2);

        % Use actual PMU measurements for denominator if available
        % H_V block: rows 1..2*nZB of Z_E, 2 rows per PMU bus
        % Row 2k-1 = e measurement of pmu_buses(k)
        % Row 2k   = f measurement of pmu_buses(k)
        if has_ze && has_psys
            pmu_buses = PowerSys.pmu_buses;
            k_idx = find(pmu_buses == b, 1);
            if ~isempty(k_idx)
                % This bus has a PMU voltage measurement
                e_meas = Z_E(:, 2*k_idx-1);
                f_meas = Z_E(:, 2*k_idx);
                den_e  = sum((e_meas(idx) - e_true(idx)).^2);
                den_f  = sum((f_meas(idx) - f_true(idx)).^2);
            else
                % No PMU at this bus: use 2% noise approximation
                % but only if the signal is large enough
                sig_e = std(e_true);
                sig_f = std(f_true);
                den_e = S * (0.02 * max(sig_e, 1e-4))^2;
                den_f = S * (0.02 * max(sig_f, 1e-4))^2;
            end
        else
            % Fallback: 2% noise approximation with signal-based scaling
            sig_e = std(e_true);
            sig_f = std(f_true);
            den_e = S * (0.02 * max(sig_e, 1e-4))^2;
            den_f = S * (0.02 * max(sig_f, 1e-4))^2;
        end

        bus_eps1_e(b) = num_e / max(den_e, 1e-16);
        bus_eps1_f(b) = num_f / max(den_f, 1e-16);
        bus_eps2_e(b) = num_e / S_eff;
        bus_eps2_f(b) = num_f / S_eff;
    end
end

% ---- Sanity check: warn if any eps1 > 1 ----
bad_nodes = chen_nodes(node_eps1(chen_nodes) > 1);
if ~isempty(bad_nodes)
    fprintf('  WARNING: eps1 > 1 at gas nodes: ');
    fprintf('%d ', bad_nodes); fprintf('\n');
end
bad_buses = chen_buses(bus_eps1_e(chen_buses) > 1 | bus_eps1_f(chen_buses) > 1);
if ~isempty(bad_buses)
    fprintf('  WARNING: eps1 > 1 at power buses: ');
    fprintf('%d ', bad_buses); fprintf('\n');
end

% ---- Pack output ----
Metrics.node_eps1  = node_eps1;
Metrics.node_eps2  = node_eps2;
Metrics.pipe_eps1  = pipe_eps1;
Metrics.pipe_eps2  = pipe_eps2;
Metrics.bus_eps1_e = bus_eps1_e;
Metrics.bus_eps1_f = bus_eps1_f;
Metrics.bus_eps2_e = bus_eps2_e;
Metrics.bus_eps2_f = bus_eps2_f;
Metrics.chen_nodes = chen_nodes;
Metrics.chen_buses = chen_buses;
end
