function [X_E_true, Z_E, t_vec] = power_gen_trajectory(PowerSys, Sys)
% POWER_GEN_TRAJECTORY - Generate true power state trajectory and PMU measurements
% Mirrors the approach of dse_3_gen_data_leak for the gas network.
%
% At each time step:
%   1. Scale loads by a sinusoidal factor (same as gas side)
%   2. Run Matpower power flow with updated loads
%   3. Convert solved voltages to rectangular x_E = [e1,f1,...,e39,f39]
%   4. Add Gaussian noise to PMU measurements via H_E
%
% Inputs:
%   PowerSys - struct from power_load_ieee39 (must contain H_E field)
%   Sys      - system params (needs Sys.dt, Sys.Hours)
%
% Outputs:
%   X_E_true - true power state trajectory [N x 2*nB]
%   Z_E      - noisy PMU measurements      [N x nZ_E]
%   t_vec    - time vector [hours]

N    = floor(Sys.Hours * 3600 / Sys.dt);
nB   = PowerSys.nB;
nZ_E = size(PowerSys.H_E, 1);   % 140 measurements

X_E_true = zeros(N, 2*nB);
Z_E      = zeros(N, nZ_E);
t_vec    = (1:N) * Sys.dt / 3600;

% Load the base case for modifying loads each step
mpc_base = case39();

% Compute scaling factor to match GasLib-40 GTU gas capacity
% GasLib-40 BaseLoad at GTU nodes = 65.42 kg/s total
% case39 power flow GTU demand = 149.82 kg/s total -> 2.3x too large
% Scale generator outputs so total GTU gas demand matches GasLib-40 design
m_gtu_base = gtu_coupling(PowerSys.x0, PowerSys);
total_powerflow = sum([m_gtu_base.m_dot]);
total_gaslib = 0;
for g = 1:length(PowerSys.gtu)
    % Use the gas network's original BaseLoad at each GTU node as target
    % (accessed via the gas node index in PowerSys.gtu)
    total_gaslib = total_gaslib + 16.3542;  % GasLib-40 BaseLoad per GTU node
end
% gas node 34 has BaseLoad=0, so only 4 nodes contribute
total_gaslib = 4 * 16.3542;   % = 65.42 kg/s
gtu_scale = total_gaslib / total_powerflow;
fprintf('GTU scaling factor: %.4f (power flow %.2f -> target %.2f kg/s)\n', ...
    gtu_scale, total_powerflow, total_gaslib);

% PMU measurement noise standard deviations (Chen Section V-A)
% Voltage:          2% of nominal (Vm ~ 1 p.u.)
% Current (branch): 2% of nominal
% Current (inject): 2% of nominal
sigma_E = 0.02;   % 2% std dev in per-unit for all PMU channels

for k = 1:N
    hr = t_vec(k);

    % Same sinusoidal load factor as gas side (Chen uses same 24h pattern)
    lf = 1.0 + 0.05 * sin(2*pi*(hr-8)/24);

    % Scale all bus loads and generators by gtu_scale*lf
    % gtu_scale ensures GTU gas demand matches GasLib-40 capacity
    % lf provides the time-varying sinusoidal pattern
    mpc = mpc_base;
    mpc.bus(:,3) = mpc_base.bus(:,3) * gtu_scale * lf;   % Pd
    mpc.bus(:,4) = mpc_base.bus(:,4) * gtu_scale * lf;   % Qd
    mpc.gen(:,2) = mpc_base.gen(:,2) * gtu_scale * lf;   % Pg
    mpc.gen(:,3) = mpc_base.gen(:,3) * gtu_scale * lf;   % Qg

    % Run power flow
    results = runpf(mpc, mpoption('verbose',0,'out.all',0));
    if results.success ~= 1
        % If power flow fails, carry forward previous state
        if k > 1
            X_E_true(k,:) = X_E_true(k-1,:);
        else
            X_E_true(k,:) = PowerSys.x0';
        end
        Z_E(k,:) = PowerSys.H_E * X_E_true(k,:)';
        continue;
    end

    % Convert polar to rectangular
    Vm = results.bus(:,8);
    Va = results.bus(:,9) * pi/180;
    e  = Vm .* cos(Va);
    f  = Vm .* sin(Va);

    x_E = zeros(2*nB, 1);
    for i = 1:nB
        x_E(2*i-1) = e(i);
        x_E(2*i)   = f(i);
    end
    X_E_true(k,:) = x_E';

    % Generate PMU measurements: z_E = H_E * x_E + noise
    noise = sigma_E * randn(nZ_E, 1);
    Z_E(k,:) = (PowerSys.H_E * x_E + noise)';
end

fprintf('Power trajectory generated: %d steps, %d PMU measurements per step\n', N, nZ_E);
end
