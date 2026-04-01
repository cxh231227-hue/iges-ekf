function [Hist_True, Z, t_vec, Leak_True, P_GTU] = dse_3_gen_data_leak(Nodes, Pipes, Compressors, Sys, Leaks, GTU, PowerSys, X_E_true)
% DSE_3_GEN_DATA_LEAK - Generate synthetic measurement data with leaks
%
% New optional inputs (Chen 2022, Section V):
%   PowerSys  - struct from power_load_ieee39
%   X_E_true  - true power state trajectory [N x 2*nB] from power_gen_trajectory
%               if provided, GTU loads are computed from power flow via Eq 22
%               if absent, falls back to original sinusoidal estimate
%
% Outputs:
%   Hist_True - true gas state history [N x nS]
%   Z         - noisy gas measurements [N x nS]
%   t_vec     - time vector [hours]
%   Leak_True - actual leak rates per pipe [N x nP]
%   P_GTU     - GTU power output [N x nG]

LHV = 50;

N  = floor(Sys.Hours * 3600 / Sys.dt);
nN = height(Nodes);
nP = height(Pipes);
nS = nN + nP;
nG = height(GTU);

Hist_True = zeros(N, nS);
Z         = zeros(N, nS);
Leak_True = zeros(N, nP);
P_GTU     = zeros(N, max(nG,1));
t_vec     = (1:N) * Sys.dt / 3600;

% Check if power coupling is available
has_powersys = nargin >= 7 && ~isempty(PowerSys);
has_xe_true  = nargin >= 8 && ~isempty(X_E_true);

x = dse_steady_solver(Nodes, Pipes, Compressors, Sys);

% Warm-up: run gas network with power-flow GTU loads for 6 hours
% so initial state matches the actual operating point
n_warmup = round(12 * 3600 / Sys.dt);
if has_powersys && has_xe_true
    % Use average GTU load from first few steps of X_E_true for warm-up
    load_warmup = Nodes.BaseLoad * (1.0 + 0.05*sin(2*pi*(-8)/24));
    x_E_avg = mean(X_E_true(1:min(6,size(X_E_true,1)), :), 1)';
    m_gtu_warmup = gtu_coupling(x_E_avg, PowerSys);
    for g = 1:length(m_gtu_warmup)
        load_warmup(m_gtu_warmup(g).gas_node_id) = m_gtu_warmup(g).m_dot;
    end
    for w = 1:n_warmup
        [A, B, u] = dse_2_build_model(x, Nodes, Pipes, Compressors, Sys, load_warmup);
        x = A \ (B*x + u);
    end
end

for t = 1:N
    hr  = t_vec(t);
    sec = t * Sys.dt;

    % Base gas load (non-GTU nodes)
    lf       = 1.0 + 0.05 * sin(2*pi*(hr-8)/24);
    load_now = Nodes.BaseLoad .* lf;

    % GTU gas consumption
    if has_powersys && has_xe_true
        % Chen Eq 22: m_dot = P_G / eta, where P_G comes from power flow
        % Use gtu_coupling with true power state at this step
        x_E_t = X_E_true(t, :)';
        m_gtu = gtu_coupling(x_E_t, PowerSys);
        for g = 1:length(m_gtu)
            node_id        = m_gtu(g).gas_node_id;
            load_now(node_id) = m_gtu(g).m_dot;
            % Record GTU power for output
            idx = find([PowerSys.gtu.gas_node_id] == node_id, 1);
            if ~isempty(idx) && g <= nG
                P_GTU(t, g) = m_gtu(g).m_dot * PowerSys.gtu(g).eta;
            end
        end
    else
        % Fallback: original sinusoidal estimate
        elf = 0.7 + 0.3 * (sin(2*pi*(hr-6)/12).^2 .* (hr>=6 & hr<=22));
        for g = 1:nG
            if GTU.Status(g) ~= 1, continue; end
            cap  = GTU.Capacity_MW(g);
            pmin = GTU.MinLoad(g) / 100;
            pmax = GTU.MaxLoad(g) / 100;
            pwr  = cap * (pmin + (pmax-pmin) * elf);
            P_GTU(t, g)           = pwr;
            gas                   = pwr * GTU.HeatRate(g) * 1000 / 3600 / LHV;
            load_now(GTU.NodeID(g)) = gas;
        end
    end

    % Leak-induced loads
    leak_load = zeros(nN, 1);
    leak_pipe = zeros(nP, 1);
    for lk = 1:height(Leaks)
        pid = Leaks.PipeID(lk);
        if sec >= Leaks.StartTime_s(lk) && sec <= Leaks.EndTime_s(lk)
            rate = Leaks.LeakRate_kg_s(lk);
            pos  = Leaks.Position(lk);
            leak_pipe(pid) = leak_pipe(pid) + rate;
            n1 = Pipes.From(pid);
            n2 = Pipes.To(pid);
            if Nodes.Type(n1) ~= 1
                leak_load(n1) = leak_load(n1) + rate*(1-pos);
            end
            if Nodes.Type(n2) ~= 1
                leak_load(n2) = leak_load(n2) + rate*pos;
            end
        end
    end
    Leak_True(t,:) = leak_pipe';

    total_load = load_now + leak_load;

    % State noise
    noise_state_p = sqrt(1.5e-4) * randn(1,nN) * 1e5 / Sys.c2;
    noise_state_m = sqrt(4e-3)   * randn(1,nP);

    [A, B, u] = dse_2_build_model(x, Nodes, Pipes, Compressors, Sys, total_load);
    x = A \ (B*x + u + [noise_state_p'; noise_state_m']);
    Hist_True(t,:) = x';

    % Measurement noise
    noise_p  = sqrt(0.04) * randn(1,nN) * 1e5 / Sys.c2;
    noise_m  = sqrt(2)    * randn(1,nP);
    Z(t,:)   = Hist_True(t,:) + [noise_p, noise_m];
end
end
