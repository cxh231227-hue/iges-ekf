function [Hist_Est, Load_Used, X_E_est] = dse_normal_ekf(Z, Nodes, Pipes, Compressors, Sys, t_vec, GTU, mismatch, PowerSys, Z_E)
% DSE_NORMAL_EKF - Standard EKF with joint power-gas state estimation
%
% When PowerSys and Z_E are provided, implements Chen (2022) joint DSE:
%   x_I = [x_E; x_G],  P_I is full joint covariance (not block-diagonal)
%   H_I = blkdiag(H_E, eye(nS))
%   F_I = blkdiag(alpha_s*I, F_G)
% Cross-covariance between power and gas states is preserved, so
% gas measurements improve power estimates and vice versa.

LHV = 50;

Q_p = 0.15;
Q_m = 0.4;
R_p = 1;
R_m = 10;

if mismatch < 1
    R_m = Q_m;
    R_p = Q_p;
elseif mismatch > 1
    Q_m = R_m;
    Q_p = R_p;
end

N  = length(t_vec);
nN = height(Nodes);
nP = height(Pipes);
nS = nN + nP;

has_gtu      = nargin >= 7 && ~isempty(GTU) && height(GTU) > 0;
has_powersys = nargin >= 9 && ~isempty(PowerSys);
has_ze       = nargin >= 10 && ~isempty(Z_E);

% Output storage
X_E_est = [];
Hist_Est  = zeros(N, nS);
Load_Used = zeros(N, nN);

% Initialize gas state estimate
init_n = min(5, N);
x_G_k = mean(Z(1:init_n,:), 1)';

if has_powersys
    alpha_s = 0.5;
    beta_s  = 0.4;
    nE      = 2 * PowerSys.nB;          % 78 for 39-bus
    H_E     = PowerSys.H_E;             % 140 x 78, constant
    nZ_E    = size(H_E, 1);             % 140
    nI      = nE + nS;                  % joint state dimension

    % Holt smoothing state
    x_E_k  = PowerSys.x0;
    L_holt = PowerSys.x0;
    T_holt = zeros(nE, 1);

    % Warm-up gas state to match power-flow GTU operating point
    n_warmup = round(12 * 3600 / Sys.dt);
    load_warmup = Nodes.BaseLoad * (1.0 + 0.05*sin(2*pi*(-8)/24));
    m_gtu_wu = gtu_coupling(PowerSys.x0, PowerSys);
    for g = 1:length(m_gtu_wu)
        load_warmup(m_gtu_wu(g).gas_node_id) = m_gtu_wu(g).m_dot;
    end
    for w = 1:n_warmup
        [A_w, B_w, u_w] = dse_2_build_model(x_G_k, Nodes, Pipes, Compressors, Sys, load_warmup);
        x_G_k = A_w \ (B_w * x_G_k + u_w);
    end
    x_G_k(1:nN) = max(x_G_k(1:nN), 0.5);

    % Joint covariance P_I = blkdiag(P_E, P_G) initially
    % After first update cross-terms become non-zero
    P_E_init = eye(nE);
    P_G_init = eye(nS);
    P_G_init(1:nN, 1:nN)     = 10 * eye(nN);
    P_G_init(nN+1:end,nN+1:end) = 25 * eye(nP);
    P_I = blkdiag(P_E_init, P_G_init);

    % Joint process noise Q_I = blkdiag(Q_E, Q_G)
    % Q_E is small (Chen uses 1e-5), Q_G is our gas tuning
    Q_E = 1e-5 * eye(nE);
    Q_G = blkdiag(Q_p * eye(nN), Q_m * eye(nP));
    Q_I = blkdiag(Q_E, Q_G);

    % Joint measurement noise R_I = blkdiag(R_E, R_G)
    R_E = (0.02^2) * eye(nZ_E);
    R_G = blkdiag(R_p * eye(nN), R_m * eye(nP));
    R_I = blkdiag(R_E, R_G);

    % Joint measurement matrix H_I = [H_E, 0; 0, eye(nS)]
    H_I = [H_E,         zeros(nZ_E, nS); ...
           zeros(nS, nE), eye(nS)];

    X_E_est = zeros(N, nE);

else
    % Gas-only initialization
    n_warmup = 0;
    P_k = eye(nS);
    P_k(1:nN, 1:nN)     = 10 * eye(nN);
    P_k(nN+1:end,nN+1:end) = 25 * eye(nP);
    Q = blkdiag(Q_p * eye(nN), Q_m * eye(nP));
    R = blkdiag(R_p * eye(nN), R_m * eye(nP));
end

for k = 1:N
    z_G_k = Z(k, :)';
    hr    = t_vec(k);

    % Base gas load
    lf     = 1.0 + 0.05 * sin(2*pi*(hr-8)/24);
    load_k = Nodes.BaseLoad * lf;

    if has_powersys
        % --- JOINT ESTIMATION (Chen 2022 Section IV-B) ---

        % Step 1: Holt predict x_E (Chen Eq 1-4)
        [x_E_pred, L_holt, T_holt] = power_holt_smooth(x_E_k, L_holt, T_holt, alpha_s, beta_s);

        % Step 2: Compute GTU boundary condition using predicted x_E (Chen Eq 24)
        m_gtu = gtu_coupling(x_E_pred, PowerSys);
        for g = 1:length(m_gtu)
            load_k(m_gtu(g).gas_node_id) = m_gtu(g).m_dot;
        end

        % Step 3: Gas network prediction (Chen Eq 20)
        [A_G, B_G, u_G] = dse_2_build_model(x_G_k, Nodes, Pipes, Compressors, Sys, load_k);
        x_G_pred = A_G \ (B_G * x_G_k + u_G);
        F_G      = A_G \ B_G;

        % Step 4: Joint state prediction
        x_I_pred = [x_E_pred; x_G_pred];

        % Step 5: Joint covariance prediction (Chen Eq 27)
        % F_I = blkdiag(alpha_s * I_nE, F_G)
        F_I  = blkdiag(alpha_s * eye(nE), F_G);
        P_I_pred = F_I * P_I * F_I' + Q_I;
        P_I_pred = (P_I_pred + P_I_pred') / 2;

        % Step 6: Joint measurement update (Chen Eq 28-30)
        if has_ze
            z_I_k = [Z_E(k,:)'; z_G_k];
        else
            % No PMU: use predicted x_E as measurement (zero innovation)
            z_I_k = [H_E * x_E_pred; z_G_k];
        end

        S_I = H_I * P_I_pred * H_I' + R_I;
        S_I = (S_I + S_I') / 2;
        K_I = P_I_pred * H_I' / S_I;

        x_I_update = x_I_pred + K_I * (z_I_k - H_I * x_I_pred);
        P_I = (eye(nI) - K_I * H_I) * P_I_pred;
        P_I = (P_I + P_I') / 2 + 1e-6 * eye(nI);

        % Step 7: Extract updated states
        x_E_k = x_I_update(1:nE);
        x_G_k = x_I_update(nE+1:end);

        % Physical constraint on gas pressures
        x_G_k(1:nN) = max(x_G_k(1:nN), 0.5);

        X_E_est(k, :) = x_E_k';

    elseif has_gtu
        % Fallback: sinusoidal GTU estimate
        elf = 0.7 + 0.3 * (sin(2*pi*(hr-6)/12).^2 * (hr>=6 && hr<=22));
        for g = 1:height(GTU)
            if GTU.Status(g) ~= 1, continue; end
            cap  = GTU.Capacity_MW(g);
            pmin = GTU.MinLoad(g) / 100;
            pmax = GTU.MaxLoad(g) / 100;
            pwr  = cap * (pmin + (pmax - pmin) * elf);
            gas  = pwr * GTU.HeatRate(g) * 1000 / 3600 / LHV;
            load_k(GTU.NodeID(g)) = gas;
        end

        % Gas-only KF
        [A, B, u] = dse_2_build_model(x_G_k, Nodes, Pipes, Compressors, Sys, load_k);
        x_pred = A \ (B * x_G_k + u);
        F_k    = A \ B;
        P_pred = F_k * P_k * F_k' + Q;
        H      = eye(nS);
        nu_k   = z_G_k - H * x_pred;
        S      = H * P_pred * H' + R;
        K      = P_pred * H' / S;
        x_G_k  = x_pred + K * nu_k;
        P_k    = (eye(nS) - K * H) * P_pred;
        P_k    = (P_k + P_k') / 2 + 1e-4 * eye(nS);
        x_G_k(1:nN) = max(x_G_k(1:nN), 0.5);
    end

    Hist_Est(k, :)  = x_G_k';
    Load_Used(k, :) = load_k';
end

% Post-processing
for i = 1:nP
    Hist_Est(:, nN+i) = movmean(Hist_Est(:, nN+i), 3);
end

end
