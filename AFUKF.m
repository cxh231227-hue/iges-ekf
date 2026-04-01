function [Hist_Est, Leak_Est, Load_Compensated, Mu_History, MuR_History, X_E_est] = AFUKF(Z, Nodes, Pipes, Compressors, Sys, t_vec, GTU, Det_Signal, mismatch, PowerSys, Z_E)
% 
LHV = 50;

Q_p = 0.15;
Q_m = 0.4;
R_p = 1;
R_m = 10;
% Q_p = 1;     % Process noise for pressure states
% Q_m = 10;      % Process noise for flow states
% R_p = 0.15;      % Measurement noise for pressure
% R_m = 0.4;       % Measurement noise for flow

%adj_rate=.5;
%Q_p=mismatch*Q_p;
if(mismatch<1)
    R_m=Q_m;
%R_p=adj_rate*R_p;
    R_p=Q_p;
elseif(mismatch>1)
    Q_m=R_m;
    Q_p=R_p;
end
%R_p=adj_rate*R_p;
%R_m=adj_rate*R_m;
%Q_p = 2;     % Process noise for pressure states
%Q_m = 15;      % Process noise for flow states
%R_p = 2;      % Measurement noise for pressure
%R_m = 15;       % Measurement noise for flow


delta_th = 0.35;     % deadzone [Bar]
alpha_dk = 0.95;     % smoothing
kappa    = 40;       % compensation gain
init_protect_hours = 5.0;

% Outlier suppression parameters
k_outlier = 3;
k2_outlier = 5;
k3_outlier = 10;
c_mad = 1.483;
alpha_sigma_mad = 0.99;
window_size = 25;

fadingFlag = true;
measLengthForfadingFactorEst = 8;

% Let bt move (as you used)
Q_ab = [1 0; 0 1e4];


stateConstrainLB = 0.999; 
stateConstrainUB = 1.5;   % alp in [0.99, 1]
measConstrainLB  = 0.999; 
measConstrainUB  = 1.5;   % bt  in [0.99, 1.1]


N = length(t_vec);
[nZ_N, nS] = size(Z);
if nZ_N ~= N
    error('Z rows (%d) must equal length(t_vec) (%d).', nZ_N, N);
end

nN = height(Nodes);
nP = height(Pipes);
if nS ~= (nN + nP)
    error('Z columns (%d) must equal nN+nP (%d).', nS, nN+nP);
end

H = eye(nS);

has_gtu      = (nargin >= 7) && ~isempty(GTU) && height(GTU) > 0;
has_detector = (nargin >= 8) && ~isempty(Det_Signal);
has_powersys = (nargin >= 10) && ~isempty(PowerSys);
has_ze       = (nargin >= 11) && ~isempty(Z_E);

init_protect_steps = round(init_protect_hours * 3600 / Sys.dt);

init_n = min(5, N);
x_k = mean(Z(1:init_n,:), 1)';

% Warm-up gas state to match power-flow GTU operating point
if has_powersys
    n_warmup = round(12 * 3600 / Sys.dt);
    load_warmup = Nodes.BaseLoad * (1.0 + 0.05*sin(2*pi*(-8)/24));
    m_gtu_warmup = gtu_coupling(PowerSys.x0, PowerSys);
    for g = 1:length(m_gtu_warmup)
        load_warmup(m_gtu_warmup(g).gas_node_id) = m_gtu_warmup(g).m_dot;
    end
    for w = 1:n_warmup
        [A_w, B_w, u_w] = dse_2_build_model(x_k, Nodes, Pipes, Compressors, Sys, load_warmup);
        x_k = A_w \ (B_w * x_k + u_w);
    end
    x_k(1:nN) = max(x_k(1:nN), 0.5);
end

P_k = eye(nS);
P_k(1:nN, 1:nN) = 10 * eye(nN);
P_k(nN+1:end, nN+1:end) = 25 * eye(nP);

Q = diag([Q_p*ones(1,nN), Q_m*ones(1,nP)]);
R0 = diag([R_p*ones(1,nN), R_m*ones(1,nP)]);  % baseline R
R_curr = R0;                                   % AFKF cumulative R

% Joint covariance for power-gas coupling
if has_powersys
    nE_af   = 2 * PowerSys.nB;
    nZ_E_af = size(PowerSys.H_E, 1);
    nI_af   = nE_af + nS;
    P_I_af  = blkdiag(eye(nE_af), P_k);
    Q_I_af  = blkdiag(1e-5*eye(nE_af), Q);
    R_E_af  = (0.02^2) * eye(nZ_E_af);
    H_I_af  = [PowerSys.H_E, zeros(nZ_E_af,nS); zeros(nS,nE_af), eye(nS)];
end


d_k = zeros(nN, 1);


sigma_est = ones(nS, 1);
innovation_history = zeros(window_size, nS);
init_protect_history=zeros(init_protect_hours,nS);
hist_idx = 1;
hist_count = 0;


rzd = zeros(nS, N);
z1  = zeros(1, N);
z2  = zeros(1, N);
Co  = nan(1, N);

u_ab_pr = zeros(2, N);
u_ab_po = zeros(2, N);
P_ab_pr = zeros(2, 2, N);
P_ab_po = zeros(2, 2, N);
K_ab    = zeros(2, N);

u_ab_pr(:,1)   = [1; 1];
P_ab_pr(:,:,1) = eye(2);


Hist_Est = zeros(N, nS);
Leak_Est = zeros(N, nN);
Load_Compensated = zeros(N, nN);
Mu_History  = zeros(N, nS);
MuR_History = zeros(N, nS);
X_E_est = [];

% Initialize Holt smoothing for power states
if has_powersys
    alpha_s = 0.5;
    beta_s  = 0.4;
    x_E_k   = PowerSys.x0;
    L_holt  = PowerSys.x0;
    T_holt  = zeros(size(PowerSys.x0));
    nE      = 2 * PowerSys.nB;
end

for k = 1:N
    z_k = Z(k, :)';
    hr  = t_vec(k);

    % ---- nominal load 
    lf = 1.0 + 0.05 * sin(2*pi*(hr-8)/24);
    load_k = Nodes.BaseLoad * lf;

    % ---- GTU consumption
    if has_powersys
        % Step 1: Holt smoothing predicts x_E at t+1 (Chen Eq 1-4)
        [x_E_pred, L_holt, T_holt] = power_holt_smooth(x_E_k, L_holt, T_holt, alpha_s, beta_s);

        % Step 2: Compute GTU boundary using predicted x_E (Chen Eq 24)
        m_gtu = gtu_coupling(x_E_pred, PowerSys);
        for g = 1:length(m_gtu)
            load_k(m_gtu(g).gas_node_id) = m_gtu(g).m_dot;
        end
        x_E_k = x_E_pred;   % updated in joint step below

    elseif has_gtu
        elf = 0.7 + 0.3 * (sin(2*pi*(hr-6)/12).^2 * (hr>=6 && hr<=22));
        for g = 1:height(GTU)
            if ismember('Status', GTU.Properties.VariableNames)
                if GTU.Status(g) ~= 1, continue; end
            end
            cap  = GTU.Capacity_MW(g);
            pmin = GTU.MinLoad(g) / 100;
            pmax = GTU.MaxLoad(g) / 100;
            pwr  = cap * (pmin + (pmax - pmin) * elf);
            gas  = pwr * GTU.HeatRate(g) * 1000 / 3600 / LHV;
            load_k(GTU.NodeID(g)) = gas;
        end
    end

    % ---- Apply leak compensation to load 
    for i = 1:nN
        if Nodes.Type(i) ~= 1
            load_k(i) = load_k(i) + d_k(i);
        end
    end

    % ---- EKF prediction ----
    [A, B, u] = dse_2_build_model(x_k, Nodes, Pipes, Compressors, Sys, load_k);
    x_pred = A \ (B * x_k + u);
    F = A \ B;
    if has_powersys
        F_I_af      = blkdiag(alpha_s * eye(nE_af), F);
        P_I_pred_af = F_I_af * P_I_af * F_I_af' + Q_I_af;
        P_I_pred_af = (P_I_pred_af + P_I_pred_af') / 2;
        P_pred = P_I_pred_af(nE_af+1:end, nE_af+1:end);  % gas block for fading/outlier
    else
        P_pred = F * P_k * F' + Q;
    end
    P_pred = (P_pred + P_pred')/2;

    % ---- innovation 
    nu_k = z_k - H * x_pred;

    % ---- convert innovation to physical units 
    nu_pressure = nu_k(1:nN) * Sys.c2 / 1e5;   % [Bar]
    nu_flow     = nu_k(nN+1:end);              % [kg/s]
    nu_physical = [nu_pressure; nu_flow];

    % Update sigma estimate
    init_protect_history(k,:)=abs(nu_physical');
    innovation_history(hist_idx, :) = abs(nu_physical');
    hist_idx = mod(hist_idx, window_size) + 1;
    hist_count = min(hist_count + 1, window_size);
    
    if k>= init_protect_steps%window_size
        med_vals = median(innovation_history, 1)';
		finite_corr=1+5/(window_size-1); % finite correction factor
        sigma_est = alpha_sigma_mad * sigma_est + (1 - alpha_sigma_mad) * finite_corr* c_mad * med_vals;
        %% Detect outlier
        is_outlier = abs(nu_physical) > k_outlier * sigma_est;
        is_big_outlier=abs(nu_physical) > k2_outlier * sigma_est;
        is_very_big_outlier=abs(nu_physical) > k3_outlier * sigma_est;
    elseif k >= window_size
        med_vals = median(init_protect_history(1:k,:), 1)';
		finite_corr=1+5/(k-1); % finite correction factor
        sigma_est=c_mad*finite_corr*med_vals;
        is_outlier = zeros(size(nu_physical));
        is_big_outlier=zeros(size(nu_physical));
        is_very_big_outlier=zeros(size(nu_physical));
    else
        is_outlier = zeros(size(nu_physical));
        is_big_outlier=zeros(size(nu_physical));
        is_very_big_outlier=zeros(size(nu_physical));
    end
    %sigma_est(1:nN) = max(sigma_est(1:nN), 0.05);
    %sigma_est(nN+1:end) = max(sigma_est(nN+1:end), 0.5);

    % ---- outlier mask 
    %is_outlier = abs(nu_physical) > k_outlier * sigma_est;
    % ---- replace outlier measurements by prediction (DO NOT modify R)
    z_use = z_k;                 % default: original measurement
    y_pred = H * x_pred;         % predicted measurement

    for m = 1:nS
        if is_very_big_outlier(m)
            z_use(m) = y_pred(m);   % hard replacement
        end
    end

    % recompute innovation using replaced measurement
    nu_use = z_use - y_pred;


 
    % rzd(:,k) = z_k - H*x_pred;
    rzd(:,k) = nu_use;
    z1(k) = trace(H*P_pred*H');
    % z2(k) = trace(R_curr);
    z2(k) = trace(R_curr(nN+1:nS, nN+1:nS));


    if fadingFlag && k > measLengthForfadingFactorEst
        idx1 = k - measLengthForfadingFactorEst;
        idx2 = k;
        rzd_seg = rzd(:, idx1:idx2);

        Co(k) = trace((1/(measLengthForfadingFactorEst-1)) * (rzd_seg * rzd_seg'));

        if k == measLengthForfadingFactorEst + 1
            u_ab_pr(:,measLengthForfadingFactorEst)   = [1;1];
            P_ab_pr(:,:,measLengthForfadingFactorEst) = eye(2);
        end

        F_ab = eye(2);
        H_ab = [z1(k) z2(k)];
        R_ab = 0.01;

        u_ab_po(:,k)   = F_ab*u_ab_pr(:,k-1);
        P_ab_po(:,:,k) = F_ab*P_ab_pr(:,:,k-1)*F_ab' + Q_ab;

        K_ab(:,k) = P_ab_po(:,:,k)*H_ab' * inv(H_ab*P_ab_po(:,:,k)*H_ab' + R_ab);

        u_ab_pr(:,k)   = u_ab_po(:,k) + K_ab(:,k)*(Co(k) - H_ab*u_ab_po(:,k));
        P_ab_pr(:,:,k) = P_ab_po(:,:,k) - K_ab(:,k)*H_ab*P_ab_po(:,:,k);

        u_ab_pr(1,k) = min(max(u_ab_pr(1,k), stateConstrainLB), stateConstrainUB);
        u_ab_pr(2,k) = min(max(u_ab_pr(2,k), measConstrainLB),  measConstrainUB);

        alp = u_ab_pr(1,k);
        bt  = u_ab_pr(2,k);
    else
        alp = 1;
        bt  = 1;
    end

    % ---- apply fading (to gas block only for joint estimation)
    P_pred_f = alp * P_pred;
    R_curr   = bt  * R_curr;   % cumulative
    if has_powersys
        % Apply fading to gas-gas block of joint covariance
        P_I_pred_af(nE_af+1:end, nE_af+1:end) = P_pred_f;
    end
    %trace(P_pred_f)
    %trace(R_curr)
    % ---- adaptive R from MAD outliers
    % R_adaptive = R_curr;
    % for m = 1:nS
    %     if is_outlier(m)
    %         R_adaptive(m,m) = R_curr(m,m) * 100;
    % 
    %     end
    % end
    R_adaptive = R_curr;
    for m = 1:nS
        if is_outlier(m)
            R_adaptive(m, m) = abs(nu_physical(m))/k_outlier; % Huber weight function
        elseif is_big_outlier(m)
            R_adaptive(m, m) = (k3_outlier-k2_outlier)/(k_outlier*(k3_outlier/abs(nu_physical(m))-1)); %% BUG FIX: was k3_ouliter (typo in Andrew's template)
        elseif is_very_big_outlier(m)
            R_adaptive(m, m) = 1e6;
        end
    end
    % ---- EKF update
    Mu_History(k, :)  = diag(R_adaptive)' ./ diag(R0)';
    MuR_History(k, :) = diag(R_adaptive)';
    if has_powersys
        R_I_af = blkdiag(R_E_af, R_adaptive);
        x_I_pr_af = [x_E_pred; x_pred];
        if has_ze
            z_I_af = [Z_E(k,:)'; z_use];
        else
            z_I_af = [PowerSys.H_E * x_E_pred; z_use];
        end
        S_I = H_I_af * P_I_pred_af * H_I_af' + R_I_af;
        S_I = (S_I + S_I') / 2;
        K_I = P_I_pred_af * H_I_af' / S_I;
        x_I_upd = x_I_pr_af + K_I * (z_I_af - H_I_af * x_I_pr_af);
        P_I_af  = (eye(nI_af) - K_I * H_I_af) * P_I_pred_af;
        P_I_af  = (P_I_af + P_I_af') / 2 + 1e-6 * eye(nI_af);
        x_E_k   = x_I_upd(1:nE_af);
        x_k     = x_I_upd(nE_af+1:end);
        P_k     = P_I_af(nE_af+1:end, nE_af+1:end);
        if isempty(X_E_est), X_E_est = zeros(N, nE_af); end
        X_E_est(k,:) = x_E_k';
    else
        S = H * P_pred_f * H' + R_adaptive;
        K = P_pred_f * H' / S;
        x_k = x_pred + K * nu_use;
        P_k = (eye(nS) - K * H) * P_pred_f;
        P_k = (P_k + P_k')/2 + 1e-6 * eye(nS);
    end

    in_init_period = (k <= init_protect_steps);

    leak_detected = false;
    if has_detector && k <= length(Det_Signal)
        leak_detected = Det_Signal(k);
    end

    for i = 1:nN
        if Nodes.Type(i) == 1
            continue;  % source node
        end

        if in_init_period
            d_k(i) = 0;
            continue;
        end

        % Freeze during outlier 
        if is_outlier(i)
            d_k(i) = d_k(i) * 0.98;
            continue;
        end

        nu_p_i = nu_pressure(i);  % [Bar]



        if leak_detected
            if nu_p_i < -delta_th
                leak_signal = -nu_p_i;
                d_k(i) = alpha_dk * d_k(i) + (1 - alpha_dk) * kappa * leak_signal;
                d_k(i) = max(0, min(d_k(i), 80));
            elseif nu_p_i > delta_th && d_k(i) > 0
                d_k(i) = d_k(i) * 0.85;
            elseif abs(nu_p_i) <= delta_th
                d_k(i) = d_k(i) * 0.995;
            end
        else
            d_k(i) = d_k(i) * 0.95;
        end

        if d_k(i) < 0.1
            d_k(i) = 0;
        end
    end

    % ---- physical constraints ----
    x_k(1:nN) = max(x_k(1:nN), 0.5);

    % ---- store ----
    Hist_Est(k, :) = x_k';
    Leak_Est(k, :) = d_k';
    Load_Compensated(k, :) = load_k';
end

% ---- post smoothing
for i = 1:nP
    Hist_Est(:, nN+i) = movmean(Hist_Est(:, nN+i), 3);
end

end



% function [Hist_Est, Leak_Est] = AFUKF(Z, Nodes, Pipes, Compressors, Sys, t_vec, GTU, leak_node, Q_p, Q_m, R_p, R_m)
% % AFUKF (actually AFEKF)
% % EKF state estimation + Adaptive Fading (AFKF logic strictly following your original code)
% % + leak detection/estimation (kept as your previous style)
% %
% % Only changes vs your version:
% %   1) allow bt < 1 (measConstrainLB < 1)
% %   2) increase Q_ab(2,2) so bt can move
% %   3) add R clamp for numerical safety (still R = bt*R accumulation)
% 
% LHV = 50;
% 
% % =========================
% % 0) Dimensions
% % =========================
% N  = length(t_vec);
% [nZ_N, nS] = size(Z);
% if nZ_N ~= N
%     error('Z rows (%d) must equal length(t_vec) (%d).', nZ_N, N);
% end
% 
% nN = height(Nodes);
% nP = height(Pipes);
% if nS ~= (nN + nP)
%     error('Z columns (%d) must equal nN+nP (%d).', nS, nN+nP);
% end
% 
% H = eye(nS);
% 
% % Make "z(:,t)" style available exactly like your original code
% z_all = Z.';          % [nS x N]
% 
% % =========================
% % 1) Noise / EKF init
% % =========================
% R_p_init = R_p;
% R_m_init = R_m;
% 
% x_t_mimus = z_all(:,1);                       % initial state
% P_t_minus = blkdiag(10*eye(nN), 25*eye(nP));   % initial covariance
% 
% Q = blkdiag(Q_p*eye(nN), Q_m*eye(nP));
% R = blkdiag(R_p_init*eye(nN), R_m_init*eye(nP));
% 
% % =========================
% % 2) AF (fading) parameters  (STRICTLY follow your AFKF logic)
% % =========================
% fadingFlag = true;
% measLengthForfadingFactorEst = 15;   % window length
% 
% % ---- KEY CHANGE #1: make bt movable ----
% % Your original comment logic:
% %   - if state covariance inaccurate -> make Q_ab(1,1) big
% %   - if measurement noise inaccurate -> make Q_ab(2,2) big
% % Here we want bt to actually move, so give bt larger process noise too.
% Q_ab = [1 0; 0 1e4];   % (changed from [1e4 0;0 1] to let bt move)
% 
% % ---- KEY CHANGE #2: allow bt < 1 so R can decrease ----
% stateConstrainLB = 1;
% stateConstrainUB = 1.50;     % "not too large"
% measConstrainLB  = 1;     % < 1 is crucial
% measConstrainUB  = 1.50;
% 
% 
% % =========================
% % 3) Leak detection settings (keep your style)
% % =========================
% do_leak_est = true;
% if nargin < 8 || isempty(leak_node), leak_node = 7; end
% 
% win_size = 6;
% thres = 3.0;
% confirm_cnt = 3;
% 
% res_hist = zeros(win_size, nN);
% base_std = ones(1,nN)*0.5; 
% 
% is_leak = true;
% leak_cnt = 0;
% est_leak = 0;
% 
% % =========================
% % 4) Buffers for AFKF (prealloc)
% % =========================
% rzd = zeros(nS, N);
% z1  = zeros(1, N);
% z2  = zeros(1, N);
% Co  = nan(1, N);
% 
% u_ab_pr = zeros(2, N);
% u_ab_po = zeros(2, N);
% P_ab_pr = zeros(2, 2, N);
% P_ab_po = zeros(2, 2, N);
% K_ab    = zeros(2, N);
% 
% % initial AF state at t=1 (safe)
% u_ab_pr(:,1)   = [1; 1];
% P_ab_pr(:,:,1) = eye(2);
% 
% % =========================
% % 5) Outputs
% % =========================
% Hist_Est = zeros(N, nS);
% Leak_Est = zeros(N, 1);
% 
% % =========================
% % 6) Main filtering loop
% % =========================
% has_gtu = (nargin >= 7) && ~isempty(GTU) && height(GTU) > 0;
% 
% for t = 1:N
% 
%     z = z_all(:,t);
%     hr = t_vec(t);
% 
%     % ---- Load logic ----
%     lf = 1.0 + 0.05*sin(2*pi*(hr-8)/24);
%     load_cur = Nodes.BaseLoad * lf;
% 
%     if has_gtu
%         elf = 0.7 + 0.3*(sin(2*pi*(hr-6)/12).^2 * (hr>=6 && hr<=22));
%         for g = 1:height(GTU)
%             pwr = GTU.Capacity_MW(g) * (GTU.MinLoad(g)/100 + (GTU.MaxLoad(g)-GTU.MinLoad(g))/100*elf);
%             load_cur(GTU.NodeID(g)) = pwr * GTU.HeatRate(g) * 1000/3600/LHV;
%         end
%     end
% 
%     if is_leak && do_leak_est
%         load_cur(leak_node) = load_cur(leak_node) + est_leak;
%     end
% 
%     % =====================================================
%     % step-1: prediction (EKF with your implicit model)
%     % =====================================================
%     [A, B, u] = dse_2_build_model(x_t_mimus, Nodes, Pipes, Compressors, Sys, load_cur);
% 
%     x_t_pre = A \ (B*x_t_mimus + u);     % predicted state
%     F = A \ B;                           % Jacobian
% 
%     P_t_pre = F*P_t_minus*F' + Q;
%     P_t_pre = (P_t_pre + P_t_pre')/2;
% 
%     % =====================================================
%     % step-2: covariance compensation (STRICT AFKF logic)
%     % =====================================================
%     rzd(:,t) = z_all(:,t) - H*x_t_pre;   % residual
%     z1(t) = trace(H*P_t_pre*H');
%     z2(t) = trace(R);
% 
%     if fadingFlag && t > measLengthForfadingFactorEst
% 
%         % empirical error covariance over window [t-L : t]
%         idx1 = t - measLengthForfadingFactorEst;
%         idx2 = t;
%         rzd_seg = rzd(:, idx1:idx2);
% 
%         Co(t) = trace( (1/(measLengthForfadingFactorEst-1)) * (rzd_seg * rzd_seg') );
% 
%         % initialize AF-KF state at index measLengthFor... (ONLY ONCE)
%         if t == measLengthForfadingFactorEst + 1
%             u_ab_pr(:,measLengthForfadingFactorEst) = [1;1];
%             P_ab_pr(:,:,measLengthForfadingFactorEst) = eye(2);
%         end
% 
%         % model for fading factor estimation
%         F_ab = eye(2);
%         H_ab = [z1(t) z2(t)];
%         R_ab = 0.001;
% 
%         % prediction
%         u_ab_po(:,t)   = F_ab*u_ab_pr(:,t-1);
%         P_ab_po(:,:,t) = F_ab*P_ab_pr(:,:,t-1)*F_ab' + Q_ab;
% 
%         % correction (use inv exactly as your code)
%         K_ab(:,t) = P_ab_po(:,:,t)*H_ab' * inv(H_ab*P_ab_po(:,:,t)*H_ab' + R_ab);
% 
%         u_ab_pr(:,t)   = u_ab_po(:,t) + K_ab(:,t)*(Co(t) - H_ab*u_ab_po(:,t));
%         P_ab_pr(:,:,t) = P_ab_po(:,:,t) - K_ab(:,t)*H_ab*P_ab_po(:,:,t);
% 
%         % constraints
%         if u_ab_pr(1,t) < stateConstrainLB, u_ab_pr(1,t) = stateConstrainLB; end
%         if u_ab_pr(1,t) > stateConstrainUB, u_ab_pr(1,t) = stateConstrainUB; end
% 
%         if u_ab_pr(2,t) < measConstrainLB,  u_ab_pr(2,t) = measConstrainLB;  end
%         if u_ab_pr(2,t) > measConstrainUB,  u_ab_pr(2,t) = measConstrainUB;  end
% 
%         alp = u_ab_pr(1,t);
%         bt  = u_ab_pr(2,t);   % <-- add semicolon to avoid spam
% 
%     else
%         alp = 1; bt = 1;
% 
%         % keep recursion stable
%         if t > 1
%             u_ab_pr(:,t)   = u_ab_pr(:,t-1);
%             P_ab_pr(:,:,t) = P_ab_pr(:,:,t-1);
%         end
%     end
% 
%     % Apply fading (STRICT)
%     P_t_pre = alp * P_t_pre;
%     R = bt * R;     % R accumulation
% 
% 
% 
%     % =====================================================
%     % step-3: correction (STRICT form)
%     % =====================================================
%     K = P_t_pre*H' * inv(H*P_t_pre*H' + R);
%     x_t_pos = x_t_pre + K*(z_all(:,t) - H*x_t_pre);
%     P_t_pos = P_t_pre - K*H*P_t_pre;
% 
%     % update state and covariance
%     x_t_mimus = x_t_pos;
%     P_t_minus = (P_t_pos + P_t_pos')/2;
% 
%     % =====================================================
%     % Leak detection / estimation (your style)
%     % =====================================================
%     innov = z_all(:,t) - x_t_pre;           % since H = I
%     p_innov = innov(1:nN);
% 
%     res_hist = circshift(res_hist,1,1);
%     res_hist(1,:) = p_innov';
% 
%     if t > win_size
%         cur_std = std(res_hist,0,1);
%         cur_std(cur_std < 0.1) = 0.1;
% 
%         norm_res = abs(p_innov') ./ max(cur_std, base_std);
% 
%           if any(norm_res > thres)
%             leak_cnt = leak_cnt + 1;
%         else
%             leak_cnt = max(0,leak_cnt-1);
%         end
% 
%         if leak_cnt >= confirm_cnt && ~is_leak
%             is_leak = true;
%         end
% 
%          if is_leak && leak_cnt == 0 && all(norm_res < thres*0.5)
%             is_leak = false;
%             est_leak = 0;
%         end
% 
%         % --- Adaptive Leak Rate Estimation ---
%         if is_leak && do_leak_est
%             dp = mean(base_std) - mean(p_innov);
%             delta = dp * 2;
%             est_leak = 0.8*est_leak + 0.2*max(delta,0);  
%             if est_leak > 50, est_leak = 50; end        
%         end
%     elseif t == win_size
%          base_std = std(res_hist,0,1);
%         base_std(base_std < 0.1) = 0.1;
%     end
% 
%      x_t_mimus(1:nN) = max(x_t_mimus(1:nN), 0.5);
% 
%     Hist_Est(t,:) = x_t_mimus';
%     Leak_Est(t) = est_leak;
% 
% end
% 
% end

% function [Hist_Est, Leak_Est, Load_Compensated] = AFUKF(Z, Nodes, Pipes, Compressors, Sys, t_vec, GTU, Det_Signal)
% 
% LHV = 50;
% 
% Q_p = 0.15;
% Q_m = 0.4;
% R_p = 1;
% R_m = 10;
% 
% % Innovation feedback parameters - for leakage
% delta_th = 0.35;     % deadzone [Bar]
% alpha_dk = 0.95;     % smoothing
% kappa    = 40;       % compensation gain
% init_protect_hours = 5.0;
% 
% % Outlier suppression parameters
% k_outlier = 4;
% c_mad = 1.4826;
% alpha_sigma_mad = 0.95;
% window_size = 10;
% 
% fadingFlag = true;
% measLengthForfadingFactorEst = 15;
% 
% % Let bt move (as you used)
% Q_ab = [1 0; 0 1e3];
% 
% stateConstrainLB = 0.995;  stateConstrainUB = 1.0;   % alp in [0.99, 1]
% measConstrainLB  = 0.999;  measConstrainUB  = 1.0;   % bt  in [0.99, 1.1]
% 
% 
% N = length(t_vec);
% [nZ_N, nS] = size(Z);
% if nZ_N ~= N
%     error('Z rows (%d) must equal length(t_vec) (%d).', nZ_N, N);
% end
% 
% nN = height(Nodes);
% nP = height(Pipes);
% if nS ~= (nN + nP)
%     error('Z columns (%d) must equal nN+nP (%d).', nS, nN+nP);
% end
% 
% H = eye(nS);
% 
% has_gtu = (nargin >= 7) && ~isempty(GTU) && height(GTU) > 0;
% has_detector = (nargin >= 8) && ~isempty(Det_Signal);
% 
% init_protect_steps = round(init_protect_hours * 3600 / Sys.dt);
% 
% init_n = min(5, N);
% x_k = mean(Z(1:init_n,:), 1)';
% 
% P_k = eye(nS);
% P_k(1:nN, 1:nN) = 10 * eye(nN);
% P_k(nN+1:end, nN+1:end) = 25 * eye(nP);
% 
% Q = diag([Q_p*ones(1,nN), Q_m*ones(1,nP)]);
% R0 = diag([R_p*ones(1,nN), R_m*ones(1,nP)]);  % baseline R
% R_curr = R0;                                   % AFKF cumulative R
% 
% 
% d_k = zeros(nN, 1);
% 
% 
% sigma_est = ones(nS, 1);
% innovation_history = zeros(window_size, nS);
% hist_idx = 1;
% hist_count = 0;
% 
% 
% rzd = zeros(nS, N);
% z1  = zeros(1, N);
% z2  = zeros(1, N);
% Co  = nan(1, N);
% 
% u_ab_pr = zeros(2, N);
% u_ab_po = zeros(2, N);
% P_ab_pr = zeros(2, 2, N);
% P_ab_po = zeros(2, 2, N);
% K_ab    = zeros(2, N);
% 
% u_ab_pr(:,1)   = [1; 1];
% P_ab_pr(:,:,1) = eye(2);
% 
% 
% Hist_Est = zeros(N, nS);
% Leak_Est = zeros(N, nN);
% Load_Compensated = zeros(N, nN);
% 
% for k = 1:N
%     z_k = Z(k, :)';
%     hr  = t_vec(k);
% 
%     % ---- nominal load 
%     lf = 1.0 + 0.05 * sin(2*pi*(hr-8)/24);
%     load_k = Nodes.BaseLoad * lf;
% 
%     % ---- GTU consumption 
%     if has_gtu
%         elf = 0.7 + 0.3 * (sin(2*pi*(hr-6)/12).^2 * (hr>=6 && hr<=22));
%         for g = 1:height(GTU)
%             if ismember('Status', GTU.Properties.VariableNames)
%                 if GTU.Status(g) ~= 1, continue; end
%             end
%             cap  = GTU.Capacity_MW(g);
%             pmin = GTU.MinLoad(g) / 100;
%             pmax = GTU.MaxLoad(g) / 100;
%             pwr  = cap * (pmin + (pmax - pmin) * elf);
%             gas  = pwr * GTU.HeatRate(g) * 1000 / 3600 / LHV;
%             load_k(GTU.NodeID(g)) = gas;
%         end
%     end
% 
%     % ---- Apply leak compensation to load 
%     for i = 1:nN
%         if Nodes.Type(i) ~= 1
%             load_k(i) = load_k(i) + d_k(i);
%         end
%     end
% 
%     % ---- EKF prediction ----
%     [A, B, u] = dse_2_build_model(x_k, Nodes, Pipes, Compressors, Sys, load_k);
%     x_pred = A \ (B * x_k + u);
%     F = A \ B;
%     P_pred = F * P_k * F' + Q;
%     P_pred = (P_pred + P_pred')/2;
% 
%     % ---- innovation 
%     nu_k = z_k - H * x_pred;
% 
%     % ---- convert innovation to physical units 
%     nu_pressure = nu_k(1:nN) * Sys.c2 / 1e5;   % [Bar]
%     nu_flow     = nu_k(nN+1:end);              % [kg/s]
%     nu_physical = [nu_pressure; nu_flow];
% 
%     % ---- MAD sigma estimate 
%     innovation_history(hist_idx, :) = abs(nu_physical');
%     hist_idx = mod(hist_idx, window_size) + 1;
%     hist_count = min(hist_count + 1, window_size);
% 
%     if hist_count >= window_size
%         med_vals = median(innovation_history, 1)';
%         sigma_est = alpha_sigma_mad * sigma_est + (1 - alpha_sigma_mad) * c_mad * med_vals;
%     end
%     sigma_est(1:nN) = max(sigma_est(1:nN), 0.05);
%     sigma_est(nN+1:end) = max(sigma_est(nN+1:end), 0.5);
% 
%     % ---- outlier mask 
%     is_outlier = abs(nu_physical) > k_outlier * sigma_est;
% 
% 
%     rzd(:,k) = z_k - H*x_pred;
%     z1(k) = trace(H*P_pred*H');
%     z2(k) = trace(R_curr);
% 
%     if fadingFlag && k > measLengthForfadingFactorEst
%         idx1 = k - measLengthForfadingFactorEst;
%         idx2 = k;
%         rzd_seg = rzd(:, idx1:idx2);
% 
%         Co(k) = trace((1/(measLengthForfadingFactorEst-1)) * (rzd_seg * rzd_seg'));
% 
%         if k == measLengthForfadingFactorEst + 1
%             u_ab_pr(:,measLengthForfadingFactorEst)   = [1;1];
%             P_ab_pr(:,:,measLengthForfadingFactorEst) = eye(2);
%         end
% 
%         F_ab = eye(2);
%         H_ab = [z1(k) z2(k)];
%         R_ab = 1e-3;
% 
%         u_ab_po(:,k)   = F_ab*u_ab_pr(:,k-1);
%         P_ab_po(:,:,k) = F_ab*P_ab_pr(:,:,k-1)*F_ab' + Q_ab;
% 
%         K_ab(:,k) = P_ab_po(:,:,k)*H_ab' * inv(H_ab*P_ab_po(:,:,k)*H_ab' + R_ab);
% 
%         u_ab_pr(:,k)   = u_ab_po(:,k) + K_ab(:,k)*(Co(k) - H_ab*u_ab_po(:,k));
%         u_ab_pr(:,k)
%         P_ab_pr(:,:,k) = P_ab_po(:,:,k) - K_ab(:,k)*H_ab*P_ab_po(:,:,k);
% 
%         u_ab_pr(1,k) = min(max(u_ab_pr(1,k), stateConstrainLB), stateConstrainUB);
%         u_ab_pr(2,k) = min(max(u_ab_pr(2,k), measConstrainLB),  measConstrainUB);
% 
%         alp = u_ab_pr(1,k);
%         bt  = u_ab_pr(2,k);
%     else
%         alp = 1;
%         bt  = 1;
%     end
% 
%     % ---- apply fading 
%     P_pred_f = alp * P_pred ;
%     R_curr   = bt  * R_curr ;   % cumulative
%     %trace(P_pred_f)
%     %trace(R_curr)
%     % ---- adaptive R from MAD outliers
%     R_adaptive = R_curr;
%     for m = 1:nS
%         if is_outlier(m)
%             R_adaptive(m,m) = R_curr(m,m) * 100;
%         end
%     end
% 
%     % ---- EKF update (use nu_k unchanged
%     S = H * P_pred_f * H' + R_adaptive;
%     K = P_pred_f * H' / S;
% 
%     x_k = x_pred + K * nu_k;
%     P_k = (eye(nS) - K * H) * P_pred_f;
%     P_k = (P_k + P_k')/2 + 1e-6 * eye(nS);
% 
%     in_init_period = (k <= init_protect_steps);
% 
%     leak_detected = false;
%     if has_detector && k <= length(Det_Signal)
%         leak_detected = Det_Signal(k);
%     end
% 
%     for i = 1:nN
%         if Nodes.Type(i) == 1
%             continue;  % source node
%         end
% 
%         if in_init_period
%             d_k(i) = 0;
%             continue;
%         end
% 
%         % Freeze during outlier 
%         if is_outlier(i)
%             d_k(i) = d_k(i) * 0.98;
%             continue;
%         end
% 
%         nu_p_i = nu_pressure(i);  % [Bar]
% 
%         if leak_detected
%             if nu_p_i < -delta_th
%                 leak_signal = -nu_p_i;
%                 d_k(i) = alpha_dk * d_k(i) + (1 - alpha_dk) * kappa * leak_signal;
%                 d_k(i) = max(0, min(d_k(i), 80));
%             elseif nu_p_i > delta_th && d_k(i) > 0
%                 d_k(i) = d_k(i) * 0.85;
%             elseif abs(nu_p_i) <= delta_th
%                 d_k(i) = d_k(i) * 0.995;
%             end
%         else
%             d_k(i) = d_k(i) * 0.95;
%         end
% 
%         if d_k(i) < 0.1
%             d_k(i) = 0;
%         end
%     end
% 
%     % ---- physical constraints ----
%     x_k(1:nN) = max(x_k(1:nN), 0.5);
% 
%     % ---- store ----
%     Hist_Est(k, :) = x_k';
%     Leak_Est(k, :) = d_k';
%     Load_Compensated(k, :) = load_k';
% end
% 
% % ---- post smoothing
% for i = 1:nP
%     Hist_Est(:, nN+i) = movmean(Hist_Est(:, nN+i), 3);
% end
% 
% end



% function [Hist_Est, Leak_Est] = AFUKF(Z, Nodes, Pipes, Compressors, Sys, t_vec, GTU, leak_node, Q_p, Q_m, R_p, R_m)
% % AFUKF (actually AFEKF)
% % EKF state estimation + Adaptive Fading (AFKF logic strictly following your original code)
% % + leak detection/estimation (kept as your previous style)
% %
% % Only changes vs your version:
% %   1) allow bt < 1 (measConstrainLB < 1)
% %   2) increase Q_ab(2,2) so bt can move
% %   3) add R clamp for numerical safety (still R = bt*R accumulation)
% 
% LHV = 50;
% 
% % =========================
% % 0) Dimensions
% % =========================
% N  = length(t_vec);
% [nZ_N, nS] = size(Z);
% if nZ_N ~= N
%     error('Z rows (%d) must equal length(t_vec) (%d).', nZ_N, N);
% end
% 
% nN = height(Nodes);
% nP = height(Pipes);
% if nS ~= (nN + nP)
%     error('Z columns (%d) must equal nN+nP (%d).', nS, nN+nP);
% end
% 
% H = eye(nS);
% 
% % Make "z(:,t)" style available exactly like your original code
% z_all = Z.';          % [nS x N]
% 
% % =========================
% % 1) Noise / EKF init
% % =========================
% R_p_init = R_p;
% R_m_init = R_m;
% 
% x_t_mimus = z_all(:,1);                       % initial state
% P_t_minus = blkdiag(10*eye(nN), 25*eye(nP));   % initial covariance
% 
% Q = blkdiag(Q_p*eye(nN), Q_m*eye(nP));
% R = blkdiag(R_p_init*eye(nN), R_m_init*eye(nP));
% 
% % =========================
% % 2) AF (fading) parameters  (STRICTLY follow your AFKF logic)
% % =========================
% fadingFlag = true;
% measLengthForfadingFactorEst = 15;   % window length
% 
% % ---- KEY CHANGE #1: make bt movable ----
% % Your original comment logic:
% %   - if state covariance inaccurate -> make Q_ab(1,1) big
% %   - if measurement noise inaccurate -> make Q_ab(2,2) big
% % Here we want bt to actually move, so give bt larger process noise too.
% Q_ab = [1 0; 0 1e4];   % (changed from [1e4 0;0 1] to let bt move)
% 
% % ---- KEY CHANGE #2: allow bt < 1 so R can decrease ----
% stateConstrainLB = 1;
% stateConstrainUB = 1.50;     % "not too large"
% measConstrainLB  = 1;     % < 1 is crucial
% measConstrainUB  = 1.50;
% 
% 
% % =========================
% % 3) Leak detection settings (keep your style)
% % =========================
% do_leak_est = true;
% if nargin < 8 || isempty(leak_node), leak_node = 7; end
% 
% win_size = 6;
% thres = 3.0;
% confirm_cnt = 3;
% 
% res_hist = zeros(win_size, nN);
% base_std = ones(1,nN)*0.5; 
% 
% is_leak = true;
% leak_cnt = 0;
% est_leak = 0;
% 
% % =========================
% % 4) Buffers for AFKF (prealloc)
% % =========================
% rzd = zeros(nS, N);
% z1  = zeros(1, N);
% z2  = zeros(1, N);
% Co  = nan(1, N);
% 
% u_ab_pr = zeros(2, N);
% u_ab_po = zeros(2, N);
% P_ab_pr = zeros(2, 2, N);
% P_ab_po = zeros(2, 2, N);
% K_ab    = zeros(2, N);
% 
% % initial AF state at t=1 (safe)
% u_ab_pr(:,1)   = [1; 1];
% P_ab_pr(:,:,1) = eye(2);
% 
% % =========================
% % 5) Outputs
% % =========================
% Hist_Est = zeros(N, nS);
% Leak_Est = zeros(N, 1);
% 
% % =========================
% % 6) Main filtering loop
% % =========================
% has_gtu = (nargin >= 7) && ~isempty(GTU) && height(GTU) > 0;
% 
% for t = 1:N
% 
%     z = z_all(:,t);
%     hr = t_vec(t);
% 
%     % ---- Load logic ----
%     lf = 1.0 + 0.05*sin(2*pi*(hr-8)/24);
%     load_cur = Nodes.BaseLoad * lf;
% 
%     if has_gtu
%         elf = 0.7 + 0.3*(sin(2*pi*(hr-6)/12).^2 * (hr>=6 && hr<=22));
%         for g = 1:height(GTU)
%             pwr = GTU.Capacity_MW(g) * (GTU.MinLoad(g)/100 + (GTU.MaxLoad(g)-GTU.MinLoad(g))/100*elf);
%             load_cur(GTU.NodeID(g)) = pwr * GTU.HeatRate(g) * 1000/3600/LHV;
%         end
%     end
% 
%     if is_leak && do_leak_est
%         load_cur(leak_node) = load_cur(leak_node) + est_leak;
%     end
% 
%     % =====================================================
%     % step-1: prediction (EKF with your implicit model)
%     % =====================================================
%     [A, B, u] = dse_2_build_model(x_t_mimus, Nodes, Pipes, Compressors, Sys, load_cur);
% 
%     x_t_pre = A \ (B*x_t_mimus + u);     % predicted state
%     F = A \ B;                           % Jacobian
% 
%     P_t_pre = F*P_t_minus*F' + Q;
%     P_t_pre = (P_t_pre + P_t_pre')/2;
% 
%     % =====================================================
%     % step-2: covariance compensation (STRICT AFKF logic)
%     % =====================================================
%     rzd(:,t) = z_all(:,t) - H*x_t_pre;   % residual
%     z1(t) = trace(H*P_t_pre*H');
%     z2(t) = trace(R);
% 
%     if fadingFlag && t > measLengthForfadingFactorEst
% 
%         % empirical error covariance over window [t-L : t]
%         idx1 = t - measLengthForfadingFactorEst;
%         idx2 = t;
%         rzd_seg = rzd(:, idx1:idx2);
% 
%         Co(t) = trace( (1/(measLengthForfadingFactorEst-1)) * (rzd_seg * rzd_seg') );
% 
%         % initialize AF-KF state at index measLengthFor... (ONLY ONCE)
%         if t == measLengthForfadingFactorEst + 1
%             u_ab_pr(:,measLengthForfadingFactorEst) = [1;1];
%             P_ab_pr(:,:,measLengthForfadingFactorEst) = eye(2);
%         end
% 
%         % model for fading factor estimation
%         F_ab = eye(2);
%         H_ab = [z1(t) z2(t)];
%         R_ab = 0.001;
% 
%         % prediction
%         u_ab_po(:,t)   = F_ab*u_ab_pr(:,t-1);
%         P_ab_po(:,:,t) = F_ab*P_ab_pr(:,:,t-1)*F_ab' + Q_ab;
% 
%         % correction (use inv exactly as your code)
%         K_ab(:,t) = P_ab_po(:,:,t)*H_ab' * inv(H_ab*P_ab_po(:,:,t)*H_ab' + R_ab);
% 
%         u_ab_pr(:,t)   = u_ab_po(:,t) + K_ab(:,t)*(Co(t) - H_ab*u_ab_po(:,t));
%         P_ab_pr(:,:,t) = P_ab_po(:,:,t) - K_ab(:,t)*H_ab*P_ab_po(:,:,t);
% 
%         % constraints
%         if u_ab_pr(1,t) < stateConstrainLB, u_ab_pr(1,t) = stateConstrainLB; end
%         if u_ab_pr(1,t) > stateConstrainUB, u_ab_pr(1,t) = stateConstrainUB; end
% 
%         if u_ab_pr(2,t) < measConstrainLB,  u_ab_pr(2,t) = measConstrainLB;  end
%         if u_ab_pr(2,t) > measConstrainUB,  u_ab_pr(2,t) = measConstrainUB;  end
% 
%         alp = u_ab_pr(1,t);
%         bt  = u_ab_pr(2,t);   % <-- add semicolon to avoid spam
% 
%     else
%         alp = 1; bt = 1;
% 
%         % keep recursion stable
%         if t > 1
%             u_ab_pr(:,t)   = u_ab_pr(:,t-1);
%             P_ab_pr(:,:,t) = P_ab_pr(:,:,t-1);
%         end
%     end
% 
%     % Apply fading (STRICT)
%     P_t_pre = alp * P_t_pre;
%     R = bt * R;     % R accumulation
% 
% 
% 
%     % =====================================================
%     % step-3: correction (STRICT form)
%     % =====================================================
%     K = P_t_pre*H' * inv(H*P_t_pre*H' + R);
%     x_t_pos = x_t_pre + K*(z_all(:,t) - H*x_t_pre);
%     P_t_pos = P_t_pre - K*H*P_t_pre;
% 
%     % update state and covariance
%     x_t_mimus = x_t_pos;
%     P_t_minus = (P_t_pos + P_t_pos')/2;
% 
%     % =====================================================
%     % Leak detection / estimation (your style)
%     % =====================================================
%     innov = z_all(:,t) - x_t_pre;           % since H = I
%     p_innov = innov(1:nN);
% 
%     res_hist = circshift(res_hist,1,1);
%     res_hist(1,:) = p_innov';
% 
%     if t > win_size
%         cur_std = std(res_hist,0,1);
%         cur_std(cur_std < 0.1) = 0.1;
% 
%         norm_res = abs(p_innov') ./ max(cur_std, base_std);
% 
%           if any(norm_res > thres)
%             leak_cnt = leak_cnt + 1;
%         else
%             leak_cnt = max(0,leak_cnt-1);
%         end
% 
%         if leak_cnt >= confirm_cnt && ~is_leak
%             is_leak = true;
%         end
% 
%          if is_leak && leak_cnt == 0 && all(norm_res < thres*0.5)
%             is_leak = false;
%             est_leak = 0;
%         end
% 
%         % --- Adaptive Leak Rate Estimation ---
%         if is_leak && do_leak_est
%             dp = mean(base_std) - mean(p_innov);
%             delta = dp * 2;
%             est_leak = 0.8*est_leak + 0.2*max(delta,0);  
%             if est_leak > 50, est_leak = 50; end        
%         end
%     elseif t == win_size
%          base_std = std(res_hist,0,1);
%         base_std(base_std < 0.1) = 0.1;
%     end
% 
%      x_t_mimus(1:nN) = max(x_t_mimus(1:nN), 0.5);
% 
%     Hist_Est(t,:) = x_t_mimus';
%     Leak_Est(t) = est_leak;
% 
% end
% 
% end