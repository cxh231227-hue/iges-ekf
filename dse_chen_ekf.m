function [Hist_Est, Load_Used, R_History, Dtt_History, X_E_est] = dse_chen_ekf(Z, Nodes, Pipes, Compressors, Sys, t_vec, GTU, mismatch, PowerSys, Z_E)
% DSE_CHEN_EKF - Chen robust KF with joint power-gas estimation
%
% Same joint P_I structure as dse_normal_ekf, but with time-varying
% scalar matrix mu applied to the GAS measurement noise R_G only
% (PMU measurements are assumed reliable, Chen Section IV-C).

LHV = 50;

Q_p = 0.15;
Q_m = 0.4;
R_p = 1;
R_m = 10;

if mismatch < 1
    R_m = Q_m; R_p = Q_p;
elseif mismatch > 1
    Q_m = R_m; Q_p = R_p;
end

m_w           = 25;
mu_min        = 1;
mu_max        = 1000;
regularization = 1e-6;

N  = length(t_vec);
nN = height(Nodes);
nP = height(Pipes);
nS = nN + nP;

has_gtu      = nargin >= 7 && ~isempty(GTU) && height(GTU) > 0;
has_powersys = nargin >= 9 && ~isempty(PowerSys);
has_ze       = nargin >= 10 && ~isempty(Z_E);

X_E_est  = [];
Hist_Est  = zeros(N, nS);
Load_Used = zeros(N, nN);
R_History   = zeros(N, nS);
Dtt_History = zeros(N, nS);

init_n = min(5, N);
x_G_k = mean(Z(1:init_n,:), 1)';

if has_powersys
    alpha_s = 0.5;
    beta_s  = 0.4;
    nE      = 2 * PowerSys.nB;
    H_E     = PowerSys.H_E;
    nZ_E    = size(H_E, 1);
    nI      = nE + nS;
    nZ_I    = nZ_E + nS;

    x_E_k  = PowerSys.x0;
    L_holt = PowerSys.x0;
    T_holt = zeros(nE, 1);

    % Warm-up
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

    P_I = blkdiag(eye(nE), [10*eye(nN), zeros(nN,nP); zeros(nP,nN), 25*eye(nP)]);
    Q_I = blkdiag(1e-5*eye(nE), blkdiag(Q_p*eye(nN), Q_m*eye(nP)));
    R_E = (0.02^2) * eye(nZ_E);
    R_G_base = blkdiag(R_p*eye(nN), R_m*eye(nP));
    H_I = [H_E, zeros(nZ_E,nS); zeros(nS,nE), eye(nS)];

    % Innovation buffer for mu adaptation (gas measurements only)
    innov_buf   = zeros(m_w, nS);
    buf_idx     = 1;
    buf_count   = 0;
    mu_diag_gas = ones(nS, 1);

    X_E_est = zeros(N, nE);

else
    P_k = blkdiag(10*eye(nN), 25*eye(nP));
    Q   = blkdiag(Q_p*eye(nN), Q_m*eye(nP));
    R   = blkdiag(R_p*eye(nN), R_m*eye(nP));
    innov_buf  = zeros(m_w, nS);
    buf_idx    = 1;
    buf_count  = 0;
    mu_diag    = ones(nS, 1);
end

for k = 1:N
    z_G_k = Z(k, :)';
    hr    = t_vec(k);
    lf    = 1.0 + 0.05 * sin(2*pi*(hr-8)/24);
    load_k = Nodes.BaseLoad * lf;

    if has_powersys
        % Predict
        [x_E_pred, L_holt, T_holt] = power_holt_smooth(x_E_k, L_holt, T_holt, alpha_s, beta_s);
        m_gtu = gtu_coupling(x_E_pred, PowerSys);
        for g = 1:length(m_gtu)
            load_k(m_gtu(g).gas_node_id) = m_gtu(g).m_dot;
        end

        [A_G, B_G, u_G] = dse_2_build_model(x_G_k, Nodes, Pipes, Compressors, Sys, load_k);
        x_G_pred = A_G \ (B_G * x_G_k + u_G);
        F_G      = A_G \ B_G;
        x_I_pred = [x_E_pred; x_G_pred];
        F_I      = blkdiag(alpha_s * eye(nE), F_G);
        P_I_pred = F_I * P_I * F_I' + Q_I;
        P_I_pred = (P_I_pred + P_I_pred') / 2;

        % Compute gas innovation for mu adaptation
        nu_G = z_G_k - x_G_pred;   % gas innovation (H_G = eye)
        innov_buf(buf_idx, :) = nu_G';
        buf_idx   = mod(buf_idx, m_w) + 1;
        buf_count = min(buf_count + 1, m_w);

        % Adapt mu for gas measurements only
        if buf_count >= m_w
            P_F_emp = zeros(nS);
            for i = 1:m_w
                e_i = innov_buf(i,:)';
                P_F_emp = P_F_emp + e_i * e_i';
            end
            P_F_emp = (P_F_emp + regularization*eye(nS)) / m_w;
            P_G_block = P_I_pred(nE+1:end, nE+1:end);
            P_F_theo  = P_G_block + R_G_base;
            diff_P    = P_F_emp - P_F_theo;
            for i = 1:nS
                mu_raw = diff_P(i,i) / R_G_base(i,i);
                mu_diag_gas(i) = max(mu_min, min(mu_max, mu_raw));
            end
        end

        % Build R_I with adaptive scaling on gas block only
        R_G_scaled = diag(mu_diag_gas) * R_G_base;
        R_I_k      = blkdiag(R_E, R_G_scaled);

        % Joint update
        if has_ze
            z_I_k = [Z_E(k,:)'; z_G_k];
        else
            z_I_k = [H_E * x_E_pred; z_G_k];
        end
        S_I = H_I * P_I_pred * H_I' + R_I_k;
        S_I = (S_I + S_I') / 2;
        K_I = P_I_pred * H_I' / S_I;
        x_I_update = x_I_pred + K_I * (z_I_k - H_I * x_I_pred);
        P_I = (eye(nI) - K_I * H_I) * P_I_pred;
        P_I = (P_I + P_I') / 2 + 1e-6 * eye(nI);

        x_E_k = x_I_update(1:nE);
        x_G_k = x_I_update(nE+1:end);
        x_G_k(1:nN) = max(x_G_k(1:nN), 0.5);

        X_E_est(k,:) = x_E_k';
        R_History(k,:)   = diag(R_G_base)';
        Dtt_History(k,:) = diag(R_G_scaled)';

    elseif has_gtu
        elf = 0.7 + 0.3*(sin(2*pi*(hr-6)/12).^2*(hr>=6 && hr<=22));
        for g = 1:height(GTU)
            if GTU.Status(g)~=1, continue; end
            cap=GTU.Capacity_MW(g); pmin=GTU.MinLoad(g)/100; pmax=GTU.MaxLoad(g)/100;
            pwr=cap*(pmin+(pmax-pmin)*elf);
            load_k(GTU.NodeID(g))=pwr*GTU.HeatRate(g)*1000/3600/LHV;
        end

        [A,B,u] = dse_2_build_model(x_G_k,Nodes,Pipes,Compressors,Sys,load_k);
        x_pred = A\(B*x_G_k+u);  F_k=A\B;
        P_pred = F_k*P_k*F_k'+Q;
        H=eye(nS); nu_k=z_G_k-H*x_pred;
        innov_buf(buf_idx,:)=nu_k'; buf_idx=mod(buf_idx,m_w)+1; buf_count=min(buf_count+1,m_w);
        if buf_count>=m_w
            P_F_emp=zeros(nS);
            for i=1:m_w, e_i=innov_buf(i,:)'; P_F_emp=P_F_emp+e_i*e_i'; end
            P_F_emp=(P_F_emp+regularization*eye(nS))/m_w;
            diff_P=P_F_emp-(P_pred+R);
            for i=1:nS, mu_diag(i)=max(mu_min,min(mu_max,diff_P(i,i)/R(i,i))); end
        end
        mu_R=diag(mu_diag)*R;
        S=H*P_pred*H'+mu_R; K=P_pred*H'/S;
        x_G_k=x_pred+K*nu_k;
        P_k=(eye(nS)-K*H)*P_pred; P_k=(P_k+P_k')/2+1e-4*eye(nS);
        x_G_k(1:nN)=max(x_G_k(1:nN),0.5);
        R_History(k,:)=diag(R)'; Dtt_History(k,:)=diag(mu_R)';
    end

    Hist_Est(k,:)  = x_G_k';
    Load_Used(k,:) = load_k';
end

for i = 1:nP
    Hist_Est(:,nN+i) = movmean(Hist_Est(:,nN+i),3);
end
end
