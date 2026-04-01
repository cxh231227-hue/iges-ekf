function [x_E_pred, L_new, T_new] = power_holt_smooth(x_E, L, T, alpha_s, beta_s)
% POWER_HOLT_SMOOTH - Holt double exponential smoothing for power state prediction
% Implements Chen (2022) Eq 1-4:
%
%   L_t   = alpha_s * x_E_t + (1 - alpha_s) * (L_{t-1} + T_{t-1})   (Eq 1)
%   T_t   = beta_s  * (L_t - L_{t-1}) + (1 - beta_s) * T_{t-1}      (Eq 2)
%   x_E_{t+1} = L_t + T_t                                             (Eq 3)
%
% Which in matrix form is:
%   x_E_{t+1} = alpha_s * x_E_t + u_E_{t+1}                          (Eq 4)
%   where u_E_{t+1} = (1-alpha_s)*(L_{t-1}+T_{t-1}) + T_t
%
% Inputs:
%   x_E     - current power state estimate [e1,f1,...,e39,f39] (78x1)
%   L       - current level vector (78x1)
%   T       - current trend vector (78x1)
%   alpha_s - level smoothing parameter (0 < alpha_s < 1), Chen uses 0.5
%   beta_s  - trend smoothing parameter (0 < beta_s < 1), Chen uses 0.4
%
% Outputs:
%   x_E_pred - predicted power state at t+1 (78x1)
%   L_new    - updated level (78x1), carry to next step
%   T_new    - updated trend (78x1), carry to next step

% Update level (Eq 1)
L_new = alpha_s * x_E + (1 - alpha_s) * (L + T);

% Update trend (Eq 2)
T_new = beta_s * (L_new - L) + (1 - beta_s) * T;

% Predict next state (Eq 3)
x_E_pred = L_new + T_new;

end
