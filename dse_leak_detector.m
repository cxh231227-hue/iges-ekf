function [Det, Diag, Leak_Est_Nodes] = dse_leak_detector(H_Est, Z, Nodes, Pipes, Sys, t_vec)
% DSE_LEAK_DETECTOR - Three-layer leak detection algorithm
%
% Inputs:
%   H_Est   - Estimated states [N x nS]
%   Z       - Measurements [N x nS]
%   Nodes   - Node table
%   Pipes   - Pipe table
%   Sys     - System parameters
%   t_vec   - Time vector [hours]
%
% Outputs:
%   Det            - Detection flag vector [N x 1]
%   Diag           - Diagnostic structure
%   Leak_Est_Nodes - Leak estimation at nodes [N x nN]

N = length(t_vec);
nN = height(Nodes);
nP = height(Pipes);
nS = nN + nP;

% Compute residuals
Res = Z - H_Est;
Res_P = abs(Res(:, 1:nN)) * Sys.c2 / 1e5;   % [Bar]
Res_M = abs(Res(:, nN+1:end));               % [kg/s]
Res_all = [Res_P, Res_M];

% Algorithm parameters
alpha = 0.99;
c = 1.4826;
k_2 = 3.0;
k_3 = 6.0;
kappa = 1.0;
window = 25;
init_period = 50;
alpha_dk = 0.90;

% Initialize arrays
sigma_copy = zeros(N, nS);
d_k = zeros(N, nS);
detection_threshold = zeros(N, nS);
Out = false(N, nS);
Leak_mask = false(N, nS);
Outlier_mask = false(N, nS);

for i = 1:N
    % Step 1: Update sigma estimate
    if i <= init_period
        for m = 1:nS
            sigma_copy(i, m) = c * median(Res_all(1:init_period, m));
        end
    else
        start_idx = max(1, i - window);
        window_data = Res_all(start_idx:i, :);
        med_vals = median(window_data, 1);
        sigma_copy(i, :) = alpha * sigma_copy(i-1, :) + (1 - alpha) * c * med_vals;
    end
    
    % Apply sigma lower bounds
    sigma_copy(i, 1:nN) = max(sigma_copy(i, 1:nN), 0.1);
    sigma_copy(i, nN+1:end) = max(sigma_copy(i, nN+1:end), 1.5);
    
    % Step 2: Compute threshold
    detection_threshold(i, :) = k_3 * sigma_copy(i, :);
    
    % Step 3: Outlier detection
    Out(i, :) = Res_all(i, :) > detection_threshold(i, :);
    
    % Step 4: Three-layer classification
    if i <= init_period
        d_k(i, :) = 0;
        Leak_mask(i, :) = false;
        Outlier_mask(i, :) = false;
    else
        for m = 1:nS
            res_val = Res_all(i, m);
            sigma_val = sigma_copy(i, m);
            
            if res_val > k_3 * sigma_val
                Outlier_mask(i, m) = true;
                d_k(i, m) = d_k(i-1, m) * 0.9;
            elseif res_val > k_2 * sigma_val
                Leak_mask(i, m) = true;
                d_k(i, m) = alpha_dk * d_k(i-1, m) + (1 - alpha_dk) * kappa * res_val;
            else
                d_k(i, m) = d_k(i-1, m) * 0.95;
            end
        end
    end
    
    d_k(i, :) = max(d_k(i, :), 0);
end

% Output detection flag
Det = any(Leak_mask(:, 1:nN), 2);
Det(1:init_period) = false;

% Leak estimation
Leak_Est_Nodes = d_k(:, 1:nN) * 2.0;

% Diagnostic structure
Diag.Res_Pressure_Bar = Res_P;
Diag.Res_Flow = Res_M;
Diag.Res_all = Res_all;
Diag.sigma_copy = sigma_copy;
Diag.detection_threshold = detection_threshold;
Diag.Out = Out;
Diag.Leak_mask = Leak_mask;
Diag.Outlier_mask = Outlier_mask;
Diag.d_k = d_k;

end