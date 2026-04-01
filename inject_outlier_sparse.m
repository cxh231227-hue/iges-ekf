function [Z_outlier, outlier_info] = inject_outlier_sparse(Z, config, Sys, magnitude_dB)
% INJECT_OUTLIER_SPARSE - Inject sparse sensor outliers
%
% dB definition: dB = 20 * log10(A_outlier / A_signal)
% Conversion:    A_outlier = A_signal * 10^(dB/20)
%
% Reference table:
%   10 dB -> 3.16x
%   20 dB -> 10x
%   30 dB -> 31.6x
%
% Inputs:
%   Z           - Original measurement data [N x nS]
%   config      - Configuration structure
%   Sys         - System parameters
%   magnitude_dB - Outlier magnitude in dB
%
% Outputs:
%   Z_outlier    - Data with injected outliers
%   outlier_info - Information about injected outliers

Z_outlier = Z;
[N, nS] = size(Z);

start_k = config.start_step;
end_k = min(config.end_step, N);

target_nodes = config.target_nodes;
target_pipes = config.target_pipes;
nN = config.nN;

% Convert dB to linear amplitude ratio: dB = 20*log10(A_out/A_sig)
linear_ratio = 10^(magnitude_dB / 20);

% Pre-compute fixed outlier magnitude per signal column: std(Z,1) * linear_ratio
%Z_std = std(Z, 1);
Z_std = mean(abs(Z), 1);  % 1 x nS row vector, std along time dimension

% Calculate spacing
steps_per_hour = 3600 / Sys.dt;
if isfield(config, 'outliers_per_hour') && config.outliers_per_hour > 0
    spacing = max(1, round(steps_per_hour / config.outliers_per_hour));
else
    spacing = 3;
end

node_outliers = [];
pipe_outliers = [];

available_steps = start_k:spacing:end_k;
n_available = length(available_steps);
n_nodes = length(target_nodes);

% Node outliers
for i = 1:n_available
    k = available_steps(i);
    for j = 1:n_nodes
        node_id = target_nodes(j);
        if node_id <= nN
            original_value = Z(k, node_id);
            % Fixed magnitude per node: std(Z,1) * linear_ratio
            outlier_magnitude = Z_std(node_id) * linear_ratio;
            sign_val = 2 * (rand() > 0.5) - 1;
            Z_outlier(k, node_id) = original_value + sign_val * outlier_magnitude;
            node_outliers = [node_outliers; k, node_id, Z_outlier(k,node_id), original_value, sign_val];
        end
    end
end

% Pipe outliers
n_pipes = length(target_pipes);
pipe_spacing = round(steps_per_hour);

for p_idx = 1:n_pipes
    pipe_id = target_pipes(p_idx);
    state_idx = nN + pipe_id;
    if state_idx > nS, continue; end
    
    % Fixed magnitude per pipe: std(Z,1) * linear_ratio
    outlier_magnitude = Z_std(state_idx) * linear_ratio;
    
    pipe_offset = round((p_idx - 1) * pipe_spacing / n_pipes);
    pipe_steps = (start_k + pipe_offset):pipe_spacing:end_k;
    
    for i = 1:length(pipe_steps)
        k = pipe_steps(i);
        if k > end_k, break; end
        
        original_value = Z(k, state_idx);
        sign_val = 2 * (rand() > 0.5) - 1;
        Z_outlier(k, state_idx) = original_value + sign_val * outlier_magnitude;
        pipe_outliers = [pipe_outliers; k, pipe_id, Z_outlier(k,state_idx), original_value, sign_val];
    end
end

% Output structure
outlier_info.node_outliers = node_outliers;
outlier_info.pipe_outliers = pipe_outliers;
outlier_info.start_step = start_k;
outlier_info.end_step = end_k;
outlier_info.spacing = spacing;
outlier_info.target_nodes = target_nodes;
outlier_info.target_pipes = target_pipes;
outlier_info.nN = nN;
outlier_info.magnitude_dB = magnitude_dB;
outlier_info.linear_ratio = linear_ratio;

end