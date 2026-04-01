function Z_outlier = inject_outlier_sustained(Z, config, Sys, magnitude_kgs)
% INJECT_OUTLIER_SUSTAINED - 注入持续的传感器异常
% 按照Dr. Andrew要求：直接使用绝对值(kg/s)，基于实际信号方差而非理论噪声
%
% Inputs:
%   Z              - 原始测量数据 [N x nS]
%   config.start_step  - 异常开始时间步
%   config.end_step    - 异常结束时间步  
%   config.node_id     - 目标节点
%   magnitude_kgs      - 异常幅度 (kg/s), 例如300或500
%
% Output:
%   Z_outlier      - 注入异常后的数据

Z_outlier = Z;

start_k = config.start_step;
end_k = config.end_step;
node_id = config.node_id;

% 计算注入时长
duration_steps = end_k - start_k + 1;
duration_hours = duration_steps * Sys.dt / 3600;

fprintf('\n持续异常注入详情:\n');
fprintf('  开始时间: t=%.1fh (step %d)\n', start_k * Sys.dt / 3600, start_k);
fprintf('  结束时间: t=%.1fh (step %d)\n', end_k * Sys.dt / 3600, end_k);
fprintf('  持续时间: %.1fh (%d steps)\n', duration_hours, duration_steps);
fprintf('  目标节点: Node %d\n', node_id);
fprintf('  异常幅度: %.0f kg/s (直接加到测量值上，模拟传感器故障)\n', magnitude_kgs);

% 计算baseline统计
baseline_data = Z(1:start_k-1, node_id);
baseline_mean = mean(baseline_data);
baseline_std = std(baseline_data);

fprintf('  Baseline统计 (0到%.1fh):\n', (start_k-1)*Sys.dt/3600);
fprintf('    Mean = %.2f\n', baseline_mean);
fprintf('    Std  = %.2f\n', baseline_std);
fprintf('    异常相当于 %.1f倍标准差\n', magnitude_kgs / baseline_std);

% 注入持续异常（直接加上固定偏差，模拟传感器bias）
for k = start_k:min(end_k, size(Z,1))
    Z_outlier(k, node_id) = Z(k, node_id) + magnitude_kgs;
end

fprintf('  ✓ 异常注入完成\n');

end