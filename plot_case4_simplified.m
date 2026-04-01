function plot_case3_simplified(H_True, Z, H_Normal, H_Chen, H_Adaptive, Nodes, Pipes, Sys, t, GTU, Leaks, outlier_config, outlier_info)
% PLOT_CASE3_SIMPLIFIED - Plot Case 3 results (Outlier + Leak)
% Outputs 5 figures: Pressure, Flow Pipe 1, Pipe 5, Pipe 6, Pipe 9

nN = height(Nodes);
nP = height(Pipes);

tgt_node = 6;
pipe1_idx = 1;
pipe5_idx = 5;
pipe6_idx = 6;
pipe9_idx = 9;

has_outlier_info = nargin >= 13 && ~isempty(outlier_info) && isfield(outlier_info, 'node_outliers');

if isfield(outlier_config, 'start_step') && isfield(outlier_config, 'end_step')
    outlier_start_time = outlier_config.start_step * Sys.dt / 3600;
    outlier_end_time = outlier_config.end_step * Sys.dt / 3600;
else
    outlier_start_time = 6;
    outlier_end_time = 12;
end

if height(Leaks) > 0
    leak_start_time = Leaks.StartTime_s(1) / 3600;
    leak_end_time = Leaks.EndTime_s(1) / 3600;
    if isinf(leak_end_time), leak_end_time = 24; end
else
    leak_start_time = NaN;
    leak_end_time = NaN;
end

% =========================================================================
% Figure 1: Pressure
% =========================================================================
figure('Name', 'Case 3: Pressure', 'Color', 'w', 'Position', [100 100 1200 700]);
hold on; box on;

Z_pressure = Z(:,tgt_node)*Sys.c2/1e5;
y_data = [H_True(:,tgt_node)*Sys.c2/1e5; H_Normal(:,tgt_node)*Sys.c2/1e5; 
          H_Chen(:,tgt_node)*Sys.c2/1e5; H_Adaptive(:,tgt_node)*Sys.c2/1e5];
y_min = min(y_data) * 0.98; y_max = max(y_data) * 1.02;
y_range = y_max - y_min;
y_lim = [y_min - 0.05*y_range, y_max + 0.20*y_range];

fill([outlier_start_time, outlier_end_time, outlier_end_time, outlier_start_time], ...
     [y_lim(1), y_lim(1), y_max, y_max], [1 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', 'Outlier period');
if ~isnan(leak_start_time)
    fill([leak_start_time, leak_end_time, leak_end_time, leak_start_time], ...
         [y_lim(1), y_lim(1), y_max, y_max], [0.9 0.9 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', 'Leak period');
end

plot(t, H_True(:,tgt_node)*Sys.c2/1e5, 'g-', 'LineWidth', 2.5, 'DisplayName', 'True');
Z_normal_mask = (Z_pressure >= y_lim(1)) & (Z_pressure <= y_max);
plot(t(Z_normal_mask), Z_pressure(Z_normal_mask), 'k.', 'MarkerSize', 6, 'DisplayName', 'Measurements');
plot(t, H_Normal(:,tgt_node)*Sys.c2/1e5, 'b:', 'LineWidth', 2, 'DisplayName', 'Standard EKF');
plot(t, H_Chen(:,tgt_node)*Sys.c2/1e5, 'm-.', 'LineWidth', 1.8, 'DisplayName', 'Chen Robust EKF');
plot(t, H_Adaptive(:,tgt_node)*Sys.c2/1e5, 'r--', 'LineWidth', 1.8, 'DisplayName', 'EKF-LE');

if has_outlier_info && ~isempty(outlier_info.node_outliers)
    node_outliers = outlier_info.node_outliers(outlier_info.node_outliers(:,2) == tgt_node, :);
    if ~isempty(node_outliers)
        outlier_steps = node_outliers(:, 1);
        outlier_times = outlier_steps * Sys.dt / 3600;
        marker_y = y_max + 0.10*y_range;
        for i = 1:length(outlier_times)
            plot(outlier_times(i), marker_y, 'o', 'MarkerSize', 16, ...
                'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', 'LineWidth', 2.5, 'HandleVisibility', 'off');
            plot([outlier_times(i), outlier_times(i)], [marker_y-0.02*y_range, y_max], ...
                'r:', 'LineWidth', 1.5, 'HandleVisibility', 'off');
            val_to_show = node_outliers(i, 3) * Sys.c2 / 1e5;
            text(outlier_times(i), marker_y + 0.04*y_range, sprintf('%.0f', val_to_show), ...
                'FontSize', 12, 'Color', 'r', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        end
        plot(NaN, NaN, 'o', 'MarkerSize', 16, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', 'LineWidth', 2.5, ...
            'DisplayName', sprintf('Outliers (n=%d)', length(outlier_times)));
    end
end

xline(outlier_start_time, 'c--', 'LineWidth', 1.5, 'Alpha', 0.6, 'HandleVisibility', 'off');
if ~isnan(leak_start_time)
    xline(leak_start_time, 'k:', 'LineWidth', 2, 'Alpha', 0.5, 'HandleVisibility', 'off');
end

title(sprintf('Case 3: Pressure at Node %d (Outlier + Leak)', tgt_node), 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Time (h)', 'FontSize', 12); ylabel('Pressure (Bar)', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 9); grid on;
xlim([0 24]); ylim(y_lim); set(gca, 'FontSize', 11);
savefig('Case3_Pressure.fig');
fprintf('  Saved: Case3_Pressure.fig\n');

% =========================================================================
% Helper: flow plot for Case 3
% =========================================================================
    function plot_flow_case3(pipe_idx, fig_name, fig_title, fig_pos, save_name, show_outlier_markers)
        figure('Name', fig_name, 'Color', 'w', 'Position', fig_pos);
        hold on; box on;
        
        Z_flow = Z(:,nN+pipe_idx);
        y_data_f = [H_True(:,nN+pipe_idx); H_Normal(:,nN+pipe_idx);
                    H_Chen(:,nN+pipe_idx); H_Adaptive(:,nN+pipe_idx)];
        y_min_f = min(y_data_f) * 0.98; y_max_f = max(y_data_f) * 1.02;
        y_range_f = y_max_f - y_min_f;
        y_lim_f = [y_min_f - 0.05*y_range_f, y_max_f + 0.20*y_range_f];
        
        fill([outlier_start_time, outlier_end_time, outlier_end_time, outlier_start_time], ...
             [y_lim_f(1), y_lim_f(1), y_max_f, y_max_f], [1 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', 'Outlier period');
        if ~isnan(leak_start_time)
            fill([leak_start_time, leak_end_time, leak_end_time, leak_start_time], ...
                 [y_lim_f(1), y_lim_f(1), y_max_f, y_max_f], [0.9 0.9 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', 'Leak period');
        end
        
        plot(t, H_True(:,nN+pipe_idx), 'g-', 'LineWidth', 2.5, 'DisplayName', 'True');
        Z_normal_mask_f = (Z_flow >= y_lim_f(1)) & (Z_flow <= y_max_f);
        plot(t(Z_normal_mask_f), Z_flow(Z_normal_mask_f), 'k.', 'MarkerSize', 6, 'DisplayName', 'Measurements');
        plot(t, H_Normal(:,nN+pipe_idx), 'b:', 'LineWidth', 2, 'DisplayName', 'Standard EKF');
        plot(t, H_Chen(:,nN+pipe_idx), 'm-.', 'LineWidth', 1.8, 'DisplayName', 'Chen Robust EKF');
        plot(t, H_Adaptive(:,nN+pipe_idx), 'r--', 'LineWidth', 1.8, 'DisplayName', 'EKF-LE');
        
        % Outlier markers
        if show_outlier_markers && has_outlier_info && ~isempty(outlier_info.pipe_outliers)
            pipe_outliers = outlier_info.pipe_outliers(outlier_info.pipe_outliers(:,2) == pipe_idx, :);
            if ~isempty(pipe_outliers)
                outlier_steps_p = pipe_outliers(:, 1);
                outlier_times_p = outlier_steps_p * Sys.dt / 3600;
                marker_y_p = y_max_f + 0.10*y_range_f;
                for i = 1:length(outlier_times_p)
                    plot(outlier_times_p(i), marker_y_p, 'o', 'MarkerSize', 16, ...
                        'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', 'LineWidth', 2.5, 'HandleVisibility', 'off');
                    plot([outlier_times_p(i), outlier_times_p(i)], [marker_y_p-0.02*y_range_f, y_max_f], ...
                        'r:', 'LineWidth', 1.5, 'HandleVisibility', 'off');
                    val_to_show = pipe_outliers(i, 3);
                    text(outlier_times_p(i), marker_y_p + 0.04*y_range_f, sprintf('%.0f', val_to_show), ...
                        'FontSize', 12, 'Color', 'r', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
                end
                plot(NaN, NaN, 'o', 'MarkerSize', 16, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', 'LineWidth', 2.5, ...
                    'DisplayName', sprintf('Outliers (n=%d)', length(outlier_times_p)));
            end
        end
        
        xline(outlier_start_time, 'c--', 'LineWidth', 1.5, 'Alpha', 0.6, 'HandleVisibility', 'off');
        if ~isnan(leak_start_time)
            xline(leak_start_time, 'k:', 'LineWidth', 2, 'Alpha', 0.5, 'HandleVisibility', 'off');
            xline(leak_end_time, 'k:', 'LineWidth', 2, 'Alpha', 0.5, 'HandleVisibility', 'off');
        end
        
        title(fig_title, 'FontSize', 13, 'FontWeight', 'bold');
        xlabel('Time (h)', 'FontSize', 12); ylabel('Flow Rate (kg/s)', 'FontSize', 12);
        legend('Location', 'best', 'FontSize', 9); grid on;
        xlim([0 24]); ylim(y_lim_f); set(gca, 'FontSize', 11);
        savefig(save_name);
        fprintf('  Saved: %s\n', save_name);
    end

% Figure 2: Flow Pipe 1 (with outlier markers)
plot_flow_case3(pipe1_idx, 'Case 3: Flow Pipe 1', ...
    'Case 3: Source Flow (Pipe 1, Outlier + Leak)', ...
    [150 150 1200 700], 'Case3_Flow_Pipe1.fig', true);

% Figure 3: Flow Pipe 5 (no outlier injection on this pipe)
plot_flow_case3(pipe5_idx, 'Case 3: Flow Pipe 5', ...
    'Case 3: Flow at Pipe 5 (No Outlier Injection)', ...
    [200 200 1200 700], 'Case3_Flow_Pipe5.fig', false);

% Figure 4: Flow Pipe 6 (Compressor + GTU-1)
plot_flow_case3(pipe6_idx, 'Case 3: Flow Pipe 6', ...
    'Case 3: Flow at Pipe 6 (Compressor + GTU-1, Outlier + Leak)', ...
    [250 250 1200 700], 'Case3_Flow_Pipe6.fig', true);

% Figure 5: Flow Pipe 9 (GTU-2 feed)
plot_flow_case3(pipe9_idx, 'Case 3: Flow Pipe 9', ...
    'Case 3: Flow at Pipe 9 (GTU-2 Feed, Outlier + Leak)', ...
    [300 300 1200 700], 'Case3_Flow_Pipe9.fig', true);

end