function plot_outlier_pipes(Z_clean, Z_outlier, outlier_info, Nodes, Sys, t, figdir)
% PLOT_OUTLIER_PIPES - Plot outlier injection for all target pipes
%
% Shows clean signal vs outlier-injected signal for every pipe in
% outlier_info.target_pipes, with red markers at injection points.
%
% Inputs:
%   Z_clean      - Original clean measurement data [N x nS]
%   Z_outlier    - Data with injected outliers     [N x nS]
%   outlier_info - Structure returned by inject_outlier_sparse
%   Nodes        - Node table
%   Sys          - System parameters (needs Sys.dt)
%   t            - Time vector (hours) [N x 1]
%   figdir       - Directory to save .fig files (optional)

if nargin < 7 || isempty(figdir)
    figdir = '';
end

nN            = outlier_info.nN;
target_pipes  = outlier_info.target_pipes;
pipe_outliers = outlier_info.pipe_outliers;   % [k, pipe_id, Z_val, orig_val, sign_val]
n_pipes       = length(target_pipes);

% Outlier window in hours
start_h = (outlier_info.start_step - 1) * Sys.dt / 3600;
end_h   = (outlier_info.end_step   - 1) * Sys.dt / 3600;

% ----------------------------------------------------------------
% One figure per injected pipe
% ----------------------------------------------------------------
for p_idx = 1:n_pipes
    pipe_id   = target_pipes(p_idx);
    state_idx = nN + pipe_id;

    % Time of outlier events for this pipe
    if ~isempty(pipe_outliers)
        rows     = pipe_outliers(pipe_outliers(:,2) == pipe_id, :);
        out_k    = rows(:, 1);                    % step indices
        out_t    = (out_k - 1) * Sys.dt / 3600;  % hours
        out_Z    = rows(:, 3);                    % injected values
    else
        out_t = []; out_Z = [];
    end

    fig = figure('Name', sprintf('Outlier Injection - Pipe %d', pipe_id), ...
                 'Color', 'w', 'Position', [100 + p_idx*30, 100 + p_idx*30, 950, 480]);
    hold on; box on;

    % Clean signal
    plot(t, Z_clean(:, state_idx), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Clean Signal');

    % Outlier-injected signal
    plot(t, Z_outlier(:, state_idx), 'k-', 'LineWidth', 1.0, 'DisplayName', 'With Outliers');

    % Mark outlier injection points
    if ~isempty(out_t)
        plot(out_t, out_Z, 'rv', 'MarkerSize', 8, 'MarkerFaceColor', 'r', ...
             'DisplayName', 'Injected Outlier');
    end

    % Shade the outlier injection window
    yl = ylim;
    patch([start_h end_h end_h start_h], [yl(1) yl(1) yl(2) yl(2)], ...
          [1 0.85 0.85], 'FaceAlpha', 0.25, 'EdgeColor', 'none', ...
          'DisplayName', 'Outlier Window');
    ylim(yl);

    % Vertical boundary lines
    xline(start_h, 'r--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    xline(end_h,   'r--', 'LineWidth', 1.2, 'HandleVisibility', 'off');

    % Labels
    title(sprintf('Pipe %d  —  Outlier Injection  (%.0f dB,  ratio=%.2f)', ...
          pipe_id, outlier_info.magnitude_dB, outlier_info.linear_ratio), ...
          'FontSize', 13, 'FontWeight', 'bold');
    xlabel('Time (h)', 'FontSize', 12);
    ylabel('Flow Rate (kg/s)', 'FontSize', 12);
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    xlim([0, max(t)]);
    set(gca, 'FontSize', 11);

    % Save
    if ~isempty(figdir)
        fname = fullfile(figdir, sprintf('Outlier_Pipe%d.fig', pipe_id));
        savefig(fig, fname);
        fprintf('  Saved: %s\n', fname);
    end
end

% ----------------------------------------------------------------
% Summary table printed to command window
% ----------------------------------------------------------------
fprintf('\n========================================================\n');
fprintf('  Outlier Injection Summary  (%.0f dB  |  ratio = %.2fx)\n', ...
        outlier_info.magnitude_dB, outlier_info.linear_ratio);
fprintf('  Injection window: %.1f h  –  %.1f h\n', start_h, end_h);
fprintf('========================================================\n');
fprintf('%-8s  %-10s  %-12s  %-12s  %-12s\n', ...
        'Pipe', 'N_outliers', 'Std_Signal', 'Magnitude', 'Max_Dev%');
fprintf('%s\n', repmat('-', 1, 60));

for p_idx = 1:n_pipes
    pipe_id   = target_pipes(p_idx);
    state_idx = nN + pipe_id;

    sig_std   = std(Z_clean(:, state_idx), 1);
    magnitude = sig_std * outlier_info.linear_ratio;

    if ~isempty(pipe_outliers)
        rows = pipe_outliers(pipe_outliers(:,2) == pipe_id, :);
        n_out = size(rows, 1);
        if n_out > 0 && mean(abs(rows(:,4))) > 0
            max_dev_pct = max(abs(rows(:,3) - rows(:,4)) ./ abs(rows(:,4))) * 100;
        else
            max_dev_pct = NaN;
        end
    else
        n_out = 0; max_dev_pct = NaN;
    end

    fprintf('%-8d  %-10d  %-12.4f  %-12.4f  %-12.1f\n', ...
            pipe_id, n_out, sig_std, magnitude, max_dev_pct);
end
fprintf('========================================================\n\n');

end
