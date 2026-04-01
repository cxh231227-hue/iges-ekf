function plot_case1(H_True, Z, H_Normal, H_Chen, H_Adaptive, Nodes, Pipes, Sys, t, GTU)
% PLOT_CASE1 - Plot Case 1 results (Clean data, no outliers, no leaks)
% Creates 5 figures: Pressure, Flow Pipe 1, Pipe 5, Pipe 6, Pipe 9

nN = height(Nodes);
nP = height(Pipes);

tgt_node = 6;
pipe1_idx = 1;
pipe5_idx = 5;
pipe6_idx = 6;
pipe9_idx = 9;

% Figure 1: Pressure Comparison
figure('Name', 'Case 1: Pressure - Clean Data', 'Color', 'w', 'Position', [100 100 1000 600]);
hold on; box on;

plot(t, H_True(:,tgt_node)*Sys.c2/1e5, 'g-', 'LineWidth', 2.5, 'DisplayName', 'True');
plot(t, Z(:,tgt_node)*Sys.c2/1e5, 'k.', 'MarkerSize', 4, 'DisplayName', 'Measurements');
plot(t, H_Normal(:,tgt_node)*Sys.c2/1e5, 'b:', 'LineWidth', 2, 'DisplayName', 'Standard EKF');
plot(t, H_Chen(:,tgt_node)*Sys.c2/1e5, 'm-.', 'LineWidth', 1.8, 'DisplayName', 'Chen Robust EKF');
plot(t, H_Adaptive(:,tgt_node)*Sys.c2/1e5, 'r--', 'LineWidth', 1.8, 'DisplayName', 'EKF-LE');

title(sprintf('Case 1: Pressure at Node %d (Clean Data)', tgt_node), 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (h)', 'FontSize', 12);
ylabel('Pressure (Bar)', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0 24]);
set(gca, 'FontSize', 11);

savefig('Case1_Pressure.fig');
fprintf('  Saved: Case1_Pressure.fig\n');

% Figure 2: Flow Comparison at Pipe 1
figure('Name', 'Case 1: Flow Pipe 1 - Clean Data', 'Color', 'w', 'Position', [150 150 1000 600]);
hold on; box on;

plot(t, H_True(:,nN+pipe1_idx), 'g-', 'LineWidth', 2.5, 'DisplayName', 'True');
plot(t, Z(:,nN+pipe1_idx), 'k.', 'MarkerSize', 4, 'DisplayName', 'Measurements');
plot(t, H_Normal(:,nN+pipe1_idx), 'b:', 'LineWidth', 2, 'DisplayName', 'Standard EKF');
plot(t, H_Chen(:,nN+pipe1_idx), 'm-.', 'LineWidth', 1.8, 'DisplayName', 'Chen Robust EKF');
plot(t, H_Adaptive(:,nN+pipe1_idx), 'r--', 'LineWidth', 1.8, 'DisplayName', 'EKF-LE');

title('Case 1: Source Flow (Pipe 1, Clean Data)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (h)', 'FontSize', 12);
ylabel('Flow Rate (kg/s)', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0 24]);
set(gca, 'FontSize', 11);

savefig('Case1_Flow_Pipe1.fig');
fprintf('  Saved: Case1_Flow_Pipe1.fig\n');

% Figure 3: Flow Comparison at Pipe 5
figure('Name', 'Case 1: Flow Pipe 5 - Clean Data', 'Color', 'w', 'Position', [200 200 1000 600]);
hold on; box on;

plot(t, H_True(:,nN+pipe5_idx), 'g-', 'LineWidth', 2.5, 'DisplayName', 'True');
plot(t, Z(:,nN+pipe5_idx), 'k.', 'MarkerSize', 4, 'DisplayName', 'Measurements');
plot(t, H_Normal(:,nN+pipe5_idx), 'b:', 'LineWidth', 2, 'DisplayName', 'Standard EKF');
plot(t, H_Chen(:,nN+pipe5_idx), 'm-.', 'LineWidth', 1.8, 'DisplayName', 'Chen Robust EKF');
plot(t, H_Adaptive(:,nN+pipe5_idx), 'r--', 'LineWidth', 1.8, 'DisplayName', 'EKF-LE');

title('Case 1: Flow at Pipe 5 (Clean Data)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (h)', 'FontSize', 12);
ylabel('Flow Rate (kg/s)', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0 24]);
set(gca, 'FontSize', 11);

savefig('Case1_Flow_Pipe5.fig');
fprintf('  Saved: Case1_Flow_Pipe5.fig\n');

% Figure 4: Flow Comparison at Pipe 6 (Compressor + GTU-1)
figure('Name', 'Case 1: Flow Pipe 6 - Clean Data', 'Color', 'w', 'Position', [250 250 1000 600]);
hold on; box on;

plot(t, H_True(:,nN+pipe6_idx), 'g-', 'LineWidth', 2.5, 'DisplayName', 'True');
plot(t, Z(:,nN+pipe6_idx), 'k.', 'MarkerSize', 4, 'DisplayName', 'Measurements');
plot(t, H_Normal(:,nN+pipe6_idx), 'b:', 'LineWidth', 2, 'DisplayName', 'Standard EKF');
plot(t, H_Chen(:,nN+pipe6_idx), 'm-.', 'LineWidth', 1.8, 'DisplayName', 'Chen Robust EKF');
plot(t, H_Adaptive(:,nN+pipe6_idx), 'r--', 'LineWidth', 1.8, 'DisplayName', 'EKF-LE');

title('Case 1: Flow at Pipe 6 (Compressor + GTU-1, Clean Data)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (h)', 'FontSize', 12);
ylabel('Flow Rate (kg/s)', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0 24]);
set(gca, 'FontSize', 11);

savefig('Case1_Flow_Pipe6.fig');
fprintf('  Saved: Case1_Flow_Pipe6.fig\n');

% Figure 5: Flow Comparison at Pipe 9 (GTU-2 feed)
figure('Name', 'Case 1: Flow Pipe 9 - Clean Data', 'Color', 'w', 'Position', [300 300 1000 600]);
hold on; box on;

plot(t, H_True(:,nN+pipe9_idx), 'g-', 'LineWidth', 2.5, 'DisplayName', 'True');
plot(t, Z(:,nN+pipe9_idx), 'k.', 'MarkerSize', 4, 'DisplayName', 'Measurements');
plot(t, H_Normal(:,nN+pipe9_idx), 'b:', 'LineWidth', 2, 'DisplayName', 'Standard EKF');
plot(t, H_Chen(:,nN+pipe9_idx), 'm-.', 'LineWidth', 1.8, 'DisplayName', 'Chen Robust EKF');
plot(t, H_Adaptive(:,nN+pipe9_idx), 'r--', 'LineWidth', 1.8, 'DisplayName', 'EKF-LE');

title('Case 1: Flow at Pipe 9 (GTU-2 Feed, Clean Data)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (h)', 'FontSize', 12);
ylabel('Flow Rate (kg/s)', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0 24]);
set(gca, 'FontSize', 11);

savefig('Case1_Flow_Pipe9.fig');
fprintf('  Saved: Case1_Flow_Pipe9.fig\n');

end