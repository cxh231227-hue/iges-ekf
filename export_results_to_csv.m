function export_results_to_csv(Stats_c1, Stats_c2, Stats_c3, output_dir)
% EXPORT_RESULTS_TO_CSV - Export detailed 4-method comparison to CSV
% 输出控制台显示的完整分段RMSE（Normal vs Leak Period）
% Case 4 已禁用；20dB 案例已移除；Pipe 6/9 已从结果中删除

if nargin < 4
    output_dir = './';
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% ===== Table 1: Case1 和 Case2 完整结果 (Pipe1-10, 不含Pipe6/9) =====
T1 = table();
T1.Case = {'Case1_Clean'; 'Case2_20dB'};

stats_c12 = {Stats_c1, Stats_c2};
methods = {'Normal','Chen','Adaptive','AFEKF'};
method_names = {'Standard','Chen','EKF_LE','AFEKF'};

for pp = [1,2,3,4,5,7,8,10]
    for mi = 1:4
        col_name = sprintf('Pipe%d_%s', pp, method_names{mi});
        vals = zeros(2,1);
        for ci = 1:2
            vals(ci) = stats_c12{ci}.(methods{mi}).(sprintf('Pipe%d_RMSE', pp));
        end
        T1.(col_name) = vals;
    end
end

% Flow RMSE
for mi = 1:4
    col_name = sprintf('Flow_%s', method_names{mi});
    vals = zeros(2,1);
    for ci = 1:2
        vals(ci) = stats_c12{ci}.(methods{mi}).M_RMSE;
    end
    T1.(col_name) = vals;
end
% Original filename
originalFilename = 'Case1_Case2_Full_Results.csv';
% Get the current date and time
currentDateTime = datetime('now');
% Format the timestamp
timestampStr = datetime(currentDateTime,'Format','d-MMM-y-HH-mm-ss');
% Create new filename by appending the timestamp
[filepath, name, ext] = fileparts(originalFilename); % Split into components
newFilename = fullfile(filepath, sprintf('%s_%s%s', name, timestampStr, ext));
output_dir1=[output_dir 'C1/'];
if ~exist(output_dir1, 'dir')
    mkdir(output_dir1);
end
writetable(T1, fullfile(output_dir1, newFilename));

%% ===== Table 2: Case3 完整分段结果（控制台格式）=====
T2 = table();
T2.Period = {
    'Full_Period';
    'Normal_0to6h_18to24h';
    'Leak_12to18h'
};

% Pressure RMSE
T2.P_RMSE_Standard = [
    Stats_c3.Normal.P_RMSE;
    get_field_safe(Stats_c3.Normal, 'P_RMSE_Normal');
    get_field_safe(Stats_c3.Normal, 'P_RMSE_Leak')
];
T2.P_RMSE_Chen = [
    Stats_c3.Chen.P_RMSE;
    get_field_safe(Stats_c3.Chen, 'P_RMSE_Normal');
    get_field_safe(Stats_c3.Chen, 'P_RMSE_Leak')
];
T2.P_RMSE_EKF_LE = [
    Stats_c3.Adaptive.P_RMSE;
    get_field_safe(Stats_c3.Adaptive, 'P_RMSE_Normal');
    get_field_safe(Stats_c3.Adaptive, 'P_RMSE_Leak')
];
T2.P_RMSE_AFEKF = [
    Stats_c3.AFEKF.P_RMSE;
    get_field_safe(Stats_c3.AFEKF, 'P_RMSE_Normal');
    get_field_safe(Stats_c3.AFEKF, 'P_RMSE_Leak')
];

% Flow RMSE
T2.M_RMSE_Standard = [
    Stats_c3.Normal.M_RMSE;
    get_field_safe(Stats_c3.Normal, 'M_RMSE_Normal');
    get_field_safe(Stats_c3.Normal, 'M_RMSE_Leak')
];
T2.M_RMSE_Chen = [
    Stats_c3.Chen.M_RMSE;
    get_field_safe(Stats_c3.Chen, 'M_RMSE_Normal');
    get_field_safe(Stats_c3.Chen, 'M_RMSE_Leak')
];
T2.M_RMSE_EKF_LE = [
    Stats_c3.Adaptive.M_RMSE;
    get_field_safe(Stats_c3.Adaptive, 'M_RMSE_Normal');
    get_field_safe(Stats_c3.Adaptive, 'M_RMSE_Leak')
];
T2.M_RMSE_AFEKF = [
    Stats_c3.AFEKF.M_RMSE;
    get_field_safe(Stats_c3.AFEKF, 'M_RMSE_Normal');
    get_field_safe(Stats_c3.AFEKF, 'M_RMSE_Leak')
];

% Pipe1 RMSE
T2.Pipe1_Standard = [
    Stats_c3.Normal.Pipe1_RMSE;
    get_field_safe(Stats_c3.Normal, 'Pipe1_RMSE_Normal');
    get_field_safe(Stats_c3.Normal, 'Pipe1_RMSE_Leak')
];
T2.Pipe1_Chen = [
    Stats_c3.Chen.Pipe1_RMSE;
    get_field_safe(Stats_c3.Chen, 'Pipe1_RMSE_Normal');
    get_field_safe(Stats_c3.Chen, 'Pipe1_RMSE_Leak')
];
T2.Pipe1_EKF_LE = [
    Stats_c3.Adaptive.Pipe1_RMSE;
    get_field_safe(Stats_c3.Adaptive, 'Pipe1_RMSE_Normal');
    get_field_safe(Stats_c3.Adaptive, 'Pipe1_RMSE_Leak')
];
T2.Pipe1_AFEKF = [
    Stats_c3.AFEKF.Pipe1_RMSE;
    get_field_safe(Stats_c3.AFEKF, 'Pipe1_RMSE_Normal');
    get_field_safe(Stats_c3.AFEKF, 'Pipe1_RMSE_Leak')
];

% Pipe5 RMSE
T2.Pipe5_Standard = [
    Stats_c3.Normal.Pipe5_RMSE;
    get_field_safe(Stats_c3.Normal, 'Pipe5_RMSE_Normal');
    get_field_safe(Stats_c3.Normal, 'Pipe5_RMSE_Leak')
];
T2.Pipe5_Chen = [
    Stats_c3.Chen.Pipe5_RMSE;
    get_field_safe(Stats_c3.Chen, 'Pipe5_RMSE_Normal');
    get_field_safe(Stats_c3.Chen, 'Pipe5_RMSE_Leak')
];
T2.Pipe5_EKF_LE = [
    Stats_c3.Adaptive.Pipe5_RMSE;
    get_field_safe(Stats_c3.Adaptive, 'Pipe5_RMSE_Normal');
    get_field_safe(Stats_c3.Adaptive, 'Pipe5_RMSE_Leak')
];
T2.Pipe5_AFEKF = [
    Stats_c3.AFEKF.Pipe5_RMSE;
    get_field_safe(Stats_c3.AFEKF, 'Pipe5_RMSE_Normal');
    get_field_safe(Stats_c3.AFEKF, 'Pipe5_RMSE_Leak')
];
% Original filename
originalFilename = 'Case3_Outlier_Leak_Full_Results.csv';
% Get the current date and time
currentDateTime = datetime('now');
% Format the timestamp
timestampStr = datetime(currentDateTime,'Format','d-MMM-y-HH-mm-ss');
% Create new filename by appending the timestamp
[filepath, name, ext] = fileparts(originalFilename); % Split into components
newFilename = fullfile(filepath, sprintf('%s_%s%s', name, timestampStr, ext));
output_dir3=[output_dir 'C3/'];
if ~exist(output_dir3, 'dir')
    mkdir(output_dir3);
end
writetable(T2, fullfile(output_dir3, newFilename));

%% ===== Table 3: Case4 (DISABLED) =====
% T3 = table();
% T3.Period = {
%     'Full_Period';
%     'Normal_0to12h';
%     'Leak_12to24h'
% };
% 
% % Pressure RMSE
% T3.P_RMSE_Standard = [
%     Stats_c4.Normal.P_RMSE;
%     get_field_safe(Stats_c4.Normal, 'P_RMSE_Normal');
%     get_field_safe(Stats_c4.Normal, 'P_RMSE_Leak')
% ];
% T3.P_RMSE_Chen = [
%     Stats_c4.Chen.P_RMSE;
%     get_field_safe(Stats_c4.Chen, 'P_RMSE_Normal');
%     get_field_safe(Stats_c4.Chen, 'P_RMSE_Leak')
% ];
% T3.P_RMSE_EKF_LE = [
%     Stats_c4.Adaptive.P_RMSE;
%     get_field_safe(Stats_c4.Adaptive, 'P_RMSE_Normal');
%     get_field_safe(Stats_c4.Adaptive, 'P_RMSE_Leak')
% ];
% T3.P_RMSE_AFEKF = [
%     Stats_c4.AFEKF.P_RMSE;
%     get_field_safe(Stats_c4.AFEKF, 'P_RMSE_Normal');
%     get_field_safe(Stats_c4.AFEKF, 'P_RMSE_Leak')
% ];
% 
% % Flow RMSE
% T3.M_RMSE_Standard = [
%     Stats_c4.Normal.M_RMSE;
%     get_field_safe(Stats_c4.Normal, 'M_RMSE_Normal');
%     get_field_safe(Stats_c4.Normal, 'M_RMSE_Leak')
% ];
% T3.M_RMSE_Chen = [
%     Stats_c4.Chen.M_RMSE;
%     get_field_safe(Stats_c4.Chen, 'M_RMSE_Normal');
%     get_field_safe(Stats_c4.Chen, 'M_RMSE_Leak')
% ];
% T3.M_RMSE_EKF_LE = [
%     Stats_c4.Adaptive.M_RMSE;
%     get_field_safe(Stats_c4.Adaptive, 'M_RMSE_Normal');
%     get_field_safe(Stats_c4.Adaptive, 'M_RMSE_Leak')
% ];
% T3.M_RMSE_AFEKF = [
%     Stats_c4.AFEKF.M_RMSE;
%     get_field_safe(Stats_c4.AFEKF, 'M_RMSE_Normal');
%     get_field_safe(Stats_c4.AFEKF, 'M_RMSE_Leak')
% ];
% 
% % Pipe1 RMSE
% T3.Pipe1_Standard = [
%     Stats_c4.Normal.Pipe1_RMSE;
%     get_field_safe(Stats_c4.Normal, 'Pipe1_RMSE_Normal');
%     get_field_safe(Stats_c4.Normal, 'Pipe1_RMSE_Leak')
% ];
% T3.Pipe1_Chen = [
%     Stats_c4.Chen.Pipe1_RMSE;
%     get_field_safe(Stats_c4.Chen, 'Pipe1_RMSE_Normal');
%     get_field_safe(Stats_c4.Chen, 'Pipe1_RMSE_Leak')
% ];
% T3.Pipe1_EKF_LE = [
%     Stats_c4.Adaptive.Pipe1_RMSE;
%     get_field_safe(Stats_c4.Adaptive, 'Pipe1_RMSE_Normal');
%     get_field_safe(Stats_c4.Adaptive, 'Pipe1_RMSE_Leak')
% ];
% T3.Pipe1_AFEKF = [
%     Stats_c4.AFEKF.Pipe1_RMSE;
%     get_field_safe(Stats_c4.AFEKF, 'Pipe1_RMSE_Normal');
%     get_field_safe(Stats_c4.AFEKF, 'Pipe1_RMSE_Leak')
% ];
% 
% % Pipe5 RMSE
% T3.Pipe5_Standard = [
%     Stats_c4.Normal.Pipe5_RMSE;
%     get_field_safe(Stats_c4.Normal, 'Pipe5_RMSE_Normal');
%     get_field_safe(Stats_c4.Normal, 'Pipe5_RMSE_Leak')
% ];
% T3.Pipe5_Chen = [
%     Stats_c4.Chen.Pipe5_RMSE;
%     get_field_safe(Stats_c4.Chen, 'Pipe5_RMSE_Normal');
%     get_field_safe(Stats_c4.Chen, 'Pipe5_RMSE_Leak')
% ];
% T3.Pipe5_EKF_LE = [
%     Stats_c4.Adaptive.Pipe5_RMSE;
%     get_field_safe(Stats_c4.Adaptive, 'Pipe5_RMSE_Normal');
%     get_field_safe(Stats_c4.Adaptive, 'Pipe5_RMSE_Leak')
% ];
% T3.Pipe5_AFEKF = [
%     Stats_c4.AFEKF.Pipe5_RMSE;
%     get_field_safe(Stats_c4.AFEKF, 'Pipe5_RMSE_Normal');
%     get_field_safe(Stats_c4.AFEKF, 'Pipe5_RMSE_Leak')
% ];
% originalFilename = 'Case4_Pure_Leak_Full_Results.csv';
% currentDateTime = datetime('now');
% timestampStr = datetime(currentDateTime,'Format','d-MMM-y-HH-mm-ss');
% [filepath, name, ext] = fileparts(originalFilename);
% newFilename = fullfile(filepath, sprintf('%s_%s%s', name, timestampStr, ext));
% output_dir4=[output_dir 'C4/'];
% if ~exist(output_dir4, 'dir')
%     mkdir(output_dir4);
% end
% writetable(T3, fullfile(output_dir4, newFilename));

%% ===== Table 4: 汇总对比 (Case4 disabled) =====
T4 = table();
% Case1&2: Pipe1-10 (不含6/9), Case3: Pipe1&5
row_names = {};
std_vals = []; chen_vals = []; ekfle_vals = []; afekf_vals = [];

stats_c12 = {Stats_c1, Stats_c2};
case_names_c12 = {'Case1_Clean','Case2_20dB'};

% Case1&2 Pipe (不含6/9)
for pp = [1,2,3,4,5,7,8,10]
    fn = sprintf('Pipe%d_RMSE', pp);
    for ci = 1:2
        row_names{end+1} = sprintf('%s_Pipe%d', case_names_c12{ci}, pp);
        std_vals(end+1)   = stats_c12{ci}.Normal.(fn);
        chen_vals(end+1)  = stats_c12{ci}.Chen.(fn);
        ekfle_vals(end+1) = stats_c12{ci}.Adaptive.(fn);
        afekf_vals(end+1) = stats_c12{ci}.AFEKF.(fn);
    end
end

% Case3 Pipe1&5 only
for pp = [1, 5]
    fn = sprintf('Pipe%d_RMSE', pp);
    row_names{end+1} = sprintf('Case3_Outlier_Leak_Pipe%d', pp);
    std_vals(end+1) = Stats_c3.Normal.(fn);
    chen_vals(end+1) = Stats_c3.Chen.(fn);
    ekfle_vals(end+1) = Stats_c3.Adaptive.(fn);
    afekf_vals(end+1) = Stats_c3.AFEKF.(fn);
end

T4.Case = row_names';
T4.Standard = std_vals';
T4.Chen = chen_vals';
T4.EKF_LE = ekfle_vals';
T4.AFEKF = afekf_vals';
% Original filename
originalFilename = 'Summary_All_Cases.csv';
% Get the current date and time
currentDateTime = datetime('now');
% Format the timestamp
timestampStr = datetime(currentDateTime,'Format','d-MMM-y-HH-mm-ss');
% Create new filename by appending the timestamp
[filepath, name, ext] = fileparts(originalFilename); % Split into components
newFilename = fullfile(filepath, sprintf('%s_%s%s', name, timestampStr, ext));
output_dir5=[output_dir 'Summary/'];
if ~exist(output_dir5, 'dir')
    mkdir(output_dir5);
end
writetable(T4, fullfile(output_dir5, newFilename));

fprintf('\n========================================\n');
fprintf('CSV files exported to: %s\n', output_dir);
fprintf('========================================\n');
fprintf('  ✓ Case1_Case2_Full_Results.csv\n');
fprintf('  ✓ Case3_Outlier_Leak_Full_Results.csv (with Normal/Leak breakdown)\n');
fprintf('  ✓ Summary_All_Cases.csv\n');
fprintf('========================================\n\n');

end


function val = get_field_safe(S, fieldname)
% Safely get field value, return NaN if not exists
if isfield(S, fieldname)
    val = S.(fieldname);
else
    val = NaN;
end
end