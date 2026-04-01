function save_comprehensive_results(Stats_c1, Stats_c2_10, Stats_c2_20, Stats_c3, filename)
% SAVE_COMPREHENSIVE_RESULTS - Save comprehensive results to Excel file

if nargin < 5
    filename = 'Comprehensive_Results.xlsx';
end

% Sheet 1: Pressure RMSE comparison
T1 = table();
T1.Case = {'Case1_Clean'; 'Case2_10dB_Burst'; 'Case2_20dB_Burst'; 'Case3_Burst_Leak'};
T1.Standard_EKF = [
    Stats_c1.Normal.P_RMSE;
    Stats_c2_10.Normal.P_RMSE;
    Stats_c2_20.Normal.P_RMSE;
    Stats_c3.Normal.P_RMSE
];
T1.Chen_Robust_EKF = [
    Stats_c1.Chen.P_RMSE;
    Stats_c2_10.Chen.P_RMSE;
    Stats_c2_20.Chen.P_RMSE;
    Stats_c3.Chen.P_RMSE
];
T1.Adaptive_EKF = [
    Stats_c1.Adaptive.P_RMSE;
    Stats_c2_10.Adaptive.P_RMSE;
    Stats_c2_20.Adaptive.P_RMSE;
    Stats_c3.Adaptive.P_RMSE
];
T1.Chen_vs_Standard_pct = (T1.Standard_EKF - T1.Chen_Robust_EKF) ./ T1.Standard_EKF * 100;
T1.Adaptive_vs_Standard_pct = (T1.Standard_EKF - T1.Adaptive_EKF) ./ T1.Standard_EKF * 100;

writetable(T1, filename, 'Sheet', 'Pressure_RMSE_Comparison');

% Sheet 2: Flow RMSE comparison
T2 = table();
T2.Case = {'Case1_Clean'; 'Case2_10dB_Burst'; 'Case2_20dB_Burst'; 'Case3_Burst_Leak'};
T2.Standard_EKF = [
    Stats_c1.Normal.M_RMSE;
    Stats_c2_10.Normal.M_RMSE;
    Stats_c2_20.Normal.M_RMSE;
    Stats_c3.Normal.M_RMSE
];
T2.Chen_Robust_EKF = [
    Stats_c1.Chen.M_RMSE;
    Stats_c2_10.Chen.M_RMSE;
    Stats_c2_20.Chen.M_RMSE;
    Stats_c3.Chen.M_RMSE
];
T2.Adaptive_EKF = [
    Stats_c1.Adaptive.M_RMSE;
    Stats_c2_10.Adaptive.M_RMSE;
    Stats_c2_20.Adaptive.M_RMSE;
    Stats_c3.Adaptive.M_RMSE
];
T2.Chen_vs_Standard_pct = (T2.Standard_EKF - T2.Chen_Robust_EKF) ./ T2.Standard_EKF * 100;
T2.Adaptive_vs_Standard_pct = (T2.Standard_EKF - T2.Adaptive_EKF) ./ T2.Standard_EKF * 100;

writetable(T2, filename, 'Sheet', 'Flow_RMSE_Comparison');

% Sheet 3: Pressure MAE comparison
T3 = table();
T3.Case = {'Case1_Clean'; 'Case2_10dB_Burst'; 'Case2_20dB_Burst'; 'Case3_Burst_Leak'};
T3.Standard_EKF = [
    Stats_c1.Normal.P_MAE;
    Stats_c2_10.Normal.P_MAE;
    Stats_c2_20.Normal.P_MAE;
    Stats_c3.Normal.P_MAE
];
T3.Chen_Robust_EKF = [
    Stats_c1.Chen.P_MAE;
    Stats_c2_10.Chen.P_MAE;
    Stats_c2_20.Chen.P_MAE;
    Stats_c3.Chen.P_MAE
];
T3.Adaptive_EKF = [
    Stats_c1.Adaptive.P_MAE;
    Stats_c2_10.Adaptive.P_MAE;
    Stats_c2_20.Adaptive.P_MAE;
    Stats_c3.Adaptive.P_MAE
];

writetable(T3, filename, 'Sheet', 'Pressure_MAE_Comparison');

% Sheet 4: Flow MAE comparison
T4 = table();
T4.Case = {'Case1_Clean'; 'Case2_10dB_Burst'; 'Case2_20dB_Burst'; 'Case3_Burst_Leak'};
T4.Standard_EKF = [
    Stats_c1.Normal.M_MAE;
    Stats_c2_10.Normal.M_MAE;
    Stats_c2_20.Normal.M_MAE;
    Stats_c3.Normal.M_MAE
];
T4.Chen_Robust_EKF = [
    Stats_c1.Chen.M_MAE;
    Stats_c2_10.Chen.M_MAE;
    Stats_c2_20.Chen.M_MAE;
    Stats_c3.Chen.M_MAE
];
T4.Adaptive_EKF = [
    Stats_c1.Adaptive.M_MAE;
    Stats_c2_10.Adaptive.M_MAE;
    Stats_c2_20.Adaptive.M_MAE;
    Stats_c3.Adaptive.M_MAE
];

writetable(T4, filename, 'Sheet', 'Flow_MAE_Comparison');

% Sheet 5: Summary statistics
T5 = table();
T5.Metric = {
    'Avg_Pressure_RMSE_Bar';
    'Avg_Flow_RMSE_kgs';
    'Avg_Pressure_MAE_Bar';
    'Avg_Flow_MAE_kgs';
    'Max_Pressure_RMSE_Bar';
    'Max_Flow_RMSE_kgs'
};

T5.Standard_EKF = [
    mean([Stats_c1.Normal.P_RMSE, Stats_c2_10.Normal.P_RMSE, Stats_c2_20.Normal.P_RMSE, Stats_c3.Normal.P_RMSE]);
    mean([Stats_c1.Normal.M_RMSE, Stats_c2_10.Normal.M_RMSE, Stats_c2_20.Normal.M_RMSE, Stats_c3.Normal.M_RMSE]);
    mean([Stats_c1.Normal.P_MAE, Stats_c2_10.Normal.P_MAE, Stats_c2_20.Normal.P_MAE, Stats_c3.Normal.P_MAE]);
    mean([Stats_c1.Normal.M_MAE, Stats_c2_10.Normal.M_MAE, Stats_c2_20.Normal.M_MAE, Stats_c3.Normal.M_MAE]);
    max([Stats_c1.Normal.P_RMSE, Stats_c2_10.Normal.P_RMSE, Stats_c2_20.Normal.P_RMSE, Stats_c3.Normal.P_RMSE]);
    max([Stats_c1.Normal.M_RMSE, Stats_c2_10.Normal.M_RMSE, Stats_c2_20.Normal.M_RMSE, Stats_c3.Normal.M_RMSE])
];

T5.Chen_Robust_EKF = [
    mean([Stats_c1.Chen.P_RMSE, Stats_c2_10.Chen.P_RMSE, Stats_c2_20.Chen.P_RMSE, Stats_c3.Chen.P_RMSE]);
    mean([Stats_c1.Chen.M_RMSE, Stats_c2_10.Chen.M_RMSE, Stats_c2_20.Chen.M_RMSE, Stats_c3.Chen.M_RMSE]);
    mean([Stats_c1.Chen.P_MAE, Stats_c2_10.Chen.P_MAE, Stats_c2_20.Chen.P_MAE, Stats_c3.Chen.P_MAE]);
    mean([Stats_c1.Chen.M_MAE, Stats_c2_10.Chen.M_MAE, Stats_c2_20.Chen.M_MAE, Stats_c3.Chen.M_MAE]);
    max([Stats_c1.Chen.P_RMSE, Stats_c2_10.Chen.P_RMSE, Stats_c2_20.Chen.P_RMSE, Stats_c3.Chen.P_RMSE]);
    max([Stats_c1.Chen.M_RMSE, Stats_c2_10.Chen.M_RMSE, Stats_c2_20.Chen.M_RMSE, Stats_c3.Chen.M_RMSE])
];

T5.Adaptive_EKF = [
    mean([Stats_c1.Adaptive.P_RMSE, Stats_c2_10.Adaptive.P_RMSE, Stats_c2_20.Adaptive.P_RMSE, Stats_c3.Adaptive.P_RMSE]);
    mean([Stats_c1.Adaptive.M_RMSE, Stats_c2_10.Adaptive.M_RMSE, Stats_c2_20.Adaptive.M_RMSE, Stats_c3.Adaptive.M_RMSE]);
    mean([Stats_c1.Adaptive.P_MAE, Stats_c2_10.Adaptive.P_MAE, Stats_c2_20.Adaptive.P_MAE, Stats_c3.Adaptive.P_MAE]);
    mean([Stats_c1.Adaptive.M_MAE, Stats_c2_10.Adaptive.M_MAE, Stats_c2_20.Adaptive.M_MAE, Stats_c3.Adaptive.M_MAE]);
    max([Stats_c1.Adaptive.P_RMSE, Stats_c2_10.Adaptive.P_RMSE, Stats_c2_20.Adaptive.P_RMSE, Stats_c3.Adaptive.P_RMSE]);
    max([Stats_c1.Adaptive.M_RMSE, Stats_c2_10.Adaptive.M_RMSE, Stats_c2_20.Adaptive.M_RMSE, Stats_c3.Adaptive.M_RMSE])
];

T5.Chen_Improvement_pct = (T5.Standard_EKF - T5.Chen_Robust_EKF) ./ T5.Standard_EKF * 100;
T5.Adaptive_Improvement_pct = (T5.Standard_EKF - T5.Adaptive_EKF) ./ T5.Standard_EKF * 100;

writetable(T5, filename, 'Sheet', 'Summary_Statistics');

fprintf('Results saved to: %s\n', filename);

end