function [Z_biased, Z_E_biased] = inject_chen_bias(Z, Z_E, Sys, Nodes, PowerSys, bias_config)
% INJECT_CHEN_BIAS - Inject non-zero mean biases for Chen (2022) Case b
%
% Chen Section V-C biases:
%   Node 24 pressure:  +0.5 bar
%   Node 4  mass flow: +0.4 kg/s
%   Bus 12  voltage e: +0.02 p.u.
%   Bus 12  voltage f: +0.02 p.u.
%
% Inputs:
%   Z           - gas measurements [N x nS]
%   Z_E         - PMU measurements [N x nZ_E]
%   Sys         - system params
%   Nodes       - node table
%   PowerSys    - power system struct
%   bias_config - optional struct to override defaults

nN = height(Nodes);

% Default biases from Chen Section V-C
if nargin < 6 || isempty(bias_config)
    bias_config.node24_pressure_bar = 0.5;
    bias_config.node4_flow_kgs      = 0.4;
    bias_config.bus12_voltage_pu    = 0.02;
end

Z_biased  = Z;
Z_E_biased = Z_E;

% Node 24 pressure bias: convert bar to density units
bias_rho = bias_config.node24_pressure_bar * 1e5 / Sys.c2;
Z_biased(:, 24) = Z_biased(:, 24) + bias_rho;

% Node 4 mass flow bias
Z_biased(:, nN + 4) = Z_biased(:, nN + 4) + bias_config.node4_flow_kgs;

% Bus 12 voltage bias in PMU measurements
% H_E rows 1..2*nZB are voltage measurements (2 rows per PMU bus)
% Find bus 12 in pmu_buses list
pmu_buses = PowerSys.pmu_buses;
idx12 = find(pmu_buses == 12, 1);
if ~isempty(idx12)
    row_e = 2*idx12 - 1;
    row_f = 2*idx12;
    Z_E_biased(:, row_e) = Z_E_biased(:, row_e) + bias_config.bus12_voltage_pu;
    Z_E_biased(:, row_f) = Z_E_biased(:, row_f) + bias_config.bus12_voltage_pu;
end

end
