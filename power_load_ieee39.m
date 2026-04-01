function PowerSys = power_load_ieee39()
% POWER_LOAD_IEEE39 - Build power system data structure for IEEE 39-bus system
% Uses MATPOWER to run power flow and extract Y-bus matrix.
%
% Output:
%   PowerSys  - struct with fields:
%     .nB           number of buses (39)
%     .baseMVA      system MVA base (100)
%     .Y            complex admittance matrix (39x39)
%     .G            real part of Y
%     .B            imaginary part of Y
%     .x0           initial state vector [e1,f1,...,e39,f39] (78x1) from power flow
%     .pmu_buses    buses with PMU (voltage + injected current measured)
%     .pmu_branches branch pairs [from, to] with current measurement
%     .branch       branch list [from, to] from MATPOWER (for H_IB indexing)
%     .gtu          struct array with fields: bus_id, gas_node_id, eta (MW*s/kg)

% Run power flow using MATPOWER
mpc = case39();
results = runpf(mpc, mpoption('verbose', 0, 'out.all', 0));
if results.success ~= 1
    error('Power flow did not converge.');
end

PowerSys.nB      = size(results.bus, 1);   % 39
PowerSys.baseMVA = results.baseMVA;         % 100 MVA

% Build Y-bus
[Y, ~, ~] = makeYbus(results.baseMVA, results.bus, results.branch);
PowerSys.Y = full(Y);
PowerSys.G = real(PowerSys.Y);
PowerSys.B = imag(PowerSys.Y);

% Store branch list [from, to] for building H_IB later
PowerSys.branch = results.branch(:, 1:2);  % columns 1,2 are fbus, tbus

% Convert solved polar voltages to rectangular (e, f)
Vm = results.bus(:, 8);          % voltage magnitude (p.u.)
Va = results.bus(:, 9) * pi/180; % voltage angle (rad)
e  = Vm .* cos(Va);
f  = Vm .* sin(Va);

% State vector layout: [e1, f1, e2, f2, ..., e39, f39]
x0 = zeros(2 * PowerSys.nB, 1);
for i = 1:PowerSys.nB
    x0(2*i-1) = e(i);
    x0(2*i)   = f(i);
end
PowerSys.x0 = x0;

% PMU configuration from Chen (2022) Section V-A
% Buses equipped with PMU - voltage and injected current are measured
PowerSys.pmu_buses = [2, 4, 6, 9, 12, 15, 18, 21, 22, 25, 28, ...
                      30, 31, 32, 33, 34, 35, 36, 37, 38, 39];

% Branch current measurements [from_bus, to_bus]
% Listed as in Chen Section V-A
PowerSys.pmu_branches = [
    39,  1;
    25,  2;
    30,  2;
     6,  5;
     7,  6;
    11,  6;
    31,  6;
     9,  8;
    39,  9;
    11, 10;
    13, 10;
    32, 10;
    14, 13;
    15, 14;
    17, 16;
    18, 17;
    27, 17;
    20, 19;
    33, 19;
    34, 20;
    35, 22;
    24, 23;
    36, 23;
    26, 25;
    37, 25;
    28, 26;
    29, 26;
    38, 29;
];

% GTU coupling: bus_id <-> gas_node_id
% Energy conversion coefficient eta = 20.148 MW*s/kg (40% efficiency, Chen Sec V-A)
% Units: P_G [MW] = eta [MW*s/kg] * m_dot [kg/s]
eta_val = 20.148;

gtu_map = [
    31, 31;
    32, 24;
    34,  4;
    33, 34;
    36, 30;
];

for k = 1:size(gtu_map, 1)
    PowerSys.gtu(k).bus_id      = gtu_map(k, 1);
    PowerSys.gtu(k).gas_node_id = gtu_map(k, 2);
    PowerSys.gtu(k).eta         = eta_val;
end

% Build and store H_E so all EKF functions can use it without recomputing
PowerSys.H_E = power_meas_matrix(PowerSys);

fprintf('Power system loaded: %d buses, %d PMU buses, %d branch meters, %d GTUs\n', ...
    PowerSys.nB, length(PowerSys.pmu_buses), ...
    size(PowerSys.pmu_branches, 1), length(PowerSys.gtu));
end
