function H_E = power_meas_matrix(PowerSys)
% POWER_MEAS_MATRIX - Build PMU measurement matrix H_E (Chen 2022, Eq 5, Appendix A)
%
% State vector:  x_E = [e1, f1, e2, f2, ..., e39, f39]  (2*nB x 1)
% Measurement:   z_E = [z_V; z_IB; z_IN]
%   z_V  : voltage real/imag at PMU buses          (2*nZB x 1)
%   z_IB : branch current real/imag at PMU branches (2*nZC x 1)
%   z_IN : injected current real/imag at PMU buses  (2*nZB x 1)
%
% H_E is constant (PMU measurements are linear in rectangular coordinates).

nB  = PowerSys.nB;
G   = PowerSys.G;
B   = PowerSys.B;

pmu_buses    = PowerSys.pmu_buses;
pmu_branches = PowerSys.pmu_branches;   % [from, to] as listed in Chen
branch_list  = PowerSys.branch;         % [from, to] from MATPOWER

nZB = length(pmu_buses);
nZC = size(pmu_branches, 1);
nZ  = 2*nZB + 2*nZC + 2*nZB;   % total measurements

H_E = zeros(nZ, 2*nB);

row = 1;

% --- Block 1: H_V (voltage measurements, Eq 38 in Appendix A) ---
% z_V has rows [e_b, f_b] for each PMU bus b
% h_V(2b_M-1, 2i) = 1 if this measurement is voltage of bus i
for k = 1:nZB
    b = pmu_buses(k);
    H_E(row,   2*b-1) = 1;   % e_b
    H_E(row+1, 2*b  ) = 1;   % f_b
    row = row + 2;
end

% --- Block 2: H_IB (branch current measurements, Eq 39 in Appendix A) ---
% For branch i->j:
%   I_BR_real = (g_ij+g_i0)*e_i - b_ij*f_i - g_ij*e_j + b_ij*f_j  (row 2l-1)
%   I_BR_imag = (b_ij+b_i0)*f_i + g_ij*e_i - b_ij*e_j - g_ij*f_j  (row 2l)
% where g_ij, b_ij come from Y-bus off-diagonal: Y_ij = -(g_ij + j*b_ij)
% and g_i0, b_i0 are shunt at bus i: Y_ii = sum of g+jb including shunt

for k = 1:nZC
    from_bus = pmu_branches(k, 1);
    to_bus   = pmu_branches(k, 2);

    % Find this branch in MATPOWER branch list (allow either direction)
    idx = find( (branch_list(:,1)==from_bus & branch_list(:,2)==to_bus) | ...
                (branch_list(:,1)==to_bus   & branch_list(:,2)==from_bus), 1);

    if isempty(idx)
        warning('Branch %d-%d not found in branch list, skipping.', from_bus, to_bus);
        row = row + 2;
        continue;
    end

    actual_from = branch_list(idx, 1);
    actual_to   = branch_list(idx, 2);

    % Branch admittance from Y-bus off-diagonal: Y_ij = -(g_ij + j*b_ij)
    g_ij = -G(actual_from, actual_to);
    b_ij = -B(actual_from, actual_to);

    % Shunt at from-bus: Y_ii(diag) = g_ii + j*b_ii includes all connected branches + shunt
    % g_i0, b_i0 extracted as: Y_ii - sum of -Y_ij for all j~=i
    % Simplified: use g_i0 = G(i,i) - sum_j(g_ij), b_i0 = B(i,i) - sum_j(b_ij)
    % For H_IB we use i = from_bus of the branch direction in MATPOWER
    i = actual_from;
    j = actual_to;

    g_i0 = G(i,i) + sum(G(i, [1:i-1, i+1:nB]));   % shunt conductance at bus i
    b_i0 = B(i,i) + sum(B(i, [1:i-1, i+1:nB]));   % shunt susceptance at bus i

    % If Chen's listed direction matches MATPOWER direction, use directly.
    % If reversed, negate (current flows the other way).
    direction = 1;
    if from_bus == actual_to
        direction = -1;
    end

    % Row 2l-1: real part of branch current
    H_E(row, 2*i-1) =  direction * (g_ij + g_i0);
    H_E(row, 2*i  ) =  direction * (-b_ij - b_i0);
    H_E(row, 2*j-1) =  direction * (-g_ij);
    H_E(row, 2*j  ) =  direction * (b_ij);

    % Row 2l: imaginary part of branch current
    H_E(row+1, 2*i-1) =  direction * (b_ij + b_i0);
    H_E(row+1, 2*i  ) =  direction * (g_ij + g_i0);
    H_E(row+1, 2*j-1) =  direction * (-b_ij);
    H_E(row+1, 2*j  ) =  direction * (-g_ij);

    row = row + 2;
end

% --- Block 3: H_IN (injected current measurements, Eq 40 in Appendix A) ---
% Injected current at bus b:
%   I_IN_real(b) = sum_j [ G_bj * e_j - B_bj * f_j ]
%   I_IN_imag(b) = sum_j [ G_bj * f_j + B_bj * e_j ]
% This equals the b-th row of Y * x_E in rectangular form.
for k = 1:nZB
    b = pmu_buses(k);

    for j = 1:nB
        % Real part of injected current at bus b
        H_E(row,   2*j-1) =  G(b, j);   % coefficient of e_j
        H_E(row,   2*j  ) = -B(b, j);   % coefficient of f_j

        % Imaginary part of injected current at bus b
        H_E(row+1, 2*j-1) =  B(b, j);   % coefficient of e_j
        H_E(row+1, 2*j  ) =  G(b, j);   % coefficient of f_j
    end
    row = row + 2;
end

fprintf('H_E built: %d measurements x %d states\n', size(H_E,1), size(H_E,2));
end
