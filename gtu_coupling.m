function m_gtu = gtu_coupling(x_E, PowerSys)
% GTU_COUPLING - Compute gas mass flow demand at GTU sink nodes from power states
% Implements Chen (2022) Eq 24:
%
%   m_dot_s = (1/eta_i) * sum_j[ e_i*(G_ij*e_j - B_ij*f_j)
%                               + f_i*(G_ij*f_j + B_ij*e_j) ]
%
% where i is the GTU bus, j loops over all buses connected to i.
% The inner sum is exactly the real power injection P_G at bus i.
%
% Inputs:
%   x_E      - power state vector [e1,f1,e2,f2,...,e39,f39] (78x1)
%   PowerSys - struct from power_load_ieee39
%
% Output:
%   m_gtu    - struct array, one entry per GTU:
%                .bus_id      power bus
%                .gas_node_id gas network sink node
%                .m_dot       mass flow rate [kg/s]

G   = PowerSys.G;
B   = PowerSys.B;
nB  = PowerSys.nB;

% Unpack rectangular voltages from state vector
e = x_E(1:2:2*nB);   % real parts  [e1, e2, ..., e39]
f = x_E(2:2:2*nB);   % imag parts  [f1, f2, ..., f39]

m_gtu = PowerSys.gtu;   % copy struct, will fill in m_dot

for k = 1:length(PowerSys.gtu)
    i   = PowerSys.gtu(k).bus_id;
    eta = PowerSys.gtu(k).eta;   % [MW*s/kg]

    % Real power injection at bus i (Eq 23 summed = Eq 24 numerator)
    % P_G_i = sum_j [ e_i*(G_ij*e_j - B_ij*f_j) + f_i*(G_ij*f_j + B_ij*e_j) ]
    P_G = 0;
    for j = 1:nB
        P_G = P_G + e(i)*(G(i,j)*e(j) - B(i,j)*f(j)) ...
                  + f(i)*(G(i,j)*f(j) + B(i,j)*e(j));
    end

    % Convert power to mass flow: m_dot = P_G / eta
    % P_G is in per-unit, convert to MW first: P_G_MW = P_G * baseMVA
    P_G_MW = P_G * PowerSys.baseMVA;

    % m_dot [kg/s] = P_G [MW] / eta [MW*s/kg]
    m_dot = P_G_MW / eta;

    % Physical constraint: mass flow cannot be negative
    m_dot = max(m_dot, 0);

    m_gtu(k).m_dot = m_dot;
end

end
