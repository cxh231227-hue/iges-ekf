function x0 = dse_steady_solver(Nodes, Pipes, Compressors, Sys)

% DSE_STEADY_SOLVER - Compute initial steady-state solution
% Uses iterative fixed-point method to find equilibrium state
% The steady state serves as initial condition for dynamic simulation
% Input:  Network topology and system parameters
% Output: x0 - Steady state vector [rho_1..rho_N, M_1..M_P]

nN = height(Nodes);
nP = height(Pipes);

rho0 = ones(nN,1) * 50e5/Sys.c2;  
M0 = ones(nP,1) * 50;              
x = [rho0; M0];

% Compute load at t=0 using the daily load profile
load0 = Nodes.BaseLoad * (1 + 0.05*sin(2*pi*(0-8)/24));

% Use larger time step for faster convergence to steady state
Sys_tmp = Sys;
Sys_tmp.dt = 1000;  

for iter = 1:100
    [A, B, u] = dse_2_build_model(x, Nodes, Pipes, Compressors, Sys_tmp, load0);
    xnew = A \ (B*x + u);
 
    if norm(xnew - x) < 1e-3
        x = xnew;
        break
    end
    x = xnew;
end

x0 = x;
end