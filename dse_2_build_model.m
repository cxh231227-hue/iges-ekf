function [A, B, u] = dse_2_build_model(x, Nodes, Pipes, Compressors, Sys, Loads)
% DSE_2_BUILD_MODEL - Build linearized state-space model for gas network
% Constructs matrices A, B, u for the implicit discrete-time model: A*x_new = B*x + u
% State vector x = [rho_1..rho_N, M_1..M_P] (nodal densities + pipe mass flows)
nN = height(Nodes);
nP = height(Pipes);
nS = nN + nP;

% Extract current state
rho = x(1:nN);
M = x(nN+1:end);

% Initialize sparse matrices for efficiency
A = spalloc(nS,nS,5*nS);
B = spalloc(nS,nS,5*nS);
u = zeros(nS,1);

% Get list of pipes with compressors
if ~isempty(Compressors)
    comp_pipes = Compressors.PipeID;
else
    comp_pipes = [];
end

for i = 1:nN
    if Nodes.Type(i) == 1
         A(i,i) = 1;
        u(i) = Nodes.InitP(i) / Sys.c2;
    else
         out_p = find(Pipes.From == i);
        in_p = find(Pipes.To == i);
        conn = [out_p; in_p];
        
        % Compute effective node volume from connected pipes
        Vol = sum(Pipes.L(conn) .* (pi/4 .* Pipes.D(conn).^2)) * 0.5;
        if Vol < 1, Vol = 100; end
        
        dt_V = Sys.dt / Vol;
        A(i,i) = 1;
        B(i,i) = 1;
        u(i) = -dt_V * Loads(i);    % Load withdrawal term
        
        % Coupling with pipe flows (outflow is positive, inflow is negative)
        for k = out_p'
            A(i,nN+k) = dt_V;
        end
        for k = in_p'
            A(i,nN+k) = -dt_V;
        end
    end
end

% For compressor pipes: Pressure ratio constraint (P_out = ratio * P_in)
% For regular pipes: d(M)/dt = S*c2*(rho_in - rho_out)/L - friction*M
for k = 1:nP
    row = nN + k;
    i_from = Pipes.From(k);
    i_to = Pipes.To(k);
    
    if ismember(k,comp_pipes)
        % Compressor: Algebraic pressure ratio constraint
        idx = find(Compressors.PipeID == k);
        cr = Compressors.Ratio(idx);
        A(row,i_to) = 1;
        A(row,i_from) = -cr;
        A(row,row) = 1e-9;   
    else
        % Regular pipe: Discretized momentum equation
        L = Pipes.L(k);
        D = Pipes.D(k);
        S = pi/4*D^2;         
        
        coef = Sys.dt * S * Sys.c2 / L; 
        
        % Average density for friction calculation
        rho_avg = (rho(i_from) + rho(i_to)) / 2;
        if rho_avg < 0.1, rho_avg = 0.1; end
        
        % Friction term (Darcy-Weisbach based)
        f = Pipes.Fric(k);
        vel = abs(M(k)) / (S * rho_avg);
        if vel < 0.1, vel = 0.1; end
        lam = f * vel / (2*D);
        
        % Build momentum equation: A(row,:)*x_new = B(row,:)*x
        A(row,i_to) = coef;
        A(row,i_from) = -coef;
        A(row,row) = 1 + Sys.dt*lam;
        B(row,row) = 1;
    end
end
end