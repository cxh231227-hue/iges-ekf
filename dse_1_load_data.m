function [Nodes, Pipes, Compressors, GTU] = dse_1_load_data(filename)
% DSE_1_LOAD_DATA - Load gas network topology from Excel file
% Reads network components: Nodes, Pipes, Compressors, and Gas Turbine Units
% Input:  filename - path to Excel file (default: 'gas_data.xlsx')
% Output: Nodes, Pipes, Compressors, GTU tables
if nargin < 1, filename = 'gas_data.xlsx'; end
if ~exist(filename,'file')
    error('File not found: %s', filename);
end

sheets = sheetnames(filename);
opts = detectImportOptions(filename);
opts.VariableNamingRule = 'preserve';

% Load Node Data (sources, loads, compressor stations, GTU nodes)
raw = readtable(filename, opts, 'Sheet', 'Nodes');
Nodes = table();
Nodes.ID = get_col(raw, {'ID','NodeID'}, 1);
Nodes.Type = get_col(raw, {'Type','NodeType'}, 1);           % 1=Source, 2=Load, 3=Compressor, 4=GTU
Nodes.BaseLoad = get_col(raw, {'BaseLoad_kg_s','BaseLoad','Load'}, 0, 0);
Nodes.InitP = get_col(raw, {'InitPressure_Pa','InitP','Pressure'}, 0, 50e5);

% Load Pipe Data (network edges with physical properties)
raw = readtable(filename, opts, 'Sheet', 'Pipes');
Pipes = table();
Pipes.ID = get_col(raw, {'ID','PipeID'}, 1);
Pipes.From = get_col(raw, {'FromNode','From'}, 1);
Pipes.To = get_col(raw, {'ToNode','To'}, 1);
Pipes.L = get_col(raw, {'Length_m','Length','L'}, 1);        % Pipe length [m]
Pipes.D = get_col(raw, {'Diameter_m','Diameter','D'}, 1);    % Pipe diameter [m]
Pipes.Fric = get_col(raw, {'FrictionFactor','Friction','Fric','f'}, 0, 0.012);

% Load Compressor Data (optional - pressure boosting stations)
try
    raw = readtable(filename, opts, 'Sheet', 'Compressors');
    Compressors = table();
    Compressors.PipeID = get_col(raw, {'PipeID','Pipe','ID'}, 1);
    Compressors.Ratio = get_col(raw, {'Ratio','CompressionRatio'}, 1);
catch
    Compressors = table([],[],'VariableNames',{'PipeID','Ratio'});
end

% Load GTU Data (optional - Gas Turbine Units for power generation)
GTU = table([],[],[],[],[],[],[],[],...
    'VariableNames',{'ID','NodeID','Capacity_MW','HeatRate','MinLoad','MaxLoad','BaseFlow','Status'});

if any(strcmpi(sheets,'GTU'))
    try
        raw = readtable(filename,'Sheet','GTU','VariableNamingRule','preserve');
        if height(raw) > 0
            GTU = table();
            GTU.ID = get_col(raw,{'GTU_ID','GTUID','ID'},1);
            GTU.NodeID = get_col(raw,{'NodeID','Node'},1);
            GTU.Capacity_MW = get_col(raw,{'Capacity_MW','CapacityMW','Capacity','P_max','Pmax'},1);
            GTU.HeatRate = get_col(raw,{'HeatRate_MJ_kWh','HeatRateMJkWh','HeatRate','HR'},0,8.0);
            GTU.MinLoad = get_col(raw,{'MinLoad_pct','MinLoadpct','MinLoad','Pmin'},0,30);
            GTU.MaxLoad = get_col(raw,{'MaxLoad_pct','MaxLoadpct','MaxLoad','Pmax'},0,100);
            GTU.BaseFlow = get_col(raw,{'BaseFlow_kg_s','BaseFlowkgs','BaseFlow','Flow'},1);
            GTU.Status = get_col(raw,{'Status','On','Active'},0,1);
        end
    catch
    end
end
end


function data = get_col(T, names, required, defval)
cols = T.Properties.VariableNames;
cols_c = lower(regexprep(cols,'[^a-zA-Z0-9]',''));

% Try each possible column name
for i = 1:length(names)
    key = lower(regexprep(names{i},'[^a-zA-Z0-9]',''));
    idx = find(strcmp(cols_c,key),1);
    if ~isempty(idx)
        tmp = T.(cols{idx});
        % Handle different data types (cell, string, numeric)
        if iscell(tmp)
            data = str2double(string(tmp));
        elseif isstring(tmp)
            data = str2double(tmp);
        else
            data = double(tmp);
        end
        if nargin >= 4
            data(isnan(data)) = defval;
        end
        return
    end
end
if required
    error('Column not found: %s', names{1});
end
if nargin < 4, defval = 0; end
data = repmat(defval, height(T), 1);
end