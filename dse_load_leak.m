function Leaks = dse_load_leak(fname)
% =========================================================================
% DSE_LOAD_LEAK - Load leak scenario configuration from Excel file
% Reads leak parameters from 'Leaks' sheet in the specified Excel file
% Input:  fname - path to Excel file (default: 'leak.xlsx')
% Output: Leaks table with columns: PipeID, StartTime_s, EndTime_s, LeakRate_kg_s, Position
% =========================================================================
if nargin < 1, fname = 'leak.xlsx'; end

% Define empty table template for error cases
empty_tbl = @() table([],[],[],[],[],...
    'VariableNames',{'PipeID','StartTime_s','EndTime_s','LeakRate_kg_s','Position'});

% Return empty table if file doesn't exist
if ~exist(fname,'file')
    Leaks = empty_tbl();
    return
end

try
  
    % Read Excel File and Parse Leak Parameters
    opts = detectImportOptions(fname);
    opts.VariableNamingRule = 'preserve';
    raw = readtable(fname,opts,'Sheet','Leaks');
    
    if isempty(raw) || height(raw) == 0
        Leaks = empty_tbl();
        return
    end
    
    % Extract columns with flexible name matching
    Leaks = table();
    Leaks.PipeID = get_leak_col(raw,{'PipeID','Pipe','ID','PipeNo'});
    Leaks.StartTime_s = get_leak_col(raw,{'StartTime_s','StartTime','Start','T_start'});
    Leaks.EndTime_s = get_leak_col(raw,{'EndTime_s','EndTime','End','T_end'});
    Leaks.LeakRate_kg_s = get_leak_col(raw,{'LeakRate_kg_s','LeakRate','Rate','Q_leak'});
    Leaks.Position = get_leak_col(raw,{'Position','Pos','Location','X'});  % 0-1 position along pipe
    
    % Validate and Clean Data
    ok = Leaks.PipeID > 0 & ~isnan(Leaks.PipeID) & Leaks.LeakRate_kg_s > 0;
    Leaks = Leaks(ok,:);
    
    if height(Leaks) == 0
        Leaks = empty_tbl();
        return
    end
    
    % Treat very large end times as infinite (permanent leaks)
    Leaks.EndTime_s(Leaks.EndTime_s > 1e8) = Inf;
catch
    Leaks = empty_tbl();
end
end


function data = get_leak_col(T, names)
% GET_LEAK_COL - Extract column from table with flexible name matching
% Tries multiple possible column names and handles type conversion
cols = T.Properties.VariableNames;
cols_c = lower(regexprep(cols,'[^a-zA-Z0-9]',''));

% Try each possible column name
for i = 1:length(names)
    key = lower(regexprep(names{i},'[^a-zA-Z0-9]',''));
    idx = find(strcmp(cols_c,key),1);
    if ~isempty(idx)
        tmp = T.(cols{idx});
        % Handle different data types
        if iscell(tmp)
            data = str2double(string(tmp));
        elseif isstring(tmp)
            data = str2double(tmp);
        else
            data = double(tmp);
        end
        data(isnan(data)) = 0;
        return
    end
end

% Column not found - return zeros
data = zeros(height(T),1);
end