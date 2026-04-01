function [Nodes, Pipes, Compressors, GTU] = dse_load_gaslib40(net_file, scn_file)
% DSE_LOAD_GASLIB40  读取 GasLib-40 的 .net 和 .scn 文件
% 输出与原 dse_1_load_data 完全兼容的表格结构
%
% 用法:
%   [Nodes,Pipes,Compressors,GTU] = dse_load_gaslib40( ...
%       'GasLib-40-v1-20211130.net','GasLib-40-v1-20211130.scn');
%
% 单位换算:
%   长度  km  → m   (×1000)
%   直径  mm  → m   (÷1000)
%   流量  1000m³/h → kg/s  (×NORM_DENSITY/3.6)
%   压力  bar → Pa  (×1e5)
%   摩阻  由 Swamee-Jain 从管壁粗糙度计算

if nargin < 1, net_file = 'GasLib-40-v1-20211130.net'; end
if nargin < 2, scn_file = 'GasLib-40-v1-20211130.scn'; end

NORM_DENSITY = 0.785;
% 源节点压力来自 Chen et al. (2022) Table V 描述：
%   source_1 = source_2 = 41.48 bar, source_3 = 42.16 bar
% 取三者均值作为统一初始压力
P0_SOURCE   = 41.5e5;    % Pa，~41.5 bar（原来70bar导致流量虚高10倍以上）
P0_OTHER    = 40.0e5;    % Pa，非源节点初始压力（略低于源压，给稳态求解提供合理起点）

% ── 1. 解析拓扑 (.net) ─────────────────────────────────────────────────
net = xmlread(net_file);

all_ids   = {};
all_type  = [];
all_initP = [];

node_tags = {'source','sink','innode'};
node_types = [1, 2, 2];

for ti = 1:3
    els = net.getElementsByTagName(node_tags{ti});
    for k = 0:els.getLength-1
        nd = els.item(k);
        all_ids{end+1}  = char(nd.getAttribute('id'));
        all_type(end+1) = node_types(ti);
        if node_types(ti) == 1
            all_initP(end+1) = P0_SOURCE;
        else
            all_initP(end+1) = P0_OTHER;
        end
    end
end

nN     = length(all_ids);
id2idx = containers.Map(all_ids, 1:nN);

% ── 2. 读取场景边界条件 (.scn) ──────────────────────────────────────────
scn       = xmlread(scn_file);
base_load = zeros(nN,1);

scn_nodes = scn.getElementsByTagName('node');
for k = 0:scn_nodes.getLength-1
    nd  = scn_nodes.item(k);
    nid = char(nd.getAttribute('id'));
    typ = char(nd.getAttribute('type'));
    if ~isKey(id2idx, nid), continue; end

    flow_el = nd.getElementsByTagName('flow');
    if flow_el.getLength == 0, continue; end
    flow_val = str2double(char(flow_el.item(0).getAttribute('value')));
    flow_kgs = flow_val * 1000 * NORM_DENSITY / 3600;

    idx = id2idx(nid);
    if strcmp(typ,'entry')
        base_load(idx) = -flow_kgs;
    else
        base_load(idx) =  flow_kgs;
    end
end

Nodes = table((1:nN)', all_type', base_load, all_initP', ...
    'VariableNames', {'ID','Type','BaseLoad','InitP'});

% ── 3. 解析管道 ─────────────────────────────────────────────────────────
pipes_xml = net.getElementsByTagName('pipe');
nP = pipes_xml.getLength;

pid_v  = zeros(nP,1);
from_v = zeros(nP,1);
to_v   = zeros(nP,1);
L_v    = zeros(nP,1);
D_v    = zeros(nP,1);
f_v    = zeros(nP,1);

for k = 0:nP-1
    pd   = pipes_xml.item(k);
    pnum = sscanf(char(pd.getAttribute('id')),'pipe_%d');

    from_id = char(pd.getAttribute('from'));
    to_id   = char(pd.getAttribute('to'));

    L_m  = str2double(char(pd.getElementsByTagName('length').item(0).getAttribute('value'))) * 1e3;
    D_m  = str2double(char(pd.getElementsByTagName('diameter').item(0).getAttribute('value'))) / 1e3;

    % Chen (2022) Section V-A: friction factor γ = 0.003 (constant for all pipes)
    f_dw = 0.003;

    pid_v(k+1)  = pnum;
    from_v(k+1) = id2idx(from_id);
    to_v(k+1)   = id2idx(to_id);
    L_v(k+1)    = L_m;
    D_v(k+1)    = D_m;
    f_v(k+1)    = f_dw;
end

Pipes = table(pid_v, from_v, to_v, L_v, D_v, f_v, ...
    'VariableNames', {'ID','From','To','L','D','Fric'});
[~,si] = sort(Pipes.ID);
Pipes  = Pipes(si,:);

% ── 4. 解析压缩机（追加为虚拟管道） ────────────────────────────────────
% 压比标定说明：
%   XML 中 pressureInMin/pressureOutMax 是允许范围（31~71 bar），不是运行值。
%   Chen (2022) 给出全网运行压力约 41.5 bar，相邻节点压差约 1~3 bar。
%   因此运行压比 ≈ (41.5+Δ)/41.5，Δ 取 1~3 bar，对应 ratio = 1.02~1.07。
%
%   comp_1: innode_6 → sink_25  （西部主分配中心，压差稍大）ratio = 1.05
%   comp_2: sink_11  → innode_1 （东部次级）                ratio = 1.05
%   comp_3: sink_19  → innode_2 （东北部）                  ratio = 1.05
%   comp_4: source_3 → innode_4 （气源出口，压差小）        ratio = 1.02
%   comp_5: source_2 → innode_7 （气源出口，压差小）        ratio = 1.02
%   comp_6: sink_3   → innode_8 （source_1 出口）           ratio = 1.02
COMP_RATIOS = [1.05, 1.05, 1.05, 1.02, 1.02, 1.02];

comp_xml      = net.getElementsByTagName('compressorStation');
nC            = comp_xml.getLength;
comp_pipe_ids = zeros(nC,1);
comp_ratios   = zeros(nC,1);
extra_rows    = table();

for k = 0:nC-1
    cd      = comp_xml.item(k);
    from_id = char(cd.getAttribute('from'));
    to_id   = char(cd.getAttribute('to'));

    d_in  = str2double(char(cd.getElementsByTagName('diameterIn').item(0).getAttribute('value')));
    d_out = str2double(char(cd.getElementsByTagName('diameterOut').item(0).getAttribute('value')));
    D_avg = (d_in + d_out) / 2 / 1e3;

    new_id             = nP + k + 1;
    comp_pipe_ids(k+1) = new_id;
    if k+1 <= length(COMP_RATIOS)
        comp_ratios(k+1) = COMP_RATIOS(k+1);
    else
        comp_ratios(k+1) = 1.05;
    end

    row = table(new_id, id2idx(from_id), id2idx(to_id), 100, D_avg, 0.012, ...
        'VariableNames', {'ID','From','To','L','D','Fric'});
    extra_rows = [extra_rows; row];
end

if ~isempty(extra_rows)
    Pipes = [Pipes; extra_rows];
end

Compressors = table(comp_pipe_ids, comp_ratios, 'VariableNames', {'PipeID','Ratio'});

% ── 5. GTU（GasLib-40 无 GTU） ──────────────────────────────────────────
GTU = table([],[],[],[],[],[],[],[], ...
    'VariableNames',{'ID','NodeID','Capacity_MW','HeatRate','MinLoad','MaxLoad','BaseFlow','Status'});

fprintf('GasLib-40 loaded: %d nodes, %d pipes (%d regular + %d compressor)\n', ...
    nN, height(Pipes), nP, nC);
fprintf('  Total supply: %.1f kg/s  |  Total demand: %.1f kg/s\n', ...
    -sum(base_load(base_load<0)), sum(base_load(base_load>0)));
end
