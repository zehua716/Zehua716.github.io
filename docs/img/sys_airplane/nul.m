set(0,'defaultfigurecolor','w')
%% 定义传递函数和开环系统分析
% 定义传递函数
s = tf('s');
P_pitch = (1.151*s + 0.1774)/(s^3 + 0.739*s^2 + 0.921*s);

% 绘制开环系统的阶跃响应
t = 0:0.01:10;  % 时间向量
figure;
step(P_pitch, t);
ylabel('俯仰角 (度)');
xlabel('时间 (秒)');
title('开环系统的阶跃响应');
grid on;  % 增加网格便于观察
set(gca, 'FontSize', 12); % 增大字体，便于阅读

% 检查传递函数的极点
disp('开环系统的极点为:');
open_loop_poles = pole(P_pitch);
disp(open_loop_poles);
%% 闭环系统分析
% 定义闭环系统（单位反馈）
sys_cl = feedback(P_pitch, 1);

% 绘制闭环系统的阶跃响应
t = 0:0.01:60;  % 使用更长时间范围分析系统收敛特性
figure;
step(sys_cl, t);
ylabel('俯仰角 (度)');
xlabel('时间 ');
title('闭环系统的阶跃响应');
grid on;
set(gca, 'FontSize', 12);

% 检查闭环系统的极点
disp('闭环系统的极点为:');
closed_loop_poles = pole(sys_cl);
disp(closed_loop_poles);
%% 分解闭环系统传递函数
% 定义阶跃输入 R(s) = 1/s
R = 1/s;

% 分解闭环输出 Y(s)
Y = zpk(sys_cl * R);
disp('闭环系统输出传递函数的零极点增益形式:');
disp(Y);
% 部分分数展开
[r, p, k] = residue([1.151 0.1774], [1 0.739 2.072 0.1774 0]);

disp('部分分数展开的残差 (零点分布):');
disp(r);
disp('部分分数展开的极点:');
disp(p);
disp('多项式余项 (通常为空):');
disp(k);

% 提取复共轭极点部分
[num, den] = residue(r(1:2), p(1:2), k);
disp('复共轭极点部分的分子与分母:');
disp('分子:');
disp(num);
disp('分母:');
disp(den);

% 确定系统分量 A, B, C, D
disp('部分分数展开各分量:');
disp(['A = 1']);
disp(['B = -0.881']);
disp(['C = -0.5605']);
disp(['D = -0.4036']);

%% 求解闭环系统的时域响应
% 求解闭环系统的时域响应
syms s t
F = 1/s - 0.881/(s + 0.08805) - (0.5605*s + 0.4036)/(s^2 + 0.6509*s + 2.015);
y_t = ilaplace(F);

disp('闭环系统的时域响应 (解析表达式):');
disp(vpa(y_t, 4));  % 显示高精度表达式

%% 绘制闭环系统的时域响应曲线
% 定义时间向量
t3 = 0:0.1:70;

% 时域响应的计算
yy = 1 - (1121*exp(-(6509*t3)/20000).*(cos((763632919^(1/2)*t3)/20000) + ...
    (8847411*763632919^(1/2)*sin((763632919^(1/2)*t3)/20000))/856032502199))/2000 - ...
    (881*exp(-(1761*t3)/20000))/1000;

% 绘制时域响应曲线
figure;
plot(t3, yy, 'LineWidth', 1.5); % 增加曲线宽度
xlabel('时间 (秒)');
ylabel('俯仰角 (度)');
title('闭环系统的时域响应曲线');
grid on;
set(gca, 'FontSize', 12);

% 标注响应特性
text(15, 0.8, ' 稳态响应近似为1', 'FontSize', 12, 'Color', 'red');

%% 分析补充
% 系统性能分析
disp('闭环系统的性能分析:');
step_info = stepinfo(sys_cl);  % 获取阶跃响应性能
disp(step_info);

% 检查闭环系统的稳定性
if all(real(closed_loop_poles) < 0)
    disp('闭环系统是稳定的 (所有极点实部均小于0)');
else
    disp('闭环系统是不稳定的 (存在极点实部大于等于0)');
end

%%
%% 根轨迹法 - 开环根轨迹设计
% 定义传递函数
s = tf('s');
P_pitch = (1.151*s + 0.1774)/(s^3 + 0.739*s^2 + 0.921*s);

% 打开控制系统设计器进行根轨迹分析
controlSystemDesigner('rlocus', P_pitch);

% 注释: 打开控制系统设计器后，可编辑补偿器C(s)并设计满足要求的区域。
% 设计中添加零点z和极点p的超前补偿器：
% C(s) = K * (s + z) / (s + p)
% 根据性能要求调整根轨迹和极点分布，观察输出响应。



%% 频域分析法 - Bode图与补偿器设计
% 定义传递函数
s = tf('s');
P_pitch = (1.151*s + 0.1774)/(s^3 + 0.739*s^2 + 0.921*s);

% 初始开环系统的Bode图与裕度分析
figure;
margin(P_pitch);
grid on;
title('开环系统的Bode图 (未补偿)');
xlabel('频率 ');
ylabel('幅度与相位');

% 设置增益K并观察频域响应
K = 10;  % 增益设置
figure;
margin(K * P_pitch);
grid on;
title('加增益后的Bode图 (K=10)');
xlabel('频率 ');
ylabel('幅度与相位');

% 定义超前补偿器参数
alpha = 0.1;  % 零点与极点的比值
T = 0.52;      % 时间常数

% 超前补偿器传递函数
C_lead = K * (T*s + 1) / (alpha*T*s + 1);

% 补偿后的开环传递函数
P_lead = C_lead * P_pitch;

% 绘制补偿后的Bode图
figure;
margin(P_lead);
grid on;
title('加入超前补偿器后的Bode图');
xlabel('频率 ');
ylabel('幅度与相位');

%% 频域分析法 - 闭环系统响应
% 补偿后的闭环系统
sys_cl_lead = feedback(P_lead, 1);

% 绘制闭环阶跃响应
figure;
step(sys_cl_lead, 0:0.01:10);
grid on;
title('超前补偿器设计的闭环阶跃响应');
ylabel('俯仰角 (度)');
xlabel('时间 (秒)');

% 获取闭环系统的性能指标
disp('超前补偿器设计后的闭环系统性能:');
stepinfo(sys_cl_lead);



%% 连续状态空间法 - 能控性与极点布置
% 定义连续状态空间模型
A = [-0.313 56.7 0; -0.0139 -0.426 0; 0 56.7 0];
B = [0.232; 0.0203; 0];
C = [0 0 1];
D = [0];

% 检查系统能控性
co = ctrb(A, B); % 能控性矩阵
Controllability = rank(co);
disp(['连续系统能控性: Rank = ', num2str(Controllability)]);

%% 连续状态空间法 - LQR设计
% 定义LQR权重矩阵
p = 50; % 状态加权因子
Q = p * C' * C; % 状态成本加权矩阵
R = 1;          % 控制权重矩阵

% 求解状态反馈增益矩阵K
[K] = lqr(A, B, Q, R);
disp('LQR设计的反馈增益矩阵K:');
disp(K);

% 构造闭环系统
sys_cl = ss(A - B * K, B, C, D);
figure;
step(sys_cl);
grid on;
title('连续状态空间下的LQR闭环阶跃响应');
xlabel('时间 ');
ylabel('俯仰角 (度)');

%% 连续状态空间法 - 添加预补偿器
% 计算预补偿器Nbar
Nbar = rscale(A, B, C, D, K); % 官方函数 rscale.m
disp(['预补偿器比例因子Nbar = ', num2str(Nbar)]);

% 构造加入预补偿器后的闭环系统
sys_cl_comp = ss(A - B * K, B * Nbar, C, D);
figure;
step(sys_cl_comp);
grid on;
title('加入预补偿器后的LQR闭环阶跃响应');
xlabel('时间 ');
ylabel('俯仰角 (度)');

%% 离散状态空间法 - 离散化与能控性检查
% 将连续系统离散化
Ts = 0.01; % 采样时间 (秒)
sys_d = c2d(ss(A, B, C, D), Ts, 'zoh');

% 检查离散系统能控性
co_d = ctrb(sys_d.A, sys_d.B);
Controllability_d = rank(co_d);
disp(['离散系统能控性: Rank = ', num2str(Controllability_d)]);

%% 离散状态空间法 - LQR设计
% LQR离散版本
p = 50; % 状态加权因子
Q_d = p * sys_d.C' * sys_d.C; % 离散状态成本加权矩阵
R_d = 1;                     % 控制权重矩阵

% 求解离散状态反馈增益矩阵K
[K_d] = dlqr(sys_d.A, sys_d.B, Q_d, R_d);
disp('离散LQR设计的反馈增益矩阵K:');
disp(K_d);

% 构造离散闭环系统
sys_cl_d = ss(sys_d.A - sys_d.B * K_d, sys_d.B, sys_d.C, sys_d.D, Ts);
time = 0:Ts:10; % 仿真时间
theta_des = ones(size(time)); % 阶跃输入
[y, t] = lsim(sys_cl_d, theta_des, time);

% 绘制离散闭环系统的响应
figure;
stairs(t, y);
grid on;
title('离散LQR下的闭环阶跃响应');
xlabel('时间 (秒)');
ylabel('俯仰角 (度)');

%% 离散状态空间法 - 添加预补偿器
% 手动调整预补偿器比例因子
Nbar_d = 6.95; % 根据仿真调整的比例因子
disp(['离散系统预补偿器比例因子Nbar = ', num2str(Nbar_d)]);

% 构造加入预补偿器后的离散闭环系统
sys_cl_comp_d = ss(sys_d.A - sys_d.B * K_d, sys_d.B * Nbar_d, sys_d.C, sys_d.D, Ts);
[y_comp, t_comp] = lsim(sys_cl_comp_d, theta_des, time);

% 绘制加入预补偿器后的离散闭环响应
figure;
stairs(t_comp, y_comp);
grid on;
title('加入预补偿器后的离散LQR闭环阶跃响应');
xlabel('时间 (秒)');
ylabel('俯仰角 (度)');













