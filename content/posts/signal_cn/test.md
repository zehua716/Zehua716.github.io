---
title: "反问题 TP1"
author: "Zehua"
date: "2024-11-05T16:25:17+01:00"
lastmod: "2024-11-06T17:12:35+08:00"
draft: false
lang: "zh"
summary: "摘要"
description: "测试"
tags: ["信号处理", "图像处理"]
# categories: "posts"
# cover:
#     image: "images/.jpg"
# comments: true
# hideMeta: false
# searchHidden: false
# ShowBreadCrumbs: true
# ShowReadingTime: false
---



**图像反卷积：Wiener-Hunt 方法**

主要针对图像去模糊问题，即从模糊(带噪)图像中恢复清晰图像。这属于逆问题的范畴，一半出现在处理真实测量系统时。由于每个测量系统（如温度计、CCD相机、光谱仪等）都受到基础物理学的限制，比如有限精度、有限动态范围、非零响应时间等。这意味着测得的量或多或少都有扭曲。因此这部分是对感兴趣物理量的某种扭曲程度的度量。

大多数情况下，测量系统直接给出的测量数据通常具有足够的精度和鲁棒性。但是，也存在测量结果不准确的情况。为了解决精度问题，或者说至少部分地优化它，已经开发了特殊的信号和图像处理技术。在接下来的内容中，我们将通过一个简单的例子来展示此类方法。

我们有一张未聚焦的图像。这种情况下，点的图像实际上会是一个斑点。捕获的图像将会因为模糊而受损，因为它是由真实图像中每个点生成的斑点的叠加结果。

描述这种转换的最简单模型是线性不变滤波器，即卷积。示意图如下图所示。

在上面的示意图中， $x_{n,m}$ 代表真实或原始图像，$y_{n,m}$ 代表观测到的图像，或者更确切地说，是我们通过相机拍到的未聚焦图像。添加分量 $b_{n,m}$ 是为了考虑测量和建模误差。

描述测量过程的方程(二维)如下：


$$
y_{n,m} = \sum_{p=-P}^{P} \sum_{q=-Q}^{Q} h_{p,q} x_{n-p,m-q} + b_{n,m}
$$


$y_{n,m}$ 是对于每个观测到的像素 $(n, m)$ 。在这个公式中，$P$ 和 $Q$ 是给定的整数。

注意，滤波器通常来说都是低通滤波器，这就意味着它们无法准确地在输出中再现输入信号或图像中的所有分量，因为高频分量要么被强烈衰减，要么完全被拒绝，这也就是为什么 ‘‘恢复真实信号’‘或者说’‘图像的逆问题’’ 是如此困难：必须恢复那些要么完全不存在、要么“错误”观测到的高频分量。

在下面的例子中，我们用线性方法来解决图像反卷积问题。这些线性方法依赖于最小二乘准则，并结合了二次惩罚。我们先介绍其理论部分，包括这些准则及其最小化器。此外，展示背后的技术细节，并提出了一种基于循环近似的方法，以实现快速的数值计算。

**1. 一维反卷积**

为了简化理论概念，我们先讨论在一维情况下的信号反卷积。这种简化情况允许对反卷积问题的分析更加深入，同时更容易掌握概念和思路。随后再引入二维情况，并将其视为一维情况的扩展。Matlab 实现部分仅涉及二维情况。

**1.1 一维建模**

在一维情况下，(1) 中给出的观测模型变为：


$$
y_n = \sum_{p=-P}^{P} h_p x_{n-p} + b_n
$$


如果我们有 $N$ 个样本，可以将相应的 $N$ 个方程写成矩阵形式：


$$
\mathbf{y} = \mathbf{H} \mathbf{x} + \mathbf{b}
$$


​	•	向量 $\mathbf{y}$ 包含了所有的 $N$ 个观测值（在二维情况下，它将包含模糊的图像）

​	•	向量 $\mathbf{x}$ 包含了恢复图像的样本，而 $\mathbf{b}$ 是噪声样本。

​	•	矩阵 $\mathbf{H}$ ，称为模糊矩阵，具有以下经典结构：


$$
H = \begin{bmatrix}
h_P & \cdots & h_0 & \cdots & h_{-P} & 0 & 0 & 0 & \cdots \\\\
0 & h_P & \cdots & h_0 & \cdots & h_{-P} & 0 & 0 & \cdots \\\\
0 & 0 & h_P & \cdots & h_0 & \cdots & h_{-P} & 0 & \cdots \\\\
0 & 0 & 0 & h_P & \cdots & h_0 & \cdots & h_{-P} & \cdots \\\\
\cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots
\end{bmatrix}
$$


​	•	$\mathbf{H}$ 是一个 $N \times N$ 的方阵，并且具有 Toeplitz 结构。

因此，信号反卷积问题可以重新表述为: 在已知观测信号 $\mathbf{y}$ 并知道卷积矩阵 $H$ 的情况下，估计向量 $\mathbf{x}$

**1.2 带惩罚的最小二乘法**

提出的重建策略 (损失函数) 是一种带惩罚的最小二乘法。它包含两个部分：

​	•	一个重构损失项，用于量化恢复信号 $\mathbf{x}$ 与观测信号 $\mathbf{y}$ 进行重新卷积后的相似性，从而确保恢复的信号与观测信号一致。

​	•	一个惩罚项，用于限制恢复信号的连续样本之间的差异，确保其具有一定的规则性结构。

该准则采用以下表达式：


$$
J_{\text{PLS}}(x) = | y - Hx |^2 + \mu | Dx |^2 = (y - Hx)^t (y - Hx) + \mu x^t D^t D x
$$


其中，$D$ 是阶数为 1，大小为 $(N - 1) \times N$ 的差分矩阵，定义如下：


$$
D = \begin{bmatrix}
\cdots & -1 & 1 & 0 & 0 & \cdots \\\\
\cdots & 0 & -1 & 1 & 0 & \cdots \\\\
\cdots & \cdots & \cdots & \cdots & \cdots & \cdots \\\\
\cdots & 0 & 0 & -1 & 1 & \cdots\\\\
\end{bmatrix}
$$


带惩罚的最小二乘准则的最小化器为：


$$
\hat{x} = (H^t H + \mu D^t D)^{-1} H^t y
$$


证明:


$$
\hat{x} = (H^T H + \mu D^T D)^{-1} H^T y
$$




我们已知损失准则为:




$$
J_{\text{PLS}}(x) = y^T y - 2x^T H^T y + x^T H^T H x + \mu x^T D^T D x
$$




求偏导为0:




$$
\hat{x} = (H^T H + \mu D^T D)^{-1} H^T y
$$




特殊情况：



在特定情况下 $\mu = 0$ 时准则和最小化器的结果表达式变为 $J_{\text{PLS}}(x) = |y - Hx|^2$



最小化器方程为：




$$
\hat{x} = (H^T H)^{-1} H^T y
$$




此时准则变为经典的最小二乘问题，没有正则化项，也就是说，模型仅考虑最小化观测值与预测值之间的误差，而不会惩罚解的复杂度或平滑性。其解是经典的 **最小二乘解**。



**1.2.1 循环近似**



回到公式 (9) ，我们已知矩阵 $H^T H + \mu D^T D$ 的大小为 $N \times N$ ，当 $N$ 很大时，这种反演在计算上非常昂贵甚至不可行。



比如说在处理图像时，比如对于 $1000 \times 1000$ 的图像，矩阵的大小为 $10^6 \times 10^6$ ，计算不了。在三维情况下，更复杂。



因此为了计算 $\hat{x}$ ，有几种方法可以克服这种大计算量的困难。



下面我们考虑使用循环矩阵的特性，因为我们可以用对角矩阵来 “替换” 公式 (9) 中的矩阵 (通过快速傅里叶变换 FFT 可以将循环矩阵 “转化”为对角矩阵)。因此，使用对角矩阵进行计算时，乘法或反演等矩阵运算的复杂度将大大降低。



但是，这需要先将矩阵 $H$ 和 $D$ 近似为循环矩阵 $\tilde{H}$ 和 $\tilde{D}$ 。



循环近似涉及修改矩阵的右上角和/或左下角部分，使其具有循环结构。这种近似的核心假设是信号或图像在开始和结束部分是周期性的，即信号的末尾与开头相连接，形成一个环状结构。



循环卷积矩阵 $\tilde{H}$ 和 $\tilde{D}$ 在傅里叶基下可以轻松对角化：




$$
\tilde{H} = F^T \Lambda_h F \quad \text{和} \quad \tilde{D} = F^T \Lambda_d F
$$




矩阵 $\Lambda_h$ 是对角矩阵，其对角线上元素是 $H$ 的特征值。



特征值可以通过对矩阵 $H$ 第一行进行快速傅里叶变换（FFT）获得，即计算脉冲响应的 $N$ 点 FFT，这些响应代表频率响应的样本。



同样适用于矩阵 $\tilde{D}$ 及其特征值，其中将脉冲响应替换为 $[-1, 1]$。



通过在 (9) 中用 (18) 代替，并使用简单的矩阵操作，可以得到：




$$
\overset{\circ}{\hat{x}} = (\Lambda_{h}^{\dagger} \Lambda_{h} + \mu \Lambda_{d}^{\dagger} \Lambda_{d})^{-1} \Lambda_{h}^{\dagger} \overset{\circ}{y}
$$




在此基础上再进一步:




$$
\overset{\circ}{\hat{x}} = \left[ \Lambda_h^{\dagger} \Lambda_h + \mu \left( \Lambda_{d_c}^{\dagger} \Lambda_{d_c} + \Lambda_{d_r}^{\dagger} \Lambda_{d_r} \right) \right] \Lambda_h^{\dagger} \dot{\mathbf{y}}
$$




```matlab
FT_x = 1 ./ (abs(Lambda_H).^2 + mu * (abs(Lambda_dC).^2 + abs(Lambda_dR).^2)).* (conj(Lambda_H).* FT_Data);
```





$\Lambda_h \Rightarrow$ FT_IR = MyFFT2RI(IR, 256);



$\Lambda_{d_c} \Rightarrow$ FT_Dc = MyFFT2RI(Dc, 256);



$\Lambda_{d_r} \Rightarrow$ FT_Dr = MyFFT2RI(Dr, 256);



回顾性质:




$$
\Lambda_H^{+} = (\Lambda_H^T)^{} = \Lambda_H^{} = \operatorname{conj}(\Lambda_H)
$$




因此有:




$$
\Lambda_H^{+} \cdot \Lambda_H = \Lambda_H^{*} \cdot \Lambda_H = |\Lambda_H|^2
$$




证明：




$$
\overset{\circ}{\hat{x}} = (\Lambda_{h}^{\dagger} \Lambda_{h} + \mu \Lambda_{d}^{\dagger} \Lambda_{d})^{-1} \Lambda_{h}^{\dagger} \overset{\circ}{y}
$$




补充知识: $\tilde{H}$ 是一个实矩阵，因此其复共轭等于它本身，$\tilde{H} = \tilde{H}^T$ 。$\tilde{D}$ 同理



在近似为循环矩阵后，公式 (15) 变为:




$$
\hat{x} = (\tilde{H}^{T}\tilde{H} + \mu \tilde{D}^{T}\tilde{D})^{-1}\tilde{H}^{T}y
$$




其中: $\tilde{H} = F^T \Lambda_h F$ 且 $\tilde{D} = F^T \Lambda_d F$ 将其带入上述公式，得:




$$
\hat{x} = ((F^{T}\Lambda_{h}F)^{T}(F^{T}\Lambda_{h}F) + \mu (F^{T}\Lambda_{d}F)^{T}(F^{T}\Lambda_{d}F))^{-1}(F^{T}\Lambda_{h}F)^{T}y
$$




因为傅里叶矩阵 $F$ 是一个正交矩阵，具有 $F F^{T} = I$ 的性质，即 $F^{T} = F^{-1}$



并且我们有 $(F^{T}\Lambda_{h}F)^{T} = F^{T}\Lambda_{h}^{T}F$ 以及 $(F^{T}\Lambda_{d}F)^{T} = F^{T}\Lambda_{d}^{T}F$



所以:



令 $\quad \overset{\circ}{\hat{x}} = F \hat{x}, \quad \overset{\circ}{y} = F y$



那么:




$$
\overset{\circ}{\hat{x}} = (\Lambda_{h}^{\dagger} \Lambda_{h} + \mu \Lambda_{d}^{\dagger} \Lambda_{d})^{-1} \Lambda_{h}^{\dagger} \overset{\circ}{y}
$$




证毕。



特殊情况：



当 $\mu = 0$ 时，正则化项消失，公式简化为：




$$
\overset{\circ}{\hat{x}} = (\Lambda_{h}^{\dagger} \Lambda_{h})^{-1} \Lambda_{h}^{\dagger} \overset{\circ}{y}
$$




这意味着我们仅仅执行了经典的 Wiener 去卷积，没有考虑图像的正则化。



$$\hat{x} = (\tilde{H}^{T}\tilde{H} )^{-1}\tilde{H}^{T}y\ $$



 



没有正则化时，虽然理论上可以恢复原始信号，但是实际上收到的噪声影响很大，且没有约束来将其消除



为了完成我们的讨论，我们首先构建向量 $g_{\text{PLS}}$ ，其分量定义如下：


$$
g_{PLS}^{n} = \frac{\overset{\circ}{h}_{n}^{*}}{|\overset{\circ}h_n|^2 + \mu |\overset{\circ}d_n|^2}\quad \text{for } n = 1, 2, \dots, N
$$


因此，向量 $\overset{\circ}{\hat{x}}$ 是通过向量 $g_{\text{PLS}}$ 和 $\overset{\circ}{\hat{y}}$ 之间逐元素相乘得到的：




$$
\overset{\circ}{\hat{x}} = g_{\text{PLS}} .* \overset{\circ}{\hat{y}}
$$




反卷积问题可以表述为在傅里叶域中进行的滤波操作，其中 $g_{\text{PLS}}$ 代表离散传递函数。



反卷积问题总结如下：

​	•	① 构建 $\mathring{h}$ 作为脉冲响应的 $N$ 点 FFT

​	•	② 构建 $\mathring{d}$ 作为 $[1; -1]$ 的 $N$ 点 FFT

​	•	③ 构建包含传递函数 $g_{\text{PLS}}$ 的向量

​	•	④ 构建观测值的 FFT $\mathring{y}$

​	•	⑤ 计算 $\mathring{\hat{x}}$ ，作为传递函数 $g_{\text{PLS}}$ 和 $\mathring{\hat{y}}$ 的乘积

​	•	⑥ 计算 $\mathring{\hat{x}}$ 的 IFFT 以在空间域中获得解 $\hat{x}$



**2 实现**



**2.1 二维方法**



对于二维情况，其方程类似于一维情况。但是，所涉及的block-Tœplitz矩阵结构更加复杂: 每个块本身也具有 Tœplitz 结构。这使得在两个方向上进行循环近似变得更加困难。因此，二维情况仅作为一维情况的扩展来展示，重点是 Matlab 实现，理论部分暂不讨论。



注意以下几点：

​	•	图像、脉冲响应、正则化项都是二维的，这意味着必须使用 FFT-2D 而不是 FFT。

​	•	更确切地说，如果要恢复的图像有 $N$ 行和 $N$ 列，那么 FFT-2D 必须在 $N$ 行和 $N$ 列上计算。

​	•	频率传递函数也是二维的，每个空间频率有一个维度。



特别说明 — 在任何情况下，矩阵 $H$ 和 $D$ 都不应在 Matlab 代码中构建。



**2.2 观测图像**



​	•	第一步是加载数据 **Data1** 和 **Data2** 。使用 **load** 函数来完成。每个数据文件内部包含：模糊图像 (**Data**)、用于比较的真实图像 (**TrueIma**)，以及卷积滤波器的脉冲响应 (**IR**)。现在分析每个数据集及其相互关系。



```matlab
clear all 

close all

clc

%% Load Data

Data1 = load('DataOne.mat')

Data2 = load('DataTwo.mat')

% Create a window to display the image

figure()

subplot(1,2,1)

% Display the image, grayscale

imagesc(Data1.Data);

colormap('gray');

% Scale the axes and eliminate the grading

axis('square','off')

title('Observed Image - Data1');



subplot(1,2,2)

% Display the image, grayscale

imagesc(Data2.Data);

colormap('gray');

% Scale the axes and eliminate the grading

axis('square','off')

title('Observed Image - Data2');
```

![TP1_1](/img/Problem_inverse/TP1/TP1_1.png)

两观测图像都有明显的模糊，具体是边缘和结构细节处。这种模糊表明图像中的高频信息被卷积或散射了，导致图像细节的丢失，这就是由于卷积效应或低通滤波的影响而造成的。

​	•	为了进一步分析，肯定要到频域看

```matlab
%% Analyse the images in the frequency domain (linear and a logarithmic scale)

% % Get the sizes of the images

% [M1,N1] = size(Data1.Data);

% [M2,N2] = size(Data2.Data);

% % Generate frequency axes for Data1

% u1 = (-M1/2:M1/2-1)/M1; % Normalized frequencies in y-direction

% v1 = (-N1/2:N1/2-1)/N1; % Normalized frequencies in x-direction

% % Generate frequency axes for Data2

% u2 = (-M2/2:M2/2-1)/M2;

% v2 = (-N2/2:N2/2-1)/N2;



% Generate normalized frequency axes using linspace

Nu = linspace(-0.5, 0.5, 256);



% Compute the 2D FFT of observed images and shift zero frequency to center

FFT_Data1 = MyFFT2(Data1.Data);

FFT_Data2 = MyFFT2(Data2.Data);

% Magnitude spectra

Mag_Data1 = abs(FFT_Data1)

Mag_Data2 = abs(FFT_Data2)

% Display magnitude spectra in linear scales

figure();

subplot(1,2,1)

imagesc(Nu,Nu,Mag_Data1); % Use frequency axes v1 and u1

xlabel('X');

ylabel('Y');

colormap('gray');

title('Magnitude (Linear Scale) - Data1');



subplot(1,2,2)

imagesc(Nu,Nu,Mag_Data2); % Use frequency axes v2 and u2

xlabel('X');

ylabel('Y');

colormap('gray');

title('Magnitude (Linear Scale) - Data2');



% Display magnitude spectra in logarithmic scales

figure;

subplot(1,2,1)

imagesc(Nu,Nu,log(1+Mag_Data1)); % Use log scale



xlabel('X');

ylabel('Y');

colormap('gray');

title('Magnitude (logarithmic Scale) - Data1');



subplot(1,2,2)

imagesc(Nu,Nu,log(1+Mag_Data2));

xlabel('X');

ylabel('Y');

colormap('gray');

title('Magnitude (logarithmic Scale) - Data2');


```

可见只有两组频谱图的中心部分有强烈亮度，即低频分量较强，即图像大范围平滑信息较多，而高频部分的细节没有

​	•	我们继续看两个脉冲响应 $h_{n,m}$ 及其关联的传递函数 $H(\nu_x, \nu_y)$ 。首先使用 imagesc 函数，然后使用 plot 函数来进行分析。



```matlab
%% Calcul impulse responses and associated transfer functions using impulse

% Load the impulse responses

IR1 = Data1.IR;

IR2 = Data2.IR;



% Display the impulse responses

figure;

subplot(1,2,1)

imagesc(IR1);

colormap('gray');

title('Impulse Response - IR1');



subplot(1,2,2)

imagesc(IR2);

colormap('gray');

title('Impulse Response - IR2');



% Compute and display the transfer functions

Long = 256;

H1 = MyFFT2RI(IR1, Long);

H2 = MyFFT2RI(IR2, Long);



% Compute the magnitude spectra

Mag_H1 = abs(H1);

Mag_H2 = abs(H2);



% Display the transfer functions using imagesc

figure;

subplot(1,2,1)

imagesc(Nu, Nu, Mag_H1);

xlabel('X');

ylabel('Y');

title('Transfer Function Magnitude - H1');

colormap('gray');



subplot(1,2,2)

imagesc(Nu, Nu, Mag_H2);

xlabel('X');

ylabel('Y');

title('Transfer Function Magnitude - H2');

colormap('gray');
```



很明显，IR1 和 IR2 都是低通滤波器，只允许低频分量通过，高频分量进行衰减或完全抑制。IR1 的传递函数 H1 的分布更加平滑，说明它对低频分量的保留更为均匀。IR2 的滤波特性更倾向于选择性地保留某些低频分量，而非均匀地平滑整个低频区域。因此，相较于 IR1，IR2 对图像的细节影响更强。不过这么看看不出来具体的道道儿，还是得看下面的切片。

```matlab
%% Plot slices through the transfer functions using plot

% Midpoint index (since Long = 256)

mid_index = Long / 2 + 1; % This will be 129



% Plot slices for H1

figure;

subplot(2,1,1)

plot(Nu, Mag_H1(mid_index, :));

xlabel('Normalized Frequency X');

ylabel('Magnitude');

title('Transfer Function along Central Row - H1');

grid on;



subplot(2,1,2)

plot(Nu, Mag_H1(:, mid_index));

xlabel('Normalized Frequency Y');

ylabel('Magnitude');

title('Transfer Function along Central Column - H1');

grid on;


```

可见前面的结论是正确的，H1 幅值响应曲线变化相对平滑，适合图像整体的模糊处理任务。



```matlab
% Plot slices for H2

figure;

subplot(2,1,1)

plot(Nu, Mag_H2(mid_index, :));

xlabel('Normalized Frequency X');

ylabel('Magnitude');

title('Transfer Function along Central Row - H2');

grid on;



subplot(2,1,2)

plot(Nu, Mag_H2(:, mid_index));

xlabel('Normalized Frequency Y');

ylabel('Magnitude');

title('Transfer Function along Central Column - H2');

grid on;
```



在 H2 的切片图中，幅值响应曲线有更明显的周期性波动，适合特定方向的模糊或增强效果。



**2.3 Implementation**

我们将在二维情况下实现反卷积，并使用带有二次惩罚项的最小二乘法，同时使用循环近似进行最小化，在前面已经进行过了总结。



关于正则化项，它依赖于图像列和行上相邻像素之间的差异。其表达式为：




$$
| D x |^2 = \sum_{n,m} (x_{n,m} - x_{n,m+1})^2 + (x_{n,m} - x_{n+1,m})^2
$$


​	•	正则化项 $| D x |^2$ 代表了水平方向和垂直方向相邻像素值的平方差之和。



因此它将基于两个滤波器(水平和垂直)来实现：




$$
D_{horiz} = \begin{bmatrix} 0 & 0 & 0 \\\\ 0 & -1 & 1 \\\\ 0 & 0 & 0 \end{bmatrix} \quad D_{vert} = \begin{bmatrix} 0 & 0 & 0 \\\\ 0 & -1 & 0 \\\\ 0 & 1 & 0 \end{bmatrix}
$$




这两个滤波器分别计算行和列上像素之间的差异。



当然也可以从以下脉冲响应滤波器中选一个，它们都可以实现图像梯度的近似。




$$
\begin{bmatrix}
0 & -1 & 0 \\\\
-1 & 4 & -1 \\\\
0 & -1 & 0
\end{bmatrix}
\quad \text{或} \quad
\begin{bmatrix}
-1 & -1 & -1 \\\\
-1 & 8 & -1 \\\\
-1 & -1 & -1
\end{bmatrix}
\quad \text{或} \quad
\begin{bmatrix}
1 & -2 & 1 \\\\
-2 & 4 & -2 \\\\
1 & -2 & 1
\end{bmatrix}
$$




回顾前面，我们得到：




$$
\hat{x} = (H^{T}H + \mu D^{T}D)^{-1}H^{T}y
$$




现在进行傅里叶变换下的去卷积




$$
\hat{x}(\nu_{x}, \nu_{y}) = \frac{\hat{H}^{*}(\nu_{x}, \nu_{y}) \hat{y}(\nu_{x}, \nu_{y})}{|\hat{H}(\nu_{x}, \nu_{y})|^2 + \mu |\hat{D}(\nu_{x}, \nu_{y})|^2}
$$


​	•	$\hat{H}(\nu_{x}, \nu_{y})$ 是卷积矩阵的傅里叶变换

​	•	$\hat{D}(\nu_{x}, \nu_{y})$ 是差分矩阵的傅里叶变换

​	•	$\hat{x}(\nu_{x}, \nu_{y})$ 是频域中的恢复图像



我们先写一个反卷积函数，将观测数据 (Data)、脉冲响应 (IR) 和正则化参数 (mu) 作为输入。



```matlab
function [x] = deconvolve_image(Data,IR,mu)



  Long = length(Data);

  Nu = linspace(-0.5, 0.5, Long);



  TF_Data = MyFFT2(Data);



  TF_IR = MyFFT2RI(IR,Long);

   

  % figure(1)

  % subplot(2, 1, 1)

  % imagesc(Nu, Nu, abs(TF_IR))

  % title('Frequency response')

  % xlabel('\nu_x')

  % ylabel('\nu_y')

  % axis square

  % subplot(2, 1, 2)

  % plot(Nu, abs(TF_IR(round(length(TF_IR)/2), :)))



  % Define the regularization filters (difference operators)

  % Horizontal difference filter

  Dh = [0 0 0; 0 -1 1; 0 0 0];



  % Vertical difference filter

  Dv = [0 0 0; 0 -1 0; 0 1 0];



  % Compute the FFTs of the regularization filters using MyFFT2RI

  TF_Dh = MyFFT2RI(Dh, Long);

  TF_Dv = MyFFT2RI(Dv, Long);

   



  % Compute |Dh|^2 and |Dv|^2

  abs_Dh_squared = abs(TF_Dh).^2;

  abs_Dv_squared = abs(TF_Dv).^2;



  % Total regularization term |D|^2 = |Dh|^2 + |Dv|^2

  abs_D_squared = abs_Dh_squared + abs_Dv_squared;



  % Compute the denominator of the Wiener filter

  denom = abs(TF_IR).^2 + mu * abs_D_squared;



  % Compute the numerator

  numerator = conj(TF_IR) .* TF_Data;



  % Compute X_hat in the frequency domain

  TF_X = numerator ./ denom;



  % Compute the inverse FFT to get the deconvolved image

  % Since MyFFT2 uses fftshift, we need to use ifftshift before ifft2

  x = MyIFFT2(TF_X);



end
```



​	•	然后应用这个逆卷积函数，查看反卷积后的图像去噪效果。



```matlab
%% Question 5

mu = 0.004;

% Deconvolve Data1

x_1 = deconvolve_image(Data1.Data, Data1.IR, mu);

% Display the deconvolved image

figure;

subplot(1,2,1)

% Display the image, grayscale

imagesc(Data1.Data);

colormap('gray');

% Scale the axes and eliminate the grading

axis('square','off')

title(['Observed Image - Data1, \mu = ', num2str(mu)]);

subplot(1,2,2)

imagesc(x_1);

colormap('gray');

title(['Deconvolved Image - Data1, \mu = ', num2str(mu)]);

axis('square','off')





% Deconvolve Data2

x_2 = deconvolve_image(Data2.Data, Data2.IR, mu);

% Display the deconvolved image

figure;

subplot(1,2,1)

% Display the image, grayscale

imagesc(Data2.Data);

colormap('gray');

% Scale the axes and eliminate the grading

axis('square','off')

title(['Observed Image - Data2, \mu = ', num2str(mu)]);



subplot(1,2,2)

imagesc(x_2);

colormap('gray');

title(['Observed Image - Data2, \mu = ', num2str(mu)]);

axis('square','off')
```



可见效果挺理想，但是之前这个 $\mu$ 是随便设的，现在探讨这个 $\mu$ 值对结果的影响。

​	•	首先考虑简单的逆滤波器情况，即 $\mu = 0$ 。



前面我们得到： $\hat{x} = (H^{T}H + \mu D^{T}D)^{-1}H^{T}y$



现在变成： $\hat{x} = (H^{T}H)^{-1}H^{T}y$




$$
\hat{x}(\nu_{x}, \nu_{y}) = \frac{\hat{H}^{*}(\nu_{x}, \nu_{y}) \hat{y}(\nu_{x}, \nu_{y})}{|\hat{H}(\nu_{x}, \nu_{y})|^2 }
$$




```matlab
%% Question 6  Simple Inverse Filter (μ = 0)

mu = 0;

% Deconvolve Data1

x_1 = deconvolve_image(Data1.Data, Data1.IR, mu);

% Display the deconvolved image

figure;

subplot(1,2,1)

% Display the image, grayscale

imagesc(Data1.Data);

colormap('gray');

% Scale the axes and eliminate the grading

axis('square','off')

title(['Observed Image - Data1, \mu = ', num2str(mu)]);

subplot(1,2,2)

imagesc(x_1);

colormap('gray');

title(['Deconvolved Image - Data1, \mu = ', num2str(mu)]);

axis('square','off')
```



由于 $\mu = 0$ 等价于丢失了正则化项，反卷积结果会直接依赖于卷积核 $H$ 的傅里叶系数 $|\hat{H}(\nu_{x}, \nu_{y})|^2$ ，如果 $H$ 的某些频率分量（尤其是高频）接近零，那么在这些频率上，分母 $|\hat{H}(\nu_{x}, \nu_{y})|^2$ 会非常小，导致反卷积结果中放大这些频率分量的噪声。反卷积图像中可以看到明显的颗粒状噪声。这是由于直接逆滤波在某些频率分量上产生了极大值，放大了噪声。



```matlab
% Deconvolve Data2

x_2 = deconvolve_image(Data2.Data, Data2.IR, mu);

% Display the deconvolved image

figure;

subplot(1,2,1)

% Display the image, grayscale

imagesc(Data2.Data);

colormap('gray');

% Scale the axes and eliminate the grading

axis('square','off')

title(['Observed Image - Data2, \mu = ', num2str(mu)]);



subplot(1,2,2)

imagesc(x_2);

colormap('gray');

title(['Observed Image - Data2, \mu = ', num2str(mu)]);

axis('square','off')
```



​	•	我们因此取不同的 $\mu$ 值(在 $\log_{10}$ 刻度上取值)。并且根据结果，确定合适的 $\mu$ 值。



```matlab
%% Effect of Varying μ on Deconvolution Results

% Define a range of mu values on a log scale

mu_values = logspace(-11, 0, 12); % From 1e-10 to 1

% Deconvolution and display for Data1

figure('Name', 'Deconvolution Results for Data1');

for i = 1:length(mu_values)

  mu = mu_values(i);

  x_1 = deconvolve_image(Data1.Data, Data1.IR, mu);

   

  subplot(3,4,i);

  imagesc(x_1);

  colormap('gray');

  axis('square','off');

  title(['\mu = ', num2str(mu, '%.1e')]);

end

sgtitle('Deconvolved Images - Data1 with Different \mu Values');



% Deconvolution and display for Data2

figure('Name', 'Deconvolution Results for Data2');

for i = 1:length(mu_values)

  mu = mu_values(i);

  x_2 = deconvolve_image(Data2.Data, Data2.IR, mu);

   

  subplot(3,4,i);

  imagesc(x_2);

  colormap('gray');

  axis('square','off');

  title(['\mu = ', num2str(mu, '%.1e')]);

end

sgtitle('Deconvolved Images - Data2 with Different \mu Values');


```

小的 $\mu$ 值时，可见图像噪声多，甚至都有可能完全被噪声埋没; 中等 $\mu$ 值 时，图像细节和噪声之间达到了较好的平衡; 当 $\mu$ 值过大时，图像变得过于平滑，细节逐渐丢失，尤其是当 $\mu = 1$ 时。我们目前只能通过视觉方法来选出较好的 $\mu$ 值，两数据最优 $\mu$ 值都大约等于 0.01，但是这样肯定是不行的，是不严谨的，是无法成为一名科研奇才的，因此我们和真实图像做数学比较。



2.4 超参数的作用



上一点使我们能够评估与反卷积问题相关的内在难度。它还表明，考虑重建图像期望正则性的先验信息可以获得更好的结果。这种方法因此使我们能够在两种信息源之间进行折衷：观测数据和可用的先验信息（关于正则性）。这是通过参数 $\mu$ 的值来实现的。在下面的研究中，选择一个最合适的 $\mu$ 值，以使反卷积图像既不过于平滑也不过于不规则。



因此，可以计算反卷积图像 $\hat{x}$ 和真实图像 $x^\star$ 之间的数值差异，作为正则化参数 $\mu$ 的函数。



为此，考虑以下三种距离函数：







<div>
  $$\Delta_2(\mu) = \frac{\sum_{p,q} (\hat{x}_{p,q}(\mu) - x^{\star}_{p,q})^2}{\sum_{p,q} (x^{\star}_{p,q})^2} 
  = \frac{\|\hat{x}(\mu) - x^{\star}\|_2^2}{\|x^{\star}\|_2^2}$$
</div>
<div>
  $$\Delta_1(\mu) = \frac{\sum_{p,q} |\hat{x}_{p,q}(\mu) - x^{\star}_{p,q}|}{\sum_{p,q} |x^{\star}_{p,q}|} 
  = \frac{\|\hat{x}(\mu) - x^{\star}\|_1}{\|x^{\star}\|_1}$$
</div>
<div>
  $$\Delta_\infty(\mu) = \frac{\max_{p,q} |\hat{x}_{p,q}(\mu) - x^{\star}_{p,q}|}{\max_{p,q} |x^{\star}_{p,q}|} 
  = \frac{\|\hat{x}(\mu) - x^{\star}\|_\infty}{\|x^{\star}\|_\infty}$$
</div>




当恢复的图像类似于真实图像时，这些距离接近于 0，当恢复的图像为零时，它们接近 1。

​	•	我们设定 $\mu$ 值在 $10^{-10}$ 和 $10^{10}$ 之间取对数间隔值。



```matlab
%% Question 8

% Load TrueImage (focus on Data2)

TrueImage = Data2.TrueImage;

mu=logspace(-10, 10, 100);



for i=1:length(mu)

  val_mu=mu(i);

   

  % Deconvolve image with current mu

  x_hat_temp = deconvolve_image(Data2.Data, Data2.IR, val_mu);



  delta_2(i)=norm(x_hat_temp-TrueImage,2)/norm(TrueImage,2);

  delta_1(i)=norm(x_hat_temp-TrueImage,1)/norm(TrueImage,1);

  delta_inf(i)=norm(x_hat_temp-TrueImage,Inf)/norm(TrueImage,Inf);

end



% Find mu that minimizes each distance

[min_delta_2,ind_min_delta_2]=min(delta_2);

[min_delta_1,ind_min_delta_1]=min(delta_1);

[min_delta_inf,ind_min_delta_inf]=min(delta_inf);



mu_min_delta_2 = mu(ind_min_delta_2);

mu_min_delta_1 = mu(ind_min_delta_1);

mu_min_delta_inf = mu(ind_min_delta_inf);





% Display the results

fprintf('Delta_2 的最小值为: %e，对应的 mu 为: %e\n', min_delta_2, mu_min_delta_2);

fprintf('Delta_1 的最小值为: %e，对应的 mu 为: %e\n', min_delta_1, mu_min_delta_1);

fprintf('Delta_inf 的最小值为: %e，对应的 mu 为: %e\n', min_delta_inf, mu_min_delta_inf);



% Plot the distances as functions of mu

figure;

subplot(3,1,1)

loglog(mu, delta_2, 'b-', 'LineWidth', 2);

hold on;

loglog(mu(ind_min_delta_2), min_delta_2, 'ro', 'MarkerSize', 8, 'LineWidth', 2);

xlabel('\mu');

ylabel('\Delta_2(\mu)');

title('\Delta_2 vs \mu');

grid on;

legend('\Delta_2(\mu)', ['Min at \mu = ', num2str(mu_min_delta_2, '%.1e')]);



subplot(3,1,2)

loglog(mu, delta_1, 'g-', 'LineWidth', 2);

hold on;

loglog(mu(ind_min_delta_1), min_delta_1, 'ro', 'MarkerSize', 8, 'LineWidth', 2);

xlabel('\mu');

ylabel('\Delta_1(\mu)');

title('\Delta_1 vs \mu');

grid on;

legend('\Delta_1(\mu)', ['Min at \mu = ', num2str(mu_min_delta_1, '%.1e')]);



subplot(3,1,3)

loglog(mu, delta_inf, 'm-', 'LineWidth', 2);

hold on;

loglog(mu(ind_min_delta_inf), min_delta_inf, 'ro', 'MarkerSize', 8, 'LineWidth', 2);

xlabel('\mu');

ylabel('\Delta_\infty(\mu)');

title('\Delta_\infty vs \mu');

grid on;

legend('\Delta_\infty(\mu)', ['Min at \mu = ', num2str(mu_min_delta_inf, '%.1e')]);


```

```matlab
Delta_2 的最小值为: 4.887066e-02，对应的 mu 为: 2.983647e-03

Delta_1 的最小值为: 1.337040e-01，对应的 mu 为: 1.204504e-02

Delta_inf 的最小值为: 1.855954e-01，对应的 mu 为: 7.564633e-03
```



和之前肉眼观察得到的 $\mu$ 值相比，可见差距还是挺大的。

​	本文内容均为 ’ [Problème inverse](https://formations.u-bordeaux.fr/#/details-formation?type=enseignement&id=53344) ’ 课程下的实验部分。

​	具体实验PDF文稿见 **Jean-François Giovanelli** 老师官方网站: [http://giovannelli.free.fr](http://giovannelli.free.fr/)






