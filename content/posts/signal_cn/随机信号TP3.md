---
title: "通过维纳滤波器进行反卷积"
# author: "Zehua"
date: "2023-11-30T16:25:17+01:00"
lastmod: "2024-11-15T17:12:35+08:00"
lang: "zh"
draft: false
summary: "使用维纳反卷积技术，重建受噪声和测量误差影响的信号。"
description: ""
tags: ["信号处理","反问题", "Matlab"]
# categories: "posts"
# cover:
#     image: "images/.jpg"
# comments: true
# hideMeta: false
# searchHidden: false
# ShowBreadCrumbs: true
# ShowReadingTime: false


---



在超声波无损检测、用于地球物理学的地震反射、医学中的相干光学断层扫描、天体物理学这些领域中，很多时候，未知信号是脉冲型的(即信号是由一个或多个短时间的冲击组成)，它们表现为在几乎为零的背景上具有大幅度的脉冲。尤其适用于确定一个介质中两个界面的定位以量化其厚度。

## 目的

通过使用维纳反卷积技术，重建受噪声和测量误差影响的信号。维纳滤波在频域中可以根据信号和噪声的功率谱密度来调整增益，从而优化信号的重建。反卷积和去噪的目标是逆转信号卷积的影响，并尽可能消除噪声。



由于这种卷积和噪声的影响，观测数据的分辨率很低，相近的脉冲在观测数据中被混淆，或者淹没在强噪声中。因此，输出观测信号y和输入原始信号x差距较大。因此使用反卷积-去噪 方法，可以通过维纳滤波器实现，维纳滤波器频率增益为：具体推导过程见 随机信号(理论)

$$
\mathcal{W}(\nu) = \frac{\mathcal{H}^*(\nu)}{|\mathcal{H}(\nu)|^2 + \frac{S_B(\nu)}{S_X(\nu)}}
$$

$S_B$ 代表测量噪声的功率谱密度（DSP）假设其为白噪声 $S_B (ν) = r_B$
$S_X$ 代表待重建信号的DSP，是我们掌握的先验信息

**对于DSP在不同场景下的选择:**

- 对于一个相对平滑、连续且正相关(当前值与之前的值有很强的正相关性)的信号，我们可以依赖于一个自回归系数接近1的信号的DSP。也就是说，如果信号通常在时间或频率上变化较慢(且正相关)，我们可以假设其自回归系数接近于1



自回归模型（AR模型）是基于当前值由前面的值加上一个随机误差来决定的。如果自回归系数接近1，意味着当前值与前一个或几个值之间的关系非常强，信号变化非常平缓。其数学形式为:
$$
X_t = \phi X_{t-1} + \epsilon_t
$$


- 在处理图像时，我们通常关注的是图像的纹理，例如图像中的细节、重复的图案等。功率谱密度（DSP）可以用来反映这些纹理的频率特征，因此研究DSP就可以研究图像纹理细节。
- 我们不希望惩罚 脉冲信号的任何频率，即不希望限制或增强特定的频率成分，换句话说，希望所有频率成分都被同等对待。因此选择了一个常数DSP(即每个频率的强度都是相同的)，即也是白噪音，其功率谱密度是平坦的。



## 实现

### 一维情况

加载数据文件 **Observations.mat**，并查看滤波器在时域和傅里叶域中的特性

```matlab
clear all
clc
close all

%% 加载数据并可视化滤波器
% 加载数据文件 Observations.mat
load('Observations.mat'); % 这将加载变量 RI(滤波器) 和 Donnees(观测信号)

% 在时域中可视化滤波器
figure;
subplot(2,1,1);
plot(RI);
title('时域中的滤波器');
xlabel('时间样本点');
ylabel('幅度');

% 计算滤波器的傅里叶变换
H = MyFFT(RI);  % 使用 MyFFT 得到中心化频域的 H
N = length(RI);

% 频率轴设置
freq = (-N/2:N/2-1)/N; % 从 -0.5 到 0.5 的归一化频率轴

% 在频域中可视化滤波器的幅度
subplot(2,1,2);
plot(freq, abs(H));
title('频域中的滤波器(幅度)');
xlabel('归一化频率');
ylabel('幅度');
```

​	<img src="/img/signal_aleatoire/TP3_1.png" alt="signal aléatoire  TP 3" width="85%" />

在**时间域**，滤波器的特性通过其**冲激响应（impulse response）**展现出来。冲激响应是指滤波器对单位冲激信号（Dirac delta 函数）的响应，完全描述了该滤波器的行为。

在**频率域**，滤波器的特性通过其**频率响应（H）**的**幅值图（magnitude plot）**来表示。幅值图表明了滤波器如何处理不同的频率分量，是高通还是低通。这里明显低通



然后绘制观测信号

```matlab
%% 可视化观测信号
figure;
plot(Donnees);
title('观测信号');
xlabel('时间样本点');
ylabel('幅度');
```

​	<img src="/img/signal_aleatoire/TP3_2.png" alt="signal aléatoire  TP 3" width="85%" />

信号经过滤波器处理时，**卷积（convolution）**使信号变得模糊，也就是我们观测到的信号是模糊的，再加上噪声的干扰，非常不好，使得**单脉冲变得不易区分**，甚至有可能出现邻近脉冲的重叠问题，这里的观测信号出现了尿分叉的现象，需要使用反卷积来处理，那就要使用维纳滤波器来

**2. 维纳滤波器**

我们先固定一个 λ 的值来 绘制维纳滤波器的增益，然后将该增益曲线与滤波器 H 的特性放在一起看一下。其中，λ  是噪声功率谱密度与信号功率谱密度之间的比率:
$$
λ= S_B (ν)/S_X (ν) 
$$

```matlab
%%  计算并可视化维纳滤波增益
% 固定 lambda 的值 (lambda = S_B / S_X)
lambda = 0.9;

% 计算维纳滤波器增益
W = conj(H) ./ (abs(H).^2 + lambda);

% 绘制维纳滤波增益并与滤波器 H 对比
figure;
plot(freq, abs(W), 'LineWidth', 2);
hold on;
plot(freq, abs(H), '--', 'LineWidth', 2);
title(['维纳滤波增益与滤波器 H (\lambda = ', num2str(lambda), ')']);
xlabel('归一化频率');
ylabel('幅度');
legend('维纳滤波增益', '滤波器 H');
grid on;
```

​	<img src="/img/signal_aleatoire/TP3_3.png" alt="signal aléatoire  TP 3" width="85%" />



**2b.** 改变 λ 的值，每次绘制维纳滤波器的增益。推导出 λ 对滤波器特性的影响。

```matlab
%% 不同 lambda 对维纳滤波器的影响
% 定义一组 lambda 值
lambda_values = [0.01, 0.1, 0.9];

figure;
hold on;
for i = 1:length(lambda_values)
    lambda_val = lambda_values(i);
    W_temp = conj(H) ./ (abs(H).^2 + lambda_val);
    plot(freq, abs(W_temp), 'LineWidth', 1.5);
end
title('不同 \lambda 值下的维纳滤波增益');
xlabel('归一化频率');
ylabel('幅度');
legend('\lambda=0.01', '\lambda=0.1', '\lambda=0.9');
grid on;
```

​	<img src="/img/signal_aleatoire/TP3_4.png" alt="signal aléatoire  TP 3" width="85%" />

增益是指滤波器对信号的增强或衰减程度，**λ**的本质其实就是 **信-噪比** ，$\lambda = \frac{\sigma_{\text{noise}}^2}{\sigma_{\text{signal}}^2}$ ，也就是噪声的方差比上信号的方差。因此根据前置公式$ \mathcal{W}(\nu) = \frac{\mathcal{H}^*(\nu)}{|\mathcal{H}(\nu)|^2 + \frac{S_B(\nu)}{S_X(\nu)}}$ ，如果不加 $\lambda$  直接进行逆滤波（即除以H(f)）会导致幅值响应低的频率区域的的噪声被放大，为了抑制噪声被放大，维纳滤波器通过均衡信号和噪声的比例，如果信号功率远高于噪声功率，那么直接让$\lambda \to 0$ 都是可以的，维纳滤波器趋于完全恢复信号，但是如果$\lambda $ 很大，维纳滤波器会更加注重抑制噪声。

```matlab
%% 使用维纳滤波器对信号进行重建

lambda = 0.1; % 为重建选择合适的 lambda
W = conj(H) ./ (abs(H).^2 + lambda);

% 计算观测信号的傅里叶变换
Y = MyFFT(Donnees);    % 频域观测信号(已居中)

% 在频域中应用维纳滤波器
X_hat = Y .* W; % 不要加abs，会丢相位信息

% 在时域中重建信号
x_hat = MyIFFT(X_hat);

% 对重建的时域信号进行循环移位以校正相位
x_hat = circshift(x_hat, N/2);

% 可视化观测与重建信号
figure;
subplot(2,1,1);
plot(Donnees);
title('观测信号');
xlabel('时间样本点');
ylabel('幅度');

subplot(2,1,2);
plot(real(x_hat));
title('使用维纳滤波器重建的信号(相位已校正)');
xlabel('时间样本点');
ylabel('幅度');
```

​	<img src="/img/signal_aleatoire/TP3_5.png" alt="signal aléatoire  TP 3" width="85%" />

根据结果可见，维纳滤波器很好的从被模糊或被卷积后的信号中还原原始信号形状，让脉冲特征更加易于辨别 ，但是由于我们的原始信号不可知，所以没办法直接比较真实的原始信号来评估性能，非常遗憾



### 二维情况

```matlab
clear all
clc
close all

%% 加载数据
% 假设DataOne.mat中有变量 Data, IR, TrueImage
Data1 = load('DataOne.mat');
Y_obs = Data1.Data;       % 观察图像(已退化的)
RI = Data1.IR;            % 2D滤波器内核
X_true = Data1.TrueImage; % 与 Y_obs 尺寸相同的真实清晰图像

% 显示观测图像
figure;
imagesc(Y_obs);
colormap gray;
axis image off;
title('Observed Degraded Image');
drawnow;

%% 计算滤波器的频域表示
[rows, cols] = size(Y_obs);
Long = max(rows, cols); % 根据需要设置扩展后的大小，可与图像大小匹配

H = MyFFT2RI(RI, Long);

% 频率轴（归一化频率，仅用于绘图可选）
freq_x = (-cols/2:cols/2-1)/cols; 
freq_y = (-rows/2:rows/2-1)/rows;

%% 定义维纳滤波参数
lambda = 0.05;

% 构建维纳滤波增益W(二维)
W = conj(H) ./ (abs(H).^2 + lambda);

%% 对观测图像进行频域变换
Y = MyFFT2(Y_obs);  

%% 应用维纳滤波器恢复图像
X_hat_freq = Y .* W; 
x_hat = MyIFFT2(X_hat_freq);
%% 计算MSE和PSNR
if ~exist('X_true','var')
    error('真实图像 X_true 缺失，请后续补充提供。');
end

mse_value = mean((double(X_true(:)) - double(x_hat(:))).^2);
max_val = max(X_true(:));
psnr_value = 10*log10((max_val^2)/mse_value);

fprintf('MSE: %.4f\n', mse_value);
fprintf('PSNR: %.4f dB\n', psnr_value);
%% 可视化恢复结果
figure;
subplot(1,3,1);
imagesc(Y_obs);
colormap gray;
axis image off;
title('Observed (Degraded) Image');

subplot(1,3,2);
imagesc(real(x_hat));
colormap gray;
axis image off;
title(sprintf('Restored Image using Wiener Filter\nMSE=%.4f, PSNR=%.2f dB', mse_value, psnr_value));

subplot(1,3,3);
imagesc(X_true);
colormap gray;
axis image off;
title('True Image');
drawnow;


%% 绘制像素值随着强度变化的对比图
row_idx = floor(rows/2);
true_line = double(X_true(row_idx,:));
est_line = double(x_hat(row_idx,:));

figure;
plot(true_line, 'k-', 'LineWidth', 2);
hold on;
plot(est_line, 'b--', 'LineWidth', 2);
title('Row Intensity Profile Comparison');
xlabel('Pixel Index');
ylabel('Intensity');
legend('True','Wiener');
grid on;
```

​	<img src="/img/signal_aleatoire/TP3_6.png" alt="signal aléatoire  TP 3" width="85%" />

​	<img src="/img/signal_aleatoire/TP3_8.png" alt="signal aléatoire  TP 3" width="85%" />

自定义函数如下

```matlab

%% MyIFFT
function Temporel = MyIFFT(Frequentiel)
    % MyIFFT
    % 对一维频域数据执行逆FFT操作，包括 fftshift 和 sqrt(N) 归一化，恢复时域信号。
    %
    % 输入参数：
    %   Frequentiel : 一维频域数据(已 fftshift)
    %
    % 输出参数：
    %   Temporel : 逆FFT后的时域信号数组

    Temporel = ifft( fftshift(Frequentiel) ) * sqrt(length(Frequentiel));
end


%% MyFFTRI
function Frequentiel = MyFFTRI(Temporel)
    % MyFFTRI
    % 对一维时域数据进行：先 fftshift，再 fft，然后再 fftshift，
    % 得到频域中心化的频谱数据(不归一化)。
    %
    % 输入参数：
    %   Temporel : 一维时域数据
    %
    % 输出参数：
    %   Frequentiel : 中心化后的频域数据(不做归一化)

    Frequentiel = fftshift( fft(fftshift(Temporel)) );
end


%% MyFFT
function Frequentiel = MyFFT(Temporel)
    % MyFFT
    % 对一维时域数据进行 fft，然后 fftshift，并通过 sqrt(N) 归一化结果，
    % 生成零频居中的一维频域数据。
    %
    % 输入参数：
    %   Temporel : 一维时域数据
    %
    % 输出参数：
    %   Frequentiel : 经 fftshift 和 sqrt(N) 归一化的一维频域数据

    Frequentiel = fftshift( fft(Temporel) ) / sqrt(length(Temporel));
end

%% MyFFT2
function Frequentiel = MyFFT2(Spatial)	
    % MyFFT2
    % 对二维空间域数据 Spatial 进行 2D FFT，并对结果进行 fftshift 和归一化处理。
    %
    % 输入参数：
    %   Spatial : 二维时域(或空间域)数据矩阵
    %
    % 输出参数：
    %   Frequentiel : 已经经过 fftshift 和归一化的二维频域数据矩阵

    Frequentiel = fftshift( fft2(Spatial) ) / length(Spatial);
end


%% MyFFT2RI
function Frequentiel = MyFFT2RI(Spatial, Long)	
    % MyFFT2RI
    % 对输入的二维数据 Spatial 进行补零扩展至尺寸 Long x Long，以便在频域处理时
    % 保持良好的居中对齐。补零后对数据进行 fftshift、2D FFT 再 fftshift。
    %
    % 输入参数：
    %   Spatial : 原始二维数据矩阵
    %   Long    : 补零后的目标矩阵尺寸(长宽相等)
    %
    % 输出参数：
    %   Frequentiel : 已补零扩展并居中对齐的二维频域数据矩阵

    Taille = length(Spatial);
    Ou = 1 + Long/2 - (Taille-1)/2 : 1 + Long/2 + (Taille-1)/2;
    SpatialComplet = zeros(Long, Long);
    SpatialComplet(Ou, Ou) = Spatial;

    Frequentiel = fftshift( fft2( fftshift(SpatialComplet) ) );
end


%% MyIFFT2
function Spatial = MyIFFT2(Frequentiel)
    % MyIFFT2
    % 对二维频域数据执行逆FFT操作并进行 fftshift 和归一化，以还原回空间/时域信号。
    %
    % 输入参数：
    %   Frequentiel : 二维频域数据矩阵(已 fftshift)
    %
    % 输出参数：
    %   Spatial : 逆变换并归一化后的二维时域(空间域)数据矩阵

    Spatial = ifft2( fftshift(Frequentiel) ) * length(Frequentiel);
end

```

