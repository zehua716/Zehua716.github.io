---
title: "随机信号反卷积"
# author: "Zehua"
date: "2023-11-30T16:25:17+01:00"
lastmod: "2024-11-15T17:12:35+08:00"
lang: "zh"
draft: false
summary: "使用维纳反卷积技术，重建受噪声和测量误差影响的信号。"
description: "第二次实验内容"
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



## 通过维纳滤波器对脉冲信号进行反卷积



在超声波无损检测、用于地球物理学的地震反射、医学中的相干光学断层扫描、天体物理学这些领域中，很多时候，未知信号是脉冲型的(即信号是由一个或多个短时间的冲击组成)，它们表现为在几乎为零的背景上具有大幅度的脉冲。尤其适用于确定一个介质中两个界面的定位以量化其厚度。

#### **目的:**

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



#### 实现

**1. 信号可视化**

加载数据文件 **Observations.mat**，并查看滤波器在时域和傅里叶域中的特性

```matlab
%%Load data and visualize the filter
% Load the data file Observations.mat
load('Observations.mat'); % This loads variables RI (filter) and Donnees (observed signal)

% Time domain visualization of the filter
figure;
subplot(2,1,1);
plot(RI);
title('Filter in Time Domain');
xlabel('Time Samples');
ylabel('Amplitude');

% Compute the Fourier Transform of the filter
H = MyFFT(RI);

% Frequency axis setup
N = length(RI);
freq = (-N/2:N/2-1)/N; % Normalized frequency axis from -0.5 to 0.5

% Frequency domain visualization of the filter magnitude
subplot(2,1,2);
plot(freq, abs(H));
title('Filter in Frequency Domain (Magnitude)');
xlabel('Normalized Frequency');
ylabel('Magnitude');
```

​	<img src="/img/signal_aleatoire/TP3_1.png" alt="signal aléatoire  TP 3" width="85%" />

在**时间域**，滤波器的特性通过其**冲激响应（impulse response）**展现出来。冲激响应是指滤波器对单位冲激信号（Dirac delta 函数）的响应，完全描述了该滤波器的行为。

在**频率域**，滤波器的特性通过其**频率响应（H）**的**幅值图（magnitude plot）**来表示。幅值图表明了滤波器如何处理不同的频率分量，是高通还是低通。这里明显低通



然后绘制观测信号

```matlab
%%Visualize the observed signal
% Since the original signal x is not provided, we'll plot only the observed signal
figure;
plot(Donnees);
title('Observed Signal');
xlabel('Time Samples');
ylabel('Amplitude');
```

​	<img src="/img/signal_aleatoire/TP3_2.png" alt="signal aléatoire  TP 3" width="85%" />

信号经过滤波器处理时，**卷积（convolution）**使信号变得模糊，也就是我们观测到的信号是模糊的，再加上噪声的干扰，非常不好，使得**单脉冲变得不易区分**，甚至有可能出现邻近脉冲的重叠问题，这里的观测信号出现了尿分叉的现象，需要使用反卷积来处理，那就要使用维纳滤波器来

**2. 维纳滤波器**

我们先固定一个 λ 的值来 绘制维纳滤波器的增益，然后将该增益曲线与滤波器 H 的特性放在一起看一下。其中，λ  是噪声功率谱密度与信号功率谱密度之间的比率:
$$
λ= S_B (ν)/S_X (ν) 
$$

```matlab
%% Compute and visualize the Wiener filter gain
% Fix the value of lambda (lambda = S_B / S_X)
lambda = 0.9;

% Compute the Wiener filter gain
W = conj(H) ./ (abs(H).^2 + lambda);

% Plot the Wiener filter gain and overlay with the filter H
figure;
plot(freq, abs(W), 'LineWidth', 2);
hold on;
plot(freq, abs(H), '--', 'LineWidth', 2);
title(['Wiener Filter Gain and Filter H (\lambda = ', num2str(lambda), ')']);
xlabel('Normalized Frequency');
ylabel('Magnitude');
legend('Wiener Filter Gain', 'Filter H');
grid on;

```

​	<img src="/img/signal_aleatoire/TP3_3.png" alt="signal aléatoire  TP 3" width="85%" />



**2b.** 改变 λ 的值，每次绘制维纳滤波器的增益。推导出 λ 对滤波器特性的影响。

```matlab
%%Effect of varying lambda on the Wiener filter
% Define a range of lambda values
lambda_values = [ 0.01, 0.1, 0.9];

% Plot the Wiener filter gains for different lambda values
figure;
hold on;
for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    W = conj(H) ./ (abs(H).^2 + lambda);
    plot(freq, abs(W), 'LineWidth', 1.5);
end
title('Wiener Filter Gains for Different \lambda Values');
xlabel('Normalized Frequency');
ylabel('Magnitude');
legend('\lambda=0.01', '\lambda=0.1', '\lambda=0.9');
grid on;

```

​	<img src="/img/signal_aleatoire/TP3_4.png" alt="signal aléatoire  TP 3" width="85%" />

增益是指滤波器对信号的增强或衰减程度，**λ**的本质其实就是 **信-噪比** ，$\lambda = \frac{\sigma_{\text{noise}}^2}{\sigma_{\text{signal}}^2}$ ，也就是噪声的方差比上信号的方差。因此根据前置公式$ \mathcal{W}(\nu) = \frac{\mathcal{H}^*(\nu)}{|\mathcal{H}(\nu)|^2 + \frac{S_B(\nu)}{S_X(\nu)}}$ ，如果不加 $\lambda$  直接进行逆滤波（即除以H(f)）会导致幅值响应低的频率区域的的噪声被放大，为了抑制噪声被放大，维纳滤波器通过均衡信号和噪声的比例，如果信号功率远高于噪声功率，那么直接让$\lambda \to 0$ 都是可以的，维纳滤波器趋于完全恢复信号，但是如果$\lambda $ 很大，维纳滤波器会更加注重抑制噪声。

```matlab
%% Reconstruction of the signal using the Wiener filter
% Choose an appropriate lambda value for reconstruction
lambda = 0.1;

% Recompute the Wiener filter gain with the chosen lambda
W = conj(H) ./ (abs(H).^2 + lambda);

% Compute the Fourier Transform of the observed signal
Y = MyFFT(Donnees);

% Apply the Wiener filter in the frequency domain
X_hat = Y.*abs(W);

% Reconstruct the signal in the time domain
x_hat = MyIFFT(X_hat);

% Visualize the observed and reconstructed signals
figure;
subplot(2,1,1);
plot(Donnees);
title('Observed Signal');
xlabel('Time Samples');
ylabel('Amplitude');

subplot(2,1,2);
plot(real(x_hat));
title('Reconstructed Signal using Wiener Filter');
xlabel('Time Samples');
ylabel('Amplitude');

```

​	<img src="/img/signal_aleatoire/TP3_5.png" alt="signal aléatoire  TP 3" width="85%" />

根据结果可见，维纳滤波器很好的从被模糊或被卷积后的信号中还原原始信号形状，让脉冲特征更加易于辨别 ，但是由于我们的原始信号不可知，所以没办法直接比较真实的原始信号来评估性能，非常遗憾