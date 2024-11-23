---
title: "反问题 TP2"
# author: "Zehua"
date: "2024-11-05T16:25:17+01:00"
lastmod: "2024-11-06T17:12:35+08:00"
lang: "zh"
draft: false
summary: "继续扩展反问题中的Wiener-Hunt方法，主要关注于超参数的自动调节。通过贝叶斯解释，介绍了误差分布与信号分布的建模，详细阐述了马尔可夫链蒙特卡洛（MCMC）方法中的Gibbs采样算法"
description: "第二次实验内容"
tags: ["信号处理","统计分析","反问题", "贝叶斯方法"]
# categories: "posts"
# cover:
#     image: "images/.jpg"
# comments: true
# hideMeta: false
# searchHidden: false
# ShowBreadCrumbs: true
# ShowReadingTime: false

---

# **Wiener-Hunt 方法：无监督方面**



在上一个实践内容中，我们介绍了去卷积问题的困难，即在应用卷积或者低通滤波器后所导致的观测数据缺失高频相关信息的情况。我们使用了 $Wiener$-$Hunt$ 方法：将量化解的误差的二次项和数据相结合，并在损失函数中使用带有正则化系数的二次惩罚标准以量化解的粗糙度。我们得到了相对来说不错的结果。但是这个方法的缺点是它需要调节一个参数，即正则化参数 $\mu$。我们最开始由经验选择到观察选择，一直到最后循环找到 $\mu$ 的最优值，使得去卷积后的图像既不过于不规则也不过于平滑。下面的工作重点在于介绍一种自动调节超参数的方法。



## **1. 超参数与全后验分布**



这个方法基于 $Wiener$-$Hunt$ 解的贝叶斯解释。该解释本身基于关于误差 $e$ 和关于图像 $x$ 的两个高斯概率模型。



### **1.1 误差分布**



误差被建模为白色、零均值同质高斯向量。白色指的是像白噪音一样，在其频谱特性中，所有频率分量有相同的功率密度，即信号在不同频率上的能量分布是均匀的，在数学层面，它具有零相关性，即不同时间点的误差值是统计独立的（不相关的）。



对于高斯分布，选择了一个涉及所谓精度参数 $\gamma_e$（方差的倒数）的替代参数化。根据这种参数化，其表达式可写为：



<div>$$f(e \mid \gamma_e) = (2\pi)^{-N/2} \gamma_e^{N/2} \exp\left( -\frac{\gamma_e \|e\|^2}{2} \right)$$</div>





根据 $y = Hx + e$，数据 $y$ 和感兴趣信号 $x$ 的似然函数表达式：



<div>$$f(y \mid x, \gamma_e) = (2\pi)^{-N/2} \gamma_e^{N/2} \exp\left( -\frac{\gamma_e \|y - Hx\|^2}{2} \right)$$</div>





根据所给的似然表达式 $f(y \mid x, \gamma_e)$，量化重建物体 $x$ 相对于观测数据 $y$ 的充分性可以通过协对数（log-likelihood）的形式表示。这种表达经常用于概率模型中，特别是最大似然估计（MLE）或贝叶斯推断中，用于评估模型参数的适配程度。



**再补充一下协对数相关内容，就是似然函数取对数，对于上述似然函数表达式，取其对数后为：**



<div>$$\log f(y \mid x, \gamma_e) = \log\left( (2\pi)^{-N/2} \gamma_e^{N/2} \exp\left( -\frac{\gamma_e \|y - Hx\|^2}{2} \right) \right)$$</div>





利用对数的性质：$\ln(a \cdot b) = \ln a + \ln b$，将三部分拆分开：



<div>$$\log f(y \mid x, \gamma_e) = \log\left( (2\pi)^{-N/2} \right) + \log\left( \gamma_e^{N/2} \right) + \log\left( \exp\left( -\frac{\gamma_e \|y - Hx\|^2}{2} \right) \right)$$</div>





逐项计算：



<div>$$\log\left( (2\pi)^{-N/2} \right) = -\frac{N}{2} \log(2\pi)$$</div>





<div>$$\log\left( \gamma_e^{N/2} \right) = \frac{N}{2} \log(\gamma_e)$$</div>





<div>$$\log\left( \exp\left( -\frac{\gamma_e \|y - Hx\|^2}{2} \right) \right) = -\frac{\gamma_e}{2} \|y - Hx\|^2$$</div>





合并得到：



<div>$$\log f(y \mid x, \gamma_e) = -\frac{N}{2} \log(2\pi) + \frac{N}{2} \log(\gamma_e) - \frac{\gamma_e}{2} \|y - Hx\|^2$$</div>





回到之前的内容，我们使用了协对数来表达数据能否充分重建原信号，我们给出这种拟合程度的量化指标：



<div>$$\mathcal{J}_{LS}(x) = \|y - Hx\|^2 = -k_y \log f(y \mid x, \gamma_e) + C_y$$</div>





​	•	$|y - Hx|^2$ 是重建信号（模型参数 $x$）与观测数据 $y$ 的误差平方和，称为残差平方和（Residual Sum of Squares, RSS）。

​	•	$-k_y \log f(y \mid x, \gamma_e)$ 是协对数的负加权形式，其中 $k_y > 0$，是一个常数。

​	•	$C_y$ 也是一个常数。



我们现在给出两个常数 $k_y$ 和 $C_y$ 的对应表达式：



前面得到：



<div>$$\log f(y \mid x, \gamma_e) = -\frac{N}{2} \log(2\pi) + \frac{N}{2} \log(\gamma_e) - \frac{\gamma_e \|y - Hx\|^2}{2}$$</div>





将上述结果带入 $\mathcal{J}_{LS}(x) = -k_y \log f(y \mid x, \gamma_e) + C_y$ 得：



<div>$$\mathcal{J}_{LS}(x) = -k_y \left( -\frac{N}{2} \log(2\pi) + \frac{N}{2} \log(\gamma_e) - \frac{\gamma_e \|y - Hx\|^2}{2} \right) + C_y$$</div>





<div>$$\mathcal{J}_{LS}(x) = k_y \left( \frac{N}{2} \log(2\pi) - \frac{N}{2} \log(\gamma_e) + \frac{\gamma_e \|y - Hx\|^2}{2} \right) + C_y$$</div>





将上述结果和原公式 $\mathcal{J}_{LS}(x) = |y - Hx|^2$ 做对比得到结果：



<div>$$k_y = \frac{2}{\gamma_e} \quad\quad\quad C_y = -\frac{N}{\gamma_e} \left( \log(2\pi) - \log(\gamma_e) \right)$$</div>





### **1.2 感兴趣信号的分布**



贝叶斯解释要求为感兴趣的信号 $x$ 提供一个概率分布。其模型也是高斯分布，只是这里它不是白色的，也就是说，它的各组成部分之间存在相关性。在接下来的内容中，相关性实际上是通过协方差矩阵 $R$ 来建模的。



**贝叶斯解释**的核心思想是为感兴趣的信号 $x$ 提供一个概率分布，而不是一个确定值。这种概率分布反映了我们对 $x$ 的不确定性以及其任何可能的取值。因此我们假设在模型中，$x$ 服从一个高斯分布。



但是它是一个非白色的高斯分布，也就是它的各组成部分之间存在相关性，协方差矩阵 $R$ 是一个非对角矩阵，其非零非对角元素表示信号的不同分量之间的相关性。后续我们就使用这个协方差矩阵 $R$ 在贝叶斯框架中建模信号的相关性。



**补充一下关于协方差矩阵的相关内容**



协方差矩阵 $R$ 提供了对信号相关性的精确描述。元素 $R_{ij}$ 表示信号第 $i$ 和第 $j$ 个分量之间的协方差：



<div>$$R_{ij} = \mathbb{E}[(x_i - \mu_i)(x_j - \mu_j)]$$</div>





根据相关性，$R$ 可能是一个稀疏矩阵或者满矩阵。



在贝叶斯建模中：

​	1.	信号 $x$ 的 **先验** 分布 $p(x)$ 使用协方差矩阵 $R$ 的描述公式为：



<div>$$p(x) = \frac{1}{(2\pi)^{N/2} |R|^{1/2}} \exp\left( -\frac{1}{2} x^T R^{-1} x \right)$$</div>





其中，$R^{-1}$ 是协方差矩阵的逆，也称为精度矩阵，定义了 $x$ 的相关性强度。



​	2.	最大后验估计（MAP）：

利用先验分布 $p(x)$ 和观测数据 $y$ 的似然函数 $p(y \mid x)$，可以通过贝叶斯法则计算 $x$ 的后验概率分布 $p(x \mid y)$，并基于该分布选择最优解。



回到之前，我们通过逆矩阵 $R^{-1} = \gamma_x \Pi$ 来表示建模信号的相关性，



其中：

​	•	**精度参数** $\gamma_x$ 控制相关性的强度

​	•	**矩阵** $\Pi$ 决定了相关性的结构



我们将 $R^{-1} = \gamma_x \Pi$ 带入之前的 $f(x \mid \gamma_x)$ 公式可得：



<div>$$f(x \mid \gamma_x) = (2\pi)^{-N/2} \det[\Pi]^{1/2} \gamma_x^{N/2} \exp\left( -\frac{\gamma_x}{2} x^T \Pi x \right)$$</div>





也就是说：



<div>$$f(x \mid \gamma_x) \propto \exp\left( -\frac{\gamma_x}{2} x^T \Pi x \right)$$</div>





量化物体相对于先验信息充分性的项表现为密度的协对数：



<div>$$\mathcal{J}_0(x) = -k_x \log f(x \mid \gamma_x) + C_x = x^T \Pi x$$</div>





其中：

​	•	$\mathcal{J}_0(x)$ 是对信号 $x$ 的量化，用来描述 $x$ 相对于先验信息（即对 $x$ 的已知假设或统计特性）是否充分匹配。其形式以密度的协对数（具体是取对数的负数）表示，结合了贝叶斯模型中的先验分布。

​	•	$f(x \mid \gamma_x)$ 是 $x$ 的先验概率密度函数，反映了 $x$ 如何符合假设的先验模型，我们之前在假设 $x$ 服从高斯分布的前提下，得到了其表达式（见上面）。

​	•	**正则化项** $x^T \Pi x$ 描述了信号 $x$ 的“复杂度”或“平滑性”，由 $\Pi$ 决定其结构，精度参数 $\gamma_x$ 控制正则化的强度，当 $\gamma_x$ 较大时，正则化约束更强。



同样，在上述公式中，我们添加了加法常数 $C_x$ 和乘法常数 $k_x$。为了与之前已经计算过的 Wiener-Hunt 方法联系起来，只需选择 $\Pi = D^T D$。



现在我们要给出两个常数的对应表达式。



已知原公式：



<div>$$f(x \mid \gamma_x) = (2\pi)^{-N/2} \det(\Pi)^{1/2} \gamma_x^{N/2} \exp\left( -\frac{\gamma_x x^T \Pi x}{2} \right)$$</div>





两边取对数：



<div>$$\log f(x \mid \gamma_x) = \log\left( (2\pi)^{-N/2} \det(\Pi)^{1/2} \gamma_x^{N/2} \exp\left( -\frac{\gamma_x x^T \Pi x}{2} \right) \right)$$</div>





<div>$$\log f(x \mid \gamma_x) = -\frac{N}{2} \log(2\pi) + \frac{1}{2} \log\left( \det(\Pi) \right) + \frac{N}{2} \log(\gamma_x) - \frac{\gamma_x x^T \Pi x}{2}$$</div>





根据公式：



<div>$$\mathcal{J}_0(x) = -k_x \log f(x \mid \gamma_x) + C_x$$</div>





将上述结果带入其中得到：



<div>$$\mathcal{J}_0(x) = -k_x \left( -\frac{N}{2} \log(2\pi) + \frac{1}{2} \log\left( \det(\Pi) \right) + \frac{N}{2} \log(\gamma_x) - \frac{\gamma_x x^T \Pi x}{2} \right) + C_x$$</div>





<div>$$\mathcal{J}_0(x) = k_x \left( \frac{N}{2} \log(2\pi) - \frac{1}{2} \log\left( \det(\Pi) \right) - \frac{N}{2} \log(\gamma_x) + \frac{\gamma_x x^T \Pi x}{2} \right) + C_x$$</div>





对比：



<div>$$\mathcal{J}_0(x) = x^T \Pi x$$</div>





得到：



<div>$$k_x = \frac{2}{\gamma_x} \quad\quad\quad C_x = \frac{N}{\gamma_x} \log(2\pi) - \frac{1}{\gamma_x} \log\left( \det(\Pi) \right) - \frac{N}{\gamma_x} \log(\gamma_x)$$</div>





但是严格来说，上述解释并不完全正确，因为 $D^T D$。常量图像只不过是对应于特征值为零的特征向量（这对应于零频率）。严格的发展要求引入一个用于零频率下能量的惩罚项（通过一个可以设定为任意小值的参数）。**不懂，后面补充**



上面提到的这个 $f(x \mid \gamma_x) = (2\pi)^{-N/2} \det(\Pi)^{1/2} \gamma_x^{N/2} \exp\left( -\frac{\gamma_x x^T \Pi x}{2} \right)$ 先验分布（a priori），因为它使人们能够处理先验信息，从而更倾向于具有更高规则性的图像。对于给定图像的概率越高，则图像越规则。



其中的 $\gamma_x$ 精度参数我们非常关注，因为它控制着图像的平滑度，进而影响着概率分布的整体趋势。

​	•	当 $\gamma_x$ 较大时，指数项中的 $x^T \Pi x$ 会被放大。

​	•	当 $\gamma_x$ 较小时，指数项中的 $x^T \Pi x$ 对总的概率密度的影响较小。**待补充**



### **1.3 后验分布**



借助前面定义的两个成分，并使用概率的乘法规则，现在可以构造重构信号 $x$ 和数据 $y$ 的联合密度：



<div>$$f(x, y \mid \gamma_e, \gamma_x) = f(y \mid x, \gamma_e) f(x \mid \gamma_x)$$</div>





将之前得到的结果代入：



<div>$$f(x, y \mid \gamma_e, \gamma_x) = (2\pi)^{-N} \det[\Pi]^{1/2} \gamma_x^{N/2} \gamma_e^{N/2} \exp\left( -\left[ \gamma_e \|y - Hx\|^2 + \gamma_x x^T \Pi x \right] / 2 \right)$$</div>





这个表达式由两个精度参数 $\gamma_e$ 和 $\gamma_x$ 参数化。可以注意到在指数项内部，我们得到了加权最小二乘准则的表达式：



<div>$$\mathcal{J}_{PLS}(x) = \mathcal{J}_{LS}(x) + \mu \mathcal{J}_0(x)$$</div>





<div>$$\mathcal{J}_{PLS}(x) = \|y - Hx\|^2 + \mu x^T \Pi x$$</div>





其中，正则化参数 $\mu$ 表示为信噪比的倒数 $\mu = \gamma_x / \gamma_e$。正则化参数 $\mu$ 在 $\gamma_e$ 和 $\gamma_x$ 中的作用是？



**待补充**



通过贝叶斯定理可以确定感兴趣信号的后验分布（后验概率分布）：



<div>$$f(x \mid y, \gamma_e, \gamma_x) = \frac{f(x, y \mid \gamma_e, \gamma_x)}{f(y \mid \gamma_e, \gamma_x)} \propto \exp\left( -\gamma_e \mathcal{J}_{PLS}(x) / 2 \right)$$</div>





这就是给定数据（已观测到的）和参数下的感兴趣信号的分布。



我们希望为感兴趣信号构造的任何估计器都基于上述分布。最常见的估计器是后验分布的均值、中位数或众数（即后验的最大化者）。在当前情况下，当后验分布是高斯分布时，这三者是相等的。众数或后验最大化者（MAP），记为 $\hat{x} _{MAP}$ 。也就是最小化准则 $\mathcal{J} _{PLS}(x)$ 的解



<div>$$\hat{x}_{MAP} = \arg \max_{x} f(x \mid y, \gamma_e, \gamma_x) = \arg \min_{x} \mathcal{J}_{PLS}(x) = \hat{x}_{PLS}$$</div>





结论是，最小二乘准则的解 	 $\hat{x} _{MAP}$  	就是之前的工作中推导出来的后验分布的众数 $ \hat{x} _{MAP}$。



### **1.4 扩展的后验分布**



到目前为止，贝叶斯方法只允许我们对已经存在的超参数值的估计给出概率解释。将之前的框架扩展到包含超参数的估计，需要为两个精度参数 $\gamma_e$ 和 $\gamma_x$ 引入一个先验分布。在多种可选方案中，接下来我们将重点关注伽马分布：



<div>$$f(\gamma) = \frac{\beta^\alpha}{\Gamma(\alpha)} \gamma^{\alpha - 1} \exp[-\beta \gamma] \cdot 1_{\mathbb{R}^+}(\gamma)$$</div>





它由两个正实数参数 $(\alpha, \beta)$ 驱动，具有均值 $\alpha / \beta$ 和方差 $\alpha / \beta^2$。这种选择的理由如下：

​	•	选择伽马分布作为先验分布确保了条件后验分布也是伽马分布（我们正在讨论共轭先验）。在算法上，这意味着只需要更新分布参数的值（具体见下面）。

​	•	这种选择允许在参数值的信息较少（也称为“非信息先验”）或精确（如名义值或某种不确定性）的情况下进行处理。该工作中特别感兴趣的是“非信息先验”的极限情况，即 $(\alpha, \beta) = (0, 0)$。



此外，关于变量 $\gamma_e$ 和 $\gamma_x$ 的组合，它们被建模为独立的。



从伽马分布：



<div>$$f(\gamma) = \frac{\beta^\alpha}{\Gamma(\alpha)} \gamma^{\alpha - 1} \exp[-\beta \gamma] \cdot 1_{\mathbb{R}^+}(\gamma)$$</div>





和部分联合分布：



<div>$$f(x, y \mid \gamma_e, \gamma_x) = f(y \mid x, \gamma_e) f(x \mid \gamma_x) = (2\pi)^{-N} \det[\Pi]^{1/2} \gamma_x^{N/2} \gamma_e^{N/2} \exp\left( -\left[ \gamma_e \|y - Hx\|^2 + \gamma_x x^T \Pi x \right] / 2 \right)$$</div>





的表达式出发，我们推导出对于 $y, x, \gamma_e$ 和 $\gamma_x$ 的完整联合分布的表达式为：



<div>$$f(y, x, \gamma_e, \gamma_x) = f(y, x \mid \gamma_e, \gamma_x) f(\gamma_e) f(\gamma_x)$$</div>





其明确表达为：



<div>$$f(x, y, \gamma_e, \gamma_x) = (2\pi)^{-N} \det[\Pi]^{1/2} \frac{\beta_e^{\alpha_e} \beta_x^{\alpha_x}}{\Gamma(\alpha_e) \Gamma(\alpha_x)} \gamma_e^{\alpha_e + N/2 - 1} \gamma_x^{\alpha_x + N/2 - 1} \exp\left( -\left[ \gamma_e \left( \beta_e + \|y - Hx\|^2 / 2 \right) + \gamma_x \left( \beta_x + x^T \Pi x / 2 \right) \right] \right)$$</div>





注意: 这个密度非常重要，因为它允许推导出所有相关的联合、边缘和条件密度。

现在我们可以通过贝叶斯规则推导出完整的后验分布，即给定观测数据$y$时，感兴趣信号$x$和超参数$\gamma_e$, $\gamma_x$的分布：

<div>$$f(x, \gamma_e, \gamma_x | y) = \frac{f(x, y, \gamma_e, \gamma_x)}{f(y)}$$</div>

<div>$$f(x, \gamma_e, \gamma_x | y) \quad \propto \quad \gamma_e^{\alpha_e + N/2 - 1} \gamma_x^{\alpha_x + N/2 - 1} \exp \left( - \left[ \gamma_e \left( \beta_e + \|y - Hx\|^2 / 2 \right) + \gamma_x \left( \beta_x + \|x\|_\Pi^2 / 2 \right) \right] \right)$$</div>

这汇总了所有关于感兴趣信号和超参数在数据视角下的可用信息：对于三重项 $x$, $\gamma_e$, $\gamma_x$，它量化了后验密度，即在给定观测数据下三重项的概率。感兴趣信号和超参数的估计器是从这个分布中构造出来的。我们可以查看后验分布的均值、中位数或众数。每种方法都有其优缺点。在接下来的内容中，我们将重点讨论均值。

### **1.5 后验均值**

考虑到后验分布（上面这个）的复杂性，获得均值的解析公式是不可行的。为了计算后验均值，有几种方法可用，在这里我们将重点关注随机采样技术。最终，它归结为对后验分布进行随机采样，然后取样本的经验均值，从而近似后验均值。

后验分布的采样可以通过**马尔可夫链蒙特卡洛（MCMC）方法**来实现。它要求构建一个迭代过程，以生成随机样本，经过一定的时间（称为 burn-in），这些样本将根据目标分布进行分布。构建这样一个过程并不容易，但在当前情况下，存在一个标准算法可以轻松使用：Gibbs 采样算法。它将对三重项 $x$, $\gamma_e$, $\gamma_x$ 的后验分布进行采样的问题，转换为它们三个各自的更简单分布的采样问题。每个分布实际上是条件分布，给定其余参数的条件下对其中一个参数进行采样。该算法的工作原理在下表中给出，接下来的部分我们将详细说明这些步骤。

<div>$$\begin{aligned} &\bullet \, \text{Initialize } x^{[0]} = y \\ &\bullet \, \text{For } k = 1, 2, \dots \, \text{repeat} \\ &\quad \text{(a) \ sample } \gamma_e^{[k]} \text{ under } f(\gamma_e \mid \gamma_x^{[k-1]}, x^{[k-1]}, y) \\ &\quad \text{(b) \ sample } \gamma_x^{[k]} \text{ under } f(\gamma_x \mid \gamma_e^{[k]}, x^{[k-1]}, y) \\ &\quad \text{(c) \ sample } x^{[k]} \text{ under } f(x \mid \gamma_e^{[k]}, \gamma_x^{[k]}, y) \end{aligned}$$</div>

#### **1.5.1 采样逆误差功率**

采样对应于步骤 (a) 的超参数 $\gamma_e$ 需要从条件后验分布 $f(\gamma_e | x, \gamma_x, y)$ 中采样。该分布由完整的联合分布 $f(x, y, \gamma_e, \gamma_x)$ 获得，如下所示：

<div>$$\text{posterior distribution }: \quad f(\gamma_e | x, \gamma_x, y) = \frac{f(x, y, \gamma_e, \gamma_x)}{f(x, \gamma_x, y)}$$</div>

仅保留包含 $\gamma_e$ 的项（与 $\gamma_e$ 相关的部分），并且由于分母不依赖于 $\gamma_e$，我们得到

<div>$$f(\gamma_e | x, \gamma_x, y) \propto f(x, y, \gamma_e, \gamma_x)$$</div>

<div>$$f(\gamma_e | x, \gamma_x, y) \quad \propto \quad \gamma_e^{\alpha_e + N/2 - 1} \exp \left( - \gamma_e \left( \beta_e + \| y - Hx \|^2 / 2 \right) \right)$$</div>

由此获得的条件分布实际上是具有参数 $(\alpha, \beta)$ 的伽马分布：

<div>$$\alpha = \alpha_e + N/2 \quad \text{and} \quad \beta = \beta_e + \| y - Hx \|^2 / 2$$</div>

在先验参数 $(\alpha_e, \beta_e)$ 等于 $(0, 0)$ 的极限情况下，后验的参数为：

<div>$$\alpha = N/2 \quad \text{and} \quad \beta = \| y - Hx \|^2 / 2$$</div>

因此我们关注于这个 $f(\gamma_e | x, \gamma_x, y)$ 分布的均值和方差表达式，并将它与输出误差 $y - Hx$ 的功率相关联。

已知伽马分布的概率密度函数：

<div>$$f(\gamma) = \frac{\beta^\alpha}{\Gamma(\alpha)} \gamma^{\alpha - 1} \exp[-\beta \gamma] \cdot 1_{\mathbb{R}+}(\gamma)$$</div>

其中：$\alpha$ 是形状参数，$\beta$ 是尺度参数。

对于伽马分布 $\text{Gamma}(\alpha, \beta)$，其均值和方差的标准表达式分别为：

均值：

<div>$$\mathbb{E}[\gamma_e] = \frac{\alpha}{\beta}$$</div>

方差：

<div>$$\text{Var}[\gamma_e] = \frac{\alpha}{\beta^2}$$</div>

已知：

<div>$$\alpha = N/2 \quad \text{and} \quad \beta = \| y - Hx \|^2 / 2$$</div>

误差精度参数 $\gamma_e$ 的均值：

<div>$$\mathbb{E}[\gamma_e] = \frac{\alpha_e + N/2}{\beta_e + \frac{\| y - Hx \|^2}{2}}$$</div>

误差精度参数 $\gamma_e$ 的方差：

<div>$$\text{Var}[\gamma_e] = \frac{\alpha_e + N/2}{\left( \beta_e + \frac{\| y - Hx \|^2}{2} \right)^2}$$</div>

伽马分布中的 $\beta$ 参数直接依赖于 $\| y - Hx \|^2$。

- 当误差 $\| y - Hx \|^2$ 较大时，$\beta$ 参数也会增大，表示模型拟合较差，这意味着伽马分布的均值和方差会减小，反映了误差增加时对数据的信任度降低。
- 反之，误差 $\| y - Hx \|^2$ 较小时，精度参数 $\gamma_e$ 的均值增大，表示对数据的信任度较高。

因此为了实现步骤 (a)：$\gamma_e^{[k]}$ 的样本从具有上述两个参数 $\alpha$ 和 $\beta$ 的伽马分布中抽取，我们可以使用 Matlab 中的 `RNDGamma(Alpha, Beta);` 函数，具体代码为：

```matlab
function SamplePrecision = RNDGamma(Alpha,Beta)	
    % The Precision variable is a sample of the gamma distribution with parameters Alpha and Beta
    % Tirage d'un échantillon Gamma approché par du Gauss (JFG+TBC)
    SamplePrecision = Alpha/Beta + sqrt( Alpha/(Beta*Beta) ) * randn;
```

注意：计算参数 $\beta$ 涉及计算建模误差 $\| y - Hx \|^2$ 的范数。计算空间域中的范数通常成本较高，但是可以在傅里叶域中进行计算以降低成本。

#### **1.5.2 采样感兴趣信号的逆功率**

现在我们将重点放在采样超参数 $\gamma_x$ 上，对应于表 1 中算法的步骤 (b)。

这需要采样条件后验分布 $f(\gamma_x | x, \gamma_e, y)$。使用与上一节类似的方法，我们得到：

<div>$$f(\gamma_x | x, \gamma_e, y) \propto \gamma_x^{\alpha_x + N/2 - 1} \exp \left( - \gamma_x \left( \beta_x + \| x \|_\Pi^2 / 2 \right) \right)$$</div>

可以看出，这个条件后验分布也是伽马分布。在先验参数 $(\alpha_x, \beta_x)$ 等于 $(0, 0)$ 的极限情况下，后验参数为：

<div>$$\alpha = N/2 \quad \text{and} \quad \beta = \| x \|_\Pi^2 / 2$$</div>

因此我们关注于 $f(\gamma_x | x, \gamma_e, y)$ 这个分布的均值和方差表达式，并将它与输出误差 $y - Hx$ 的功率相关联。

均值：

<div>$$\mathbb{E}[\gamma_x] = \frac{\alpha_x}{\beta_x}$$</div>

方差：

<div>$$\text{Var}[\gamma_x] = \frac{\alpha_x}{\beta_x^2}$$</div>

<div>$$\mathbb{E}[\gamma_x] = \frac{\alpha_x + N/2}{\beta_x + \frac{\| x \|_{\Pi}^2}{2}}$$</div>

<div>$$\text{Var}[\gamma_x] = \frac{\alpha_x + N/2}{\left( \beta_x + \frac{\| x \|_{\Pi}^2}{2} \right)^2}$$</div>

当图像 $x$ 的二次形式较小时（即图像较为平滑），均值会相应增大。这表明我们更信任较为平滑的图像。

因此为了实现步骤 (b) $\gamma_x^{[k]}$ 的样本从具有上述两个参数的伽马分布中抽取，同样使用 `RNDGamma` 函数。

注意：计算参数 $\beta$ 涉及计算建模误差 $\| y - Hx \|^2$ 的范数。计算空间域中的范数通常成本较高，但是可以在傅里叶域中进行计算以降低成本。

### **1.6 采样感兴趣的物体**

最后但同样重要的是，我们将处理对应于表 1 中算法步骤 (c) 的感兴趣物体 $x$ 的采样。这意味着采样条件后验分布 $f(x | \gamma_x, \gamma_e, y)$，其表达式已经推导出

<div>$$f(x | \gamma_e, \gamma_x, y) = \frac{f(x, y | \gamma_e, \gamma_x)}{f(y | \gamma_e, \gamma_x)} \propto \exp \left( - \gamma_e \mathcal{J}_{PLS}(x) / 2 \right)$$</div>

并且它便捷地重新写为：

<div>$$f(x | \gamma_e, \gamma_x, y) \propto \exp \left( - \left[ \gamma_e \| y - Hx \|^2 + \gamma_x \| x \|_\Pi^2 \right] / 2 \right) = \exp \left( - \gamma_e \mathcal{J}_{PLS}(x) / 2 \right)$$</div>

该密度本身是高斯分布，因为指数项内的变量 $x$ 是正定的二次项。它由均值和协方差矩阵完全表征。在这种情况下：

- 均值 $\mu_{x|*}$ 也是众数，作为最小化 $\mathcal{J}_{PLS}(x)$ 的解，即之前实践工作中讨论的 Wiener-Hunt 解。
- 协方差矩阵 $\Sigma_{x|*}$ 通过计算 $\mathcal{J}_{PLS}(x)$ 的 Hessian（即二阶导数矩阵）获得。

我们得到以下表达式：

<div>$$\mu_{x|*} = \gamma_e \Sigma_{x|*} H^T y$$</div>

<div>$$\Sigma_{x|*} = \left( \gamma_e H^T H + \gamma_x \Pi \right)^{-1}$$</div>

当前面临的数值问题是对可能具有高维的高斯分布进行采样。这个高维度性阻止了协方差矩阵 $\Sigma_{x|*}$ 的求逆或分解，这意味着没有简单的采样方案可用。为了解决这个问题，我们采用循环矩阵的近似方法，从而能够在傅里叶域中进行快速的矩阵运算。给出了以下表达式：

<div>$$\overset{\circ}{\mu}_{x|*} = \gamma_e \Lambda_{x|*} \Lambda_H^{\dagger} \overset{\circ}{y}$$</div>

<div>$$\Lambda_{x|*} = \left( \gamma_e \Lambda_H^\dagger \Lambda_H + \gamma_x \Lambda_D^\dagger \Lambda_D \right)^{-1}$$</div>

在傅里叶域中，协方差矩阵是对角线形式的，这意味着其各个分量是解相关的。因此，每个分量是独立的，这使得可以并行采样。

{{< alert class="warning" >}} 

**接下来我们给出上面两个表达式的推导过程，基于循环矩阵对角化的思想。**

由于矩阵 $H$ 是实数矩阵，因此它也是它的复共轭矩阵，且 $H^t = H^\dagger$。$D$ 同理。

因此我们将利用循环矩阵的对角化性质，由于 $H$ 和 $D$ 是循环矩阵，且循环矩阵是具有特殊结构的方阵，矩阵的每一行元素是前一行元素循环右移一位。基于这一性质，循环矩阵的一个重要性质是，它可以通过傅里叶变换对角化。具体来说，任何 $N \times N$ 的循环矩阵 $C$ 都可以写成：

<div>$$C = F \Lambda F^\dagger$$</div>

- $F$ 是离散傅里叶变换矩阵，$F^\dagger$ 是其共轭转置（逆傅里叶变换）。
- $\Lambda$ 是一个对角矩阵，其元素为矩阵 $C$ 的特征值（傅里叶系数）。

之前我们得到的方程为：

<div>$$\mu_{x|*} = \gamma_e \Sigma_{x|*} H^T y$$</div>

<div>$$\Sigma_{x|*} = \left( \gamma_e H^T H + \gamma_x \Pi \right)^{-1}$$</div>

我们的目标是利用循环矩阵的性质将其转换到频域，从而得到新的表达式。先将 $H$ 和 $D$ 对角化：

<div>$$H = F \Lambda_H F^\dagger$$</div>

<div>$$\Pi = F \Lambda_\Pi F^\dagger$$</div>

带入原方程：

<div>$$\Sigma_{x|*} = \left( \gamma_e H^T H + \gamma_x \Pi \right)^{-1}$$</div>

将 $F$ 和 $F^\dagger$ 提到外面：

<div>$$\Sigma_{x|*} = F \left( \gamma_e \Lambda_H^\dagger \Lambda_H + \gamma_x \Lambda_\Pi \right)^{-1} F^\dagger$$</div>

第一个公式证明完毕：

<div>$$\Lambda_{x|*} = \left( \gamma_e \Lambda_H^\dagger \Lambda_H + \gamma_x \Lambda_D^\dagger \Lambda_D \right)^{-1}$$</div>

回到开始，我们有：

<div>$$H = F \Lambda_H F^\dagger$$</div>

<div>$$\Pi = F \Lambda_\Pi F^\dagger$$</div>

把上述公式带入原方程：

<div>$$\mu_{x|*} = \gamma_e \Sigma_{x|*} H^T y$$</div>

计算得：

<div>$$\mu_{x|*} = \gamma_e F \left( \gamma_e \Lambda_H^\dagger \Lambda_H + \gamma_x \Lambda_D^\dagger \Lambda_D \right)^{-1} F^\dagger F \Lambda_H^\dagger \hat{y}$$</div>

由于 $F^\dagger F = I$，可以简化为：

<div>$$\mu_{x|*} = \gamma_e \Lambda_{x|*} \Lambda_H^\dagger \hat{y}$$</div>

第二个公式证毕。

 {{< /alert >}}



在进行步骤 (c) 时，对图像 $x^{[k]}$ 的样本应从具有在傅里叶域中给定的第一和第二矩中的高斯分布中抽取。可以使用自定义的 Matlab 函数 `RNDGauss(Moy, Cov)`，`Moy` 和 `Cov` 必须在傅里叶域中给出，函数具体内容为：

```matlab
function SampleImage = RNDGauss(MoyGauss,Covariance)	

% Generate an image sample under a Gaussian distribution, with the mean given by Moy and the covariance given by Cov.
% The parameters Moy and Cov, and Image, are all in the Fourier domain, not in the spatial domain.
% Paramètre de Taille
	Taille = length(MoyGauss);
    
% Tirage d'un bruit blanc avec la bonne symétrie
	BoutGauss = randn(Taille,Taille) + sqrt(-1) * randn(Taille,Taille);
    BoutGauss = MyFFT2( real( MyIFFT2(BoutGauss) ) );

% Filtrage du bruit blanc
	SampleImage = MoyGauss + BoutGauss .* sqrt(Covariance);
```

## **2 实现**

在 Matlab 实践中，我们使用和上次内容相同的数据集，最后结果理论上应该相似。因为我们做出的改进只是自动调整正则化参数。我们之前介绍了算法步骤，并且详细解释了涉及两个超参数 $\gamma_e$ 和 $\gamma_x$ 的条件分布采样，以及图像 $x$ 的条件分布采样，所以这里直接给代码。

