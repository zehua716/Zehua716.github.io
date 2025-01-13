---
title: "反问题 TP4"
#password: "123"
# author: "Zehua"
date: "2024-12-14T16:25:17+01:00"
lastmod: "2024-12-14T17:12:35+08:00"
lang: "zh"
draft: false
summary: "对系统输出图像进行逆卷积操作试图恢复原始清晰的输入图像"
description: ""
tags: ["信号处理","图像处理"]
# categories: "posts"
#cover:
    #image: "img/signal.png"
# comments: true
# hideMeta: false
searchHidden: true
# ShowBreadCrumbs: true
# ShowReadingTime: false

---



# 									图像复原TP --- **约束方法**

### 												     前提条件：提前阅读整篇论文，通读多次

在之前的实务操作中，你已经解决了一个图像去模糊问题。为了从模糊（并且有噪声）图像恢复出清晰的图像，你使用了两种正则化方法。这两种方法都基于一个带有二次最小二乘项（用于量化解和数据之间的差异）和一个惩罚项（用于考虑解的规则性先验信息，适度的）惩罚的凸准则。第一种方法，称为 **Wiener-Hunt 方法**，涉及二次惩罚（为了优先给平滑解而不是爆炸解），而第二种方法涉及 **Huber 惩罚**（为了保留边缘并提升分辨率）。然而，在分析结果时，可以观察到某些像素的值为负，而真实图像的所有像素值都是正的。此外，与真实图像不同的是，脑外的“黑色区域”中的像素值不为零。此实务操作的目的是开发一种约束方法，该方法能带来更好的物理建模并提升分辨率。

### 1. 二次惩罚和约束

我们仍然使用以下模型来描述获取过程：

$$
y = Hx + e
$$
其中向量   y   表示数据（模糊图像），向量   x   代表未知图像（清晰图像），  H   是卷积矩阵，  e   是表示测量和建模误差的向量  

为了正则化去卷积问题，我们通过引入一个用于强空间灰度级变化惩罚的项来考虑有关未知图像空间规则性的信息。我们定义以下准则：

$$
J(x) = \|y - Hx\|^2 + \mu \|Dx\|^2
$$
其中   D   是“差分”矩阵，例如一阶差分或其他任何线性“微分”算子（例如拉普拉斯算子、Sobel 算子、Prewitt 算子等）。此外，我们还考虑了关于未知图像的另一个先验信息：

- 像素值为正，  
- 在给定支撑集   S   之外像素值为零  

因此，重建图像 $\hat{x}$ 定义为：

$$
\hat{x} = \arg \min_{x \in \mathbb{R}^P} J(x) \quad \text{s.t.} \quad \left\{
\begin{aligned}
x_p &\geq 0 \quad \forall p \\
x_p &= 0 \quad \text{for} \quad p \notin S
\end{aligned}
\right.
$$
即寻找最小化准则并满足约束条件的图像。值得注意的是，该准则严格凸且约束集是凸的，因此存在唯一解。

### 2. 优化

一般来说，优化问题包含一个目标函数（准则）和可能的一组约束，通常以等式或不等式系统的形式表达。目标是通过某种算法找到最优值，即最小化目标函数并满足约束条件。各种算法可以用来解决此类问题，参见 [1–3]，其中可以引用以下几种方法：

- 梯度投影和约束梯度法，
- 内点法和障碍法，
- 像素逐点下降法，
- 拉格朗日乘数法  

这些方法适用于工程学、成像（例如天文学、分子学，广义上的物理学）、计算机科学（计算机图形学、计算机视觉、增强现实等）、医学和生物学、经济学和金融学、统计推断和机器学习等广泛领域  

#### 2.1 ADMM：解决约束问题的高效算法

在此实务操作中，我们关注基于拉格朗日乘数和准则增强思想的算法。该算法称为 **交替方向乘子法（ADMM）**，尤其适用于解决大规模凸约束问题。以下内容摘自 [3]：

该方法起源于20世纪70年代，根源可追溯到20世纪50年代，并且等同或密切相关于许多其他算法，例如 **对偶分解法**、**乘子法**、**Douglas-Rachford 分裂法**、**Spingarn 的部分逆方法**、**Dykstra 的交替投影法**、**用于 \( l_1 \) 问题的 Bregman 迭代算法**、近端方法等。[…] 它采用一种分解-协调过程，其中对小局部子问题的解进行协调，以找到大规模全局问题的解。ADMM 可被视为对对偶分解法和增强拉格朗日法两者优点的结合尝试  

ADMM 可以通过多种方式引入。基本上，它解决了一般凸问题，带有线性等式约束：

$$
(P_p) \quad (\hat{x}, \hat{z}) = \arg \min_{x, z} \left\{ F(x) + G(z) \quad \text{s.t.} \quad A x + B z = c \right\}
$$
其中   F   和   G   是凸函数，  A   和   B   是矩阵，  c   是具有适当尺寸的向量。该问题称为原始问题，  x   和   z   是原始变量。

首先，我们写出与原始问题 (2) 相关的所谓增强拉格朗日函数   L  ：

$$
L(x, z, \lambda) = F(x) + G(z) + \lambda^t(Ax + Bz - c) + \frac{\rho}{2} \|Ax + Bz - c\|^2
$$

其中，$\lambda$ 是拉格朗日乘数向量，$\rho > 0$ 是惩罚参数。

然后我们对 $(x, z)$ 最小化 $L_\rho$，给定 $\lambda$ 的值，得出最小值：
$$
(x_{\lambda}, z_{\lambda}) = \arg\min_{x, z} L(x, z, \lambda)
$$
最小值为：
$$
L_\rho(\lambda) = \min_{x, z} L(x, z, \lambda) = L(x_{\lambda}, z_{\lambda}, \lambda)
$$




即所谓的对偶函数。接下来需要解决对偶问题：

$$
(P_d) \ \ \ \ \ \ \ \ \ \  \bar{\lambda} = \arg \max_\lambda \tilde{L}_\rho(\lambda)
$$


由此得出正确的拉格朗日乘数值。最后一步是将 (6) 代入 (4)：
$$
(\hat{x}, \hat{z}) = (\bar{x}_\lambda, \bar{z}_\lambda)
$$
这就是原始问题的解  



实际上，该算法在数值上起作用：它交替最小化原始变量   x  、  z   和对偶变量   λ  ，这就是为什么它被称为原始-对偶方法。该算法的三个关键步骤如下：

$$
(a) \ x_{k+1} = \arg \min_x L(x_k, z_k, \lambda_k) \  \\

(b) \ z_{k+1} = \arg \min_z L(x_{k+1}, z_k, \lambda_k) \\

(c) \ \lambda_{k+1} = \lambda_k + \rho(Ax_{k+1} + Bz_{k+1} - c) \\
$$

$$
请注意，λ更新（c）是可分离的、直接的，并对应于对偶函数的梯度上升步骤。
$$



### 2.2 将 ADMM 应用于带有线性（不等式和等式）约束的二次准则

现在我们重新考虑包含不等式和等式约束的原始问题 (1)：


$$
\hat{x} = \arg \min_{x \in \mathbb{R}^P} J(x) \quad \text{s.t.} \quad \left\{
\begin{aligned}
x_p &\geq 0 \quad \forall p \\
x_p &= 0 \quad \text{for} \quad p \notin S
\end{aligned}
\right.
$$

并将其重写为仅包含等式约束的严格等价形式  

通过首先考虑  $ \mathbb{R}^P$  的约束子集:

$$
C = \{ x \in \mathbb{R}^P \mid x_p \geq 0 \quad \forall p \quad \text{并且} \quad x_p = 0 \quad \text{对于} \quad p \notin S \}
$$
 然后它的指示函数为：

$$
I_C(x) =
\begin{cases} 
0 & \text{如果 } x \in C \\ 
+\infty & \text{如果 } x \notin C 
\end{cases}
$$


因此，解可以表示为 ： 


$$
\hat{x} = \arg \min_{x \in \mathbb{R}^P} J(x) + I_C(x)
$$


不需要额外的约束条件。原始约束被替换为一种惩罚，特别是对不满足约束条件的物体的无限惩罚  

从表面上看，这似乎是多余的……但我们将看到事实并非如此  

现在引入一个辅助变量向量 $z \in \mathbb{R}^P$，并将问题重新表述为包含一个额外等式约束的等价问题：
$$
\hat{x} = \arg \min_{x \in \mathbb{R}^P} \left\{ J(x) + I_C(z) \quad \text{s.t.} \quad x = z \right\}
$$


这看起来似乎也是多余的……但现在我们可以利用前面提到的 ADMM 算法。因此，我们可以写出该问题的增强拉格朗日函数：


$$
L(x, z, \lambda) = J(x) + I_C(z) + \lambda^t(x - z) + \frac{\rho}{2} \| x - z \|^2
$$

 ADMM 的步骤如下：

①初始化：
$$
z_0 = 0, \quad \lambda_0 = 0, \quad k = 0
$$
②设定容差值 $\epsilon$，例如：
$$
\epsilon = 10^{-5}
$$
③对于 $k = 1, 2, \ldots$，执行以下迭代步骤：

- 更新 $x$：

$$
x_{k+1} = \arg \min_x L(x, z_k, \lambda_k)
$$

- 更新 $z$：

$$
z_{k+1} = \arg \min_z L(x_{k+1}, z, \lambda_k)
$$

- 更新 $\lambda$：

$$
\lambda_{k+1} = \lambda_k + \rho (x_{k+1} - z_{k+1})
$$

④终止条件：
$$
\| x_k - x_{k-1} \| < \epsilon
$$


#### 2.2.1 更新目标变量   x  

在给定当前 $z$ 和 $\lambda$ 的情况下，目标变量 $x$ 的更新步骤 (a) 的细节如下：

考虑增强拉格朗日函数 $L(x, z, \lambda)$ 的结构，我们可以将 $z$ 视为固定值，并最小化以下目标函数：
$$
K_o(x) = J(x) + \lambda^T(x - z) + \frac{\rho}{2} \|x - z\|^2
$$


值得注意的是，  K_o   是关于   x   的二次函数，且最小化问题是无约束的  

- 详细说明   x   更新步骤 (a)。解释如何有效地计算它（参见你之前的实验操作）



x 更新步骤 (a) 是通过最小化增强拉格朗日函数来进行的.
$$
K_o(x) = J(x) + \lambda^t(x - z) + \frac{\rho}{2} \|x - z\|^2\\
$$

$$
J(x) = |y - Hx|^2 + \mu |Dx|^2
$$

- 其中 $D$ 是差分矩阵，$\lambda$ 是拉格朗日乘数，$z$ 是辅助变量，$\rho$ 是惩罚参数。



我们可以将该优化问题看作是一个关于  x  的无约束二次优化问题.它包含了两个二次项，一个关于数据拟合（与  y  和  Hx  相关），另一个关于正则化（与  Dx  相关）。另外，惩罚项  \|x - z\|^2  也为二次形式
$$
x_{k+1} = \arg \min_x \left( \|y - Hx\|^2 + \mu \|Dx\|^2 + \lambda^t(x - z) + \frac{\rho}{2} \|x - z\|^2 \right)
$$

为了简化分析，我们将目标函数的各项展开
$$
\left\{
\begin{aligned}
&①\quad  \|y - Hx\|^2 = (y - Hx)^T (y - Hx) = y^T y - 2y^T Hx + x^T H^T Hx, \\
&②\quad  \mu \|Dx\|^2 = \mu (Dx)^T (Dx) = \mu x^T D^T D x, \\
&③\quad  \lambda^t (x - z) = \lambda^t x - \lambda^t z, \\
&④\quad  \frac{\rho}{2} \|x - z\|^2 = \frac{\rho}{2} (x - z)^T (x - z) = \frac{\rho}{2} (x^T x - 2x^T z + z^T z).
\end{aligned}
\right.
$$


$$
K_o(x) = y^T y - 2y^T Hx + x^T H^T Hx + \mu x^T D^T D x + \lambda^t x - \lambda^t z + \frac{\rho}{2}(x^T x - 2x^T z + z^T z)
$$

我们对 $x$ 求导，并令导数为零：
$$
\frac{\partial K_o(x)}{\partial x} = -2H^T y + 2H^T Hx + 2\mu D^T D x + \lambda + \rho (x - z)
$$

$$
-2H^T y + 2H^T Hx + 2\mu D^T D x + \lambda + \rho (x - z) = 0
$$

$$
(H^T H + \mu D^T D + \rho I)x = H^T y + \rho z - \lambda
$$

矩阵 $H^T H + \mu D^T D + \rho I$ 是对称正定矩阵，因此我们可以使用共轭梯度法或矩阵分解。





#### 2.2.2 更新辅助变量   z  

现在我们来讨论辅助变量 $z$ 的更新，即上面提到的步骤 (b)，此时给定当前的 $x$ 和 $\lambda$ 值。再次考虑拉格朗日函数 $L(x, z, \lambda)$，我们可以将 $x$ 忽略，并最小化


$$
K_a(z) = I_C(z) + \lambda^t(x - z) + \frac{\rho}{2} \|x - z\|^2
$$

主要考虑以下两点：

a. 这是一个关于 $z_p$ 的可分函数，对于 $p = 1, \ldots, P$ ，并且

b. 对于每个 $z_p$，它是一个简单的二次多项式 (对于 $z_p \geq 0$) 或 $+\infty$ (对于 $z_p \leq 0$ ，如图所示)

仔细思考这一点，并明确   z   更新步骤 (b)  

$$
目标函数  K_a(z)  是可分的，也就是说它关于每个像素  z_p  的项是独立的。因此，我们可以逐个像素进行优化\\

K_a(z_p) = I_C(z_p) + \lambda_p(x_p - z_p) + \frac{\rho}{2} (x_p - z_p)^2
\\
这其中包含:\ \
\left\{
\begin{aligned}
	&•	 I_C(z_p) ：指示函数，确保  z_p \geq 0  并且在支撑集之外  z_p = 0 \\
	&•	 \lambda_p(x_p - z_p) ：拉格朗日乘数项\\
	&•	 \frac{\rho}{2} (x_p - z_p)^2 ：二次惩罚项，用于将  z_p  约束在  x_p  附近\\
	\end{aligned}
\right.\\
其中指示函数I_C(z)，确保  z  满足约束:\ \
\left\{
\begin{aligned}
&z_p \geq 0 ，即所有像素的值都是非负的\\
&z_p = 0 ，对于不在给定支撑集  S  内的像素，像素值为 0\\
		\end{aligned}
\right.\\
$$



目标函数 $K_a(z)$ 是可分的，也就是说它关于每个像素 $z_p$ 的项是独立的。因此，我们可以逐个像素进行优化：
$$
K_a(z_p) = I_C(z_p) + \lambda_p(x_p - z_p) + \frac{\rho}{2} (x_p - z_p)^2
$$
这其中包含：
$$
\left\{
\begin{aligned}
& \ I_C(z_p) ：指示函数，确保 z_p \geq 0 并且在支撑集之外 z_p = 0, \\
& \ \lambda_p(x_p - z_p) ：拉格朗日乘数项, \\
& \ \frac{\rho}{2} (x_p - z_p)^2 ：二次惩罚项，用于将 z_p 约束在 x_p 附近.
\end{aligned}
\right.
$$
其中指示函数 $I_C(z)$ 确保 $z$ 满足约束：
$$
\left\{
\begin{aligned}
&z_p \geq 0 ，即所有像素的值都是非负的, \\
&z_p = 0 ，对于不在给定支撑集 S 内的像素，像素值为 0.
\end{aligned}
\right.
$$


无约束最小化：

首先，如果不考虑约束条件（即忽略 $I_C(z_p)$ 的影响），我们可以对 $z_p$ 求导，最小化目标函数：
$$
\frac{\partial K_a(z_p)}{\partial z_p} = -\lambda_p - \rho(x_p - z_p)
$$
令其为 $0$，得到无约束的最优解：
$$
z_p = x_p + \frac{\lambda_p}{\rho}
$$
引入非负约束：

接下来，我们引入第一个约束 $z_p \geq 0$：
$$
z_p = \max(0, x_p + \frac{\lambda_p}{\rho})
$$
如果 $x_p + \frac{\lambda_p}{\rho}$ 为负数，我们将 $z_p$ 设置为 $0$，否则保留。

对于不在支撑集内的像素，直接设置为 $0$：
$$
z_p = 0 \quad \text{if} \quad p \notin S
$$




![image-20240911172815099](/Users/zehua/Library/Application Support/typora-user-images/image-20240911172815099.png)



#### 2.2.3 更新拉格朗日乘数   λ  

拉格朗日乘数的更新（步骤 (c)）是可分且直接的，非常容易并行实现为一行 Matlab 代码。它直接与等式约束的残差 $x - z$ 相关，并且当约束满足时，$\lambda$ 达到稳定值。

### 2.3 实际实现

本节涉及在 Matlab 计算环境中的实现以及结果的分析。预期的实现只考虑正值约束  

3.实现上一节中的优化方法。花时间正确构建并注释你的代码。  z  -更新步骤应类似于以下代码：

```matlab
% 无约束最小化器
Auxiliary = Object + Lambda/rho;

% 考虑正值约束
Auxiliary(Auxiliary < 0) = 0;
```

$$
\\
	\left\{
\begin{aligned}
&	1.	初始化：设  z_0 = 0 ， \lambda_0 = 0 ，并设定收敛阈值  \epsilon = 10^{-5} \\ 
\\
&	2.	对于每次迭代  k ，进行以下操作:\\
&	\left\{
\begin{aligned}
	&•	x 更新：求解线性系统来更新  x  :\ \

(H^T H + \mu D^T D + \rho I)x_{k+1} = H^T y + \rho z_k - \lambda_k\\

&•	z 更新：逐像素更新  z  :\ \

z_{k+1,p} = \max(0, x_{k+1,p} + \frac{\lambda_{k,p}}{\rho}) \quad \text{如果} \ p \in S\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ 

 对于  p \notin S ，直接令  z_{k+1,p} = 0 \\
&	•	拉格朗日乘数更新:\ \

 \lambda_{k+1} = \lambda_k + \rho (x_{k+1} - z_{k+1})\\
		\end{aligned}
\right.\\
\\
&	3.	重复上述步骤，直到满足收敛条件:\ \

 \|x_k - x_{k-1}\| < \epsilon\\
		\end{aligned}
\right.\\
$$







ADMM 算法的实现步骤如下:

1. 初始化：设 $z_0 = 0$，$\lambda_0 = 0$，并设定收敛阈值 $\epsilon = 10^{-5}$。

2. 对于每次迭代 $k$，进行以下操作：  

   - **$x$ 更新**：求解线性系统来更新 $x$：  

     $$
     (H^T H + \mu D^T D + \rho I)x_{k+1} = H^T y + \rho z_k - \lambda_k
     $$

   - **$z$ 更新**：逐像素更新 $z$：  

     $$
     z_{k+1,p} = \max(0, x_{k+1,p} + \frac{\lambda_{k,p}}{\rho}) \quad \text{如果} \ p \in S
     $$
     对于 $p \notin S$，直接令：

     $$
     z_{k+1,p} = 0
     $$

   - **拉格朗日乘数更新**：  

     $$
     \lambda_{k+1} = \lambda_k + \rho (x_{k+1} - z_{k+1})
     $$

3. 重复上述步骤，直到满足收敛条件：  

   $$
   \|x_k - x_{k-1}\| < \epsilon
   $$





4.将结果与使用标准（无约束的）Wiener-Hunt 方法得到的结果进行比较，特别是图像分辨率和大脑周围的黑色区域方面  

> [!IMPORTANT]
>
> ​	•	**ADMM 方法**：通过引入正则化项和约束条件（正值约束和支撑集约束），它可以更好地恢复图像，同时避免出现负值像素和大脑周围的黑色区域不为零的情况。ADMM 通过引入拉格朗日乘数和交替更新法，逐步逼近最优解。
>
> ​	•	**Wiener-Hunt 方法**：这是标准的去卷积方法，它通过最小化二次目标函数，得到复原图像。但是，它没有约束条件，因此可能会导致负值像素，并且无法控制图像中的支撑集外像素。







