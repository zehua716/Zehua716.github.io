---
title: 'My First Post'
date: '2024-01-14T07:07:07+01:00'
draft: false
tags: ["tag1"]
---

测试222

333333444444

```python
def add(a, b):
    a + b
```
# 			Védio 3D---(Reconstruction 3D)

# 				Vision par ordinateur





## 											第一次课

### 一、**理论部分---图像投影模型与3D重建理论**

### **I. SLAM的概念**

- SLAM（Simultaneous Localization and Mapping，即同步定位与地图构建） 
- 通过估计每个相机的位置和场景的三维点，实现对场景的重建 

---

### **II. 逆向二维图像**

​		从二维图像中提取三维场景的信息，即进行3D重建

**逆向投影**

- 图像是三维场景经过投影后的二维表示，要恢复三维信息，需要逆转这个投影过程 
- 建立一个数学模型，描述三维场景如何投影到二维图像中，然后尝试逆向求解 

---

### **III. 针孔相机模型（Modèle Sténopé）**

**1. 模型概述**

- **定义**：针孔相机模型假设所有的光线都通过一个公共点，即光心（光学中心） 
- **优点**：模型简单，易于逆向计算，在三维重建中广泛使用 

**2. 坐标系和符号规定**

- **摄像机坐标系**：
  - **原点O<sub>C</sub>**：光学中心，坐标为 (0, 0, 0) 
  - **轴方向**：建立右手坐标系，X<sub>C</sub> 向右，Y<sub>C</sub> 向下，Z<sub>C</sub> 指向后方（场景深度方向） 
  - **优势**：Z 轴指向后方，物体深度为正，符合直觉 

**3. 三维点的投影到归一化焦平面**

- **三维点表示**：
- **点 U**：坐标为 (U<sub>X</sub>, U<sub>Y</sub>, U<sub>Z</sub>)，表示空间中的一个三维点 
- **归一化焦平面**：

  - 一个与光心O<sub>C</sub>距离为 1 的平面（Z<sub>C</sub> = 1），因此称为归一化聚焦平面 
  - 将远处的三维点投影到此平面上 用m表示 

**4. 齐次坐标与非齐次坐标**

- **齐次坐标（Homogeneous Coordinates）**：

  - **定义**：在原有坐标后增加一个维度（通常为 1），方便表示投影和变换 
  - **表示**：对于二维点 m = (m<sub>X</sub>, m<sub>Y</sub>$)^T$，其齐次坐标为

  $$
    \bar{m} = (m_X, m_Y, 1)^T\
  $$

  - **作用**：在后续理论运算中，很多情况下矩阵维度不匹配(因为三维是三个坐标，二维是两个)，因此必须增加一个维度用于运算矩阵，最后运算完再将其标准化即可

- **非齐次坐标（Inhomogeneous Coordinates）**：

  - 标准的笛卡尔坐标表示法，不包含额外的维度 

---

### IV. 摄像机的线性校准（Calibration）

**1. 从归一化焦平面到图像平面**

- **目的**：将归一化焦平面上的点$m=(m_x,m_y)^T$映射到图像平面上的点$p=(p_u,p_v)^T$

- **线性变换**：

  - **变换公式**：
    $$
    \begin{cases}
    p_u = f \cdot m_x + u_0 \\
    p_v = f \cdot m_y + v_0
    \end{cases}
    $$

  - **焦距 f**

  - **m<sub>X</sub>, m<sub>Y</sub> 是归一化焦平面上的点**

  - **光学中心在图像平面中的坐标(U<sub>0</sub>, V<sub>0</sub>)**：也就是要注意，在图像平面中的光学中心并不一定是$(0,0)$

  <script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js">
  <link href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">

    