---
title: "3D Reconstruction Theory"
# author: "Zehua"
date: "2024-10-21T16:25:17+01:00"
lastmod: "2024-11-12T17:12:35+08:00"
# draft: false
summary: "In the field of computer vision within artificial intelligence, the subject focuses on theoretical topics including camera model calibration, homography estimation, epipolar geometry, bundle adjustment, etc., to achieve applications such as image calibration, stitching, and bundle calibration."
description: "Course notes, for personal study and review only."
tags: ["Image Processing", "Computer Vision"]
# categories: "posts"
# cover:
#     image: "images/.jpg"
# comments: true
# hideMeta: false
# searchHidden: false
# ShowBreadCrumbs: true
# ShowReadingTime: false


---



## **1. Image Projection Model and 3D Reconstruction Theory**



### **1.1. Concept of SLAM**



- SLAM (Simultaneous Localization and Mapping).

- By estimating each camera’s position and the scene’s 3D points, achieve scene reconstruction.



### **1.2. Reverse 2D Image**



Extract 3D scene information from a 2D image, i.e., perform $3D$ reconstruction.



**Inverse Projection**

- The image is a 2D representation of the 3D scene after projection. To recover 3D information, it is necessary to reverse this projection process.

- Establish a mathematical model describing how the 3D scene is projected onto the 2D image, then attempt to solve it inversely.



### **1.3 Pinhole Camera Model**



#### **1.3.1 Model Overview**



**Definition**: The pinhole camera model assumes that all light rays pass through a common point, the optical center.

  - **Advantages**: Simple model, easy to compute inversely, widely used in 3D reconstruction.



#### **1.3.2 Coordinate Systems and Notation Conventions**



  - **Camera Coordinate System**:





**Origin $O_C$**: Optical center, coordinates $(0, 0, 0)$.

  - **Axis Directions**: Establish a right-handed coordinate system, $X_C$ to the right, $Y_C$ downward, $Z_C$ pointing backward (scene depth direction).

  - **Advantage**: The $Z$ axis points backward, object depth is positive, which is intuitive.



#### **1.3.3 Projection of 3D Points to Normalized Focal Plane**



  - **3D Point Representation**:

  - **Point $U$**: Coordinates $(U_X, U_Y, U_Z)$, representing a 3D point in space.

  - **Normalized Focal Plane**:

  - A plane at a distance of $1$ from the optical center $O_C$ ($Z_C = 1$), called the normalized focal plane.

  - Project distant 3D points onto this plane.



#### **1.3.4 Homogeneous Coordinates and Inhomogeneous Coordinates**



**Homogeneous Coordinates**:

  - **Definition**: Adding an extra dimension (usually $1$) to the original coordinates to facilitate projection and transformation.

  - **Representation**: For a 2D point $m = (m_X, m_Y)^T$, its homogeneous coordinates are $\bar{m} = (m_X, m_Y, 1)^T$.



**Purpose**: Homogeneous coordinates facilitate matrix operations, especially during projection and transformation processes.

  - **Inhomogeneous Coordinates**:

  - Standard Cartesian coordinate representation, without extra dimensions.



### **1.4 Linear Calibration of the Camera**



#### **1.4.1 From Normalized Focal Plane to Image Plane**



  - **Image Plane**:

  - **Coordinate System**: Pixel coordinate system, typically with the image’s top-left corner as the origin, $X$ axis to the right (column index), $Y$ axis downward.

  - **Purpose**: Map points on the normalized focal plane to actual image pixel coordinates.

  - **Linear Transformation**:

  - **Transformation Formula**:



<div>$$\begin{cases} P_U = f \cdot m_X + U_0 \\ P_V = f \cdot m_Y + V_0 \end{cases}$$</div>





  - **Focal Length $f$**

  - $m_X$, $m_Y$ **are points on the normalized focal plane**



  - **Optical Center Coordinates on the Image Plane** $(U_0, V_0)$



#### **1.4.2 Camera Intrinsic Matrix**



  - Represent the above linear transformation in matrix form.



<div>$$K = \begin{pmatrix} f & 0 & U_0 \\ 0 & f & V_0 \\ 0 & 0 & 1 \end{pmatrix}$$</div>





  - **Matrix Mapping Relationship**:



<div>$$\underline{P} = K \cdot \underline{m}$$</div>





Where, $\underline{P}$ is the homogeneous coordinates of points on the image plane.



#### **1.4.3 Inverse Process**



  - **From Image Plane to Normalized Focal Plane**:



<div>$$\underline{M} = K^{-1} \cdot \underline{P}$$</div>





#### **1.4.4 Viewing Frustum**



  - Represents the spatial range that the camera can see. By converting the four corner points of the image to the normalized focal plane and then connecting them to the optical center, forming a viewing frustum.



### **1.5 Distortion Modeling and Correction**



#### **1.5.1 Sources of Camera Distortion**



  - Optical defects in actual camera lenses, especially in wide-angle lenses, causing image distortion, straight lines becoming curved, image edges stretched or compressed.



#### **1.5.2 Distortion Model**



  - **Distortion Function**:

  - Apply a distortion function to ideal points on the normalized focal plane (from ideal image to distorted image) to obtain distorted points, which are actual $2D$ points on the distorted focal plane.



<div>$$\underline{m}_d = d(\underline{m}, k)$$</div>





Where, $k$ are distortion parameters.



  - **Example: Polynomial Radial Distortion Model**



<div>$$M_d = \left(1 + k_1 \|m\|_2^2 + k_2 \|m\|_2^4 + \dots \right) m$$</div>





Where: $|m|_2^2 = m_x^2 + m_y^2$



### **1.6 Implementation of Distortion Correction**



#### **1.6.1 Task Description**



  - **Goal**: Correct the distorted actual image to an ideal undistorted image.



#### **1.6.2 Implementation Steps**



**Define Parameters**: Ideal camera intrinsic matrix $K_{\text{ideal}}$; distorted camera intrinsic matrix $K_{\text{real}}$; distortion parameters $k$.



**For each ideal image pixel coordinate, perform the following steps:**

​	1.	**Convert pixel coordinates to normalized focal plane**:



<div>$$\underline{m} _{\text{ideal}} = K_{\text{ideal}}^{-1} \cdot \underline{P} _{\text{ideal}}$$</div>







​	2.	**Apply distortion function**:



<div>$$\underline{m}_d = d(\underline{m} _{\text{ideal}}, k)$$</div>







​	3.	**Map back to actual image coordinate system**:



<div>$$\underline{P} _{\text{real}} = K_{\text{real}} \cdot \underline{m}_d$$</div>







​	4.	**Interpolation**:

Perform interpolation on $\underline{P} _{\text{real}}$ (since coordinates may be non-integer, possibly use bilinear interpolation)

​	5.	**Generate corrected image**



## **2.  2D Rigid Transformation and Homography**



### **2.1 2D Rigid Transformation**



2D rigid transformations include **translation** and **rotation**.



#### **2.1.1 Rotation**



<div>$$\mathbf{U}^c = \overrightarrow{O _c U}^c \quad \mathbf{U}^w = \overrightarrow{O _w U}^w$$</div>





<div>$$\mathbf{R} _{wc} \underline{\mathbf{U}}^c = \mathbf{R} _{wc} \cdot \overrightarrow{O _c U}^c = \overrightarrow{O _w U}^w$$</div>





Select a vector from one reference frame and then transform it to another coordinate frame.



$\mathbf{R} _{wc}$ is an orthogonal matrix.



#### **2.1.2 Translation**



<div>$$\mathbf{T} _{wc} = \overrightarrow{O _w O _c}^{w}$$</div>





#### **2.1.3 Rigid Transformation Formula**



<div>$$\mathbf{U}^w = \mathbf{R} _{wc} \cdot \mathbf{U}^c + \mathbf{T} _{wc}$$</div>





**Proof:**



<div>$$\mathbf{R} _{wc} \cdot \mathbf{U}^c + \mathbf{T} _{wc} = \mathbf{R} _{wc} \cdot \overrightarrow{O _c U}^c + \overrightarrow{O _w O _c}^w = \overrightarrow{O _c U}^w + \overrightarrow{O _w O _c}^w = \overrightarrow{O _w U}^w = \mathbf{U}^w$$</div>





#### **2.1.4 Homogeneous Coordinates:**



<div>$$\underline{\mathbf{U}}^w = \begin{bmatrix} \mathbf{U}^w \\ 1 \end{bmatrix}$$</div>





<div>$$\mathbf{M} _{wc} = \begin{bmatrix} \mathbf{R} _{wc} & \mathbf{T} _{wc} \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} r _{11} & r _{12} & r _{13} & t _{x} \\ r _{21} & r _{22} & r _{23} & t _{y} \\ r _{31} & r _{32} & r _{33} & t _{z} \\ 0 & 0 & 0 & 1 \end{bmatrix}$$</div>





#### **2.1.5 Inverse Transformation:**



<div>$$\mathbf{M} _{cw} = \mathbf{M} _{wc}^{-1}$$</div>





#### **2.1.6 Composability of Transformations:**



<div>$$\mathbf{M} _{ab} \cdot \mathbf{M} _{bc} = \mathbf{M} _{ac}$$</div>





### **2.2 Homography**



#### **2.2.1 Plane Scene Assumption**



<div>$$\mathbf{U} _i^A = z _i^A \cdot \underline{\mathbf{m}} _{Ai}$$</div>





This equation means that the point $\mathbf{U} _i ^ A$ can be represented by $\underline{\mathbf{m}} _{Ai}$ by multiplying its depth, since $\underline{\mathbf{m}} _{Ai}$ has unit depth.



#### **2.2.2 Finding the Correspondence Between $\underline{\mathbf{m}} _{Ai}$ and $\underline{\mathbf{m}} _{Bi}$**



With only this equation, how do we find the correspondence between $\underline{\mathbf{m}} _{Ai}$ and $\underline{\mathbf{m}} _{Bi}$? In simple terms, how do we perform coordinate correspondence transformations?

​	1.	**Key Formula for the Normal**

We need to recall a property to obtain a key formula involving the normal and the plane.

In reference frame $A$, the equation of plane $P$ is: $ax + by + cz + d = 0$, where $a, b, c$ are the components of the plane’s normal vector, and $d$ is a constant representing the relative distance between plane $P$ and the origin $O _A$.

In vector form, the plane equation can be simplified to:



<div>$$\mathbf{n} _A^T \mathbf{U} _i^A + d = 0$$</div>





$\mathbf{n} _A^T$ represents the normal vector of plane $P$ in reference frame $A$.

Through the vector form of the plane equation, we obtain a very important formula involving the normal vector.



​	2.	**Obtaining the Depth Expression Using Variable Substitution**

Substitute $\mathbf{U} _i^A = z _i^A \cdot \underline{\mathbf{m}} _{A,i}$ into the above equation:



<div>$$\mathbf{n} _A^T \cdot z _i^A \cdot \underline{\mathbf{m}} _{A,i} + d = 0 \quad \Rightarrow \quad z _i^A = -\dfrac{d}{\mathbf{n} _A^T \cdot \underline{\mathbf{m}} _{A,i}}$$</div>





In this way, we have introduced $\underline{\mathbf{m}} _{Ai}$, where $z _i^A = -\dfrac{d}{\mathbf{n} _A^T \cdot \underline{\mathbf{m}} _{A,i}}$ represents the depth. In other words, by using the two equations of $\mathbf{U} _i^A$, we have replaced $\mathbf{U} _i^A$, thus obtaining the depth $z _i^A$. However, this still does not solve the problem $\Rightarrow$ that is to say, having only the equation related to $\underline{\mathbf{m}} _{Ai}$ is not enough; we also need to approach from $\underline{\mathbf{m}} _{Bi}$.



​	3.	**Next, We Find the Point $\underline{\mathbf{m}} _{Bi}$ in Coordinate Frame $B$**

Starting from the rigid transformation formula $\mathbf{U}^w = \mathbf{R} _{wc} \cdot \mathbf{U}^c + \mathbf{T} _{wc}$, we can see that projecting from $c$ to $w$ only requires transforming $\mathbf{U}^c$. In other words, to obtain $\underline{\mathbf{m}} _{Bi}$, we only need to perform a rigid transformation on $\underline{\mathbf{m}} _{Ai}$.



<div>$$\underline{\mathbf{m}} _{B,i} = \Pi \left( \mathbf{R} _{BA} \mathbf{U} _i^A + \mathbf{t} _{BA} \right)$$</div>





where $\Pi(\cdot)$ is the projection function.



<div>$$\underline{\mathbf{m}} _{B,i}= \Pi \left( \mathbf{R} _{BA} \left( -\dfrac{d}{\mathbf{n} _A^T \cdot \underline{\mathbf{m}} _{A,i}} \right) \cdot \underline{\mathbf{m}} _{A,i} + \mathbf{t} _{BA} \right)$$</div>





Multiply both sides of the above equation by $-\dfrac{\mathbf{n} _A^T \cdot \underline{\mathbf{m}} _{A,i}}{d}$:



<div>$$\underline{\mathbf{m}} _{B,i}= \Pi \left( \mathbf{R} _{BA} \cdot \underline{\mathbf{m}} _{A,i} - \dfrac{\mathbf{n} _A^T \cdot \underline{\mathbf{m}} _{A,i}}{d} \cdot \mathbf{t} _{BA} \right)$$</div>





<div>$$\underline{\mathbf{m}} _{B,i} = \Pi \left( \left( \mathbf{R} _{BA} - \dfrac{\mathbf{t} _{BA} \cdot \mathbf{n} _A^T}{d} \right) \cdot \underline{\mathbf{m}} _{A,i} \right)$$</div>





This establishes the correspondence between point $A$ and point $B$ on their respective normalized planes.

**Question:** In the above formula, why does multiplying both sides by $-\dfrac{\mathbf{n} _A^T \cdot \underline{\mathbf{m}} _{A,i}}{d}$ not change the equation?

The projection function $\Pi(\cdot)$ is a scale-invariant operation (it only considers direction and relative position, not absolute scale). Therefore, even if we multiply the right side by $-\dfrac{\mathbf{n} _A^T \cdot \underline{\mathbf{m}} _{A,i}}{d}$, it does not affect the condition for the equality to hold because the projection result remains the same.



#### **2.2.3 Finding the Correspondence Between $\underline{\mathbf{P}} _{A,i}$ and $\underline{\mathbf{P}} _{B,i}$**



We know:



<div>$$\left\{ \begin{aligned} \underline{\mathbf{m}} _{A,i} = K _A^{-1} \cdot \underline{\mathbf{P}} _{A,i}\\ \underline{\mathbf{m}} _{B,i} = K _B^{-1} \cdot \underline{\mathbf{P}} _{B,i} \end{aligned} \right.$$</div>





$\underline{\mathbf{P}} _{B,i} = K _B \cdot \underline{\mathbf{m}} _{B,i} \Rightarrow$ Substitute the obtained $\underline{\mathbf{m}} _{B,i}$:



<div>$$\underline{\mathbf{P}} _{B,i} = K _B \cdot \Pi \left( \left( \mathbf{R} _{BA} - \dfrac{\mathbf{t} _{BA} \cdot \mathbf{n} _A^T}{d} \right) \cdot \underline{\mathbf{m}} _{A,i} \right)$$</div>





<div>$$\underline{\mathbf{P}} _{B,i} = K _B \cdot \Pi \left( \left( \mathbf{R} _{BA} - \dfrac{\mathbf{t} _{BA} \cdot \mathbf{n} _A^T}{d} \right) \cdot K _A^{-1} \cdot \underline{\mathbf{P}} _{A,i} \right)$$</div>





Recall the property:



<div>$$K \cdot \Pi \left( \begin{bmatrix} a \\ b \\ c \end{bmatrix} \right) = \Pi \left( K \cdot \begin{bmatrix} a \\ b \\ c \end{bmatrix} \right)$$</div>





Using this property, we get:



<div>$$\underline{\mathbf{P}} _{B,i} = \Pi \left( K _B \cdot \left( \mathbf{R} _{BA} - \dfrac{\mathbf{t} _{BA} \cdot \mathbf{n} _A^T}{d} \right) \cdot K _A^{-1} \cdot \underline{\mathbf{P}} _{A,i}\right)$$</div>





#### **2.2.4 Obtaining the Homography Matrix $\mathbf{H} _{AB}$**



Assume:



$$\mathbf{H} _{AB} = K _B \cdot \left( \mathbf{R} _{BA} - \dfrac{\mathbf{t} _{BA} \cdot \mathbf{n} _A^T}{d} \right) \cdot K _A^{-1}$$



Therefore:



<div>$$\left\{ \begin{aligned} &\underline{\mathbf{P}} _{B,i} = \Pi \left( \mathbf{H} _{BA} \cdot \underline{\mathbf{P}} _{A,i} \right) \quad \quad  A \Rightarrow B\\ &\underline{\mathbf{P}} _{A,i} = \Pi \left( \mathbf{H} _{BA}^{-1} \cdot \underline{\mathbf{P}} _{B,i} \right) = \Pi \left( \mathbf{H} _{AB} \cdot \underline{\mathbf{P}} _{B,i} \right) \quad \quad  B \Rightarrow A \\ \end{aligned} \right.$$</div>





Through the homography matrix, we can transform a point from one camera’s image coordinate system to another camera’s image coordinate system, establishing a point mapping relationship.



#### **2.2.5 Estimating and Solving the Homography Matrix**



<div>$$\mathbf{H} _{AB} = \begin{bmatrix} h _1 & h _4 & h _7 \\ h _2 & h _5 & h _8 \\ h _3 & h _6 & h _9 \end{bmatrix}$$</div>





This is a homogeneous matrix with 9 parameters $h _1$ to $h _9$. Homogeneous matrices have redundancy in scale, leading to a loss of degrees of freedom.

  - **Simple Solution – Parameterization (Parameters to Estimate = Free Parameters)**



<div>$$\mathbf{H} _{AB} = \begin{bmatrix} h _1 & h _4 & h _7 \\ h _2 & h _5 & h _8 \\ h _3 & h _6 & 1 \end{bmatrix}$$</div>





<div>$$\mathbf{h} = \begin{bmatrix} h _1 \\ \vdots \\ h _8 \end{bmatrix}$$</div>





**How to Estimate $\mathbf{h}$?**

  - In this case, as long as we know one corresponding point, we can determine $h _1$ to $h _8$.



<div>$$\underline{\mathbf{P}} _{A,i} = \Pi \left( \begin{bmatrix} h _1 & h _4 & h _7 \\ h _2 & h _5 & h _8 \\ h _3 & h _6 & 1 \end{bmatrix} \cdot \underline{\mathbf{P}} _{B,i} \right)$$</div>





Since $\underline{\mathbf{P}} _{A,i}$ and $\underline{\mathbf{P}} _{B,i}$ are homogeneous coordinates, we expand them:



<div>$$\begin{bmatrix} P _{A,i,x} \\ P _{A,i,y} \\ 1 \end{bmatrix} = \Pi \left( \begin{bmatrix} h _1 & h _4 & h _7 \\ h _2 & h _5 & h _8 \\ h _3 & h _6 & 1 \end{bmatrix} \cdot \begin{bmatrix} P _{B,i,x} \\ P _{B,i,y} \\ 1 \end{bmatrix} \right)$$</div>





<div>$$\left\{ \begin{aligned} P _{A,i,x} = \dfrac{h _1 \cdot P _{B,i,x} + h _4 \cdot P _{B,i,y} + h _7}{h _3 \cdot P _{B,i,x} + h _6 \cdot P _{B,i,y} + 1} \\ P _{A,i,y} = \dfrac{h _2 \cdot P _{B,i,x} + h _5 \cdot P _{B,i,y} + h _8}{h _3 \cdot P _{B,i,x} + h _6 \cdot P _{B,i,y} + 1} \end{aligned} \right.$$</div>





<div>$$\left\{ \begin{aligned} P _{A,i,x} \cdot \left( h _3 \cdot P _{B,i,x} + h _6 \cdot P _{B,i,y} + 1 \right) = h _1 \cdot P _{B,i,x} + h _4 \cdot P _{B,i,y} + h _7 \\ P _{A,i,y} \cdot \left( h _3 \cdot P _{B,i,x} + h _6 \cdot P _{B,i,y} + 1 \right) = h _2 \cdot P _{B,i,x} + h _5 \cdot P _{B,i,y} + h _8 \end{aligned} \right.$$</div>





<div>$$\begin{bmatrix} P _{B,i,x} & 0 & -P _{A,i,x} \cdot P _{B,i,x} & P _{B,i,y} & 0 & -P _{A,i,x} \cdot P _{B,i,y} & 1 & 0 \\ 0 & P _{B,i,x} & -P _{A,i,y} \cdot P _{B,i,x} & 0 & P _{B,i,y} & -P _{A,i,y} \cdot P _{B,i,y} & 0 & 1 \end{bmatrix} \begin{bmatrix} h _1 \\ h _2 \\ h _3 \\ h _4 \\ h _5 \\ h _6 \\ h _7 \\ h _8 \end{bmatrix} = \begin{bmatrix} P _{A,i,x} \\ P _{A,i _y} \end{bmatrix}$$</div>





Since there are 8 unknowns, we need eight independent linear equations. Each pair of corresponding points provides two corresponding equations (i.e., equations 59), so at least four pairs of corresponding points are required. That is, four matching pairs $\left( P _{A,i}, P _{B,i} \right) \quad i = 1, 2, 3, 4$ are needed.



$\Rightarrow \mathbf{h}^* = \arg\min _{\mathbf{h}} \sum _{i=1}^{4} \left\lVert M _i \mathbf{h} - P _{A,i} \right\rVert _2^2 \Rightarrow$ Linear Least Squares



## **3. Robust Homography Estimation Using the RANSAC Algorithm**



### **3.1 Objective**



  - **Image Alignment and Stitching**: Achieve automatic image stitching by estimating the homography between two images.



### **3.2 Automatically Establishing Correspondences — SIFT Algorithm**



**Interest Point Detection**



  - Use algorithms like SIFT to detect feature points in both images (this code is provided by the instructor), eliminating the need to manually label corresponding points and utilizing algorithms to automatically establish correspondences between images.

  - Therefore, we can find the most similar point pairs in both images. However, note that the point pairs may not correspond correctly.

  - That is, there may be incorrect matches (outliers). In such cases, we cannot directly use the correspondences. Instead, we will use another algorithm called RANSAC to automatically assess the correctness of corresponding points and obtain the most optimal $H$ matrix to output.



### **3.3 Robust Estimation Using the RANSAC Algorithm**



​	1.	**Algorithm Concept:**

  - **Random Sample Consensus (RANSAC)** is a robust algorithm for estimating model parameters ($H$) in the presence of outliers (incorrect points).

  - It repeatedly performs random sampling to find the best-fitting model.

​	2.	**RANSAC Procedure:**

  - Repeat $N$ times

(The number of iterations is determined based on experience or computation):

​	1.	**Randomly Select 4 Pairs of Matching Points:**

  - 4 is the minimum number of matching points required to estimate the homography matrix.

  - Randomly select four from all matching points. Since it is uncertain which correspondences are correct, a subsequent estimation and evaluation criterion (Euclidean distance) is needed.

​	2.	**Estimate the Homography Matrix $H^k$:**

  - Use the selected 4 pairs of matching points and the $DLT$ algorithm (done in the previous experiment, its purpose and function are to estimate the homography matrix given known corresponding points) to estimate the homography matrix.

​	3.	**Compute Errors and Evaluate the Model:**

  - For all matching points (including those not selected), transform the second image’s point $P _{B _i}$ using the estimated $H^k$, i.e., $H^k P _{B _i}$.

  - Calculate the Euclidean distance between the transformed point (estimated point) and the actual point $P _{A _i}$ in the first image.

  - **Define the Cost Function**: Use a binary kernel function $\phi _c(d)$ (either 0 or 1):

  - When the distance $d < \tau$, consider the match correct, and the cost is 0.

  - When the distance $d \geq \tau$, consider the match incorrect, and the cost is 1.

  - **Total Cost**: $L^k = \sum _{i} \phi _c(|P _{A _i} - H^k P _{B _i}|)$

​	4.	**Update the Best Model:**

  - If the current cost $L^k$ is less than the previous minimum cost $L$, update $L$ and the corresponding $H$.

  - **Final Output:**

  - The homography matrix $H$ with the minimum cost.

​	3.	**Selection of Threshold $\tau$:**

  - $\tau$ is the distance threshold to determine whether a match is an inlier. It is usually chosen based on image resolution and matching accuracy, generally between $0.5$ to $3$ pixels.

  - Choosing a too large $\tau$ increases incorrect matches, while a too small $\tau$ may ignore correct matches.



### **3.4 Why Not Use Traditional Quadratic Cost Functions**



  - **Sensitivity Issues:**

  - Quadratic cost functions (such as least squares) are very sensitive to outliers. If a point has a large error, it will cause the cost function value to be excessively large, making the errors of other points irrelevant.

  - **Robustness:**

  - Binary kernel functions are insensitive to those extremely large or outlying points (all equal to 1), effectively suppressing the influence of outliers and making the estimation result more robust.

  - **Other Kernel Functions:**

  - Besides binary kernel functions, there are other robust kernel functions like the $Huber$ kernel and $Lorentzian$ kernel, which can balance error magnitude and robustness to some extent.



### **3.5 Limitations of the RANSAC Algorithm**



  - **Influence of the Number of Parameters:**

  - As the number of model parameters increases, the required number of random samples grows exponentially, significantly increasing computational cost.

  - **Applicable Range:**

  - RANSAC is suitable for cases with a small number of parameters, such as line fitting, fundamental matrix estimation, and homography estimation.



**4. Epipolar Geometry in Stereo Vision**



So far, we have studied the case of planar scenes and used homography to describe the relationship between two views. However, for general three-dimensional scenes, the planar assumption no longer holds. To address this, we introduce epipolar geometry.



**4.1 Epipolar Geometry**



Epipolar geometry can be well explained through a schematic diagram:

​	1.	**Consider Two Cameras Located in Reference Frames 1 and 2 Respectively**

The optical center of Camera 1 is $O_1$, and the optical center of Camera 2 is $O_2$. A point $U$ in space is projected onto the image planes of both cameras, resulting in points $\underline{m}_1$ and $\underline{m}_2$.

​	2.	**Problem Description**:

  - In general cases, we cannot make any assumptions about point $U$ (unlike the previous planar scene).

  - We need to find a method to establish a relationship between $\underline{m}_1$ and $\underline{m}_2$ without knowing $U$.



**4.2 Epipolar Plane and Epipolar Lines**



​	1.	**Epipolar Plane**

$U$ and the optical centers $O_1$, $O_2$ define a plane $\Rightarrow$ Points $m_1$, $m_2$, $O_1$, $O_2$ are coplanar $\Rightarrow$ This plane is called the epipolar plane.

**Epipolar Constraint** = **Coplanarity**, meaning $\underline{m}_1$, $\underline{m}_2$, $O_1$, $O_2$ are coplanar.

In stereo vision, the fundamental matrix $F$ and the essential matrix $E$ both rely on the coplanarity condition for their computation.

​	2.	**Epipolar Lines**

The epipolar plane intersects the image planes of the two cameras, resulting in epipolar lines $l_1$ and $l_2$ respectively.

$m_2$ is the projection of the three-dimensional point $U$ onto the image plane of the second camera. However, according to the constraints of epipolar geometry, $m_2$ must lie on the epipolar line $l_2$.

$\Rightarrow$ Given the position of point $m_1$, the corresponding epipolar line $l_2$ can be determined using the fundamental matrix $F$: $l_2 = F \cdot m_1$.

The fundamental matrix $F$ captures the relative pose and intrinsic parameters between the two cameras. This formula indicates that given point $m_1$, one can compute the epipolar line $l_2$ on which $m_2$ must lie.



**4.3 Epipolar Constraint**



**Objective**: Utilize the above geometric relationships to formalize the epipolar constraint and establish a mathematical relationship between $m_1$ and $m_2$.



**Define Vectors:**



<div>$$\left\{ \begin{aligned} 

&\mathbf{\underline{m}_1} \text{ is the vector from the optical center } O_1 \text{ to the image point } \underline{m}_1 \quad \overrightarrow{O_1 m_1}^{1} \\ 

&\mathbf{\underline{m}_2} \text{ is the vector from the optical center } O_2 \text{ to the image point } \underline{m}_2 \quad \overrightarrow{O_2 m_2}^{2} \\ 

&\mathbf{t_{12}} = \overrightarrow{O_1 O_2}^{1} \text{ is the translation vector between the two camera optical centers} 

\end{aligned} \right.$$</div>





**Define the Normal Vector of the Epipolar Plane:**



<div>$$\left\{ \begin{aligned} 

&\text{In reference frame } 1, \quad \overrightarrow{\mathbf{n}_1}^{1} = \underline{\mathbf{m}}_1 \times \mathbf{t} _{12} \\ 

&\text{In reference frame } 2, \quad \overrightarrow{\mathbf{n}_2}^{2} = \mathbf{R} _{21} \overrightarrow{\mathbf{n}_1}^{1}, \text{ where } \mathbf{R} \text{ is the rotation matrix between the cameras} 

\end{aligned} \right.$$</div>





**Note:**

  - Here, $\times$ denotes the cross product operation between two vectors. The result of the cross product is a vector **perpendicular to both vectors involved in the operation**, with its direction determined by the right-hand rule and magnitude equal to the area of the parallelogram formed by the two vectors.

  - Coordinate system transformations for the normal vector do not need to consider the translation component because unit normal vectors are not position coordinates. Direction vectors remain unchanged in magnitude during rotation and are unaffected by translation. In summary, normal vectors only consider the rotation matrix, while points need to consider both rotation and translation.



<div>$$\overrightarrow{\mathbf{n}_2}^{2} = \mathbf{R} _{21} \cdot \overrightarrow{\mathbf{n}_1}^{1} = \mathbf{R} _{21} \cdot \left( \underline{\mathbf{m}}_1 \times \mathbf{t} _{12} \right) = \mathbf{R} _{21} \cdot \underline{\mathbf{m}}_1 \times \mathbf{R} _{21} \cdot \mathbf{t} _{12}$$</div>





Since we previously know that $\mathbf{t} _{21} = \mathbf{R}*{21} \cdot \mathbf{t} _{12}$, the above equation becomes



<div>$$\overrightarrow{\mathbf{n}_2}^{2} = \mathbf{t} _{21} \times \left( \mathbf{R} _{21} \cdot \underline{\mathbf{m}}_1 \right)$$</div>





**Recall Properties of Cross Product Operations:**



<div>$$\mathbf{a} \times \mathbf{b} = \begin{bmatrix} a_x \\ a_y \\ a_z \end{bmatrix} \times \begin{bmatrix} b_x \\ b_y \\ b_z \end{bmatrix} = \begin{bmatrix} a_y b_z - a_z b_y \\ a_z b_x - a_x b_z \\ a_x b_y - a_y b_x \end{bmatrix} _{3 \times 1} \Rightarrow \left[\mathbf{a}\right]_{\times} = \begin{bmatrix} 0 & -a_z & a_y \\ a_z & 0 & -a_x \\ -a_y & a_x & 0 \end{bmatrix}$$</div>





<div>$$\mathbf{a} \times \mathbf{b} = \left[\mathbf{a}\right]_{\times} \mathbf{b} = \begin{bmatrix} 0 & -a_z & a_y \\ a_z & 0 & -a_x \\ -a_y & a_x & 0 \end{bmatrix} \begin{bmatrix} b_x \\ b_y \\ b_z \end{bmatrix}$$</div>





Using the above property, we can see that the cross product operation can be transformed into matrix operations. Therefore, using the above property, we obtain:



<div>$$\overrightarrow{\mathbf{n}_2}^{2} = \mathbf{t} _{21} \times \left( \mathbf{R} _{21} \cdot \underline{\mathbf{m}}_1 \right) = \left[ \mathbf{t} _{21} \right]_{\times} \cdot \mathbf{R} _{21} \cdot \underline{\mathbf{m}}_1$$</div>





Because $\overrightarrow{\mathbf{n}_2}^{2}$ is the normal vector of $\mathbf{m}_2$, we have



$\Rightarrow \mathbf{m}_2^T \cdot \overrightarrow{\mathbf{n}_2}^{2} = 0$



<div>$$\mathbf{m}_2^T \cdot \left[ \mathbf{t} _{21} \right]_{\times} \cdot \mathbf{R} _{21} \cdot \underline{\mathbf{m}}_1 = 0$$</div>





<div>$$\mathbf{m}_2^T \cdot \left( \left[ \mathbf{t} _{21} \right]_{\times} \cdot \mathbf{R} _{21} \right) \cdot \underline{\mathbf{m}}_1 = 0$$</div>





**4.4 Essential Matrix (Matrice Essentielle)**



​	1.	**Formula**

Assume



<div>$$\mathbf{E} _{21} = \left[ \mathbf{t} _{21} \right]_{\times} \cdot \mathbf{R} _{21} \quad \Rightarrow \quad \text{essential matrix}$$</div>





It contains information about the relative rotation $\mathbf{R}$ and translation $\mathbf{t}$ between the two cameras.

The original equation becomes $ \underline{\mathbf{m}}*2^T \cdot \mathbf{E} _{21} \cdot \underline{\mathbf{m}}_1 = 0 $



​	2.	**Degrees of Freedom**



<div>$$

5 \text{ degrees of freedom} \\ \downarrow\\ 5 \text{ DoF} \left( \begin{array}{c} 3 , \mathbf{R} _{21} \quad \text{rotation} \\ \quad 2 , \mathbf{t} _{21} \quad \text{translation} \end{array} \right)\\ \downarrow\\ \quad \quad \| \mathbf{t} _{21} \|_2 \quad \text{ unknown}

$$</div>





**Degrees of Freedom:**



<div>$$\left\{ \begin{aligned} 

&\text{The rotation matrix } \mathbf{R} \text{ has 3 degrees of freedom} \\ 

&\text{The translation vector } \mathbf{t} \text{ has 2 degrees of freedom (since the scale is unknown)} \\ 

&\text{Therefore, } \mathbf{E} \text{ has 5 degrees of freedom} 

\end{aligned} \right.$$</div>





  - **Degrees of Freedom (DoF)** refer to the number of independent parameters required to describe the essential matrix. In geometry and linear algebra, DoF reflects the number of independent directions or ways a system can vary without constraints.

  - The rotation matrix has 3 degrees of freedom, describing rotation in three-dimensional space.

  - The translation vector theoretically has 3 degrees of freedom in three-dimensional space. However, in the essential matrix, the translation vector typically only considers direction, ignoring magnitude (unknown scale), thus leaving the translation vector with only 2 effective degrees of freedom, describing the direction of translation.



**4.5 Fundamental Matrix (Matrix Fundamental)**



Continuing the transformation of the above formula:



<div>$$\underline{\mathbf{m}}_2^T \cdot \mathbf{E} _{21} \cdot \underline{\mathbf{m}}_1 = 0$$</div>





Given:



<div>$$\left\{ \begin{aligned} 

\underline{\mathbf{m}}_2 = K^{-1} \cdot \underline{\mathbf{P}}_2 \\ 

\underline{\mathbf{m}}_1 = K^{-1} \cdot \underline{\mathbf{P}}_1 

\end{aligned} \right.$$</div>





<div>$$\underline{\mathbf{P}}_2^T \cdot (K^{-1})^T \cdot \mathbf{E} _{21} \cdot K^{-1} \cdot \underline{\mathbf{P}}_1 = 0$$</div>





When the camera intrinsics are unknown or not considered, we introduce a fundamental matrix $\mathbf{F}$ to encapsulate $K$.



Assume $\mathbf{F} _{21} = (K^{-1})^T \cdot \mathbf{E} _{21} \cdot K^{-1}$



<div>$$\mathbf{F} _{21} : \text{ fundamental matrix} \quad \Rightarrow \quad 7 \text{ DoF}\quad \left\{ \begin{aligned} 

& \text{- homogeneous matrix} \\ 

& \text{- rank}(\mathbf{F} _{21}) = 2 \quad \Rightarrow \quad \det(\mathbf{F} _{21}) = 0 

\end{aligned} \right.$$</div>





**Properties:**



<div>$$\left\{ \begin{aligned} 

&\text{Homogeneous: The fundamental matrix } \mathbf{F} \text{ is a homogeneous matrix, and it can be scaled by any non-zero scalar without changing its properties} \\ 

&\text{Rank Constraint:} \mathbf{F} \text{ has rank } 2 

\end{aligned} \right.$$</div>





The original equation becomes $ \underline{\mathbf{P}}*2^T \cdot \mathbf{F} _{21} \cdot \underline{\mathbf{P}}_1 = 0 $



<div>$$\text{Let:} \quad \mathbf{L}_2 = \mathbf{F} _{21} \cdot \underline{\mathbf{P}}_1 = \begin{bmatrix} a \\ b \\ c \end{bmatrix}$$</div>





<div>$$\underline{\mathbf{P}}_2^T \cdot \mathbf{L}_2 = 0 \quad \Leftrightarrow \quad a P_{2,x} + b P_{2,y} + c = 0$$</div>





This is the equation of a line on Camera 2’s image plane $\Rightarrow$ Epipolar Line



**4.6 Estimation of Essential and Fundamental Matrices**



  - **Camera Calibrated** $\Rightarrow$ Estimate the essential matrix $\mathbf{E}$ (5 degrees of freedom) $\Rightarrow$ 5-point correspondence algorithm

  - **Camera Uncalibrated** $\Rightarrow$ Estimate the fundamental matrix $\mathbf{F}$ (7 degrees of freedom) $\Rightarrow$ 7-point correspondence algorithm

  - **Solution** $\Rightarrow$ 8-point correspondence algorithm $\Rightarrow$ Intentionally ignoring the constraint $\det(\mathbf{F}) = 0$



**4.7 Algorithm**



**Steps of the 8-Point Correspondence Algorithm:**

​	1.	**Collect Corresponding Point Pairs:**

Although $\mathbf{F}$ has 7 degrees of freedom, the algorithm ignores the rank-2 constraint. Therefore, at least 8 pairs of corresponding points are required to estimate $\mathbf{F}$.

​	2.	**Construct a Linear System of Equations**

For each pair of corresponding points $(\mathbf{m}_1, \mathbf{m}_2)$, construct the equation:



<div>$$\underline{\mathbf{m}}_2^T \cdot \mathbf{E} _{21} \cdot \underline{\mathbf{m}}_1 = 0$$</div>





<div>$$\underline{\mathbf{P}}_2^T \cdot \mathbf{F} _{21} \cdot \underline{\mathbf{P}}_1 = 0$$</div>







​	3.	**Solve:**

Represent the system of equations as: $\underline{\mathbf{P}}_2^T \cdot \mathbf{L}_2 = 0$

​	4.	**RANSAC Algorithm Steps**

Handle outliers (incorrect matches) in the set of corresponding points and robustly estimate $\mathbf{F}$.

  - **Random Sampling**: Use the **8-point correspondence algorithm** to estimate $\mathbf{F}$.

  - **Model Evaluation**: Use the estimated $\mathbf{F}$ to calculate the epipolar constraint error for all corresponding point pairs, i.e., the distance from each point to its corresponding epipolar line.

  - **Determine Inliers**: Based on a set distance threshold, determine which corresponding points are inliers.

  - **Iteration**: Repeat the above process until the model with the highest number of inliers is found.



**5. Bundle Adjustment**



  - Bundle adjustment is a technique that simultaneously optimizes camera parameters (including position, orientation, and intrinsic parameters) and the positions of three-dimensional points in the scene.

  - Its core idea is to make the optimized model more consistent with actual observations by minimizing the re-projection error of three-dimensional points onto the images.

  - Remember these five words: **Minimize Re-projection Error**



**5.1 Case of Two Cameras**



**5.1.1 Data**



$$\left( P_{A,i}, P_{B,i} \right) _{i=1,\dots,N} \implies N \text{ correspondences}$$



**5.1.2 Parameters to Estimate**



Camera poses and the three-dimensional point cloud data set.



<div>$$\mathbf{R} _{W1} \quad \mathbf{t} _{W1} \quad \mathbf{R} _{W2} \quad \mathbf{t} _{W2} \quad \left\{\mathbf{U}^w_i \right\} _{i=1,\dots,N}$$</div>





**5.1.3 Loss Function**



<div>$$\mathcal{L} \left( \mathbf{R} _{w1}, \mathbf{t} _{w1}, \mathbf{R} _{w2}, \mathbf{t} _{w2}, \left\{ \mathbf{U}^w_i \right\} _{i=1,\dots,N} \right) = \sum_{i=1}^{N} \left( \left\lVert P_{1,i} - K_1 \Pi \left( \mathbf{R} _{w1}^T \mathbf{U}_i^{w} - \mathbf{R} _{w1}^T \mathbf{t} _{w1} \right) \right\rVert_2^2 + \left\lVert P_{2,i} - K_2 \Pi \left( \mathbf{R} _{w2}^T \mathbf{U}_i^{w} - \mathbf{R} _{w2}^T \mathbf{t} _{w2} \right) \right\rVert_2^2 \right)$$</div>





Where:

  - $K_A$ and $K_B$ are the intrinsic matrices of cameras $A$ and $B$, respectively.

  - $\Pi(\cdot)$ is the projection function that projects three-dimensional points onto a two-dimensional plane.

  - $\mathbf{R} _{w1}^T$ and $\mathbf{R} _{w2}^T$ are equivalent to $\mathbf{R} _{1w}$ and $\mathbf{R} _{2w}$, which transform points from the world coordinate system to the camera coordinate system.

  - $\mathbf{R} _{w1}^T \mathbf{t} _{w1}$ is equivalent to $\mathbf{t} _{1w}$, representing the translation vector.

  - $\mathbf{U}_i^{1} = \mathbf{R} _{w1}^T \cdot \mathbf{U}_i^{w} - \mathbf{R} _{w1}^T \cdot \mathbf{t} _{w1}$, which transforms $\mathbf{U}_i^{w}$ to $\mathbf{U}_i^{1}$, i.e., from the world coordinate system to the camera coordinate system.

  - **Difference**: The image coordinates in camera $A$ or $B$ (actual) minus the estimated image coordinates obtained through three-dimensional space rotation and transformation equals the re-projection error.



**5.2 Case of Multiple Cameras**



**5.2.1 Data**



Detected points in each image are:



<div>$$\left\{ \left\{ P_{m,i} \right\} _{i=1,\dots,N_m} \right\} _{m=1,\dots,M}$$</div>





  - These points form tracks across different viewpoints.

  - Points detected by the $m$-th camera, where $N_m$ is the number of points detected by the $m$-th camera.



<div>$$\left\{ \text{p2d-id}_m, \ \text{p3d-id}_m \right\} _{m=1,\dots,M}$$</div>





Where:

  - $\text{p2d-id}_m$ is the index of the two-dimensional point in the image.

  - $\text{p3d-id}_m$ is the index of the corresponding three-dimensional point in the point cloud.

  - Both have dimensions of $C_m \times 1$.



**5.2.2 Parameters to Estimate**



  - **Camera Extrinsics**:



<div>$$ \left\{ \left( \mathbf{R} _{wm}, \mathbf{t} _{wm} \right) \right\} _{m=1,\dots,M} $$</div>







  - **Positions of Three-Dimensional Points**:



<div>$$ \left\{ \mathbf{U}_i^{w} \right\} _{i=1,\dots,N} $$</div>









**5.2.3 Loss Function**



The cost function extends to calculate errors across all cameras and all detected points, minimizing the distance between projected points and actual observed points:



<div>$$\mathcal{L}(x) = \sum_{m=1}^{M} \sum_{c=1}^{C_m} \left\| P_{m,\ \text{p2d-id}_m(c)} - K_m \Pi \left( \mathbf{R} _{wm}^\top \mathbf{U} _{\text{p3d-id}_m(c)}^{w} - \mathbf{R} _{wm}^\top \mathbf{t} _{wm} \right) \right\|_2^2$$</div>





  - $C_m$ is the number of observations for the $m$-th camera.

  - $\mathbf{U} _{\text{p3d-id}_m(c)}^{w}$ is the three-dimensional point corresponding to the observation.



We can simply simplify the above cost function to:



<div>$$\mathcal{L}(x) = \sum_{i=1}^{N} \left\| f_i(x) \right\|_2^2 \quad \left\{ \begin{array}{l} x \in \mathbb{R}^D \\ f_i : \mathbb{R}^D \rightarrow \mathbb{R}^B \end{array} \right.$$</div>





  - $x$ represents all parameters to be optimized (camera parameters and three-dimensional point coordinates).

  - $f_i(x)$ is the $i$-th residual function, representing the re-projection error of the $i$-th observation.

  - Our goal is to find $x$ that minimizes $\mathcal{L}(x)$. This is a non-linear least squares problem, typically solved using iterative methods.



**5.3 Gauss-Newton Algorithm**



  - An iterative optimization algorithm used for non-linear least squares problems. $\Rightarrow$ **Iterative** $\quad \delta_{k+1} = \delta_k + d_k$



​	1.	**Linearization of** $f_i$:



<div>$$f_i(x_k + d_k) \approx f_i(x_k) + \mathbf{J}_i(x_k) \cdot d_k$$</div>





  - $\delta x$ is the increment of the parameters to be solved.

  - For each iteration, we perform a Taylor expansion of $f_i(x)$ around the current estimate $x_k$ and ignore higher-order terms.

  - $f_i(x_k + d_k) \in \mathbb{R}^B$

  - $f_i(x_k) \in \mathbb{R}^B$

  - $\mathbf{J}_i(x_k) \in \mathbb{R}^{B \times D}$

  - $d_k \in \mathbb{R}^D$



​	2.	**Jacobian Matrix:**



<div>$$\mathbf{J}_i(x_k) = \frac{\partial f_i(x_k + d_k)}{\partial d_k} \bigg|_{d_k=0}$$</div>





  - Represents the partial derivatives of the function $f_i$ with respect to $d_k$ at the point $x_k$.

  - Describes the linear rate of change of the function $f_i$ at the point $x_k$.



​	3.	**Linear Least Squares:**

$$

L_k(d_k) = \sum_{i=1}^{N} \left| f_i(x_k) + \mathbf{J}_i(x_k) \cdot d_k \right|_2^2

$$



<div>$$\quad \mathbf{J}_k = 

\begin{bmatrix}

  \quad J_1(x_k) \\

  \quad J_2(x_k) \\

  \quad J_3(x_k) \\

  \vdots \\

  \quad J_N(x_k)

\end{bmatrix} \quad \mathbf{b}_k = 

\begin{bmatrix}

 \quad f_1(x_k) \\

 \quad f_2(x_k) \\

 \quad \vdots \\

 \quad f_N(x_k)

\end{bmatrix}$$</div>





  - $\mathbf{b}_k$ is the combination of all residuals.

  - The linear least squares problem becomes:



<div>$$

L_k(d_k) = \lVert \mathbf{b}_k + \mathbf{J}_k \cdot d_k \rVert_2^2

$$</div>





By minimizing $L_k(d_k)$, we obtain the linear system of equations:



<div>$$\mathbf{J}_k^\top \cdot \mathbf{J}_k \cdot d_k = -\mathbf{J}_k^\top \cdot \mathbf{b}_k \quad $$</div>





  - Here, the left matrix $\mathbf{J}_k^T \mathbf{J}_k$ is an approximation of the Hessian matrix, and the right vector $-\mathbf{J}_k^T \mathbf{b}_k$ is the negative of the gradient.

  - Solving this linear system yields the parameter update $d_k$.



​	4.	**Levenberg-Marquardt Algorithm**

Introduces a damping factor $\lambda$ to the Gauss-Newton algorithm, allowing the optimization process to exhibit the fast convergence of Gauss-Newton when close to the solution and the stability of gradient descent when far from the solution.

  - Commonly used for iterative optimization in non-linear least squares problems.

  - **Objective Function:**



<div>$$L_k(d_k) = \lVert \mathbf{b}_k + \mathbf{J}_k d_k \rVert_2^2 + \lambda \lVert d_k \rVert_2^2 \quad \quad \Rightarrow (J_k^T J_k + \lambda I_k) d_k = -J_k^T b_k$$</div>







  - $\lambda$ is the damping factor.



<div>$$\begin{cases} 

  \text{Si } \lambda = 0 & \Rightarrow \text{Gauss-Newton} \\

  \text{Si } \lambda \rightarrow +\infty & \Rightarrow \lambda d_k \rightarrow -J_k^T b_k \quad \text{descente de gradient}

\end{cases}$$</div>





  - **If the new cost function value decreases** (indicating an effective update), reduce $\lambda$ to make the algorithm closer to Gauss-Newton, accelerating convergence.

  - **If the cost function value does not decrease**, increase $\lambda$ to make the algorithm closer to gradient descent, ensuring stability.



**5.4 Summary of Algorithm Steps**



In practical applications, the steps of the Levenberg-Marquardt algorithm are as follows:

​	1.	**Initialization**:

  - Set initial parameters $x$ and damping factor $\lambda$.

  - Compute the initial cost function $L_{\min}$.

​	2.	**Iteration**:

  - **Compute Jacobian Matrix** $\mathbf{J}$ and residuals $\mathbf{b}$.

  - **Solve Linear System**:



<div>$$ (J^T J + \lambda I_d) d = -J^T b$$</div>







  - **Update Parameters**:



<div>$$x' = x + d$$</div>







  - **Compute New Cost Function** $L’$



​	3.	**Evaluate Update Effectiveness**:

  - **If $L’ < L_{\min}$** (cost function decreases):

  - Accept the update: $x = x’$, $L_{\min} = L’$.

  - Decrease $\lambda$: $\lambda = \lambda / 2$.

  - Continue iteration.

  - **Otherwise** (cost function does not decrease):

  - Reject the update, do not change $x$.

  - Increase $\lambda$: $\lambda = 2\lambda$.

  - Check if $\lambda$ exceeds the maximum value; if so, stop iteration.

​	4.	**Termination Conditions**:

  - When $\lambda$ exceeds a preset maximum value or the parameter updates become smaller than a threshold, stop iterating.



All content in this document is theoretical and part of the ‘Vidéo 3D - **Computer Vision**’ course, taught by instructor **Guillaume Bourmaud**. All rights are reserved by the instructor.



For specific experimental content and complete code, please refer to **Guillaume Bourmaud**’s official website: https://gbourmaud.github.io/teaching/
