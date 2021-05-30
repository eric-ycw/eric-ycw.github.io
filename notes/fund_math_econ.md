---
layout: subpage
title: Fundamental Methods of Mathematical Economics
---
**Table of Contents**
- [Chapter 5: Linear Models and Matrix Algebra (Continued)](#chapter-5)
  - [5.7: Leontief Input-Output Models](#57-leontief-input-output-models)
- [Chapter 7: Rules of Differentiation and Their Use in Comparative Statics](#chapter-7)
  - [7.4: Partial Differentiation](#74-partial-differentiation)
  - [7.5: Applications to Comparative-Static Analysis](#75-applications-to-comparative-static-analysis)
  - [7.6: Note on Jacobian Determinants](#76-note-on-jacobian-determinants)
- [Chapter 8: Comparative-Static Analysis of General-Function Models](#chapter-8)
  - [8.2: Total Differentials](#82-total-differentials)
  - [8.4: Total Derivatives](#84-total-derivatives)
  - [8.5: Derivatives of Implicit Functions](#85-derivatives-of-implicit-functions)

## **Chapter 5**
## 5.7: Leontief Input-Output Models
### 5.7.1
- Leontief's input-output analysis answers the question: "What level of output should each industry in an economy produce to sufficiently satisfy total demand for that product?"
- The nuance lies within the fact that the output of any industry (e.g. the steel industry) is needed as an input in other industries (and vice versa)
- We can create an $n\times n$ input-coefficient matrix $A$ for an n-industry economy, where each **column** specifies the input requirements for the production of one unit by that industry
- For example, in an economy with industries I, II, and III, each producing oil, plastic, and metal respectively, we have:

$$
  \begin{bmatrix}
  0.2 & 0.3 & 0.2 \\
  0.4 & 0.1 & 0.2 \\
  0.1 & 0.3 & 0.2 \\
  \end{bmatrix}
$$

- The second column, for example, states that to produce 1 unit of plastic, the inputs needed are 0.3 units of oil, 0.1 units of plastic, and 0.3 units of metal

### 5.7.2
- In order to incorporate final demand (e.g. consumer demand) and primary inputs (e.g. labor), we must include an **open sector** outside of the n-industry network
- With an open sector, the column sum $\sum_{i=1}^n a_{ij}$ (representing the partial input cost) incurred in producing a unit worth of some commodity must be less than 1 (i.e. $\sum_{i=1}^n a_{ij} < 1$)
- Since the output value must be fully absorbed by all factors of production, the value of the primary input is $ 1 - \sum_{i=1}^n a_{ij}$
- For an industry to produce an output just sufficient to meet the input requirements of all industries as well as the final demand $d$, we have:

$$x_n = a_{n1}x_1 + a_{n2}x_2 + \cdots + a_{nn}x_n + d_n$$

- Rearranging:

$$ -a_{n1}x_1 - a_{n2}x_2 - \cdots + (1-a_{nn})x_n = d_n$$

- We can solve the system of linear equations to find the optimal output levels for this economy

$$
  \begin{bmatrix}
  (1-{a_11}) & -a_{12} & \cdots & -a_{1n} \\
  -a_{21} & (1-a_{22}) & \cdots & -a_{2n} \\
  \vdots & \vdots & \ddots & \vdots \\
  -a_{n1} & a_{n2} & \cdots & (1-a_{nn}) \\
  \end{bmatrix}

  \begin{bmatrix}
  x_1 \\
  x_2 \\
  \vdots \\
  x_n \\
  \end{bmatrix}

  =

  \begin{bmatrix}
  d_1 \\
  d_2 \\
  \vdots \\
  d_n \\
  \end{bmatrix}
$$

- Note that the left matrix is simply the identity matrix minus the input-coefficient matrix $A$, so we have:

$$(I - A)x = d$$

$$x^* = (I - A)^{-1}d$$

- $I - A$ is the **Leontief matrix**
- The required primary input to meet the specific final demand is $\sum_{j=1}^n a_{0j}x_j^*$, where $a_{0j} = 1 - \sum_{i=1}^n a_{ij}$

### 5.7.3
- It is clear that in order for the solution values to make economic sense, the output levels $x_j^*$ should all be non-negative
- This is true if and only if the Leontief matrix $I - A$ satisfies the **Hawkins-Simon condition**:

  1. All **off-diagonal elements** in $I - A$ are non-positive
  2. All elements in the demand vector $d$ are non-negative
  3. All the **leading principal minors** of $I - A$ are positive

- The leading principal minors of a $n\times n$ matrix $M$ are the minors with dimension $m\times m$ ($m \le n$) that include the **top-left corner** of $M$
- The Hawkins-Simon condition specifies practicability and viability restrictions for production

### 5.7.4
- If the exogenous sector of the open model is absorbed into the system as another industry, the model becomes **closed**
- Hence, all the final demands will be zero, and the system of equations becomes homogeneous
- The Leontief matrix in this case must be singular, and there will be infinitely many solutions with $x_1^*, \cdots, x_j^\*$ in proportion to one another

## **Chapter 7**
## 7.4: Partial Differentiation
- Partial differentiation is simply differentiating a function with respect to a single variable while holding all other independent variables constant
- The **gradient** $\nabla f(x_1, x_2, \cdots, x_n)$ is a vector with all the partial derivatives of a function as its components

$$
\nabla f(x_1, x_2, \cdots, x_n) =
\begin{bmatrix}
\frac{\partial f}{\partial x_1}(x_1, x_2, \cdots, x_n) \\
\frac{\partial f}{\partial x_2}(x_1, x_2, \cdots, x_n) \\
\vdots \\
\frac{\partial f}{\partial x_n}(x_1, x_2, \cdots, x_n) \\
\end{bmatrix}
$$

## 7.5: Applications to Comparative-Static Analysis
### 7.5.1
- Comparative-static analysis concerns how the equilibrium value of an endogenous variable will change when the exogenous variables change
- For example, in a one-commodity market model:

$$
Q_D = a - bP \quad (a, b > 0) \\
Q_S = -c + dP \quad (c, d > 0) \\
$$

$$
P^* = \frac{a + c}{b + d} \\
Q^* = \frac{ad - bc}{b + d}
$$

- We can use partial derivatives to better understand how changing the parameters affect $P^\*$ and $Q^\*$, which is useful for higher dimensional cases where graphical representation is impossible

$$
\frac{\partial P^*}{\partial a} = \frac{\partial P^*}{\partial c} = \frac{1}{b + d} > 0 \\
\frac{\partial P^*}{\partial b} = \frac{\partial P^*}{\partial d} = \frac{-(a + c)}{(b + d)^2} < 0
$$

- The same applies to a national-income model:

$$
Y = C + I_0 + G_0 \\
C = \alpha + \beta(Y - T) \quad (\alpha > 0; \quad 0 < \beta < 1) \\
T = \gamma + \delta Y \quad (\gamma > 0; \quad 0 < \delta < 1) \\
$$

$$
Y^* = \frac{\alpha - \beta \gamma + I_0 + G_0}{1 - \beta + \beta \delta}
$$

- We can obtain the **government-expenditure multiplier**, the **nonincome-tax multiplier**, and the extent to which increasing the income tax rate $\delta$ will lower the equilibrium income $Y^*$

$$
\frac{\partial Y^*}{\partial G_0} = \frac{1}{1 - \beta + \beta \delta} > 0 \\
\frac{\partial Y^*}{\partial \gamma} = \frac{-\beta}{1 - \beta + \beta \delta} < 0 \\
\frac{\partial Y^*}{\partial \delta} = \frac{-\beta Y^*}{1 - \beta + \beta \delta} < 0 \\
$$

### 7.5.2
- Returning to the open Leontief input-output model, the inverse of the Leontief matrix actually represents the comparative-static derivatives of our output values to the final demand

$$
\frac{\partial x^*}{\partial d} = (I - A)^{-1}
$$

- This is useful in revising output goals in response to changes in planning targets

## 7.6: Note on Jacobian Determinants
- Partial derivatives can be used to test whether there is dependence between a set of functions
- The Jacobian determinant is defined as:

$$
|J| \equiv \left| \frac{\partial (y_1, y_2, \cdots, y_n)}{\partial (x_1, x_2, \cdots, x_n)}\right|
\equiv

\begin{vmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\vdots & & \vdots \\
\frac{\partial y_n}{\partial x_1} & \cdots & \frac{\partial y_n}{\partial x_n}
\end{vmatrix}

$$

- $\|J\| = 0$ for all values of $x_1, x_2, \cdots, x_n$ if and only if the functions are **functionally (linearly or nonlinearly) dependent**
- The determinant test for systems of linear equations can be interpreted as a special application of the Jacobian test

## **Chapter 8**
## 8.2: Total Differentials
- Total differentials give the total change in a function with respect to all of its arguments
- For example, consider a production function $Q = Q(K, L, t)$
- The total differential is:

$$
dQ = \frac{\partial Q}{\partial K}dK + \frac{\partial Q}{\partial L}dL + \frac{\partial Q}{\partial t}dt
$$

## 8.4: Total Derivatives
- Hence, it is clear that we can find the total derivative with respect to a specific variable just by dividing by its differential, given that the other variables are functionally dependent on the specific variable (i.e. $K = K(t)$ and $L = L(t)$ in this example)

$$
\frac{dQ}{dt} = \frac{\partial Q}{\partial K}\frac{dK}{dt} +
\frac{\partial Q}{\partial L}\frac{dL}{dt} +
\frac{\partial Q}{\partial t}
$$

## 8.5: Derivatives of Implicit Functions
