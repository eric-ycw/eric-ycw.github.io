---
layout: subpage
title: Fundamental Methods of Mathematical Economics
---
**Table of Contents**
- [Chapter 5: Linear Models and Matrix Algebra (Continued)](#chapter-5)
  - [5.7: Leontief Input-Output Models](#57-leontief-input-output-models)
- [Chapter 7: Rules of Differentiation and Their Use in Comparative Statics](#chapter-7)
  - [7.5: Applications to Comparative-Static Analysis](#75-applications-to-comparative-static-analysis)

# **Chapter 5**
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

# **Chapter 7**
## 7.5: Applications to Comparative-Static Analysis
