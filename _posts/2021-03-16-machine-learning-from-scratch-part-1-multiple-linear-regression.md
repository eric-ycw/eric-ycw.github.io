---
layout: post
title: "Machine Learning From Scratch Part 1: Multiple Linear Regression"
tags: [math, statistics, machine-learning]
---
**Table of Contents**
- [Introduction](#introduction)

- [What is linear regression?](#what-is-linear-regression)
  - [The simple case](#the-simple-case)
  - [Multiple linear regression](#multiple-linear-regression)
  - [Linear regression in matrix form](#linear-regression-in-matrix-form)

- [What is gradient descent?](#what-is-gradient-descent)
  - [The hill analogy](#the-hill-analogy)
  - [Gradient vectors and loss functions](#gradient-vectors-and-loss-functions)

- [Implementation](#implementation)
  - [Laying the groundwork](#laying-the-groundwork)
  - [Data preprocessing](#data-preprocessing)
  - [Feature scaling](#feature-scaling)
  - [Results](#results)

- [Footnotes](#footnotes)

## Introduction

This is a series of posts where I attempt to build machine learning algorithms in Python from scratch without the use of any external libraries (except NumPy). The goal is to have a solid understanding of the math/statistics and design principles that govern how these algorithms work.

Note that the actual implementation will primarily focus on readability and instructiveness rather than efficiency. No prerequisite knowledge in machine learning is required, although knowing some calculus, statistics and linear algebra will be helpful. You can check out the full repository [here](https://github.com/eric-ycw/solomon).

To start off with something relatively simple, we will first look at **linear regression**, as well as an introduction to **gradient descent** and **feature scaling**.

## What is linear regression?
### The simple case
<br>
<p align="center">
<img src="/assets/images/simple-linear-regression.png" alt="Simple linear regression" width="600">
</p>
You have probably come across the concept of a best-fit line, which models a linear relationship between an **explanatory variable** $x$ and a **response variable** $y$. The idea is similar to the equation for a straight line:

$$
y_i = \beta_0 + \beta_1 x_i + \epsilon_i
$$

We can see that $\beta_0$ is the intercept and $\beta_1$ is the slope, and the **error term**  $\epsilon_i$ is defined as the difference between the predicted value $\hat y_i$ given by our regression line and the actual observed value $y_i$. This difference is known as the **residual**.

We want to find $\beta_0$ and $\beta_1$ that can best predict the response variable. This is usually done with **ordinary least squares (OLS)**. OLS is a common way to fit a line to given data by minimizing the sum of the squares of residuals.

$$
\beta_0 = \underset{\beta_0}{\mathrm{argmin}}\, \sum_{i=1}^{m}\epsilon_i^2 \\
\beta_1 = \underset{\beta_1}{\mathrm{argmin}}\, \sum_{i=1}^{m}\epsilon_i^2
$$

There are many reasons behind squaring the residuals, one of them being that it penalizes large errors more, such that the model will prefer a large number of small errors over a small number of large errors. The function itself is also continuously differentiable, which will come in handy later on.

**N.B.** There is a closed-form solution for OLS by solving for $\beta$ algebraically. Technically, we don't need any machine learning at all to fit a linear regression model, but it serves as a good entry point to understand and implement machine learning principles.

### Multiple linear regression
<br>
<p align="center">
<img src="/assets/images/multiple-linear-regression.png" alt="multiple linear regression" width="350">
</p>

Multiple linear regression is an extension of the simple linear regression model, but with more explanatory variables.

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_n x_{in} + \epsilon_i
$$

Now is probably a good time to bring up some key assumptions made in a linear regression model using OLS, other than the obvious assumption that a linear relationship exists. These include:

1. **No perfect multicollinearity** [^1] (i.e. no two explanatory variables are perfectly correlated)
2. **Homoscedasticity** (i.e. all explanatory variables have the same variance/noise)
3. **Independent errors** (i.e. the errors are not correlated)

Often, these assumptions do not hold when dealing with real data, and there are other methods (ridge regression, Bayesian linear regression) to circumvent these constraints. However, OLS is good enough for most use cases, but it's always a nice idea to consider any possible pitfalls or misapplications.

You might think: "Okay, this is great, but how do we actually find the set of coefficients that minimizes error? And what does this have to do with machine learning?" In order to answer these two questions, we will need a bit of linear algebra and calculus to help us out.

### Linear regression in matrix form

Rewriting the equation above in matrix form, we have:

$$
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_m
\end{bmatrix}
=
\begin{bmatrix}
\beta_0 + \beta_1 x_{11} + \cdots + \beta_n x_{1n} \\
\beta_0 + \beta_1 x_{21} + \cdots + \beta_n x_{2n} \\
\vdots \\
\beta_0 + \beta_1 x_{m1} + \cdots + \beta_n x_{mn} \\
\end{bmatrix}
+
\begin{bmatrix}
\epsilon_1 \\
\epsilon_2 \\
\vdots \\
\epsilon_m
\end{bmatrix}
$$

$$
\quad \quad \quad \,
=

\begin{bmatrix}
1 & x_{11} & \cdots & x_{1n} \\
1 & x_{21} & \cdots & x_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{m1} & \cdots & x_{mn} \\
\end{bmatrix}

\begin{bmatrix}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_n \\
\end{bmatrix}
+
\begin{bmatrix}
\epsilon_1 \\
\epsilon_2 \\
\vdots \\
\epsilon_m
\end{bmatrix}
$$

Therefore, we have:

$$
\mathbf Y = \mathbf X \beta + \epsilon
$$

The response vector $\mathbf Y$ can be obtained simply by multiplying the **design matrix** $\mathbf X$ with the vector of parameters $\beta$ and adding the error vector $\epsilon$. Note that an extra column of ones is added to form $\mathbf X$ in order for the multiplication to work. $m$ is the total number of cases while $n$ is the number of parameters to be estimated.

We haven't done anything new or amazing here, but a matrix representation allows us to exploit the convenience and speed of NumPy's ndarrays in our implementation. With this is mind, we can move on to how we can optimize $\beta$, the vector of parameters.

## What is gradient descent?
### The hill analogy
<br>
<p align="center">
<img src="/assets/images/gradient-descent.gif" alt="Gradient descent" width="300">
</p>
To gain an intuitive understanding of gradient descent, we can imagine a person trying to find their way down a hill. They have no idea where they are, but they can get a sense of direction by heading down the steepest direction.

Since it's annoying to change directions all the time, the person should only readjust their walking direction once in a while. But if they do that too infrequently, they might overshoot and end up wandering. When they reach flat ground, it's probable that they've arrived at the bottom of the hill, but if the hill terrain is complex, they might end up somewhere else that isn't the lowest point of the entire mountain (e.g. a saddle or a hole).

### Gradient vectors and loss functions

Let's rephrase that in a more mathematical way. We are trying to minimize a **loss function** $L(\beta)$ that measures how incorrect our model is. By computing the **gradient vector** $\nabla L(\beta)$, we'll know how to adjust $\beta$ locally to make $L(\beta)$ decrease the fastest.

The gradient is defined as a vector with all the partial derivatives of a function as its components.

$$
\nabla L(\beta) =

\begin{bmatrix}
\frac{\partial L(\beta)}{\partial \beta_0} \\
\frac{\partial L(\beta)}{\partial \beta_1} \\
\vdots \\
\frac{\partial L(\beta)}{\partial \beta_n} \\
\end{bmatrix}

$$

Since it'll take ages to converge if we update our parameters too often, we have to set a **learning rate** $\alpha$ that determines how much we "walk" before we compute $\nabla L(\beta)$ again. If $\alpha$ is too large, we might never converge at all. Also, if $L(\beta)$ is non-convex, we might get stuck in a saddle point or a local minimum, rather than the global minimum that we hope to reach [^2].

Choosing the best way to update $\alpha$ and $\beta$ is an entire art form in itself [^3], but for linear regression, it's sufficient to use a fixed $\alpha$ and update our parameters using the following formula:

$$
\beta = \beta - \alpha \nabla L(\beta)
$$

Note that we are subtracting $\alpha \nabla L(\beta)$ from $\beta$, because $\nabla L(\beta)$ points in the direction that would make $L(\beta)$ increase the fastest, so we have to add a negative sign to go the opposite way.

Now, the only missing piece is the loss function $L(\beta)$ that we have yet to define. Recall that we used the sum of the squares of residuals to measure the regression error. However, the magnitude of error increases with the number of cases, which isn't what we want. Instead, we can take the **average** of the aforementioned metric. This is the **mean squared error (MSE)** of our model, and a perfect choice for our loss function.

$$
L(\beta) = MSE = \frac{1}{m}\sum_{i=1}^{m}\epsilon_i^2
$$

$$
\epsilon_i = y_i - \beta_0 - \beta_1 x_{i1} - \beta_2 x_{i2} - \cdots - \beta_n x_{in}
$$

By considering the partial derivative of $L(\beta)$ with respect to a single parameter $\beta_j$, we have:

$$
\frac{\partial L(\beta)}{\partial \beta_j} = \frac{-2}{m}\sum_{i=1}^{m}x_{ij}\epsilon_i
$$

Hence, the gradient can be represented as such:

$$
\nabla L(\beta) = \frac{-2}{m}

\begin{bmatrix}
\sum_{i=1}^{m}\epsilon_i \\
\sum_{i=1}^{m}x_{i1}\epsilon_i \\
\vdots \\
\sum_{i=1}^{m}x_{in}\epsilon_i \\
\end{bmatrix}

=
\frac{-2}{m}\mathbf X^T \epsilon
$$

We arrive at an elegant result: we can obtain $\nabla L(\beta)$ by multiplying the transpose of the design matrix with the error vector along with a coefficient of $\frac{-2}{m}$. Having understood the math, we can finally proceed to our implementation.

## Implementation
### Laying the groundwork

In our *utils.py* file, we write out our loss function and gradient optimization algorithm. Since we're using ndarrays from NumPy, implementing matrix algebra is quite straightforward.

{% highlight python %}
import numpy as np

class MeanSquaredError(Loss):
    @staticmethod
    def loss(y, y_hat):
        return np.square(y - y_hat).mean()

    @staticmethod
    def gradient(X, y, y_hat):
        return (-2 / X.shape[0]) * X.T.dot(y - y_hat)
{% endhighlight %}

{% highlight python %}
class BatchGradientDescent(Optimization):
    @staticmethod
    def optimize(X, y, y_hat, params, alpha, loss_func):
        return params - \
               loss_func.gradient(X, y, y_hat) * alpha
{% endhighlight %}

Our gradient descent algorithm is called **batch gradient descent**, because we compute the gradient over the full set of training data (i.e. one batch).

In *linear_regression.py*, we have:

{% highlight python %}
import numpy as np
from src.utils import MeanSquaredError, BatchGradientDescent

class LinearRegression:
    def __init__(self):
        self.params = None

    def train(self, X, y, iterations=5000, alpha=0.01):
        self.params = np.zeros((X.shape[1], 1))

        for i in range(iterations):
            y_hat = X.dot(self.params)
            loss = MeanSquaredError.loss(y, y_hat)

            self.params = BatchGradientDescent.optimize(
                X, y, y_hat, self.params,
                alpha, MeanSquaredError)

        return loss

    def predict(self, X, y):
        y_hat = X.dot(self.params)
        loss = MeanSquaredError.loss(y, y_hat)

        return y_hat, loss
{% endhighlight %}

### Data preprocessing

To test our implementation on real data, we will build a linear regression model for Amazon stock prices using S&P 500 prices, as well as the stock prices of Amazon's main competitors (Walmart, Ebay, Apple). We will also compare our results and prediction error to that of scikit-learn.

{% highlight python %}
from datetime import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression as skl_LR
from sklearn.metrics import r2_score
plt.style.use('seaborn')

from src.regression.linear_regression import LinearRegression
from src.utils import ZScoreNormalization
{% endhighlight %}

{% highlight python %}
# Load data from csv files
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/AMZN.csv"))
spy = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/SPY.csv"))
wmt = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/WMT.csv"))
ebay = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/EBAY.csv"))
aapl = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/AAPL.csv"))

# Combine all price columns into one dataframe and rename them
df.drop(df.columns[[1, 2, 3, 4, 6]], axis=1, inplace=True)
df.rename(columns={'Adj Close' : 'AMZN'}, inplace=True)
df['SPY'] = spy['Adj Close']
df['WMT'] = wmt['Adj Close']
df['EBAY'] = ebay['Adj Close']
df['AAPL'] = aapl['Adj Close']

# We convert the date entries to datetime and fix them in place as an index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

plt.title('Correlation matrix')
sns.heatmap(df.corr(), annot=True)
plt.show()
{% endhighlight %}

<p align="center">
<img src="/assets/images/mlr-correlation-matrix.png" alt="Correlation matrix for our variables" width="800">
</p>

We observe a large degree of multicollinearity in the parameters we selected. This means we cannot treat our parameters as independent, and we should not interpret the regression coefficients as an accurate measure of how the parameters affect the response variable. However, this does not undermine the predictive power of our model.

{% highlight python %}
# Select the feature/target value columns, do a train-test split
X = df.iloc[:,1:]
y = df.iloc[:,0:1]
X_train, y_train = X.sample(frac=0.8, random_state=20), y.sample(frac=0.8, random_state=20)
X_test, y_test = X.drop(X_train.index), y.drop(y_train.index)

# Convert into ndarrays
X_train, y_train, X_test, y_test = X_train.values, y_train.values, X_test.values, y_test.values
{% endhighlight %}

We've picked out all relevant data from the *csv* files for training and testing. However, we still have a few things left to do before we can plug them in into our training function.

### Feature scaling
<br>
<p align="center">
<img src="/assets/images/gradient-descent-zigzag.png" alt="Zigzagging in gradient descent without feature scaling" width="400">
</p>
When our parameters vary greatly in magnitude, gradient descent converges at a slower rate. For example, we can consider the case with only two parameters, which can be visualized using a contour plot. Note that the gradient vector points perpendicularly to the contour line, so if the contour lines do not form concentric circles, we will "zigzag" before we reach the global minimum.

This can be fixed by **feature scaling**, which normalizes all of the explanatory variables to a similar range of values. The normalization method we will use is **z-score normalization (standardization)**[^4].

$$
x^{\prime} = \frac{x - \bar x}{\sigma_x}
$$

{% highlight python %}
class ZScoreNormalization(Normalization):
    @staticmethod
    def normalize(features):
        return (features - features.mean(0)) / features.std(0)
{% endhighlight %}

And in our *linear_regression.py*, we have:

{% highlight python %}
# We use z-score normalization to rescale the features set
X_train, X_test =  ZScoreNormalization.normalize(X_train),  ZScoreNormalization.normalize(X_test)

# Add a column of ones to the features set
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
{% endhighlight %}

### Results

{% highlight python %}
lr = LinearRegression()
train_loss = lr.train(X_train, y_train, display=True)
print('Training set loss:', train_loss)

y_hat, test_loss = lr.predict(X_test, y_test)
print('Test set loss:', test_loss)

print('Parameters:', np.squeeze(lr.params.T))
{% endhighlight %}

{% highlight python %}
Training set loss: 37661.92263121619
Testing set loss: 35483.63744155301
Parameters: [1660.66414379  141.08057762  259.43738789   88.4854266   292.47921795]
{% endhighlight %}

Our loss in the testing set is comparable to that of the training set, indicating no overfitting occured.

<p align="center">
<img src="/assets/images/loss-against-iterations.png" alt="Loss against iterations" width="800">
</p>

<p align="center">
<img src="/assets/images/model-results.png" alt="Model results" width="800">
</p>

Our model is able to predict the general swings in Amazon's stock price based on the S&P 500 and the stock prices of its competitors, but doesn't accurately capture the price level during some periods. Now to compare the results with that of scikit-learn's *LinearRegression*:

{% highlight python %}
print('Our R-squared:', r2_score(y_test, y_hat))

# Train and test using sklearn
skl_lr = skl_LR()
skl_lr.fit(X_train, y_train)
print('sklearn R-squared:', skl_lr.score(X_test, y_test))
{% endhighlight %}

{% highlight python %}
Our R-squared: 0.9428816246669391
sklearn R-squared: 0.9428859468886387
{% endhighlight %}

Our $R^2$ is nearly identical to that of scikit-learn's. For OLS linear regression, scikit-learn computes the closed-form solution directly and does not use gradient descent, which explains the slight difference.

## Footnotes

*The full code and data used in this article can be found in [this repository](https://github.com/eric-ycw/solomon).*

[^1]: The mere presence of multicollinearity among explanatory variables does not invalidate the predictive power of a linear regression model, but it makes it difficult to arrive at statistically significant conclusions, as the standard errors of the coefficients will be large. When there is perfect multicollinearity (which may happen when there is redundant data), there is no closed-form solution for OLS because the matrix of explanatory variables will be singular.
[^2]: The loss function we will pick (MSE) is convex for linear regression, so any local minimum is also a global minimum. However, MSE may not be convex when non-linearity is introduced into the mix (e.g. in neural networks).
[^3]: The gradient descent method we discuss in this article is too slow for more complex models or when dealing with many parameters. Usually, we approximate the gradient or calculate it using mini-batches of training data, we modify $\alpha$ during the optimization process to improve performance (e.g. AdaGrad, Adam), or we can consider previous states when updating $\beta$ (e.g. momentum).
[^4]: Min-max normalization can also be used, where $x^{\prime} = \frac{x - x_{min}}{x_{max} - x_{min}}$, but this doesn't work well if outliers are present.
