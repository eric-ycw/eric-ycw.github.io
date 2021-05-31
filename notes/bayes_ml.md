---
layout: subpage
title: Bayesian Methods for Machine Learning
source: HSE
tags: [math, statistics, machine-learning]
---
**Table of Contents**
- [Week 1: Introduction to Bayesian Methods and Conjugate Priors](#week-1)
  - [1.1: Introduction to Bayesian Methods](#11-introduction-to-bayesian-methods)
  - [1.2: Conjugate Priors](#12-conjugate-priors)

## **Week 1**
## 1.1: Introduction to Bayesian Methods
### 1.1.1
- A frequentist approach sees probabilities as **long-run frequencies** where parameters are constant, while a Bayesian approach sees probabilities as **degrees of belief** where parameters are variables
- For more information, check out [this fun brain teaser](https://www.behind-the-enemy-lines.com/2008/01/are-you-bayesian-or-frequentist-or.html){:target="_blank"} that demonstrates the difference

### 1.1.2
- Chain rule for random variables:

$$P(X_1, X_2, \cdots, X_n) = \prod_{k=1}^{n} P\left(X_k|\bigcap_{j=1}^{k-1}X_j\right)$$

- Sum rule (marginalization):

$$P(X) = \int_{-\infty}^{\infty} P(X, Y) dY$$

- Bayes' Theorem, where $\theta$ are parameters and $X$ are observations

$$P(\theta|X) = \frac{P(X|\theta)P(\theta)}{P(X)}$$

- $P(\theta\|X)$ is the **posterior**
- $P(\theta)$ is the **prior**
- $P(X\|\theta)$ is the **likelihood**
- $P(X)$ is the **evidence**

### 1.1.3
- We can construct a joint probability function from a **Bayesian network**, which we can use along with marginalization to find conditional probabilities
- For example, when considering the famous sprinkler example:

<p align="center">
<img src="/assets/images/bayes-sprinkler.png" alt="Bayesian network for sprinkler, rain, and wet grass">
</p>

- Our joint probability function would be:

$$P(R, G, S) = P(G|S, R)P(S|R)P(R)$$

- To find $P(R\|G)$, we can substitute the values from the conditional probability tables into the following formula:

$$P(R|G) = \frac{P(R, G)}{P(G)}=\frac{P(R, G, S) + P(R, G, S^C)}{P(R, G, S) + P(R, G, S^C) + P(R^C, G, S) + P(R^C, G, S^C)}$$

### 1.1.4
- We can frame a conditional classification problem using Bayes' Theorem

$$P(class|X_1, X_2, \cdots, X_n) = \frac{P(X_1, X_2, \cdots, X_n|class)P(class)}{P(X_1, X_2, \cdots, X_n)}$$

- Unless $n$ is extremely large, estimating the conditional probability of each observation is not feasible, so the direct application of Bayes' Theorem is intractable
- A **naive Bayes classifier** assumes all input variables are independent from each other, such that:

$$P(class|X_1, X_2, \cdots, X_n) = P(class)P(X_1|class)P(X_2|class)\cdots P(X_n|class)$$

- We then estimate the conditional probability for each feature from the dataset, usually by selecting one of three standard probability distributions
  - For **binary** data (e.g. T/F), a binomial distribution is used
  - For **categorical** data, a multinomial distribution is used
  - For **numerical** data, a Gaussian distribution is used
- If the variable does not have a well-defined distribution, we can use a **kernel density estimator** (KDE) to estimate the probability distribution

### 1.1.5
- The **maximum likelihood estimate** (MLE) is the specific value of parameter vector $\theta$ that maximizes the likelihood function $L_n = P(X\|\theta)$ (i.e. how probable the observed data $X$ is for a specific $\theta$)
- For a univariate normal distribution, $\theta = (\mu, \sigma^2)$
- It is more convenient to maximize the natural logarithm of the likelihood function (we can do this because the logarithm is monotonic)
- Hence, we can solve for the MLE by setting:

$$\frac{\partial (\log L_n)}{\partial \theta_1} = 0, \frac{\partial (\log L_n)}{\partial \theta_2} = 0, \cdots, \frac{\partial (\log L_n)}{\partial \theta_k} = 0$$

## 1.2: Conjugate Priors
### 1.2.1
- Modelling the distribution of evidence $P(X)$ may be very hard in some cases
  - For a neural network trained to play games where X is an image of the game screen, P(X) is easy to model for games like "Snake", while almost impossible for more complex games
- We can find the maximum a posteriori (MAP) $\theta_{MP}$ (i.e. the mode of the posterior distribution) by optimization, which would allow us to eliminate $P(X)$ as it does not depend on $\theta$

$$
\theta_{MP} = \underset{\theta}{\mathrm{argmax}}\, P(X|\theta)P(\theta)
$$
- However, there are problems with MAP, including not being invariant to reparameterization (e.g. applying a sigmoid function on a Gaussian distribution)
- Another problem is that we cannot compute confidence intervals for a MAP estimation
