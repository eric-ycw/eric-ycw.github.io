---
layout: subpage
title: Probability and Random Variables (MIT 18.600)
---
**Table of Contents**
- [Problem Set 1](#problem-set-1)

Link to questions [here](https://ocw.mit.edu/courses/mathematics/18-600-probability-and-random-variables-fall-2019/assignments/).

**DISCLAIMER:** All 'answers' are attempts and could be incorrect.

# Problem Set 1
### Question A1
(a) $$8!$$

(b) $$7! \cdot 2!$$

(c) $$4! \cdot \frac{5!}{(5-4)!}$$

(d) $$4! \cdot 5!$$

(e) $$4! \cdot 2! \cdot 4$$

### Question A2
Consider the set $$S = \{1, 2, \cdots, n\}$$. We partition the set into two such that $$S{_A} = \{1, 2, \cdots, k-1\}$$ and $$S{_B} = \{k, \cdots, n\}$$ where $$k \leq n$$. Consider $$i$$ where $$k \leq i \leq n$$ (i.e. $$i$$ is an element of $$S{_B}$$). The number of $$k$$-size subsets of $$S$$ where $$i$$ is the largest element is equal to $$\binom{i-1}{k-1}$$. By definition, every $$k$$-size subset of $$S$$ must include at least one element from $$S{_B}$$, and we can obtain the total number of $$k$$-size subsets of $$S$$ by summing up $$\binom{i-1}{k-1}$$ for all $$i$$. Hence:

$$\binom{n}{k} = \sum_{i=k}^{n} \binom{i-1}{k-1}$$

### Question B1
Consider a table with $n$ guests and $n$ seats with the first guest seated at a fixed position. There are $$(n-1)!$$ ways to rearrange the seating. Similarly, of permutations $\sigma$, there are $$(n-1)!$$ permutations with exactly one cycle.

*The reason we keep an arbitrary guest at a fixed position is because it doesn't matter which element a cycle begins on. Another way to think about it is to consider $$n!$$ possible seatings divided by $n$ since there are $n$ different ways to start the cycle.*

### Question B2
Using the table analogy again, imagine there are two tables with $k$ seats and $n-k$ seats respectively. We choose $k$ guests to sit at the first table and the remaining guests to sit at the second table, so the total number of possible seatings is $$\binom{n}{k} \cdot (k-1)!$$. Therefore, of permutations $\sigma$, there are $$\binom{n}{k} \cdot (k-1)!$$ permutations with exactly two cycles of length $$k$$ and length $$n-k$$ respectively.

### Question B3
If a permutation is an involution, then it must consist entirely of fixed points (since if $$\sigma(i) = i$$, then $$\sigma(\sigma(i)) = i$$) and/or cycles of length $$2$$. Imagine we have to divide $n$ guests into $k$ pairs ($0 \leq 2k \leq n$) and leave the other $n - 2k$ guests as singles. The number of possible pairings is $$\binom{n}{2k} \cdot \frac{(2k)!}{k! \cdot (2!)^k}$$. Hence, the total number is the sum of pairings for all possible values of $k$.

$$\sum_{k=0}^{\lfloor n/2 \rfloor} \binom{n}{2k} \cdot \frac{(2k)!}{k! \cdot 2^k}$$

### Question C1
$$4$$

### Question C2
There are $\binom{4}{2}$ ways to choose two suits out of four. We then pick 13 cards out of 26, and subtract the two cases where all cards come from a single suit.

$$\binom{4}{2} \cdot \left( \binom{26}{13} - 2 \right)$$

### Question C3
There are $\binom{4}{3}$ ways to choose three suits out of four. We then pick 13 cards out of 39, and subtract the $\binom{26}{13} \cdot \binom{3}{2}$ cases where all cards come from less than three suits.

$$\binom{4}{3} \cdot \left( \binom{39}{13} - \binom{26}{13} \cdot \binom{3}{2} \right)$$


### Question C4
We pick 13 cards out of the whole deck, and subtract the $\binom{39}{13} \cdot \binom{4}{3}$ cases where all cards come from less than four suits.

$$\binom{52}{13} - \binom{39}{13} \cdot \binom{4}{3}$$

### Question D

Exactly one white ball: $$\binom{5}{1} \cdot \binom{64}{4} \cdot 26$$ and $$p \approx 0.28268$$

Exactly two white balls: $$\binom{5}{2} \cdot \binom{64}{3} \cdot 26$$ and $$p \approx 0.037073$$

Exactly three white balls: $$\binom{5}{3} \cdot \binom{64}{2} \cdot 26$$ and $$p \approx 0.0017938$$

Exactly one red ball and two white balls: $$\binom{5}{2} \cdot \binom{64}{3}$$ and $$p \approx 0.0014259$$

### Question E1
*Skipped*

### Question E2

Because the Taylor series for $e^{\lambda}$ is $1 + \lambda + \frac{\lambda^2}{2!} + \cdots$, we have:

$$
\sum_{k=0}^{\infty} \frac{e^{-\lambda}\lambda^{k}}{k!} = \frac{1}{e^{\lambda}} \sum_{k=0}^{\infty} \frac{\lambda^{k}}{k!}
= \frac{1}{\sum_{k=0}^{\infty} \frac{\lambda^{k}}{k!}} \sum_{k=0}^{\infty} \frac{\lambda^{k}}{k!} = 1
$$

### Question E3

$$
(p+q)^n = \sum_{k=0}^{n} \binom{n}{k}p^{k}q^{n-k} = 1
$$

### Question E4

$$
\begin{split}
\int_{0}^{\infty} x^{n}e^{-x}dx & = -\int_{0}^{\infty} x^{n}de^{-x} \\

& = n \int_{0}^{\infty} x^{n-1}e^{-x}dx - [x^n e^{-x}]_{0}^{\infty} \\

& = n \int_{0}^{\infty} x^{n-1}e^{-x}dx \\

& = n(n-1) \int_{0}^{\infty} x^{n-2}e^{-x}dx \\

& = n! \int_{0}^{\infty} e^{-x}dx = n! \cdot [-e^{-x}]_{0}^{\infty} \\

& = n!

\end{split}
$$
