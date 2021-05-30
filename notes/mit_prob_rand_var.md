---
layout: subpage
title: Probability and Random Variables
source: MIT
---
**Table of Contents**
- [Problem Set 1](#problem-set-1)
- [Problem Set 2](#problem-set-2)

Link to questions [here](https://ocw.mit.edu/courses/mathematics/18-600-probability-and-random-variables-fall-2019/assignments/).

**DISCLAIMER:** All 'answers' are attempts and could be incorrect.

## Problem Set 1
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

### Additional Notes

The number of ways to partition k indistinguishable objects into n distinguishable groups is $$\binom{n+k-1}{k}$$. One way to think about this is to imagine $k$ "dots" and $n-1$ "bars" on a horizontal line, such that the bars divide the dots into $n$ groups. There would be $n+k-1$ locations to place a dot or bar, and we have to choose $k$ locations to put the dots (or $n-1$ locations to put the bars). This method comes in handy for a large family of problems (integer partitions, $k$ picks from $n$ objects with replacement).

There are also some useful identities that warrant attention.

$$
n \binom{n-1}{k-1} = k \binom{n}{k}
$$

This can obviously be derived algebraically, but it can also be proven using an intuitive explanation. Suppose we choose $k$ team members out of $n$ students, and designate one of the team members as the captain. This is the same as choosing a captain from $n$ students, then picking the rest of the team ($k-1$ members) from the remaining $n-1$ students.

Next, we have Vandermonde's identity.

$$
\binom{m+n}{k} = \sum_{i=0}^{k} \binom{m}{i} \binom{n}{k-i}
$$

Imagine two groups of indistinguishable objects of size $m$ and $n$. We have to pick $k$ objects from the two groups. We first pick $i$ objects ($0 \leq i \leq k$) from the group of size $m$, then pick the remaining $k-i$ objects from the group of size $n$. The total number of ways to do this is equal to the sum of $\binom{m}{i} \binom{n}{k-i}$ over all possible values of $i$.

## Problem Set 2
### Question A1

$$P(S_5) = \frac{4}{6^2} = \frac{1}{9}$$

$$P(S_7) = \frac{6}{6^2} = \frac{1}{6}$$

$$P(E) = P(S_5) + (1 - P(S_5) - P(S_7)) \cdot P(E) = \frac{2}{5}$$

### Question A2

Can be shown visually using a Venn diagram with three sets. Alternatively:

$$
\begin{split}

P(E \cup F \cup G) & = P(E \cup F) + P(G) - P((E \cup F)G) \\

& = P(E) + P(F) + P(G) - P(FG) - P(EG) - P(EF) + P(EFG) \\

& = P(E) + P(F) + P(G) - P(E^cFG) - P(EF^cG) - P(EFG^c) - 3P(EFG) + P(EFG) \\

& = P(E) + P(F) + P(G) - P(E^cFG) - P(EF^cG) - P(EFG^c) - 2P(EFG) \\

\end{split}
$$

### Question A3

If all points are equally likely with $$P(E_i) = p$$ for some $$p > 0$$, the sum of $$P(E_i)$$ over the whole set will be $$\sum^{\infty}_{i=1} P(E_i) = \infty$$, which is a contradiction since the sum should be $$1$$. Similarly, a contradiction is reached if $$p = 0$$.

All points can have positive $$p$$, for example in the case of $$P(E_i) = \frac{1}{2^i}$$.

### Question B1

There exists arbitrage opportunities where the prices for opposing bets on predictit and betfair add up to less than 100. Examples include buying predictit NO and betfair YES for Yang ($$89 + 6.1 = 95.1$$), Buttigieg ($$92 + 5 = 97$$), and Sanders ($$84 + 13.5 = 97.5$$).

### Question B2

People betting on presidential nomination contracts are subject to many cognitive biases (overconfidence, information bias), which gives room to pricing errors and arbitrage opportunities.

### Additional Notes

A classic and famous problem in probability is De Montmort's matching problem. Imagine we have sorted cards labelled $$1, 2, \cdots, n$$. We then shuffle the deck of cards. If there is at least one card that happens to be in its original position (e.g. the card labelled $$7$$ remains in the $$7th$$ position), we win the game. What is the probability of winning?

For any card, the probability of it remaining in the same position after shuffling is $$P(W_1) = \frac{(n-1)!}{n!} = \frac{1}{n}$$. For any two cards, the probability of both of them remaining in the same positions after shuffling is $$P(W_2) = \frac{(n-2)!}{n!} = \frac{1}{n(n-1)}$$.


Note that the probability of winning $$P(W)$$ is $$P(W_1 \cup W_2 \cup \cdots \cup W_n)$$. So by the inclusion-exclusion principle, we have:

$$
\begin{split}

P(W) & = P(W_1 \cup W_2 \cup \cdots \cup W_n) = P(\bigcup_{i=1}^{n} W_i) \\

& = \binom{n}{1} P(W_1) - \binom{n}{2} P(W_2) + \cdots + (-1)^{n+1} \binom{n}{n} P(W_n) \\

& = n \cdot \frac{1}{n} - \frac{n(n-1)}{2!} \frac{1}{n(n-1)} + \cdots + (-1)^{n+1} (\frac{n!}{n!}) \frac{1}{n!} \\

& = 1 - \frac{1}{2!} + \cdots + (-1)^{n+1} \frac{1}{n!} \\

& \approx 1 - e^{-1}

\end{split}
$$
