---
title: "Simple Linear Regression"
date: 2025-01-23T00:00:00Z
draft: false
math: true
---



Although a straightforward procedure, simple linear regression captures some key ideas about model fitting. Let's consider $ x \in \mathbb{R} \rightarrow y \in \mathbb{R} $ 
and one is given pairwise data $(y_i,x_i)$. The relationship we assume between $x$ and $y$ is a linear one, yielding a simple model,

$$
\begin{equation}
y=w_0 + w_1 x,
\label{eq:lin}
\end{equation}
$$

where the $w_i \in \mathbb{R}$. Geometrically, we are assuming that scaling an $x$ by $w_1$ and then translating by $w_0$ gets you a $y$. At the risk of being pedantic, one can also express \eqref{eq:lin} as,

$$
\begin{equation}
    \begin{bmatrix}
    y \\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    w_1 & w_0 \\
    0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    x \\
    1
    \end{bmatrix}=
    \begin{bmatrix}
    w_0 + w_1x \\
    1
    \end{bmatrix}.
\end{equation}
$$

As an aside, under translations, $x \rightarrow x + a \implies  y \rightarrow y + w_1 a $, and the $w_i$ are invariant to translations in $x$, i.e., all $x$ share the same $w_i$'s.

Okay, so that's the model, now the fitting. Naturally, the data provided will contain an additive noise term $\epsilon \sim  \mathcal{N}(0, \sigma^2)$. Like this,


{{< figure src="/images/linear_regression/lin_reg.png" width="350em">}}

Our job is to perform this optimization,

$$
\begin{equation}
\operatorname*{argmin}_{w_0,w_1} \sum_{i=1}^{N} \left( w_0 + w_1 x_i - y_i\right)^2
\end{equation}
$$

It's mildly tedious, but one can show that the result of the optimization is,

$$
\begin{equation}
w_0 = \bar{y} - w_1 \bar{x},
\end{equation}
$$

$$
\begin{equation}
w_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2},
\end{equation}
$$

where the bar indicates averages. Now, earlier we said that the $w_i$'s are independent of $x$, i.e., they are invariant to translations of $x$. But what happens if we change this? Let's make the following change,

$$
\begin{equation}
w_1 \rightarrow w'_1(x)
\end{equation}
$$

Further, let's assume a linear dependence, \( w'_1(x) = w_1 + w_2 x \), we then have,

$$
\begin{aligned}
y &= w_0 + w_1'(x) x \\
&= w_0 + w_1 x + w_2 x^2
\end{aligned}
$$

This is now quadratic regression! We could repeat this process and make $w_2$ dependent on $x$ to get higher-order polynomials. Note also now the weights are not shared between different values of $x$, they depend on $x$ but in a simple linear fashion. In general, one can have different types of dependence of $x$ (e.g., linear basis function models). 
