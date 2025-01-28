---
title: "Self attention and non-local means"
date: 2025-01-26T00:00:00Z
draft: false
math: true
---

Non-local means is a denoising technique that preserves details; at its core it is 
a weighted average of the image. Assuming a 1d signal we have,

$$
\begin{equation}
I(x) = \frac{1}{Z(x)} \int  K (x,\tilde{x}) I'(\tilde{x}) d \tilde{x}
\label{eq:nonlocal}
\end{equation}
$$

where,

- $I(x)$ is the filtered image
- \(I'(\tilde{x})\) is the unfiltered image
- $K (x,\tilde{x})$ is weighting function
- $Z(x)$ is a normalization factor given by $\int   K (x,\tilde{x})  d \tilde{x}$

Typically, the goal is to take an average of pixels that have similar values. A simple 
weighting function that achieves this is,

$$
\begin{equation}
    K(x,\tilde{x}) = \exp\left[ - \frac{ \left( I'(x) - I'(\tilde{x}) \right)^2 }{h^2} \right]
\label{eq:weight}    
\end{equation}
$$

where $h$ is a scale factor. 


To deepen our understanding of the self-attention mechanism, let's try to write \eqref{eq:nonlocal} and \eqref{eq:weight} with the key, query and value matrices 
that are commonly used in the transformer architecture.

Assuming discrete 1d signal \( I'\in \mathbb{R}^{n\times 1} \), we want,

$$
\begin{equation}
    -(I_i' - I_j')^2 = -I_i'^2 - I_j'^2 + 2 I_i' I_j' = (X_i W_Q) ( X_j W_K)^T 
    \label{eq:expand}
\end{equation}
$$

where,
- $W_Q \in \mathbb{R}^{d\times m}$ is the query matrix
- $W_K \in \mathbb{R}^{d\times m}$ is the key matrix
- $I'_i$ is a pixel of $I'$ 
- $X_i \in \mathbb{R}^{1 \times d}$ is a feature vector, more precisely \(X_i = \Xi(I'_i)\) where \(\Xi\) is an embedding function.

One way to achieve this is to take $m=d=3$,

$$
\begin{equation}
    X_i = \Xi(I_i') = 
    \begin{bmatrix}
        I_i'^2 & I_i' & 1 
    \end{bmatrix}, \hspace{0.5cm}
\end{equation}
$$
$$
\begin{equation}
    W_Q=\begin{bmatrix}
        1 & 0         & 0 \\
        0 & \sqrt{2}  & 0\\ 
        0 & 0         & 1
    \end{bmatrix} \hspace{0.5cm}
    W_K=\begin{bmatrix}
        0 & 0         & -1 \\
        0 & \sqrt{2} & 0\\ 
        -1 & 0         & 0
    \end{bmatrix}
\end{equation}
$$

<!-- We can also make this unnecessarily complicated by introducing the imaginary unit \(i=\sqrt{-1}\), then we have,

$$
\begin{equation}
    W_Q=\begin{bmatrix}
        i & 0         & 0 \\
        0 & \sqrt{2}  & 0\\ 
        0 & 0         & i
    \end{bmatrix}, \hspace{0.5 cm}
    W_K=\begin{bmatrix}
        0 & 0         & i \\
        0 & \sqrt{2} & 0\\ 
        i & 0         & 0
    \end{bmatrix}
\end{equation}
$$ -->

Now, we have (assuming $h=1$),

$$
\begin{align}
    \alpha_{ij} &= \frac{\exp\left[(X_i W_Q) ( X_j W_K)^T \right]}{ \sum_j \exp\left[(X_i W_Q) ( X_j W_K)^T \right] } \\
                &= \text{softmax}_j \left( (X_i W_Q) ( X_j W_K)^T   \right) \\
                &= \frac{\exp \left[-  \left(I'_i -I'_j \right)^2 \right]}{ \sum_j \exp \left[ - \left(I'_i -I'_j \right)^2  \right]} \\
                &= \frac{1}{Z(i)} K(i,j)
\end{align}
$$

where $\text{softmax}_j (.)$ means normalizing over $j$ index.

<!-- Now usually there is a value matrix, $W_V$, also, but to make the connection with non-local means it is not needed. So let's take, -->

<!-- $$
\begin{equation}
    V_i = I'_i
\end{equation}
$$ -->

In the transformer self attention mechanism we have value matrix, $W_V$. In this setup it would be simply,

$$
\begin{equation}
    W_V= \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}
\end{equation}
$$




Going back to \eqref{eq:nonlocal}, we have,

$$
\begin{align}
    I(i) &= \frac{1}{Z(i)} \sum_j K(i,j) I'_j \\
         &= \sum_j \alpha_{ij} I'_j \\
         &= \sum_j \text{softmax}_j \left( (X_i W_Q) ( X_j W_K)^T  \right)  X_j W_V
\end{align}
$$

We can go a bit further and compute the whole vector $I \in \mathbb{R}^{n\times 1}$, and bring back $h$,

$$
\begin{align}
    I =  \text{softmax} \left( \frac{Q K^T}{h^2} \right) V
\end{align}
$$

where

- $V = X W_V$, where $X \in \mathbb{R}^{n \times d}$, $W_V \in \mathbb{R}^{d \times 1}$ and $V \in \mathbb{R}^{n \times 1}$
- $K =X W_K$, where  $W_K \in \mathbb{R}^{d \times m} $ and $K \in \mathbb{R}^{n \times m} $ 
- $Q =X W_Q $, where $W_Q \in \mathbb{R}^{d \times m} $ and $Q \in \mathbb{R}^{n \times m} $ 
- $\text{softmax} \left( \frac{Q K^T}{h^2} \right) \in  \mathbb{R}^{n \times n}$ 

Here, $\text{softmax}(.)$ is a row-wise operation. Note that this self-attention is scaled with $h^2$ rather than the usual
self-attention, which is scaled with $\sqrt{d}$.