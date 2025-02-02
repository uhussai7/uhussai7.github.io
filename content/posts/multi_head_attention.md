---
title: "Multi-head self attention"
date: 2025-02-01T00:00:00Z
draft: false
math: true
---
### Math
Here we will define and carry out a PyTorch implementation of multi-head self attention. 
We introduced self attention in a [previous post]({{<relref "posts/self_attention.md">}}) exploring a connection with non-local means. 

Let us now introduce self attention as used in transformer-like models. I will experiment 
with [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) here to see if it makes things easier or more complicated. Below,
the exponents to $\mathbb{R}$ are in the same order as the indices. For a single "head", we have;

$$
\begin{align}
    \text{Attention}(Q, K, V)_{ai} = \text{softmax} \left( \frac{Q_{a j} K_{b}^{\ \ j}}{\sqrt{d_k}} \right) V^{b}_{\ \ \ i}
\end{align}
$$

where $Q_{a i}, K_{a i} \in \mathbb{R}^{N \times d_k}$, and $V_{ai} \in \mathbb{R}^{N \times d_v}$


and,

$$
\begin{align}
    \nonumber Q_{ai} &= X_{aj} \left(W_{Q} \right)_{i}^{\ \ j}, \ \ \ \ K_{ai} = X_{aj} \left(W_{K} \right)_{i}^{\ \ j}\\ 
    & \ \ \ \text{and} \ \ \ \ V_{ai} = X_{a}^{\ j} \left(W_{V}\right)_{ij}
\end{align}
$$

where,

- \(\left(W_{Q}\right)_{ij}, \left(W_{K} \right)_{ij} \in  \mathbb{R}^{d_m \times d_k} \) and  $ \left(W_{V}\right)_{ij} \in  \mathbb{R}^{d_m \times d_v}$
- $X_{ai} = \Xi\left( I\right) \in \mathbb{R}^{N \times d_m}$, where $I_{ai} \in \mathbb{R}^{N \times d_{\text{input}}}$ is the input signal and $\Xi$ is an embedding function.

Okay so that notation looks pretty busy 😵‍💫, but things appear to be more explicit. 

Going to multi-head attention is simple,

$$
\newcommand{\Q}{Q}
\newcommand{\K}{K}
\newcommand{\V}{V}
\begin{align}
    \text{MultiHead}(\Q, \K, \V)_{ai} =\left(W_O \right)^{\mu j}_{\ \ \ \ i} \  \text{Attention}(Q,K,V)_{\mu aj}
\end{align}
$$

where $(W_O)_{\mu j i} \in \mathbb{R}^{h \times d_v \times d_m}$ and $h$ is the number of heads. Here the extra index, $\mu$ runs over different heads of self attention. The attention for each head is given by,

$$
\begin{equation}
    \text{Attention}(Q, K, V)_{\mu a i} = \text{softmax} \left( \frac{Q_{\mu a j} K_{\mu b}^{\ \ \ \ j}}{\sqrt{d_k}} \right) V^{\ b}_{\mu \ \ i}
\end{equation}
$$

Note here, that although $\mu$ is repeated it is a label and a free index on the LHS.

I am going to abuse notation more; consider a tensor $A_{\mu a i j} \in \mathbb{R}^{n \times m \times p \times q}$ then $A_{\mu |ai,j|} \in \mathbb{R}^{n \times m p \times q}$, which is to be interpreted as a batch of matrices. These type of index fusions will help us keep track of large matrices that are GPU friendly.


### PyTorch implementation
