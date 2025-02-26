---
title: "Multi-head self attention"
date: 2025-02-14T00:00:00Z
draft: false
math: true
---






### Math
Here we will define and carry out a PyTorch implementation of multi-head self attention. 
We introduced self attention in a [previous post]({{<relref "posts/self_attention.md">}}) exploring a connection with non-local means. 

Let us now introduce multi head self attention as used in transformer-like models. Also, lets experiment 
with [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) here to see if it makes things easier or more complicated. We will raise and lower indices simply for clarity and not to indicate contravariant or covariant indices. For example,

$$
\begin{equation}
    A_{aj} B_b^{\ \ j} = \sum_{j} A_{aj} B_{bj}
\end{equation}
$$
basically a repeated index is summed over unless specified otherwise.  Below, the exponents to $\mathbb{R}$ are in the same order as the indices. For example, when we say $A_{aj} \in \mathbb{R}^{d_a \times d_j} $ then, \(a \in \{0, 1, ... , d_a-1\}\) and \(j \in \{0, 1, ... , d_j-1\}\).

Consider an input signal,

$$
\begin{equation}
    I_{\tau i} \in \mathbb{R}^{N \times d_{\text{input}}}
\end{equation}
$$

where $N$ is the context length. Then we have an embedding function $\Xi$ such that,

$$
\begin{equation}
    X_{\tau i} = \Xi \left( I  \right )_{\tau i} \in \mathbb{R}^{N \times d_{m}}
\end{equation}
$$

The main work horses of the self attention mechanism are the query, key and value matrices. They are, respectively,

$$
\begin{equation}
    \left(W_{Q}\right)_{ia}, \left(W_{K} \right)_{ia} \in  \mathbb{R}^{d_m \times d_k} \ \ \  \text{and}  \ \ \ \left(W_{V}\right)_{ia} \in  \mathbb{R}^{d_m \times d_v}
\end{equation}
$$

Next we have the attention scores,

$$
\begin{align}
    \tilde{A}_{\tau \tau'} &=\frac {X_{\tau}^{\ \ i} \left( W_Q \right)_{ia} \left( W_K \right)_{j}^{\ \ a} X_{\tau'}^{\ \ \ j}}{\sqrt{d_k}} \\
                           &=\frac {Q_{\tau a} K_{\tau'}^{\ \ \ a}}{\sqrt{d_k}}
\end{align}
$$

where \(Q_{\tau a} = X_{\tau}^{\ \ i} \left(W_Q \right)_{ia} \) and \( K_{\tau'}^{\ \ \ a} = \left( W_K \right)_{j}^{\ \ a} X_{\tau'}^{\ \ \ j}\).


Now to get the attention weights we apply the softmax function,

$$
\begin{equation}
    A_{\tau \tau'} = \text{softmax} ( \tilde{A}_{\tau} ) _{\tau'} = \frac{\exp \left(\tilde{A}_{\tau \tau'} \right)}{\sum_{\tau'} \exp(\tilde{A}_{\tau \tau'})}
\end{equation}
$$


The next piece of the puzzle is the role of the value matrix,

$$
\begin{equation}
    V_{\tau a} = \left(W_V\right)^{i}_{\ a} X_{\tau i} 
\end{equation}
$$

So finally we have self-attention defined as,

$$
\begin{equation}
    \text{Attention}(Q,K,V)_{\tau a} = A_{\tau \tau'} V^{\tau'}_{\ \ \ a}
\end{equation}
$$

Okay, so that's just a single "head", i.e., one set of key, query and key matrices. In general, we can have multiple heads, which means we have many 
$W_Q$, $W_K$ and $W_V$ matrices. In index notation this just means that we add a new index, $\mu$ on each of these matrices which then carries over to the 
self-attention;

$$
\begin{equation}
    \text{Attention}(Q,K,V)_{\mu \tau a} = A_{\mu \tau \tau'} V^{\ \tau'}_{\mu \ \ a}
\end{equation}
$$

In this case the notation is a bit confusing because we have a repeated index on the RHS which is not a summation, but rather is a label. This is specified
by it being a free index on the LHS.

Now, we combine all the heads,

$$
\begin{align}
    \text{MultiHead}(Q, K, V)_{\tau i} =\left(W_O \right)^{\mu a}_{\ \ \ \ i} \  \text{Attention}(Q,K,V)_{\mu \tau a}
\end{align}
$$

where, 

$$
\begin{equation}
    (W_O)_{\mu a i} \in \mathbb{R}^{h \times d_v \times d_m}
\end{equation}
$$

projects back to our embedding space, here $h$ is the number of heads.


There is one other detail; we should only have attention scores to previous tokens, we can achieve this by adding a mask to $\tilde{A}$,

$$
\begin{equation}
    \tilde{A}_{\mu \tau \tau'} \leftarrow \tilde{A}_{\mu \tau \tau'} + \tilde{M}_{\mu \tau \tau'}
\end{equation}
$$

where;

$$
\begin{equation}
    \tilde{M}_{\mu \tau \tau'} = 
            \begin{cases}
                0, & \text{if } \tau' \leq \tau \ \ \forall \ \ \mu \\
                -\infty, & \text{if } \tau' \gt \tau \ \ \forall \ \ \mu
            \end{cases}
\end{equation}
$$


#### Fusing indices
I am not sure if the following is standard notation for index fusions, if you know please reach out!

 Consider a tensor $T_{\mu a i j} \in \mathbb{R}^{n \times m \times p \times q}$ then $T_{\mu (ai)j} \in \mathbb{R}^{n \times m p \times q}$, where $(ai)$ is a new single index, given by, $(ai)= a p + i$. Now, $T_{\mu (ai)j}$ is to be interpreted as a batch of 2d matrices, where $\mu$ labels each matrix. These type of index fusions will help us keep track of large matrices that are GPU friendly. 
 
 For example consider the case $d_k=d_v$ and the tensor $W_{\mu B ia}$, where,

$$
\begin{equation}
    W_{\mu 0ia} =  \left(W_{Q} \right)_{\mu ia} ,\ \   W_{\mu 1ia}= \left(W_{K} \right)_{\mu ia} ,\ \  W_{\mu 2ia}= \left(W_{V} \right)_{\mu ia}  
\end{equation}
$$

again $\mu$ runs over all the heads. We can contract this as,

$$
\begin{equation}
    W_{(\mu B a) i} \in \mathbb{R}^{h3d_k \times d_m}
\end{equation}
$$

Why? Because all these matrices multiply $X_{\tau i}$. I can do this in one go now;

$$
\begin{equation}
    W_{(\mu Ba)i} X_\tau^{\ \ i}
\end{equation} 
$$ 
which is a 2d matrix multiplication :sunglasses:. Although, now I have to split this back to compute the self attention.


#### An illustration
Below is an illustration of all the mathematics shown above, the flow is from bottom to up. Note that there is no mask in this illustration, however, it is a straight forward extension.
{{< figure src="/images/attention/attention_down.png" width="600em">}}



<!-- 



For a single "head", we have;

$$
\begin{align}
    \text{Attention}(Q, K, V)_{ai} &= \text{softmax} \left( \frac{Q_{a j} K_{b}^{\ \ j}}{\sqrt{d_k}} \right) V^{b}_{\ \ \ i} \\
                                   &=
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
    \label{eq:multiattn}
\end{equation}
$$

Note here, that although $\mu$ is repeated it is a label and a free index on the LHS. -->



<!-- 
There is one other detail; we should only have attention scores to previous tokens, we can achieve this by,

$$
\begin{equation}
    \text{softmax} \left( \frac{Q_{\mu a j} K_{\mu b}^{\ \ \ \ j}}{\sqrt{d_k}} + M_{\mu a b}\right) 
\end{equation}
$$

where;

$$
\begin{equation}
    M_{\mu a b} = 
            \begin{cases}
                0, & \text{if } b \leq a \\
                -\infty, & \text{if } b \gt a 
            \end{cases}
\end{equation}
$$
 -->

### PyTorch implementation
Below is a simple PyTorch implementation;
```python
import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dm, dk, N, h=1):
        super(SelfAttention, self).__init__()
        self.dm = dm    # Model dimension
        self.dk = dk    # Key/Query dimension per head
        self.h = h      # Number of heads
        self.N = N      # Sequence length

        # Linear layer to project input into Q, K, V
        self.Wqkv = nn.Linear(dm, 3 * dk * h)
        
        # Final linear layer to project concatenated heads back to dm
        self.final = nn.Linear(h * dk, dm)

        # Causal mask to prevent attention to future tokens
        self.register_buffer("causal_mask", self.create_causal_mask(N))

        # Weight initialization (similar to GPT-2)
        nn.init.normal_(self.Wqkv.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.final.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        B, N, _ = x.shape  # Batch size, Sequence length, Model dimension

        # Linear projection to obtain Q, K, V
        QKV = self.Wqkv(x)
        Q, K, V = torch.split(QKV, [self.dk * self.h] * 3, dim=-1)

        # Reshape Q, K, V to [B, h, N, dk]
        Q, K, V = map(self.reshape, (Q, K, V))

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk ** 0.5)

        # Apply causal mask
        attn_scores += self.causal_mask[:N, :N].to(attn_scores.device)[None, None, :, :]

        # Apply external mask if provided (optional)
        if mask is not None:
            attn_scores += mask[:, None, :, :].masked_fill(mask == 0, float('-inf'))

        # Softmax to obtain attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Attention output
        attn_output = torch.matmul(attn_weights, V)  # [B, h, N, dk]

        # Concatenate heads and project back to dm
        attn_output = attn_output.transpose(1, 2).reshape(B, N, self.h * self.dk)
        output = self.final(attn_output)

        return output, attn_weights

    def reshape(self, A):
        """
        Reshape input from [B, N, h * dk] to [B, h, N, dk].
        """
        B, N, _ = A.shape
        return A.view(B, N, self.h, self.dk).permute(0, 2, 1, 3)

    def create_causal_mask(self, N):
        """
        Creates a causal mask to prevent attention to future tokens.
        Shape: [N, N] with -inf in the upper triangle (excluding diagonal).
        """
        mask = torch.triu(torch.full((N, N), float('-inf')), diagonal=1)
        return mask
```