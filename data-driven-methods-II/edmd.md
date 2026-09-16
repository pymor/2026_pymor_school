---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Extended Dynamic Mode Decomposition (EDMD)


## Koopman operator


Consider a discrete-time system

$$
x_{k + 1} = T(x_k)
$$

where $x_k \in \mathbb{R}^n$.


Define the Koopman operator $$\mathcal{K}[f] = f \circ T$$ for scalar-valued functions $f$ in some function space.

Note that the Koopman operator is linear, but infinite-dimensional.


## Linear systems

Suppose that $T(x) = A x$ for some matrix $A$.


How does $\mathcal{K}$ behave for $g_i(x) = x_i = e_i^T x$?

We see that

$$
\begin{align}
\mathcal{K}[g_i](x)
& = (g_i \circ T)(x) \\
& = g_i(T(x)) \\
& = g_i(A x) \\
& = e_i^T A x \\
& = \sum_{j = 1}^n a_{ij} e_j^T x \\
& = \sum_{j = 1}^n a_{ij} g_j(x).
\end{align}
$$


Thus, $$\mathcal{K}[g_i] = \sum_{j = 1}^n a_{ij} g_j.$$

Therefore, the span of $g_1, g_2, \ldots, g_n$ is an **invariant subspace** of $\mathcal{K}$ and

$$
\mathcal{K}
\begin{bmatrix}
  g_1 & g_2 & \cdots & g_n
\end{bmatrix} =
\begin{bmatrix}
  g_1 & g_2 & \cdots & g_n
\end{bmatrix}
A^T.
$$


Furthermore, we see that the matrix representation of $\mathcal{K}|_{\operatorname{span}\{g_1, g_2, \ldots, g_n\}}$ is $A^T$.


## Nonlinear systems


Idea: If $\mathcal{K}[\varphi_i] = \lambda_i \varphi_i$ is a spectral decomposition and
we can expand the state as $$x = \sum_i \varphi_i(x) v_i,$$
then $$x_k = \sum_i \lambda_i^k \varphi_i(x_0) v_i.$$


Suppose that functions $\psi_1, \psi_2, \ldots, \psi_N$ span an approximately invariant subspace of $\mathcal{K}$.


We want to find $K \in \mathbb{R}^{N \times N}$ such that

$$
\mathcal{K}
\begin{bmatrix}
  \psi_1 & \psi_2 & \cdots & \psi_N
\end{bmatrix} \approx
\begin{bmatrix}
  \psi_1 & \psi_2 & \cdots & \psi_N
\end{bmatrix}
K.
$$


Using data $x_1, x_2, \ldots, x_{m + 1}$, we construct

$$
\Psi_X =
\begin{bmatrix}
  \psi_1(x_1) & \psi_2(x_1) & \cdots & \psi_N(x_1) \\
  \psi_1(x_2) & \psi_2(x_2) & \cdots & \psi_N(x_2) \\
  \vdots & \vdots & \ddots & \vdots \\
  \psi_1(x_m) & \psi_2(x_m) & \cdots & \psi_N(x_m)
\end{bmatrix}, \quad
\Psi_Y =
\begin{bmatrix}
  \psi_1(x_2) & \psi_2(x_2) & \cdots & \psi_N(x_2) \\
  \psi_1(x_3) & \psi_2(x_3) & \cdots & \psi_N(x_3) \\
  \vdots & \vdots & \ddots & \vdots \\
  \psi_1(x_{m + 1}) & \psi_2(x_{m + 1}) & \cdots & \psi_N(x_{m + 1})
\end{bmatrix}
$$

and solve the least-squares problem

$$
K = \operatorname{arg\,min}_K \lVert \Psi_Y - \Psi_X K \rVert_F^2 = \Psi_X^+ \Psi_Y.
$$


Compute an eigenvalue decomposition $$K = \Xi \Lambda \Xi^{-1}.$$


The approximate Koopman eigenfunctions are

$$
\begin{bmatrix}
  \varphi_1 & \varphi_2 & \cdots & \varphi_N
\end{bmatrix} =
\begin{bmatrix}
  \psi_1 & \psi_2 & \cdots & \psi_N
\end{bmatrix}
\Xi.
$$


Suppose that $g_1, g_2, \ldots, g_n$ are included in the span of $\psi$s and

$$
\begin{bmatrix}
  g_1 & g_2 & \cdots & g_n
\end{bmatrix} =
\begin{bmatrix}
  \psi_1 & \psi_2 & \cdots & \psi_N
\end{bmatrix}
B.
$$


Then,

$$
\begin{align}
  x^T
  & =
  \begin{bmatrix}
    g_1(x) & g_2(x) & \cdots & g_n(x)
  \end{bmatrix} \\
  & =
  \begin{bmatrix}
    \psi_1(x) & \psi_2(x) & \cdots & \psi_N(x)
  \end{bmatrix}
  B \\
  & =
  \begin{bmatrix}
    \varphi_1(x) & \varphi_2(x) & \cdots & \varphi_N(x)
  \end{bmatrix}
  \Xi^{-1}
  B.
\end{align}
$$


Therefore,

$$
x =
\sum_{i = 1}^N
\varphi_i(x)
B^T \Xi^{-T} e_i
$$

and

$$
x_k =
\sum_{i = 1}^N
\lambda_i^k
\varphi_i(x_0)
B^T \Xi^{-T} e_i.
$$


## Exercise


Consider a nonlinear system with $T(x_1, x_2) = (p x_1, q x_2 + (p^2 - q) x_1^2)$ for fixed parameters $p, q$.

1. Show that the span of $x_1, x_2, x_1^2$ is invariant under $\mathcal{K}$ with $$K = \begin{bmatrix} p & 0 & 0 \\ 0 & q & 0 \\ 0 & p^2 - q & p^2 \end{bmatrix}$$.
2. Reconstruct $T$ using EDMD (analitically and numerically).
