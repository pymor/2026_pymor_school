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

# Dynamic Mode Decomposition (DMD)


## Motivation


Consider a discrete-time system

$$
x_{k + 1} = T(x_k)
$$

where $x_k \in \mathbb{R}^n$.


Suppose we have data as sequential snapshots $x_0, x_1, \ldots, x_m \in \mathbb{R}^n$.

The idea is to approximate the dynamics with a linear system

$$
\tilde{x}_{k + 1} = A \tilde{x}_k.
$$


Then it should be that $x_{k + 1} \approx A x_k$, which we can write as

$$
\begin{align*}
  \underbrace{
    \begin{bmatrix}
      x_1 & x_2 & \cdots & x_m
    \end{bmatrix}
  }_{=: Y}
  & \approx
  A
  \underbrace{
    \begin{bmatrix}
      x_0 & x_1 & \cdots & x_{m - 1}
    \end{bmatrix}
  }_{=: X} \\
  Y & \approx A X
\end{align*}
$$


Thus, we could determine $A$ by solving the least squares problem

$$
A =
\operatorname*{arg\,min}_{\mathcal{A} \in \mathbb{R}^{n \times n}}
\lVert Y - \mathcal{A} X \rVert_F^2,
$$

i.e.

$$
A = Y X^+.
$$


Typically, $X$ will have a low-numerical rank with a truncated SVD $X = U \Sigma V^T$.
Then $$A = Y V \Sigma^{-1} U^T,$$ and we could apply the Galerkin projection with $U$ to obtain a reduced system

$$
\hat{x}_{k + 1} = \underbrace{U^T Y V \Sigma^{-1}}_{\hat{A}} \hat{x}_k,
$$

and approximate the full solution with $x \approx U \hat{x}$.


Diagonalizing $\hat{A} = Z \Lambda Z^{-1}$ and defining a new reduced state $\hat{y} = Z^{-1} \hat{x}$ gives a new reduced system

$$
\hat{y}_{k + 1} = \Lambda \hat{y}_k
$$

and approximate full solution $x \approx U Z \hat{y}$.


Therefore,

$$
x_k \approx U Z \Lambda^k Z^{-1} \hat{x}_0
$$


Columns of $U Z$ are called *DMD modes*.

Columns of $Y V \Sigma^{-1} Z = A U Z$ were proposed as *exact DMD modes*.


## Interpretation as a data-driven eigensolver


If the goal is to find a few eigenpairs of a matrix $A$ given orthonormal vectors
$U = \begin{bmatrix} u_1 & u_2 & \cdots & u_m \end{bmatrix}$,
a standard approach would be using Ritz vectors:

> Computing an eigenvalue decomposition of $\hat{A} = U^T A U = Z \Lambda Z^{-1}$
gives approximate eigenvalues $\lambda_i$ and eigenvectors $U z_i$ (Ritz pairs) of $A$.

As we saw above, this can be done using only $X$ and $Y$, without direct access to $A$.


Additionally, exact DMD modes can be interpreted as a single iteration of the power iteration starting with Ritz vectors.


Importantly, the error in the Ritz pairs can be evaluated using the residual $A U z_i - \lambda_i U z_i$.
Note that

$$
A U z_i - \lambda_i U z_i =
Y V \Sigma^{-1} z_i - \lambda_i U z_i.
$$

Thus, norms of the residual can also be evaluated without access to $A$.

It is a good idea to use them to get an idea about the accuracy of the DMD modes (exact DMD modes might not be more accurate in this sense).


## Extensions


DMD can be extended to systems with inputs/controls (DMDc) and with inputs and outputs (ioDMD):

$$
\begin{align*}
  x_{k + 1} & = A x_k + B u_k, \\
  y_k & = C x_k + D u_k,
\end{align*}
$$

with input data $u_0, u_1, \ldots, u_m$,
state data $x_0, x_1, \ldots, x_m$,
and output data $y_0, y_1, \ldots, y_m$.


## Exercise

Apply `pymor.algorithms.dmd.dmd`.
