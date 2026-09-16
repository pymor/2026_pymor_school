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

# Sparse Identification of Nonlinear Dynamics (SINDy)


## Goal

Approximate a nonlinear system $$\dot{x}(t) = f(x(t))$$ from data.


## Idea

Physical models are often parsimonious (simple, small number of terms).
Therefore, maybe the given system can be approximated with functions from a dictionary $\mathcal{D} = \{f_1, f_2, \ldots, f_d\}$:

$$
\dot{x}(t) =
\underbrace{
  \begin{bmatrix}
    \xi_{11} & \xi_{12} & \cdots & \xi_{1d} \\
    \xi_{21} & \xi_{22} & \cdots & \xi_{2d} \\
    \vdots & \vdots & \ddots & \vdots \\
    \xi_{n1} & \xi_{n2} & \cdots & \xi_{nd}
  \end{bmatrix}
}_{\Xi}
\underbrace{
  \begin{bmatrix}
    f_1(x(t)) \\ f_2(x(t)) \\ \vdots \\ f_d(x(t))
  \end{bmatrix}
}_{F(x(t))}
$$

where every row of $\Xi$ is sparse (has only a few nonzero entries).

Thus, using data $x_i$ and $\dot{x}_i$, apply sparse regression for each row $k$:

$$
\min_{\Xi_{k:}} \sum_{i = 1}^{n_d} (\dot{x}_i[k] - \Xi_{k:} F(x_i))^2 + \lambda \lVert \Xi_{k:} \rVert_1
$$
