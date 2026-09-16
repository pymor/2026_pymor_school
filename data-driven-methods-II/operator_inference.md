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

# Operator Inference


## Goal

Approximate a nonlinear system $$\dot{x}(t) = f(x(t), u(t))$$ from data.


## Idea

Assume we have measurements (or simulation data) of $x$ and $\dot{x}$ (and $u$) and the nonlinear system can be approximate with a quadratic-bilinear system
$$\dot{x}(t) = A x(t) + N (x(t) \otimes u(t)) + H (x(t) \otimes x(t)) + B u(t).$$

Note that the residual
$$\dot{x}(t) - A x(t) - N (x(t) \otimes u(t)) - H (x(t) \otimes x(t)) - B u(t)$$
depends linearly on the unknown $A, N, H, B$.

Therefore, we can obtain $A, N, H, B$ using (regularized) least-squares
$$\min_{A, N, H, B}
\sum_{i = 1}^{n_d} \lVert \dot{x}_i - A x_i - N (x_i \otimes u_i) - H (x_i \otimes x_i) - B u_i \rVert^2 +
\lambda_A \lVert A \lVert_F^2 +
\lambda_N \lVert N \lVert_F^2 +
\lambda_H \lVert H \lVert_F^2 +
\lambda_B \lVert B \lVert_F^2.$$

To make it computationaly feasible and to obtain a reduced-order model, the data is first projected onto the POD modes:
$$\tilde{x}_i = U^T x_i, \quad \tilde{\dot{x}}_i = U^T \dot{x}_i.$$


## Lift and Learn

If the dynamics $f$ is known and one can derive a lifting map $L$ such that the dynamics in variables $L(x)$ is quadratic-bilinear,
Operator Inference can be applied to the data $L(x_i)$, $L(\dot{x}_i)$.


## Reprojection

Standard Operator Inference only approximates POD.
If the data is additionally obtained from simulation and using reprojection,
this version of Operator Inference exactly recovers POD.
See [here](https://arxiv.org/abs/1908.11233).
