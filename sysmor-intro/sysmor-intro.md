---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
# enable logging widget
%load_ext pymor.tools.jupyter
```

+++ {"slideshow": {"slide_type": "slide"}}

# System-theoretic Methods

<h2>
September 15, 2026<br/>
pyMOR School and User Meeting 2026
</h2>

+++ {"slideshow": {"slide_type": "subslide"}}

# What this actually is all about:

<center>
<img src="/files/sysmor-intro/figures/system_fom.svg" alt="system" width="60%"/>
</center>

+++ {"slideshow": {"slide_type": "subslide"}}

<center>
<img src="/files/sysmor-intro/figures/mor_system_fo_v2.svg" alt="mor" width="40%">
</center>

+++ {"slideshow": {"slide_type": "subslide"}}

# Outline

<h2>
1. Linear Time-Invariant (LTI) Systems<br/>
2. Transfer Function and Realizations<br/>
3. Projection-based Model Order Reduction<br />
4. System Analysis<br/>
5. A Selection of MOR Methods<br/>
</h2>

+++ {"slideshow": {"slide_type": "slide"}}

# Linear Time-Invariant (LTI) Systems

## Setting for this course

### First-order State-space Systems (see [`LTIModel`](https://docs.pymor.org/2026-1-0/autoapi/pymor/models/iosys/index.html#pymor.models.iosys.LTIModel))

$$
\begin{equation}\tag{$\Sigma$}
  \begin{aligned}
    E \dot{x}(t) & = A x(t) + B u(t), \\
    y(t) & = C x(t) + D u(t).
  \end{aligned}
\end{equation}
$$

Here

- $x(t) \in \mathbb{R}^{n}$ is called the *state*,
- $u(t) \in \mathbb{R}^{m}$ is called the *input*,
- $y(t) \in \mathbb{R}^{p}$ is called the *output*

of the LTI system.
Correspondingly, we have

$$
E, A \in \mathbb{R}^{n \times n}, \qquad
B \in \mathbb{R}^{n \times m}, \qquad
C \in \mathbb{R}^{p \times n}, \quad\text{and}\quad
D \in \mathbb{R}^{p \times m}.
$$

We assume
$t \in [0, \infty)$,
$x(0) = 0$,
$E$ is invertible,
$E^{-1} A$ is Hurwitz, and
$D = 0$.

Without loss of generality, we sometimes assume $E = I$.

+++

### Extensions

Discrete-time systems (e.g., see [[Antoulas '05]](https://doi.org/10.1137/1.9780898718713)):

$$
\begin{align*}
  E \dot{x}_{k+1} & = A x_k + B u_k, \\
  y_k & = C x_k + D u_k.
\end{align*}
$$

Structured systems, e.g., second-order systems:

$$
\begin{align*}
  M \ddot{x}(t) + E \dot{x}(t) + K x(t) & = B u(t), \\
  y(t) & = C_p x(t) + C_v \dot{x}(t).
\end{align*}
$$

Differential-algebraic systems, i.e., $E$ singular
(e.g., see
[[Voigt '19]](https://www.math.uni-hamburg.de/home/voigt/Modellreduktion_SoSe19/Notes_ModelReduction.pdf),
[[Gugercin/Stykel/Wyatt '13]](https://doi.org/10.1137/130906635),
[[Mehrmann/Stykel '05]](https://doi.org/10.1007/3-540-27909-1_3),
[[Stykel '04]](https://doi.org/10.1007/s00498-004-0141-4)).

Nonlinear systems, e.g., quadratic-bilinear systems:

$$
\begin{align*}
  E \dot{x}(t) & = A x(t) + N x(t) u(t) + H (x(t) \otimes x(t)) + B u(t), \\
  y(t) & = C x(t).
\end{align*}
$$

Parametric systems:

$$
\begin{align*}
  E(\mu) \dot{x}(t) & = A(\mu) x(t) + B(\mu) u(t), \\
  y(t) & = C(\mu) x(t) + D(\mu) u(t).
\end{align*}
$$

+++ {"slideshow": {"slide_type": "subslide"}}

## Examples

### Heat Equation ([MORWiki thermal block](https://modelreduction.org/morwiki/index.php/Thermal_Block))

<table>
<tr>
<td>

For $t \in (0, T)$, $\xi \in \Omega$ and initial values

$$
\theta(0, \xi) = 0,\text{ for } \xi \in \Omega,
$$

consider

$$
\begin{align*}
  \partial_t \theta(t, \xi)
  + \nabla \cdot (-\sigma(\xi) \nabla \theta(t, \xi))
  & = 0,
\end{align*}
$$

with boundary conditions

$$
\begin{align*}
  \sigma(\xi) \nabla \theta(t, \xi) \cdot n(\xi) & = u(t)
  & t \in (0, T),
  & \ \xi \in \Gamma_{\text{in}}, \\
  \sigma(\xi) \nabla \theta(t, \xi) \cdot n(\xi) & = 0
  & t \in (0, T),
  & \ \xi \in \Gamma_{\text{N}}, \\
  \theta(t, \xi) & = 0
  & t \in (0, T),
  & \ \xi \in \Gamma_{\text{D}},
\end{align*}
$$

and outputs

$$
y_i(t) = \int_{\Omega_i} \theta(t, \xi) \operatorname{d}\!{\xi}, \quad
i = 1, 2, 3, 4.
$$

</td>
<td>
<center>
<img src="/files/sysmor-intro/figures/cookie.svg" alt="cookie domain" width="30%">
<img src="/files/sysmor-intro/figures/Euler_100_Tf.png" alt="cookie snapshot" width="30%">
</center>
</td>
</tr>
</table>

+++ {"slideshow": {"slide_type": "subslide"}}

#### Finite element semi-discretization in space

- pairwise inner products of ansatz functions $\leadsto E$
- discretized spatial operator + Dirichlet boundary condition $\leadsto A$
- discretized non-zero Neumann boundary condition $\leadsto B$
- average temperatures on the inclusions $\leadsto C$

---

- $n = 7\,488$
- $m = 1$
- $p = 4$

+++ {"slideshow": {"slide_type": "subslide"}}

### Penzl Example ([MORWiki](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Penzl%27s_FOM))

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
import numpy as np
import scipy.sparse as sps
from pymor.models.iosys import LTIModel

A1 = np.array([[-1, 100], [-100, -1]])
A2 = np.array([[-1, 200], [-200, -1]])
A3 = np.array([[-1, 400], [-400, -1]])
A4 = sps.diags(np.arange(-1, -1001, -1), dtype=None)
A = sps.block_diag((A1, A2, A3, A4), format='csc')
B = np.ones((1006, 1))
B[:6] = 10
C = B.T

fom = LTIModel.from_matrices(A, B, C)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
fom
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
print(fom)
```

+++ {"slideshow": {"slide_type": "subslide"}}

We can perform time-domain simulation, but the final time and the time stepper
need to be specified in the `Model`.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper

T = 2
nt = 10000
fom = fom.with_(T=T, time_stepper=ImplicitEulerTimeStepper(nt))
```

+++ {"slideshow": {"slide_type": "subslide"}}

We first simulate the impulse response, i.e.,
the output in response to $x(0) = 0$ and $u(t) = \delta(t)$.

$$
\begin{align*}
  y(t)
  & =
    C e^{t A} x(0)
    + \int_0^t C e^{\tau A} B u(t - \tau) \operatorname{d\!}\tau
    + D u(t) \\
  & =
    C e^{t A} B
    + D \delta(t)
\end{align*}
$$

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
y_impulse = fom.impulse_resp()
print(y_impulse.shape)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import matplotlib.pyplot as plt

_ = plt.plot(np.linspace(0, T, nt + 1), y_impulse[0, :, 0])
```

+++ {"slideshow": {"slide_type": "subslide"}}

Next we simulate the response to a sinusoidal input.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
y_sin_100 = fom.output(input='sin(100 * t)')
print(y_sin_100.shape)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
_ = plt.plot(np.linspace(0, T, nt + 1), y_sin_100[0])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
y_sin_50 = fom.output(input='sin(50 * t)')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
_ = plt.plot(np.linspace(0, T, nt + 1), y_sin_50[0])
```

+++ {"slideshow": {"slide_type": "slide"}}

# Transfer Function

+++ {"slideshow": {"slide_type": "subslide"}}

## Laplace Transform

> ### Definition
>
> Let $f \colon [0, \infty) \to \mathbb{R}^{n}$ be exponentially bounded with
> bounding exponent $\alpha$.
> Then
> $$\mathcal{L}\{f\}(s) := \int_0^\infty f(\tau) e^{-s \tau} \operatorname{d}\!{\tau}$$
> for $\operatorname{Re}(s) > \alpha$ is called the ***Laplace transform*** of $f$.
> The process of forming the Laplace transform is called
> ***Laplace transformation***.

It can be shown that the integral converges uniformly in a domain with
$\operatorname{Re}(s) \ge \beta$ for all $\beta > \alpha$.

+++ {"slideshow": {"slide_type": "fragment"}}

> Allows us to map time signals to frequency signals.

+++ {"slideshow": {"slide_type": "subslide"}}

> ### Theorem
>
> Let $f, g, h \colon [0, \infty) \to \mathbb{R}^n$ be given.
> Then the following two statements hold true:
> 1. The Laplace transformation is linear, i.e.,
>    if $f$ and $g$ are exponentially bounded,
>    then $h := \gamma f + \delta g$ is also exponentially bounded and
>    $$
     \mathcal{L}\left\{h\right\} = \gamma\mathcal{L}\left\{f\right\} +
     \delta\mathcal{L}\left\{g\right\}
     $$
>    holds for all $\gamma, \delta \in \mathbb{C}$.
> 2. If $f \in \mathcal{PC}^1([0, \infty), \mathbb{R}^{n})$ and $\dot{f}$ is
>    exponentially bounded, then $f$ is exponentially bounded and
>    $$
     \mathcal{L}\bigl\{\dot{f}\bigr\}(s) = s \mathcal{L}\{f\}(s) - f(0).
     $$

+++ {"slideshow": {"slide_type": "fragment"}}

- $X(s) := \mathcal{L}\{x\}(s)$,
  $U(s) := \mathcal{L}\{u\}(s)$, and
  $Y(s) := \mathcal{L}\{y\}(s)$
- $A x(t) + B u(t) \leadsto A X(s) + B U(s)$
- $y(t) = C x(t) \leadsto Y(s) = C X(s)$
- $s X(s) := \mathcal{L}\{\dot{x}\}(s)$ (since $x(0) = 0$)

+++ {"slideshow": {"slide_type": "subslide"}}

## Transfer Function

In summary we have:

$$
\begin{align*}
  s X(s) & = A X(s) + B U(s), \\
  Y(s) & = C X(s),
\end{align*}
$$

i.e.,

$$
\begin{align*}
  (s I - A) X(s) & = B U(s), \\
  Y(s) & = C X(s).
\end{align*}
$$

Thus the mapping from inputs to outputs in frequency domain can be expressed as

$$
H(s) = C {\left(s I - A\right)}^{-1} B.
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$
H \text{ is analytic in } \mathbb{C} \setminus \Lambda(A).
$$

+++ {"slideshow": {"slide_type": "fragment"}}

### Pole-residue Form (Partial Fraction Expansion)

Let $(\lambda_{i}, w_{i}, v_{i})$ be the eigentriplets of $A$
with no degenerate eigenspaces.
Then the ***poles*** of $H$ are (some) of the eigenvalues
$\lambda_1, \ldots, \lambda_n$ and we have

$$
H(s) = \sum_{i = 1}^{n} \frac{R_{i}}{s - \lambda_{i}},
$$

where $R_{i} = (C v_{i}) (w_{i}^{\operatorname{H}} B)$,
assuming $w_{i}^{\operatorname{H}} v_{i} = 1$.

+++ {"slideshow": {"slide_type": "subslide"}}

### Example

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
fom.transfer_function
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
fom.transfer_function.eval_tf(0)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
fom.transfer_function.eval_tf(10j)
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Frequency-Domain Analysis

#### Bode Plots

The Bode plot for $H$ consists of a ***magnitude plot*** and a ***phase plot***.

> ##### Bode magnitude plot
>
> - component-wise graph of the function $\lvert H(\boldsymbol{\imath} \omega) \rvert$
>   for frequencies $\omega \in [\omega_{\min}, \omega_{\max}] \subset \mathbb{R}$.
> - $\omega$-axis is logarithmic.
> - magnitude is given in decibels, i.e., $\lvert H(\boldsymbol{\imath} \cdot) \rvert$ is
>   plotted as $20 \log_{10}(\lvert H(\boldsymbol{\imath} \cdot) \rvert)$.

> ##### Bode phase plot
>
> - component-wise graph of the function $\arg{H(\boldsymbol{\imath} \omega)}$
>   for frequencies $\omega \in [\omega_{\min}, \omega_{\max}] \subset \mathbb{R}$.
> - $\omega$-axis is logarithmic.
> - phase is given in degrees on a linear scale.

+++ {"slideshow": {"slide_type": "subslide"}}

#### Bode Plot for the Thermal Block Example

<center>
<img src="/files/sysmor-intro/figures/cookie_bode.svg" alt="cookie bode" width="60%">
</center>

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
# w = (1e-1, 1e5)
w, _ = fom.transfer_function.freq_resp((1e-1, 1e5))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
_ = fom.transfer_function.bode_plot(w)
```

+++ {"slideshow": {"slide_type": "subslide"}}

> #### (Sigma) Magnitude Plots
>
> - 2-norm-wise graph of the function $H(\boldsymbol{\imath} \omega)$
>   for frequencies $\omega \in [\omega_{\min}, \omega_{\max}] \subset \mathbb{R}$.
> - $\omega$-axis is logarithmic.

The name is due to the fact that for a given matrix $M$ the norm
$\lVert M \rVert_2$ is given by its largest singular value.

The real sigma magnitude plot depicts all singular values as functions of
$\omega$.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
_ = fom.transfer_function.mag_plot(w)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Projection-based MOR

### (Petrov-)Galerkin Projection

The LTI system can be written as

$$
\begin{align*}
  E \dot{x}(t) - A x(t) - B u(t) & = 0, \\
  y(t) - C x(t) - D u(t) & = 0,
\end{align*}
$$

or

$$
\begin{align*}
  E \dot{x}(t) - A x(t) - B u(t) & \perp \mathbb{R}^n, \quad x(t) \in \mathbb{R}^n, \\
  y(t) - C x(t) - D u(t) & = 0.
\end{align*}
$$

For $\mathcal{V}, \mathcal{W} \subseteq \mathbb{R}^n$ such that $\dim(\mathcal{V}) = \dim(\mathcal{W}) = r$, define

$$
\begin{align*}
  E \dot{\tilde{x}}(t) - A \tilde{x}(t) - B u(t) & \perp \mathcal{W}, \quad \tilde{x}(t) \in \mathcal{V}, \\
  \hat{y}(t) - C \tilde{x}(t) - D u(t) & = 0.
\end{align*}
$$

Let $V, W \in \mathbb{R}^{n \times r}$ be such that $\mathcal{V} = \operatorname{range}(V)$ and $\mathcal{W} = \operatorname{range}(W)$.
Then $\tilde{x}(t) = V \hat{x}(t)$ for some $\hat{x}(t) \in \mathbb{R}^r$ and

$$
\begin{align*}
  W^{\operatorname{T}} \left(E V \dot{\hat{x}}(t) - A V \hat{x}(t) - B u(t)\right) & = 0, \\
  \hat{y}(t) - C V \hat{x}(t) - D u(t) & = 0.
\end{align*}
$$

Rewriting gives

$$
\begin{align*}
  W^{\operatorname{T}} E V \dot{\hat{x}}(t) & = W^{\operatorname{T}} A V \hat{x}(t) + W^{\operatorname{T}} B u(t), \\
  \hat{y}(t) & = C V \hat{x}(t) + D u(t).
\end{align*}
$$

+++

- Galerkin projection (one-sided projection):

$$
\begin{align*}
  V^{\operatorname{T}} E V \dot{\hat{x}}(t) & = V^{\operatorname{T}} A V \hat{x}(t) + V^{\operatorname{T}} B u(t), \\
  \hat{y}(t) & = C V \hat{x}(t) + D u(t).
\end{align*}
$$

- Petrov-Galerkin projection (two-sided projection):

$$
\begin{align*}
  W^{\operatorname{T}} E V \dot{\hat{x}}(t) & = W^{\operatorname{T}} A V \hat{x}(t) + W^{\operatorname{T}} B u(t), \\
  \hat{y}(t) & = C V \hat{x}(t) + D u(t).
\end{align*}
$$

+++ {"slideshow": {"slide_type": "subslide"}}

<center>
<img src="/files/sysmor-intro/figures/compress_A.svg" alt="compress A" width="80%">
</center>

+++ {"slideshow": {"slide_type": "subslide"}}

### Reduced order model (ROM) (see [`LTIPGReductor`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/basic/index.html?highlight=ltipgred#pymor.reductors.basic.LTIPGReductor))

Define
$\hat{E} = W^{\operatorname{T}} E V$,
$\hat{A} = W^{\operatorname{T}} A V \in \mathbb{R}^{r \times r}$,
$\hat{B} = W^{\operatorname{T}} B \in \mathbb{R}^{r \times m}$, and
$\hat{C} = C V \in \mathbb{R}^{p \times r}$.
Then

$$
\begin{equation}\tag{ROM}
  \begin{aligned}
    \hat{E} \dot{\hat{x}}(t) & = \hat{A} \hat{x}(t) + \hat{B} u(t), \\
    \hat{y}(t) & = \hat{C} \hat{x}(t) + D u(t)
  \end{aligned}
\end{equation}
$$

approximates the dynamics of the full-order model $\Sigma$ with output error

$$
y(t) - \hat{y}(t) = e_{\text{output}}(t).
$$

- We call the corresponding transfer function $\hat{H}$.
- Model order reduction (MOR) $\leadsto$
  Find $W, V \in \mathbb{R}^{n \times r}$ such that $e_{\text{output}}(t)$ is
  small in a suitable sense.
- We will focus on eigenvalue-based, energy-based and
  interpolation-based methods today.

+++ {"slideshow": {"slide_type": "subslide"}}

### Example

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
from pymor.reductors.basic import LTIPGReductor

V = fom.solution_space.random(10)
pg = LTIPGReductor(fom, V, V)
rom_pg = pg.reduce()
```

+++ {"slideshow": {"slide_type": "subslide"}}

The resulting model `rom_pg` will again be an
[`LTIModel`](https://docs.pymor.org/2026-1-0/autoapi/pymor/models/iosys/index.html?highlight=ltimodel#pymor.models.iosys.LTIModel).

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
rom_pg
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom_pg.transfer_function.mag_plot(w, label='Random PG')
_ = plt.legend()
```

+++ {"slideshow": {"slide_type": "slide"}}

# System Analysis

## System Norms and Hardy Spaces

We have $$Y(s) = H(s) U(s)$$ and $$\hat{Y}(s) = \hat{H}(s) U(s).$$

> ### Question
>
> What are suitable norms such that
>
> $$
  \lVert y - \hat{y} \rVert
  \le
  \left\lVert H - \hat{H} \right\rVert
  \lVert u \rVert?
  $$

+++ {"slideshow": {"slide_type": "subslide"}}

### The Banach Space $\mathcal{H}_\infty^{p \times m}$

$$
\mathcal{H}_\infty^{p \times m}
:=
\left\{
  G \colon \mathbb{C}^+ \to \mathbb{C}^{p \times m} :
  G \text{ is analytic in $\mathbb{C}^+$ and }
  \sup_{s \in \mathbb{C}^+} \left\lVert G(s) \right\rVert_2 < \infty
\right\}.
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$\mathcal{H}_\infty^{p \times m}$ is a Banach space equipped with the
***$\mathcal{H}_\infty$-norm***

$$
\left\lVert G \right\rVert_{\mathcal{H}_\infty}
:= \sup_{\omega \in \mathbb{R}}
\left\lVert G(\boldsymbol{\imath} \omega) \right\rVert_2.
$$

+++ {"slideshow": {"slide_type": "fragment"}}

> Can show:
>
> $$
  \lVert y - \hat{y} \rVert_{\mathcal{L}_{2}}
  \le
  \left\lVert H - \hat{H} \right\rVert_{\mathcal{H}_{\infty}}
  \lVert u \rVert_{\mathcal{L}_{2}}.
  $$

This bound can even be shown to be sharp.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
fom.hinf_norm()
```

+++ {"slideshow": {"slide_type": "subslide"}}

### The Hilbert Space $\mathcal{H}_2^{p \times m}$

$$
\mathcal{H}_2^{p \times m}
:= \left\{
  G \colon \mathbb{C}^+ \to \mathbb{C}^{p \times m} :
  G \text{ is analytic in $\mathbb{C}^+$ and }
  \sup_{\xi > 0}
  \int_{-\infty}^\infty
  \left\lVert
  G(\xi + \boldsymbol{\imath} \omega)
  \right\rVert_{\operatorname{F}}^2
  \operatorname{d}\!{\omega}
  < \infty
\right\}.
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$\mathcal{H}_2^{p \times m}$ is a Hilbert space with the inner product

$$
\langle F, G \rangle_{\mathcal{H}_2}
:=
\frac{1}{2 \pi}
\int_{-\infty}^\infty
\operatorname{tr}\!\left(
  {F(\boldsymbol{\imath} \omega)}^{\operatorname{H}}
  G(\boldsymbol{\imath} \omega)
\right)
\operatorname{d}\!{\omega}
$$

and induced norm

$$
\left\lVert G \right\rVert_{\mathcal{H}_2}
:= \langle G, G \rangle_{\mathcal{H}_2}^{1/2}
= {
  \left(
    \frac{1}{2 \pi}
    \int_{-\infty}^\infty
    \left\lVert G(\boldsymbol{\imath} \omega) \right\rVert_{\operatorname{F}}^2
    \operatorname{d}\!{\omega}
  \right)
}^{1/2}.
$$

+++ {"slideshow": {"slide_type": "fragment"}}

> Can show:
>
> $$
  \lVert y - \hat{y} \rVert_{\mathcal{L}_{\infty}}
  \le
  \left\lVert H - \hat{H} \right\rVert_{\mathcal{H}_{2}}
  \lVert u \rVert_{\mathcal{L}_{2}}.
  $$

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
fom.h2_norm()
```

+++ {"slideshow": {"slide_type": "subslide"}}

### System Gramians and $\mathcal{H}_{2}$ trace formula

For ease of notation, consider $E=I$. A system $\Sigma$ with $\Lambda(A) \subset \mathbb{C}^{-}$ is called
***asymptotically stable***.
Then, all state trajectories decay exponentially as $t \to \infty$ and

- the infinite controllability and observability ***Gramians*** exist:

  $$
  \begin{align*}
    P
    & =
      \int_0^{\infty}
      e^{t A}
      B B^{\operatorname{T}}
      e^{t A^{\operatorname{T}}}
      \operatorname{d}\!{t} \\
    Q
    & =
      \int_0^{\infty}
      e^{t A^{\operatorname{T}}}
      C^{\operatorname{T}} C
      e^{t A}
      \operatorname{d}\!{t}.
  \end{align*}
  $$
- $P$, $Q$ solve the two ***Lyapunov equations***

  $$
  A P + P A^{\operatorname{T}} + B B^{\operatorname{T}} = 0, \qquad
  A^{\operatorname{T}} Q + Q A + C^{\operatorname{T}} C = 0
  $$
<!-- - If $(A, B)$ is controllable and $(A, C)$ is observable, -->
<!--   it moreover holds that $P = P^{\operatorname{T}} \succ 0$ and $Q = Q^{\operatorname{T}} \succ 0$. -->
<!--   (Otherwise we just have $P = P^{\operatorname{T}} \succcurlyeq 0$ and -->
<!--   $Q = Q^{\operatorname{T}} \succcurlyeq 0$.) -->
- the $\mathcal{H}_{2}$-norm can be expressed as

  $$
  \lVert H \rVert_{\mathcal{H}_{2}}^{2}
  = \operatorname{tr}\!\left(C P C^{\operatorname{T}}\right)
  = \operatorname{tr}\!\left(B^{\operatorname{T}} Q B\right).
  $$

+++ {"slideshow": {"slide_type": "subslide"}}

## Transfer Function Approximation

The transfer function $H$ is a degree-$n$ rational function

$$
H(s) = \frac{\mathcal{P}(s)}{\mathcal{Q}(s)}, \quad \mathcal{P},\mathcal{Q} \text{ polynomials}
$$

such that deg$(\mathcal{P}) \leq n$ and deg$(\mathcal{Q}) \leq n$.

+++ {"slideshow": {"slide_type": "fragment"}}

MOR via rational approximation:
Find a degree-$r$ rational function $\hat{H}$ with $r \ll n$ such that

$$
H \approx \hat{H}
$$

+++ {"slideshow": {"slide_type": "subslide"}}

### Rational Interpolation

Pick some complex values $\sigma_1, \ldots, \sigma_r$ and enforce interpolation

$$
\hat{H}(\sigma_j) = H(\sigma_j) \quad \text{for } j = 1, \ldots, r.
$$

+++ {"slideshow": {"slide_type": "subslide"}}

Can also pick tangential directions $b_1, \ldots, b_r \in \mathbb{C}^m$ and
$c_1, \ldots, c_r \in \mathbb{C}^p$ and enforce bitangential Hermite
interpolation

$$
\begin{align*}
  \hat{H}(\sigma_j)b_j &= H(\sigma_j) b_j, \\
  c_j^* \hat{H}(\sigma_j) &= c_j^* H(\sigma_j), \\
  c_j^* \hat{H}'(\sigma_j) b_j &= c_j^* H'(\sigma_j) b_j, \\
\end{align*}
\quad \text{for } j=1,\ldots,r.
$$

+++ {"slideshow": {"slide_type": "subslide"}}

### Interpolation via Projection

Given $E,A,B,C$, how to enforce interpolation?

> ### Theorem
>
> Let $\hat{H}$ be the transfer function of the ROM obtained from a Petrov
> Galerkin projection using $V$ and $W$.
> For $b \in \mathbb{C}^m$, $c \in \mathbb{C}^p$ and $\sigma \in \mathbb{C}$ we
> have
>
> 1. $(\sigma E - A)^{-1} B b \in \mathrm{range}(V)$ implies $H(\sigma) b = \hat{H}(\sigma) b$
> 2. $(\sigma E - A)^{-*} C^* c \in \mathrm{range}(W)$ implies $c^* H(\sigma) = c^* \hat{H}(\sigma)$
> 3. If 1. and 2. are satisfied, then $c^* H'(\sigma) b = c^* \hat{H}'(\sigma) b$

Using bases $V$ and $W$ for rational Krylov subspaces allows for interpolatory
MOR [[Antoulas/Beattie/Güğercin '20]](https://doi.org/10.1137/1.9781611976083).

In pyMOR
[`LTIBHIReductor`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/interpolation/index.html?highlight=ltibhired#pymor.reductors.interpolation.LTIBHIReductor)
is based on projection and
[`TFBHIReductor`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/interpolation/index.html?highlight=tfbhi#pymor.reductors.interpolation.TFBHIReductor)
only uses evaluations of the transfer function $H$.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
from pymor.reductors.interpolation import LTIBHIReductor

interp = LTIBHIReductor(fom)
sigma = np.array([50, 100, 200, 400, 800])
sigma = np.concatenate((1j * sigma, -1j * sigma))
b = np.ones((len(sigma), fom.dim_input))
c = np.ones((len(sigma), fom.dim_output))
rom_interp = interp.reduce(sigma, b, c)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom_interp.transfer_function.mag_plot(w, label='Interpolation')
_ = plt.legend()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
err_interp = fom - rom_interp
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
_ = err_interp.transfer_function.mag_plot(w, label='Interpolation')
_ = plt.legend()
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Iterative Rational Krylov Algorithm (IRKA)

> #### $\mathcal{H}_2$-optimal MOR problem
>
> Find a stable $\hat{H}$ of order $r$ such that
> $\lVert H - \hat{H} \rVert_{\mathcal{H}_{2}}$
> is minimized.

+++ {"slideshow": {"slide_type": "fragment"}}

> #### Interpolatory necessary $\mathcal{H}_2$-optimality conditions
>
> Let $\hat{H}(s) = \sum_{i = 1}^r \frac{\phi_i}{s - \lambda_i}$ be an
> $\mathcal{H}_2$-optimal reduced-order model for $H$.
> Then
> \begin{align*}
    H\!\left(-\overline{\lambda_i}\right)
    & = \hat{H}\!\left(-\overline{\lambda_i}\right), \\
    H'\!\left(-\overline{\lambda_i}\right)
    & = \hat{H}'\!\left(-\overline{\lambda_i}\right),
  \end{align*}
> for $i = 1, 2, \ldots, r$.

+++ {"slideshow": {"slide_type": "subslide"}}

> ***Hermite interpolation*** of the transfer function is necessary for
> $\mathcal{H}_2$-optimality.

+++ {"slideshow": {"slide_type": "fragment"}}

> #### IRKA (see [`IRKAReductor`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/h2/index.html?highlight=irkared#pymor.reductors.h2.IRKAReductor))
>
> Fixed point iteration based on interpolatory necessary optimality conditions.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
from pymor.reductors.h2 import IRKAReductor

irka = IRKAReductor(fom)
rom_irka = irka.reduce(10)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom_interp.transfer_function.mag_plot(w, label='Interpolation')
_ = rom_irka.transfer_function.mag_plot(w, label='IRKA')
_ = plt.legend()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
err_irka = fom - rom_irka
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
_ = err_interp.transfer_function.mag_plot(w, label='Interpolation')
_ = err_irka.transfer_function.mag_plot(w, label='IRKA')
_ = plt.legend()
```

+++ {"slideshow": {"slide_type": "subslide"}}

<center>Questions?</center>

+++ {"slideshow": {"slide_type": "slide"}}

### Exercises for the hands-on session

+++ {"slideshow": {"slide_type": "subslide"}}

#### Exercise transfer function

- Use the [`poles`](https://docs.pymor.org/2026-1-0/autoapi/pymor/models/iosys/index.html#pymor.models.iosys.LTIModel.poles) method of the [`LTIModel`](https://docs.pymor.org/2026-1-0/autoapi/pymor/models/iosys/index.html#pymor.models.iosys.LTIModel) class to compute the poles of the transfer function of `fom`.
  Compute the imaginary parts of the poles.
- Evaluate the transfer function for several values on the imaginary axis.
  Select some values that correspond to the imaginary parts of the poles and
  others which are close by.

```{code-cell} ipython3
---
slideshow:
  slide_type: ''
---

```

+++ {"slideshow": {"slide_type": "subslide"}}

#### Exercise Timestepper

- Use the time stepper specified above and simulate the model using the input
  function $u(t) = e^{-t}$.
- Change the number of timesteps `nt` to `50000`. Repeat the simulation of the
  model using $u(t) = e^{-t}$.

```{code-cell} ipython3
---
slideshow:
  slide_type: ''
---

```

+++ {"slideshow": {"slide_type": "subslide"}}

#### Exercise PGReductor

- Simulate the state of the `rom_pg` with $u(t) = e^{-t}$ from $t_{start}=0$ to
  $t_{end}=2$ using its
  [`solve`](https://docs.pymor.org/2026-1-0/autoapi/pymor/models/interface/index.html#pymor.models.interface.Model.solve)
method.
- The solve method computes a
  [`VectorArray`](https://docs.pymor.org/2026-1-0/autoapi/pymor/vectorarrays/interface/index.html?highlight=vectorarray#pymor.vectorarrays.interface.VectorArray)
  with all computed state vectors from the time-domain simulation.
  Consider the last state (i.e., at time $t_{end}$) from the previous
  computation.
  Use the
  [`reconstruct`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/basic/index.html?highlight=reconstruct#pymor.reductors.basic.LTIPGReductor.reconstruct)
  method of `pg` to obtain the reconstructed state vector in the full-order
  model state space.
- Perform the same simulation with `fom`.
  Compare the final state with the state that has been reconstructed using the
  ROM simulation.

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---

```

#### Exercise IRKA

- Compute 5 random values `sigma` in the interval $[0,1000]$ using
  `np.random.rand`.
  Then use the
  [`TFBHIReductor`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/interpolation/index.html?highlight=tfbhi#pymor.reductors.interpolation.TFBHIReductor)
  to compute a ROM which interpolates `fom.transfer_function` at the random
  values and its complex conjugates along the imaginary axis (i.e., use
  interpolation points `np.concatenate((1j * sigma, -1j * sigma))`).
  Compute the relative $\mathcal{H}_{2}$-error of the resulting ROM.
  Compare it to the relative $\mathcal{H}_{2}$-error of the interpolation points
  chosen in the example above where we chose
  `sigma = np.array([50, 100, 200, 400, 800])`.
- Repeat the previous computations with 10 random values
  (i.e., a total of 20 interpolation points).
- Use the `sigma` specified below as an initial guess for the
  [`reduce`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/h2/index.html?highlight=irkared#pymor.reductors.h2.IRKAReductor.reduce)
  method of the
  [`IRKAReductor`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/h2/index.html?highlight=irkared#pymor.reductors.h2.IRKAReductor):

  ```
  sigma = np.array([50, 100, 200, 400, 800]);
  sigma = np.concatenate((1j * sigma, -1j * sigma))
  ```

```{code-cell} ipython3

```
