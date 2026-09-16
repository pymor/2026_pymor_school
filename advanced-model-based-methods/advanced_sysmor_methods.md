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

# Advanced System-theoretic Methods

<h2>
September 16, 2026<br/>
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

+++ {"slideshow": {"slide_type": "subslide"}}

### Penzl Example ([MORWiki](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Penzl%27s_FOM))

```{code-cell} ipython3
---
slideshow:
  slide_type: ''
---
import numpy as np
import scipy.sparse as sps
from pymor.models.iosys import LTIModel

A1 = np.array([[-1, 100], [-100, -1]])
A2 = np.array([[-1, 200], [-200, -1]])
A3 = np.array([[-1, 400], [-400, -1]])
A4 = sps.diags(np.arange(-1., -1001., -1.))
A = sps.block_diag((A1, A2, A3, A4), format='csc')
B = np.ones((1006, 1))
B[:6] = 10
C = B.T

fom = LTIModel.from_matrices(A, B, C)
```

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

+++ {"slideshow": {"slide_type": "slide"}}

# A Selection of MOR Methods

+++ {"slideshow": {"slide_type": "subslide"}}

## Modal Methods

```{code-cell} ipython3
import matplotlib.pyplot as plt

poles = fom.poles()
fig, ax = plt.subplots()
ax.plot(poles.real, poles.imag, 'x')
ax.set(
    xlabel='Real',
    ylabel='Imag',
    title='Poles of the FOM',
)
ax.grid()
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Dominant Poles Approximation (see [`MTReductor`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/mt/index.html?highlight=mtreductor#pymor.reductors.mt.MTReductor))

Recall the pole-residue form of the transfer function

$$
H(s) = \sum_{i = 1}^{n} \frac{R_{i}}{s - \lambda_{i}},
$$

where $R_{i} = (C v_{i})(w_{i}^{\operatorname{H}} B)$, assuming
$w_{i}^{\operatorname{H}} v_{i} = 1$
(assuming $A$ is diagonalizable).

+++ {"slideshow": {"slide_type": "fragment"}}

Suppose the modes are sorted based on the magnitude of the
$\lVert R_{i} \rVert / \operatorname{Re}(\lambda_{i})$.
Then we use the truncated pole residue form

$$
H(s) = \sum_{i = 1}^{r} \frac{R_{i}}{s - \lambda_{i}},
$$

as our ROM. This is motivated by the following

> #### Error bound
>
> $$
  \left\lVert H - \hat{H} \right\rVert_{\mathcal{H}_\infty}
  \le
  \sum_{i = r + 1}^{n}
  \frac{\lVert R_{i} \rVert_2}{\lvert \operatorname{Re}(\lambda_{i}) \rvert}
  $$

+++ {"slideshow": {"slide_type": "fragment"}}

Computation is feasible via *subspace accelerated MIMO dominant pole algorithm*
(SAMDP).

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
from pymor.reductors.mt import MTReductor

mt = MTReductor(fom)
rom_mt = mt.reduce(10)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
w, _ = fom.transfer_function.freq_resp((1e-1, 1e5))

_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom_mt.transfer_function.mag_plot(w, label='MT')
_ = plt.legend()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
err_mt = fom - rom_mt
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
_ = err_mt.transfer_function.mag_plot(w, label='MT')
_ = plt.legend()
```

+++ {"slideshow": {"slide_type": "subslide"}}

## Balancing-based MOR

### Balanced Truncation aka. Lyapunov Balancing (see [`BTReductor`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.BTReductor))

#### Idea

- The system $\Sigma$, in realization $(E = I, A, B, C)$,
  is called ***balanced***, if the solutions $P, Q$ of the Lyapunov equations

  $$
  A P + P A^{\operatorname{T}} + B B^{\operatorname{T}} = 0, \qquad
  A^{\operatorname{T}} Q + Q A + C^{\operatorname{T}} C = 0,
  $$

  satisfy:
  $P = Q = \operatorname{diag}(\sigma_1, \ldots, \sigma_n)$
  where
  $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_n > 0$.

+++ {"slideshow": {"slide_type": "fragment"}}

- $\{\sigma_1, \ldots, \sigma_n\}$ are the *Hankel singular values (HSVs)* of
  $\Sigma$.

+++ {"slideshow": {"slide_type": "fragment"}}

- A so-called ***balanced realization*** is computed via state-space transformation

  $$
  \begin{align*}
    \mathcal{T} \colon (I, A, B, C) \mapsto {} & (I, T A T^{-1}, T B, C T^{-1}) \\
    & =
      \left(
        I,
        \begin{bmatrix}
          A_{11} & A_{12} \\
          A_{21} & A_{22}
        \end{bmatrix},
        \begin{bmatrix}
          B_{1} \\
          B_{2}
        \end{bmatrix},
        \begin{bmatrix}
          C_{1} & C_{2}
        \end{bmatrix}
      \right).
  \end{align*}
  $$

+++

- In a balanced realization the state variables are sorted based on their
  contribution to the input-output mapping.

+++ {"slideshow": {"slide_type": "fragment"}}

- Truncation removes state variables which are not important for input-output
  behavior $\leadsto$ reduced order model:
  $(I, \hat{A}, \hat{B}, \hat{C}) = (I, A_{11}, B_{1}, C_{1})$.

+++ {"slideshow": {"slide_type": "subslide"}}

### Implementation: The Square Root Method

#### The SR Method

1. Compute (Cholesky) factors of the solutions to the Lyapunov equation,

   $$
   P = S^{\operatorname{T}} S, \quad
   Q = R^{\operatorname{T}} R.
   $$

+++ {"slideshow": {"slide_type": "fragment"}}

2. Compute singular value decomposition

   $$
   R S^{\operatorname{T}}
   =
   \begin{bmatrix}
     U_1 & U_2
   \end{bmatrix}
   \begin{bmatrix}
     \Sigma_1 & 0 \\
     0 & \Sigma_2
   \end{bmatrix}
   \begin{bmatrix}
     V_1^{\operatorname{T}} \\
     V_2^{\operatorname{T}}
   \end{bmatrix}.
   $$

+++ {"slideshow": {"slide_type": "fragment"}}

3. Define

   $$
   W := R^{\operatorname{T}} U_1 \Sigma_1^{-1/2}, \quad
   V := S^{\operatorname{T}} V_1 \Sigma_1^{-1/2}.
   $$
4. Then the reduced-order model is
   $(W^{\operatorname{T}} A V, W^{\operatorname{T}} B, C V)$.

+++ {"slideshow": {"slide_type": "subslide"}}

#### Properties

- Lyapunov balancing **preserves asymptotic stability**.
- We have the **a priori error bound**:
  $$
  \left\lVert H - \hat{H} \right\rVert_{\mathcal{H}_{\infty}}
  \le
  2 \sum\limits_{k = r + 1}^{n} \sigma_{k}
  $$

+++ {"slideshow": {"slide_type": "subslide"}}

#### Variants

Other versions for special classes of systems or applications exist, such as

- **positive-real balancing** (passivity-preserving, see [`PRBTReductor`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.PRBTReductor)),
- **bounded-real balancing** (contractivity-preserving, see [`BRBTReductor`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.BRBTReductor)),
- **linear-quadratic Gaussian balancing**
  (stability preserving, aims at low-order output feedback controllers, see [`LQGBTReductor`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.LQGBTReductor)).

The given ones all compute $P, Q$ as solutions of ***algebraic Riccati
equations*** of the form:

$$
\begin{align*}
  0
  & =
    \tilde{A} P \tilde{E}^{\operatorname{T}}
    + \tilde{E} P \tilde{A}^{\operatorname{T}}
    + \tilde{B} \tilde{B}^{\operatorname{T}}
    \pm \tilde{E} P \tilde{C}^{\operatorname{T}} \tilde{C} P \tilde{E}^{\operatorname{T}}, \\
  0
  & =
    \tilde{A}^{\operatorname{T}} Q \tilde{E}
    + \tilde{E}^{\operatorname{T}} Q \tilde{A}
    + \tilde{C}^{\operatorname{T}} \tilde{C}
    \pm \tilde{E}^{\operatorname{T}} Q \tilde{B} \tilde{B}^{\operatorname{T}} Q \tilde{E}.
\end{align*}
$$

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
from pymor.reductors.bt import BTReductor

bt = BTReductor(fom)
rom_bt = bt.reduce(10)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom_mt.transfer_function.mag_plot(w, label='MT')
_ = rom_bt.transfer_function.mag_plot(w, label='BT')
_ = plt.legend()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
err_bt = fom - rom_bt
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
_ = err_mt.transfer_function.mag_plot(w, label='MT')
_ = err_bt.transfer_function.mag_plot(w, label='BT')
_ = plt.legend()
```

```{code-cell} ipython3
poles_mt = rom_mt.poles()
poles_bt = rom_bt.poles()
fig, ax = plt.subplots()
ax.plot(poles_mt.real, poles_mt.imag, '.')
ax.plot(poles_bt.real, poles_bt.imag, 'x')
ax.set(
    xlabel='Real',
    ylabel='Imag',
    title='Poles of the ROMs',
)
ax.set_xscale('symlog', linthresh=1e-1)
ax.grid()
```

+++ {"slideshow": {"slide_type": "slide"}}

## Relations to RB/POD

- LTI system $\Leftrightarrow$ parametric stationary problem:

$$
\begin{align*}
  (s E - A) X(s) & = B U(s), \\
  Y(s) & = C X(s).
\end{align*}
$$

- RB $\approx$ $L_\infty$ error minimization via greedy interpolation
- POD $\approx$ controllability Gramian projection

$$
P = \int_0^\infty x_{\text{imp}}(t) x_{\text{imp}}(t)^{\operatorname{T}} \operatorname{d}\!{t}, \\
x_{\text{imp}}(t) = e^{t A} B, \quad
\dot{x}_{\text{imp}}(t) = A x_{\text{imp}}(t),\ x_{\text{imp}}(0) = B \\
(\dot{x}_{\text{imp}}(t) = A x_{\text{imp}}(t) + B \delta(t)).
$$

$$
P = \frac{1}{2 \pi} \int_{-\infty}^{\infty} (i \omega E - A)^{-1} B B^{\operatorname{T}} (i \omega E - A)^{\operatorname{H}} \operatorname{d}\!{\omega}
$$

$$
P \approx V V^{\operatorname{T}}
$$

+++ {"slideshow": {"slide_type": "subslide"}}

<center>Questions?</center>

+++ {"slideshow": {"slide_type": "slide"}}

## Exercises for the hands-on session

+++ {"slideshow": {"slide_type": "subslide"}}

### Modal MOR

- Compute the $\mathcal{H}_2$-norm of the error system `err_mt`.
  Note that the
  [`LTIModel`](https://docs.pymor.org/2026-1-0/autoapi/pymor/models/iosys/index.html?highlight=ltimodel#pymor.models.iosys.LTIModel)
  class has an
  [`h2_norm`](iosys/index.html?highlight=ltimodel#pymor.models.iosys.LTIModel.h2_norm)
  method.
- Consider the input $u(t) = e^{-t}$ which has the $\mathcal{L}_{2}$-norm
  $\lVert u \rVert_{\mathcal{L}_{2}} = \frac{\sqrt{2}}{2}$.
  Simulate the error system `err_mt` with the input $u(t) = e^{-t}$ and verify
  the input-output error bound $\lVert y - \hat{y} \rVert_{\mathcal{L}_{\infty}}
  \le
  \left\lVert H - \hat{H} \right\rVert_{\mathcal{H}_{2}}
  \lVert u \rVert_{\mathcal{L}_{2}}$.

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---

```

+++ {"slideshow": {"slide_type": "subslide"}}

### Balancing based MOR
- Aside from a desired order, the
  [`reduce`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.GenericBTReductor.reduce)
  method of the
  [`BTReductor`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.BTReductor)
  allows for specifying a truncation tolerance based on the a priori error
  bound.
  Use the `bt` instance of the
  [`BTReductor`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.BTReductor)
  to compute a ROM based on a specified tolerance `tol=1e-5`.
- Use the
  [`LQGBTReductor`](https://docs.pymor.org/2026-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.LQGBTReductor)
  to reduce `fom` using a truncation tolerance of `tol=1e-5`.
  Check the dimension of the ROM.
- Compare the $\mathcal{H}_{2}$-norms and orders of the ROMs obtained by both BT
  variants.

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---

```
