#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# enable logging widget
get_ipython().run_line_magic('load_ext', 'pymor.tools.jupyter')


# ## Instationary problems and POD-greedy
# 
# Solve time-dependent problem of the form:
# 
# $$
#     (\partial_t u_h(\mu),v_h)_{L^2} + a(u_h(\mu), v_h;\mu) = \ell(v_h)
#     \qquad \forall v_h \in V_h,
# $$
# 
# where $u_h(\mu)\colon[0,T]\to V_h$ is now a function of time.
# 
# Implicit Euler with step size $\Delta t>0$:
# 
# $$
#     (m + \Delta t\cdot a)(u_{h,n+1}(\mu),v_h) = \Delta t\cdot l(v_h) + m(u_{h,n},v_h) \qquad \forall v_h \in V_h,
# $$
# 
# with mass operator
# 
# $$
# m(u,v)=(u,v)_{L^2}.
# $$
# 
# * One elliptic problem per time step.
# * Apply projection-based MOR as before.

# ### How to compute a reduced basis?
# 
# * One solution entire time trajectory!
# * Cannot add entire trajectory to RB.
# * Adding $u_h(t^*; \mu^*)$ where
#   $$(t^*, \mu^*) := \operatorname*{arg\,max}_{(t, \mu)} \|u_h(t, \mu) - u_N(t, \mu)\|,$$
#   won't work. Same $(t^*, \mu^*)$ can be selected twice.

# ### POD-greedy ###
# * Find $\mu^*$ such that:
#   $$ \mu^* := \operatorname*{arg\,max}_{\mu} \|u_h(\mu) - u_N(\mu)\|_{\text{space-time}},$$
# * Compute the defect for the projection onto RB:
#   $$ u_h^\perp(t; \mu) := u_h(t; \mu) - P_{V_N}(u_h(t; \mu))$$
# * Compress using POD:
#   $$ W := POD([u_h^\perp(t_1; \mu), \ldots, u_h^\perp(t_{n_t}; \mu)], \varepsilon_{\text{POD}}, N_{\text{modes}})$$
# * Extend reduced space:
#   $$ V_N \leftarrow \operatorname{span}(V_n \cup W). $$

# ### Properties of POD-Greedy
# * Cannot get stuck due to orthogonal projection.
# * No exact reproduction of snapshots.
# * Same $\mu^*$ can still selected multiple times $\rightarrow$ Cache snapshot trajectories.
# * Similar quasi-optimality result as for elliptic weak

# ### Example: parametric heat equation

# On spatial domain $\Omega=[0,1]^2$, consider:
# $$
#     \partial_t u(\mu) - \nabla (d(\mu)\nabla u(\mu)) = f(t), \qquad t \in [0,1],
# $$
# with diffusivity
# $$
#     d(\mu) = 1 + \underbrace{99\cdot\mathbf{1}_{(0.45,0.55)\times(0,0.7)}}_{\text{high-conductivity channel}} + (\mu - 1)\cdot(\underbrace{\mathbf{1}_{(0.35,0.4)\times(0.3,1)}+\mathbf{1}_{(0.6,0.65)\times(0.3,1)}}_{\text{parametrized channels}}),
# $$
# for $\mu \in \mathcal{P}=[1,100]$, right-hand side
# $$
#     f(t) = 100\cdot\sin(10\pi t),
# $$
# and Neumann boundary condition
# $$
#     -\partial_n u(\mu, t, (x,y))=\begin{cases}-1000 & 0.45 < x < 0.55\\0 & \text{else}\end{cases}
# $$
# at the bottom of the domain ($\Gamma_{\text{bottom}}=[0,1]\times\{0\}$), and homogeneous Dirichlet boundary conditions everywhere else.
# 
# Initial condition:
# $$
#     u(\mu, 0) = 10\cdot \mathbf{1}_{(0.45,0.55)\times(0,0.7)}.
# $$

# ### FOM in pyMOR

# - We use pyMOR's builtin discretizer.
# - Also defined in `pymor/models/examples.py`.

# In[ ]:


from pymor.basic import *

# setup analytical problem
problem = InstationaryProblem(

    StationaryProblem(
        domain=RectDomain(top='dirichlet', bottom='neumann',
                          left='dirichlet', right='dirichlet'),

        diffusion=(ConstantFunction(1., dim_domain=2)
                   + ExpressionFunction('(0.45 < x[0] < 0.55) * (x[1] < 0.7) * 1.',
                                        dim_domain=2) * (100. - 1)
                   + ExpressionFunction('(0.35 < x[0] < 0.40) * (x[1] > 0.3) * 1. + '
                                        '(0.60 < x[0] < 0.65) * (x[1] > 0.3) * 1.',
                                        dim_domain=2)
                         * ExpressionParameterFunctional('top[0] - 1.', {'top': 1})
                  ),

        rhs=ConstantFunction(value=100., dim_domain=2) * ExpressionParameterFunctional('sin(10*pi*t[0])', {'t': 1}),

        dirichlet_data=ConstantFunction(value=0., dim_domain=2),

        neumann_data=ExpressionFunction('(0.45 < x[0] < 0.55) * -1000.', dim_domain=2),
    ),

    T=1.,

    initial_data=ExpressionFunction('(0.45 < x[0] < 0.55) * (x[1] < 0.7) * 10.', dim_domain=2)
)

# discretize using continuous finite elements
fom, data = discretize_instationary_cg(analytical_problem=problem, diameter=1/50, nt=50)

parameter_space = fom.parameters.space(1, 100)


# Diffusivity field for two different parameters:

# In[ ]:


from pymor.discretizers.builtin.cg import InterpolationOperator
diffusion_field_1 = InterpolationOperator(data['grid'], problem.stationary_part.diffusion).as_vector(fom.parameters.parse({'top': [2.]}))
diffusion_field_2 = InterpolationOperator(data['grid'], problem.stationary_part.diffusion).as_vector(fom.parameters.parse({'top': [100.]}))
fom.visualize((diffusion_field_1, diffusion_field_2))


# And the corresponding solutions:

# In[ ]:


fom.visualize((fom.solve(2.), fom.solve(100.)))


# ### POD-Greedy with pyMOR

# - Use `ParabolicRBReductor`.
#     - Reduces `InstationaryModels`.
#     - Assembles a posteriori error estimator.
#     - Expects lower bound for coercivity constant.
# - Enable caching of FOM solutions (in case a $\mu$ is selected multiple times).

# In[ ]:


fom.enable_caching('disk')

from pymor.reductors.parabolic import ParabolicRBReductor
coercivity_estimator = ExpressionParameterFunctional('1.', fom.parameters)
reductor = ParabolicRBReductor(fom, product=fom.h1_0_semi_product, coercivity_estimator=coercivity_estimator)


# - Apply `rb_greedy` as before:

# In[ ]:


training_set = parameter_space.sample_uniformly(50)
greedy_data = rb_greedy(fom, reductor, training_set, max_extensions=10)
rom = greedy_data["rom"]


# - `RBSurrogate.extend` automatically uses POD with $N_\text{modes} = 1$ for time-dependent problems.
# - Pass `extension_params={'pod_modes': k}` to `rb_greedy` to always add k POD modes.
# - Use Successive Constraint Method (`pymor.algorithms.scm`) to construct a `coercivity_estimator` when no analytical bound is known.

# ### Reduced solution and speedup

# In[ ]:


from time import perf_counter
mu = parameter_space.sample_randomly()

tic = perf_counter()
U = fom.solve(mu)
t_fom = perf_counter() - tic

tic = perf_counter()
u_RB = rom.solve(mu)
t_rom = perf_counter() - tic
U_RB = reductor.reconstruct(u_RB)


print(f"Speedup: {t_fom / t_rom}")
fom.visualize((U, U_RB, U - U_RB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
              separate_colorbars=True)


# ## Nonlinear Problems with POD-DEIM

# For stationary nonlinear problems, we need to solve nonlinear discrete equations of the form
# 
# $$ A(u_h(\mu); \mu) = l.$$
# 
# Galerkin projection leads to:
# 
# $$ V^{\operatorname{T}} \cdot A(V\cdot \underline{u}_N(\mu); \mu) = V^{\operatorname{T}} \cdot l.$$
# 
# However, cannot "pre-compute" 
# $$V^{\operatorname{T}} \cdot A(V \cdot (\ldots); \mu): \mathbb{R}^N \to \mathbb{R}^N.$$
# 
# **No offline-online decomposition!**

# ### (Discrete) Empirical Interpolation Method ((D)EIM)
# 
# - Only evaluate
#   $$A(\cdot; \mu)_{i_m} \quad\text{for $M$ interpolation DOFs} \quad i_1, \ldots i_M.$$
#   For FD, FV, FEM, et al., this only requires local low-dimensional computations!
# 
# - Approximate
#   $$A( u_h; \mu) \approx \mathcal{I}_M(A(u_h; \mu)) := \sum_{m=1}^M c_m \hat\psi_m, \qquad\text{s.t}\qquad \mathcal{I}_M(A(u_h; \mu))_{i_k} = A(u_h; \mu)_{i_k}, \quad k = 1, \ldots, M$$
#   for some "collateral" interpolation basis $\hat\psi_1, \ldots \hat\psi_M$.
# 
#   Thus, with
#   $$ \hat{V} := [\hat\psi_1, \ldots, \hat\psi_M]\qquad\text{and}\qquad B_{k,m}:=\Bigl(\hat\psi_m\Bigr)_{i_k},$$
#   we have
#   $$ \mathcal{I}_M(A(u_h; \mu)) = \hat{V} \cdot B^{-1} \cdot \Bigl[A(u_h; \mu)_{i_k}\Bigr]_{k=1}^M $$
# - Compute $i_1, \ldots, i_M$ and $\hat\psi_1, \ldots, \hat\psi_M$ offline from data (EI-Greedy).
# 
# - Solve
#   $$\underline{V}^{\operatorname{T}} \cdot \hat{V} \cdot B^{-1} \cdot \Bigl[A(\underline{u}_N(\mu); \mu)_{i_k}\Bigr]_{k=1}^M = \underline{V}^{\operatorname{T}} \cdot l$$
# 
# - Offline-online decomposition by pre-computing $\underline{V}^{\operatorname{T}} \cdot \hat{V}$, $B$, and storing rows of $V$ in "neighborhoods" of the interpolation DOFs $i_k$.

# ### Example: quasi-linear Poisson equation
# 
# Parameterized version of the FEniCS [nonlinear Poisson](https://fenics.readthedocs.io/projects/dolfin/en/2017.2.0/demos/nonlinear-poisson/python/demo_nonlinear-poisson.py.html) demo:
# 
# $$ 
# \begin{align} 
# -\nabla \cdot \left[(1 + c\cdot u(x,y;\mu)^2) \nabla u(x,y;\mu)\right] &= x\cdot \sin(y) & (x,y) &\in (0,1) \times (0,1), \\
# u(1, y) &= 1, \\
# \nabla u(0, y) \cdot n = \nabla u(x, 0) \cdot n = \nabla u(x, 1) \cdot n &= 0,
# \end{align}$$
# 
# where $c \in [0.01, 1000]$ is our parameter.

# In[ ]:


import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

mesh = mesh.create_unit_square(MPI.COMM_WORLD, 50, 50)

V = fem.functionspace(mesh, ('Lagrange', 1))

def on_boundary(x):
    return np.isclose(x[0], 1)

boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)
bc = fem.dirichletbc(1., boundary_dofs, V)

u = fem.Function(V)
v = ufl.TestFunction(V)

x = ufl.SpatialCoordinate(mesh)
f = x[0]*ufl.sin(x[1])

c = fem.Constant(mesh, 1.0)
F = ufl.inner((1 + c*u**2)*ufl.grad(u), ufl.grad(v))*ufl.dx - f*v*ufl.dx


# ### pyMOR Wrapping
# - Wrap the FEniCS objects as pyMOR `Operators`/`VectorArrays`.
# - Use generic `StationaryModel`.

# In[ ]:


from pymor.bindings.fenicsx import FenicsxOperator, FenicsxVectorSpace, FenicsxVisualizer
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import VectorOperator

space = FenicsxVectorSpace(V)
op = FenicsxOperator(F, u, params={'c': c}, bcs=(bc,), apply_lifting_with_jacobian=True)
rhs = op.range.zeros()

fom = StationaryModel(op, rhs, visualizer=FenicsxVisualizer(space))

parameter_space = fom.parameters.space((0.01, 1000.))


# Solve and visualize the solution:

# In[ ]:


U = fom.solve(1.)
fom.visualize(U)


# ### Compute snapshots for interpolation basis

# - `op` vanishes on solutions!
# - Use Newton residuals instead.

# In[ ]:


from pymor.solvers.newton import NewtonSolver
solver = NewtonSolver(rtol=1e-6, return_residuals=True)

solutions = fom.solution_space.empty()
residuals = fom.solution_space.empty()
for mu in parameter_space.sample_uniformly(10):
    U, data = solver.solve(fom.operator, fom.rhs.as_vector(), mu=mu, return_info=True)
    solutions.append(U)
    residuals.append(data['residuals'])


# ### Manual interpolation

# We first pick some arbitrary interpolation points.

# In[ ]:


def from_dofs(dofs, values=None):
    if values is None:
        values = np.ones(len(dofs))
    U = np.zeros(fom.solution_space.dim)
    U[dofs] = values.ravel()
    U = fom.solution_space.from_numpy(U)
    return U

dofs = [100, 200, 300]

fom.visualize(from_dofs(dofs))


# Pick first three residuals as interpolation basis:

# In[ ]:


V_hat = residuals[:3]


# Interpolation matrix:

# In[ ]:


B = V_hat.dofs(dofs)


# Operator evalution to approximate:

# In[ ]:


mu = fom.parameters.parse(10)
U  = fom.solve(mu=100)
AU = fom.operator.apply(U, mu)


# Solve interpolation problem:

# In[ ]:


c = np.linalg.solve(B, AU.dofs(dofs))
AU_ei = V_hat.lincomb(c)


# Visualize the approximation:

# In[ ]:


fom.visualize((AU, AU_ei))


# This looks bad!
# - Bad interpolation basis.
# - Bad interpolation DOFs.

# But does it interpolate?

# In[ ]:


fom.visualize(from_dofs(dofs, (AU-AU_ei).dofs(dofs)))


# ### Localized operator evaluation

# In[ ]:


op_restr, source_dofs = fom.operator.restricted(dofs)
print(op_restr)
print(source_dofs)
print(dofs)


# In[ ]:


fom.visualize((from_dofs(dofs), from_dofs(source_dofs)))


# We have:

# In[ ]:


op.apply(U, mu).dofs(dofs) - op_restr.apply(op_restr.source.from_numpy(U.dofs(source_dofs)), mu).to_numpy()


# In particular:

# In[ ]:


c_restr = np.linalg.solve(
    B, op_restr.apply(op_restr.source.from_numpy(U.dofs(source_dofs)), mu).to_numpy()
)
AU_ei_restr = V_hat.lincomb(c)

(AU_ei - AU_ei_restr).norm()


# Is it faster?

# In[ ]:


get_ipython().run_line_magic('timeit', 'op.apply(U, mu)')

U_source_dofs = op_restr.source.from_numpy(U.dofs(source_dofs))
get_ipython().run_line_magic('timeit', 'op_restr.apply(U_source_dofs, mu)')


# ### Implementing restricted

# - Identify mesh elements that carry selected `dofs`.
# 
# Then
# - Modify operator evaluation loop to only consider affected elements
# 
# or
# 
# - Generate submesh and evaluate operator on submesh.

# ### EI-Greedy

# In[ ]:


from pymor.algorithms.ei import ei_greedy
dofs, cb, _ = ei_greedy(residuals, rtol=1e-4)


# In[ ]:


fom.visualize(from_dofs(dofs))


# ### Using EmpiricalInterpolatedOperator

# In[ ]:


from pymor.operators.ei import EmpiricalInterpolatedOperator
ei_op = EmpiricalInterpolatedOperator(fom.operator, collateral_basis=cb, interpolation_dofs=dofs, triangular=True)


# In[ ]:


AU_ei = ei_op.apply(U, mu)
fom.visualize((AU, AU_ei))


# ### Full reduction: POD-Galerkin + EI

# Compute POD basis:

# In[ ]:


from pymor.algorithms.pod import pod
rb, svals = pod(U, rtol=1e-4)


# Reduce using `StationaryRBReductor`.
# - Replace `fom.operator` by `ei_op` before reducing.

# In[ ]:


from pymor.reductors.basic import StationaryRBReductor

fom_ei = fom.with_(operator=ei_op)
reductor = StationaryRBReductor(fom_ei, rb)

rom = reductor.reduce()


# Let's see if it works:

# In[ ]:


mu = parameter_space.sample_randomly()

U = fom.solve(mu)
U_rom = reductor.reconstruct(rom.solve(mu))

fom.visualize(U - U_rom)
print(f"Relative error: {(U - U_rom).norm().item() / U.norm().item()}")


# ### Is it faster?

# In[ ]:


get_ipython().run_line_magic('time', '_ = fom.solve(mu)')
get_ipython().run_line_magic('time', '_ = rom.solve(mu)')

