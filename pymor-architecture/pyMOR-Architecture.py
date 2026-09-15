#!/usr/bin/env python
# coding: utf-8

# # pyMOR Architecture

# ## Topics

# - pyMOR's main interface classes: `VectorArrays`, `Operators`, `Models`
# - Parameters
# - Immutability
# - Defaults
# - RuleTables
# - Caching

# ## pyMOR's main interface classes

# ### Goal
# > Implement MOR algorithms in terms of operations on abstract mathematical objects, such that the same algorithm can be used with different PDE/LA backends.
# 
# ### Needed objects
# - vectors -> `VectorArrays` (support vectorized LA such as NumPY)
# - matrices, nonlinear operators, (non-)linear solvers -> `Operators`
# - problem structure, solution algorithms -> `Models`
# 
# ### Supported PDE solvers
# 
# Official support for:
# - NumPy/SciPy-based models (built-in, scikit-fem, ...)
# - FEniCS
# - FEniCSx (no MPI yet)
# - NGSolve
# - deal.II
# 
# pyMOR has been used with various flavors of DUNE and proprietary codes.

# ## VectorArrays

# - A (short) sequence of (high-dimensional) vectors.
# - Think of tall-and-skinny instead of a short-and-fat matrix.
# - All vectors live in the same associated `VectorSpace`.
# - `VectorArrays` are always created by the `VectorSpace`.
# - Vectors in pyMOR are just `VectorArrays` of length one.
#   <br>(`ListVectorArray` manages a list of `Vector` objects, but that's just an implementation detail.)
# - `pymor.vectorarrays.interface.VectorArray` defines the `VectorArray` interface.

# ### Creating VectorArrays

# **Create a NumpyVectorSpace**

# In[1]:


from pymor.basic import NumpyVectorSpace

space = NumpyVectorSpace(5)
print(space.dim)


# **Empty array**

# In[2]:


V = space.empty()
print(V)


# **Array of zero vectors**

# In[3]:


V = space.zeros(3)
print(V)


# This looks like a regular NumPy array, but it's actually a pyMOR `VectorArray` storing the data internally as a NumPy array:

# In[4]:


V


# **Array of random vectors**

# In[5]:


V = space.random(2)
print(V)


# **Array of constant coefficient vectors**

# In[6]:


V = space.ones(1)
print(V)
V = space.full(99., 2)
print(V)


# **Create array from backend data**

# In[7]:


import numpy as np
a = np.arange(15).reshape(5, 3)
print(a)


# In[8]:


V = space.make_array(a)
V


# **Warning:** Data is not copied in most cases:

# In[9]:


print(V.impl._array is a)


# This can lead to unexpected results:

# In[10]:


a[0,0] = 999
print(V)


# (This is an implementation detail. There is no guarantee that `a` and `V` hold the same data.)

# ### Interface methods

# **Length of an array**

# In[11]:


V = space.ones(3)
len(V)


# **Vector operations**

# In[12]:


V = space.make_array(np.arange(10).reshape(5, 2))
W = space.make_array(np.arange(0, 20, 2).reshape(5, 2))


# In[13]:


print(V)
print(W)


# In[14]:


print(V + W)


# In[15]:


print(V - W)


# In[16]:


print(V * 2)


# In[18]:


print(V.lincomb(np.array([1, 2])))


# In[19]:


print(V.lincomb(np.array([[1, 99],
                          [2,  0]])))


# **VectorArrays are mutable!**

# In[21]:


V.scal(2)
print(V)


# In[22]:


V.axpy(3, W)
print(V)


# **Norms and inner products**

# In[23]:


print(V.norm())


# In[24]:


print(V.inner(W))


# In[25]:


print(V.pairwise_inner(W))


# In[26]:


print(V.gramian())  # == V.inner(V)


# **Coefficient access**

# In[27]:


print(V)
print(V.dofs([1,4]))


# Only a few `dofs` should be extracted at once. This is again motivated by the fact that the `VectorArray` may be using a backend where extracting all (a large number) of dofs is either not possible or expensive.

# In[28]:


print(V.amax())


# `dofs` and `amax` are mostly used for empirical interpolation.

# **Array operations**

# In[29]:


U = V.copy()


# `VectorArrays` use a copy-on-write mechanism. Actual data is only copied when `U` or `V` is modified.

# In[30]:


V.impl._array is U.impl._array


# In[31]:


U.append(W)
print(U)
print(V)
V._impl._array is U._impl._array


# In[32]:


del U[2]
print(U)


# Indexing and advanced indexing similar as for NumPy arrays is available.
# 
# **Any form of indexing always create a view onto the original array!**

# In[33]:


V = U[[0,2]]
print(V)
print(V.is_view)


# A view array has the same `impl` object as the original arrays and store the indices in its `ind` attribute:

# In[34]:


print(V.impl is U.impl)
print(V.ind)


# Modifying the view modifies the original array:

# In[35]:


V.scal(100)
print(V)
print(U)


# ### Conversion to NumPy array

# Depending on the `VectorArray` implementation, `to_numpy` and `from_numpy` methods may be available.
# 
# For `NumpyVectorArrays` this clearly works, and `from_numpy` and `make_array` essentially do the same thing.

# In[36]:


a = np.arange(15).reshape(5, 3)
V = NumpyVectorSpace.from_numpy(a)
print(repr(V))
print(repr(V.to_numpy()))


# If you are using only NumPy-based data in your own application, using these methods is fine. However, most pyMOR code avoids these methods since they may not be available for all `VectorArray` implementations.

# ### NumpyVectorArrays vs NumPy array
# 
# Both types of arrays are used extensively in pyMOR.
# 
# `VectorArrays` are used whenever vectors that live in a (potentially high-dimensional) state-space are represented, e.g.:
# 
# - internal state of a `Model`,
# - basis used for projection,
# - vectors that "live" in an external PDE solver.
# 
# NumPy arrays are used for (low-dimensional) data which has no direct connection to an external solver, e.g.:
# 
# - inputs and outputs of models,
# - coefficients for linear combinations,
# - computed inner products.
# 
# `VectorArrays` appear to be matrices of column vectors, but
# 
# - `VectorArrays` in pyMOR are not considered ***column or row vectors***, they are simply lists of vectors.
# - Consequence: there is no `transpose` method for `VectorArrays`.
#   <br>(Technical reason: transposing a list of MPI-distributed PETsc vectors would be expensive/unfeasible.)
# - Only typical operations for "mathematical" vectors are supported.

# ### ListVectorArray
# 
# Linear algebra backends of PDE solvers often only have the notion of a single vector.
# 
# pyMOR supports an interface for single vectors via the `Vector` class.
# 
# The `ListVectorArray` manages a Python `list` of pyMOR `Vector` instances.
# 
# Thus, for integrating a PDE solver, only a new `Vector` class needs to be implemented.

# In[37]:


from pymor.vectorarrays.list import NumpyListVectorSpace

space = NumpyListVectorSpace(4)
W = space.random(3)
print(type(W), space.vector_type)


# In[38]:


print(W.vectors)


# In[39]:


print(W.vectors[0]._array)


# ### FEniCSx VectorArrays

# We create some discrete function in a finite element space over some mesh.

# In[40]:


import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

mesh = mesh.create_unit_square(MPI.COMM_WORLD, 100, 100)
V = fem.functionspace(mesh, ('Lagrange', 3))
u = fem.Function(V)
u.interpolate(lambda x: np.sin(x[1]))


# Use FEniCSx bindings to create a `VectorArray`:

# In[41]:


from pymor.bindings.fenicsx import FenicsxVectorSpace
space = FenicsxVectorSpace(V)
U = space.make_array([u.x.petsc_vec.copy()])

print(type(U), space.vector_type)
U.vectors[0].real_part.impl


# We can use this array like any other `VectorArray`:

# In[42]:


U.append(space.ones())
V = U.lincomb(np.array([3,4]))
U.inner(V)


# ### BlockVectorArray
# 
# When working with systems, we often deal with `BlockVectorArrays`:

# In[46]:


from pymor.vectorarrays.block import BlockVectorSpace
space1 = NumpyVectorSpace(3)
space2 = NumpyListVectorSpace(2)
space = BlockVectorSpace([space1, space2])

U = space.random(2)
print(type(U))
print(U.blocks[0])
print(U.blocks[1])


# In[47]:


print(U.gramian())


# In[48]:


U.to_numpy()


# ## Operators

# - `Operators` are (non-)parametric, (non-)linear mappings between `VectorSpaces`.
# - An `Operator` can be applied to any `VectorArray` from the `Operator`'s `source` `VectorSpace`.
# - The resulting `VectorArray` lies in the `Operator`'s `range` `VectorSpace`.
# - `pymor.operators.interface.Operator` defines the `Operator` interface.
# - `Operators` are **immutable**. You cannot change their attributes.

# ### Creating operators

# **Operator from NumPy array**

# In[49]:


from pymor.operators.numpy import NumpyMatrixOperator

A = np.ones((3,2))
Aop = NumpyMatrixOperator(A)

print(Aop.range)
print(Aop.source)
print(Aop)
print(Aop.matrix)


# **Operator from SciPy sparse matrix**

# In[50]:


import scipy.sparse as sps

B = sps.diags([-2., 1, 1], [0, -1, 1], shape=(3, 3))
Bop = NumpyMatrixOperator(B)

print(Bop.range)
print(Bop.source)
print(Bop)
print(Bop.sparse)


# **Operator from arbitrary mapping**

# In[51]:


from pymor.operators.numpy import NumpyGenericOperator

def C(X):
    return X**2 - X[0, :]

Cop = NumpyGenericOperator(C, dim_range=3, dim_source=3)

print(Cop.range)
print(Cop.source)
print(Cop)
print(Cop.linear)


# ### Interface methods

# **Applying an Operator**

# In[52]:


print(Aop.matrix)
V = Aop.source.ones(1)
V.append(Aop.source.zeros(1))

Aop.apply(V)


# In[53]:


Cop.apply(Cop.source.from_numpy(np.arange(3)))


# **Applying adjoint Operator** (adjoint = transposed matrix)

# In[54]:


W = Aop.range.random()
Aop.apply_adjoint(W)


# Of course, `apply_adjoint` is only implemented for linear `Operators`.

# **Solving equations**

# In[55]:


V = Bop.range.random()
print(Bop.apply_inverse(V))


# `NumpyMatrixOperator.apply_inverse` chooses appropriate solvers for dense and sparse matrices. By default:
# 
# - `scipy.linalg.lu_factor` for dense NumPy arrays
# - `scipy.sparse.linalg.splu` for SciPy sparse matrices.
# 
# The solver can be customized by setting the Operator's `solver` or global pyMOR `defaults` (see below).

# **Solving adjoint equations**

# In[56]:


V = Bop.source.random()
print(Bop.apply_inverse_adjoint(V))


# **Operators as bilinear forms**

# In[57]:


W = Aop.range.random(2)
V = Aop.source.random(3)
print(Aop.apply2(W, V))   # evaluate W.T @ A @ V


# In[58]:


print(Aop.pairwise_apply2(W, V[:2]))


# **Operators as VectorArrays**
# 
# For linear operators with low-dimensional source space, we have:

# In[60]:


Aop_range_array = Aop.as_range_array()
print(Aop_range_array)
print((Aop_range_array.lincomb(V.to_numpy()) - Aop.apply(V)).norm())


# For linear operators with low-dimensional range space, we have:

# In[61]:


Dop = NumpyMatrixOperator(np.ones((2,5)))
Dop_source_array = Dop.as_source_array()
print(Dop_source_array)
V = Dop.source.random()
print((Dop.range.from_numpy(Dop_source_array.inner(V)) - Dop.apply(V)).norm())


# **Nonlinear Operators**

# In[62]:


from pymor.analyticalproblems.burgers import burgers_problem
from pymor.discretizers.builtin import discretize_instationary_fv
p = burgers_problem()
m, _ = discretize_instationary_fv(p, diameter=1/10, nt=10)
op = m.operator.assemble(m.parameters.parse(1.2))
op


# We can linearize a nonlinear operator using the `jacobian` method:

# In[63]:


V = op.source.random()
op.jacobian(V)


# `apply_inverse` will automatically run a Newton algorithm to solve the equation:

# In[ ]:


W = op.range.random()
V = (m.mass + 0.1 * op).apply_inverse(W)


# For empirical interpolation, we need to restrict the operator to a few DOFs:

# In[ ]:


restricted_op, source_dofs = op.restricted(np.array([5,7,11]))
print(restricted_op.range, restricted_op.source, source_dofs)


# In[ ]:


V = op.source.random()
(op.apply(V).dofs([5,7,11]) 
  - restricted_op.apply(restricted_op.source.from_numpy(V.dofs(source_dofs))).to_numpy())


# ### LincombOperator
# 
# Unlike for `VectorArrays`, adding, subtracting or scaling of `Operators` is not performed immediately but rather a `LincombOperator` is created.

# In[64]:


Aop = NumpyMatrixOperator(np.ones((3,3)))
Bop = NumpyMatrixOperator(np.eye(3))
Lop = 3*Aop - 2*Bop
Lop


# We can apply a `LincombOperator` like any other `Operator`:

# In[65]:


V = Lop.source.random()
Lop.apply(V)


# Calling `assemble` on a `LincombOperator` will return a single matrix operator when possible:

# In[66]:


Lop.assemble()


# ### Further constructions
# 
# Aside from the `LincombOperator`, there are many different types of operators in `pymor.operators.constructions`.

# In[67]:


from pymor.operators.constructions import ZeroOperator, IdentityOperator

Zop = ZeroOperator(Aop.range, Aop.source)
Iop = IdentityOperator(Aop.range, Aop.source)
Cop = Aop @ Bop

LLop = Zop + Iop + Lop + Cop
LLop


# pyMOR's `Operators` can be implemented to wrap matrices/operator in a desired PDE backend,
# 
# but also enable efficient and convenient handling of certain structured operators `BlockOperator`, `CanonicalSymplecticFormOperator`, `NumpyHankelOperator`, ...

# ### Matrix conversion
# 
# We can use `to_matrix` to try to convert a linear `Operator` into a matrix. This will generally only work for NumPy/SciPy-based `Operators`.

# In[68]:


from pymor.algorithms.to_matrix import to_matrix

to_matrix(LLop)


# ## Models

# - `Models` are collections of `Operators` and `VectorArrays` that represent a particular type of model (equation system).
# - They define mappings from inputs/parameters to solutions and outputs.
# - `pymor.models.interface.Model` defines the `Model` interface.
# - `Models` are **immutable**. You cannot change their attributes.
# - MOR algorithms typically take a `Model` as an input and compute a ROM, which will again be a `Model`.

# In[69]:


from pymor.models.examples import heat_equation_non_parametric_example

m = heat_equation_non_parametric_example().to_lti()


# ### Interface methods

# **Input/output dimensions, solution space**

# In[70]:


print(m.dim_input, m.dim_output, m.solution_space)


# **Time-domain solution**

# In[71]:


V = m.solve(input='sin(10*t)')
m.visualize(V)


# **Time-domain output**

# In[72]:


out = m.output(input='sin(10*t)')
from matplotlib import pyplot as plt
plt.plot(out[0, :])


# **Computing multiple quantities at the same time**

# In[73]:


data = m.compute(solution=True, output=True, input='sin(10*t)')
data.keys()


# **Further optional interface methods**
# - `estimate_error`: a posteriori error estimator for ROM solution
# - `estimate_output_error`: a posteriori error estimator for ROM output
# - `solution_d_mu`: derivative of solution w.r.t. parameter
# - `output_d_mu`: derivative of output w.r.t. parameter

# ## Parameters and Parametric Objects

# `Operators` and `Models` can depend on one or more parameters.

# ### Parameters
# 
# - An ordered dictionary with
#      * keys: parameter names
#      * values: corresponding dimensions.
# - It defines what parameters a parametric object depends on.
# - Names `'t'`, `'s'`, and `'input'` should only be used for time, frequency, and input, respectively.
# - Immutable

# In[74]:


from pymor.parameters.base import Parameters

parameters = Parameters({'a': 1, 'b': 2})
parameters


# ### Mu
# 
# - An ordered dictionary with
#   * keys: parameters names
#   * values: corresponding values as 1D NumPy arrays.
# - Immutable

# In[75]:


from pymor.parameters.base import Mu

mu = Mu({'a': np.array([1]), 'b': np.array([2, 3]), 'c': np.array([4, 5, 6])})
mu


# **Check if a `Mu` contains values for all `Parameters`**

# In[76]:


print(parameters)
print(mu)
print(parameters.is_compatible(mu))


# **Parsing parameters**

# In[77]:


parameters.parse({'a':1, 'b':[2,3]})


# In[78]:


parameters.parse([1,2,3])


# In[79]:


Parameters({'a': 1}).parse(99)


# ### ParameterSpace
# 
# - A box-constrained set of possible parameter values.
# - Immutable.

# **Creating a ParameterSpace**

# In[80]:


print(parameters.space(0, 1))
print(parameters.space({'a': (0, 1), 'b': (1, 2)}))


# **Sampling a ParameterSpace**

# In[81]:


parameter_space = parameters.space(0, 1)
parameter_space.sample_uniformly(2)


# In[82]:


parameter_space.sample_randomly(2)


# In[83]:


parameter_space.sample_randomly()


# ### ParameterFunctional
# 
# - A mapping from `Mus` to $\mathbb{R}$ or $\mathbb{C}$.
# - Can be used as coefficients of `LincombOperator` or `SelectionOperator`
# - Immutable.

# In[84]:


mu = parameters.parse([1.,2.,3.])
mu


# **ProjectionParameterFunctional**

# In[85]:


from pymor.parameters.functionals import ProjectionParameterFunctional

f1 = ProjectionParameterFunctional('b', size=2, index=0)
f1(mu)


# **ExpressionParameterFunctional**

# In[86]:


from pymor.parameters.functionals import ExpressionParameterFunctional

f3 = ExpressionParameterFunctional('b[0]**2', parameters)
f3(mu)


# **GenericParameterFunctional**

# In[87]:


from pymor.parameters.functionals import GenericParameterFunctional

f2 = GenericParameterFunctional(lambda mu: mu['b'][0]**2, parameters)
f2(mu)


# ### Time-Dependent Parameter Values

# `Mus` can also have time dependent componenents:

# In[88]:


print(parameters)
mu = parameters.parse([1, 'sin(t)', 0])
mu


# Time-dependent components do not show up as parameters, since a time has to be prescribed first to obtain actual values:

# In[89]:


mu.parameters()


# In[90]:


mu['b']


# In[91]:


mu_pi_half = mu.at_time(np.pi/2)
mu_pi_half


# In[92]:


print(mu_pi_half.parameters())
print(mu_pi_half['b'], mu_pi_half['t'])


# ### Parametric Operators

# `ParametricObjects` like `Operators` automatically inherit parameters from their `__init__` arguments:

# In[93]:


from pymor.operators.numpy import NumpyMatrixOperator
Aop = NumpyMatrixOperator(np.ones((3,3)))
Bop = NumpyMatrixOperator(np.eye(3))

Pop = Aop + ProjectionParameterFunctional('p') * Bop
Pop


# **Parameters of an Operator**

# In[94]:


Pop.parameters


# As a shorthand, we can check if an `Operator` depends on parameters using:

# In[95]:


Pop.parametric


# **Interface methods of Parametric Operators**
# 
# For parametric operators, we need to pass a `Mu` as the `mu` argument:

# In[96]:


V = Pop.source.ones(1)
Pop.apply(V, mu=Pop.parameters.parse(0))


# In[97]:


Pop.apply(V, mu=Pop.parameters.parse(1))


# We need to pass a `Mu` object.

# In[98]:


Pop.apply(V, mu=1)


# The same is true for `apply_adjoint`, `apply_inverse`, `apply_inverse_adjoint`, `as_source_array`, `as_range_array`, `jacobian`, `assemble`.

# ### "Parametric VectorArrays"
# 
# `VectorArrays` are just data. They cannot depend on a parameter.
# A way to represent parameter-dependent vectors is via a parametric `Operator`,
# e.g., using `VectorArrayOperators`.

# In[99]:


from pymor.operators.constructions import VectorArrayOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

space = NumpyVectorSpace(3)
U = space.ones()
V = space.full(2.)

Uop = VectorArrayOperator(U)
Vop = VectorArrayOperator(V)

print(Uop)
print(Vop)


# In[100]:


UVop = Uop + ProjectionParameterFunctional('p') * Vop
print(UVop)


# In[101]:


print(UVop.as_range_array(mu=UVop.parameters.parse(5)))


# ### Parametric Models
# 
# If a `Model` contains any parametric `Operator`, the `Model` will be parametric as well.

# In[102]:


from pymor.models.iosys import LTIModel
A = np.random.rand(10, 10)
B = np.ones((10, 1))
C = np.ones((1, 10))
fom = LTIModel.from_matrices(A, B, C)

fom


# In[103]:


fom.parameters


# In[104]:


Pop = fom.A + ProjectionParameterFunctional('p') * fom.A
fom = fom.with_(A=Pop)


# In[105]:


fom.parameters


# ## Immutable objects

# * `ImmutableObjects` are locked after their `__init__` method.
# * Private attributes (starting with `_`) can still be written. Use with care! Changing them may not affect mathematical behavior of the object.
# * The `with_` method creates a shallow copy of the object by providing new values for some of its `__init__` arguments.
# * Using `with_` is generally cheap.
# * A useful idiom is calling `self.__auto_init(locals())` to save all `__init__` arguments as attributes.

# In[107]:


from pymor.core.base import ImmutableObject

class MyImmutableObject(ImmutableObject):
    def __init__(self, a, b):
        self.__auto_init(locals()) # equivalent to self.a = a; self.b = b
        self.c = a + b


# In[108]:


o = MyImmutableObject(1, 2)


# In[109]:


o.a


# In[110]:


o.a = 99


# In[111]:


o._d = 99


# In[112]:


o.with_()


# In[113]:


o.with_(b=32)


# In[114]:


o.with_(c=8)


# ## pyMOR defaults

# - Mechanism for setting global default values for optional function arguments.
# - Used a lot throughout pyMOR to set default tolerances, enable checks, etc.

# **Listing all defaults**

# In[115]:


from pymor.core.defaults import print_defaults
print_defaults()


# **Saving/loading defaults from file**

# In[116]:


from pymor.core.defaults import write_defaults_to_file
write_defaults_to_file()


# - Defaults are automatically loaded from `pymor_defaults.py` in current working directory when pyMOR is imported.
# - An alternate path can be prescribed using the `PYMOR_DEFAULTS` environment variable.
# - Defaults can also be loaded using `load_defaults_from_file`.

# **Programmatically setting defaults**

# In[117]:


from pymor.core.defaults import set_defaults
from pymor.bindings.scipy import ScipyBicgStabSpILUSolver

set_defaults({'pymor.operators.numpy.NumpyMatrixOperator.default_sparse_solver': ScipyBicgStabSpILUSolver})


# **Writing functions with defaults**

# In[118]:


from pymor.core.defaults import defaults

@defaults('tolerance')
def some_algorithm(x, y, tolerance=1e-5):
    print(tolerance)

def test_some_algorithm(x, y, tolerance_for_some_algorithm=None):
    some_algorithm(x, y, tolerance=tolerance_for_some_algorithm)


# In[119]:


test_some_algorithm(1, 2, 42)
test_some_algorithm(1, 2)
set_defaults({'__main__.some_algorithm.tolerance': 99})
test_some_algorithm(1, 2)


# ## RuleTables

# - Dispatch mechanism for algorithms based on input type and data.
# - Used for `project`, `preassemble`, `assemble_lincomb`, `estimate_image`, `expand`, `contract`, `to_matrix`.

# ### Example: assemble_lincomb
# 
# (called in `LincombOperator.assemble`)

# In[ ]:


from pymor.algorithms.lincomb import assemble_lincomb
from pymor.tools.formatsrc import print_source

print_source(assemble_lincomb)


# In[ ]:


from pymor.algorithms.lincomb import AssembleLincombRules
AssembleLincombRules


# In[ ]:


AssembleLincombRules[1]


# ### project

# In[ ]:


from pymor.algorithms.projection import project
print_source(project)


# In[ ]:


from pymor.algorithms.projection import ProjectRules
ProjectRules


# In[ ]:


ProjectRules[2]


# ## Caching

# - Save results of long-running computations in memory / on disk.
# - When a method is called twice with same arguments, the cached value is returned.
# - Depends on immutability of underlying object.
# - Used for `Model` interface methods (`solve`, `output`, `compute`, etc.), LTI model related quantities (Gramians, norms, etc.), built-in `Grid` interface methods.

# ### Example: cached model evaluation

# In[120]:


from pymor.models.examples import thermal_block_example
m = thermal_block_example()


# In[121]:


get_ipython().run_line_magic('time', '_ = m.solve([1,2,3,4])')
get_ipython().run_line_magic('time', '_ = m.solve([1,2,3,4])')


# In[122]:


m.enable_caching('memory')  # or 'disk'
get_ipython().run_line_magic('time', '_ = m.solve([1,2,3,4])')
get_ipython().run_line_magic('time', '_ = m.solve([1,2,3,4])')


# The 'persistent' cache region is preserved over multiple program runs. A unique identifier for the model has to be set.

# In[123]:


m.enable_caching('persistent', cache_id='my_model_that_wont_change')  # or 'disk'
get_ipython().run_line_magic('time', '_ = m.solve([1,2,3,4])')
get_ipython().run_line_magic('time', '_ = m.solve([1,2,3,4])')


# ### Implementing a class with cached methods

# In[124]:


from pymor.core.cache import CacheableObject, cached

class MyCachedObj(CacheableObject):

    cache_region = 'memory'  # default cache region. 'None' if not specified

    @cached
    def expensive_evaluation(self, x):
        import time
        time.sleep(x)
        return x**2


# In[125]:


o = MyCachedObj()
get_ipython().run_line_magic('time', 'o.expensive_evaluation(2)')
get_ipython().run_line_magic('time', 'o.expensive_evaluation(4)')
get_ipython().run_line_magic('time', 'o.expensive_evaluation(2)')

