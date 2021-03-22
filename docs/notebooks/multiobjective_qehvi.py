# -*- coding: utf-8 -*-
# # Multi-objective optimization: an Expected HyperVolume Improvement Approach

# +
import trieste
import gpflow
import numpy as np
import tensorflow as tf
from tensorflow import cos, sin, sqrt
from math import pi
import matplotlib.pyplot as plt
from trieste.type import TensorType
from trieste.data import Dataset
from trieste.models import create_model

from trieste.acquisition.rule import OBJECTIVE, CONSTRAINT
from trieste.models.model_interfaces import ModelStack
from trieste.acquisition.qCEHVI import BatchMonteCarloConstraintHypervolumeExpectedImprovement

np.random.seed(1793)
tf.random.set_seed(1793)


# # -
#
# # ## The problem
# #
# # In this tutorial, we replicate one of the numerical examples in [GPflowOpt](https://github.com/GPflow/GPflowOpt/blob/master/doc/source/notebooks/multiobjective.ipynb) using acquisition function from Couckuyt, 2014 [1], which is a multi-objective optimization problem with 2 objective functions. We'll start by defining the problem parameters.
#
# def vlmop2(x: TensorType) -> TensorType:
#     transl = 1 / np.sqrt(2)
#     part1 = (x[:, 0] - transl) ** 2 + (x[:, 1] - transl) ** 2
#     part2 = (x[:, 0] + transl) ** 2 + (x[:, 1] + transl) ** 2
#     y1 = 1 - tf.exp(-1 * part1)
#     y2 = 1 - tf.exp(-1 * part2)
#     return tf.stack([y1, y2], axis=1)
#
#
# mins = [-2, -2]
# maxs = [2, 2]
# lower_bound = tf.cast(mins, gpflow.default_float())
# upper_bound = tf.cast(maxs, gpflow.default_float())
# search_space = trieste.space.Box(lower_bound, upper_bound)
#
# # We'll make an observer that outputs different objective function values, labelling each as shown.
#
# num_objective = 2
#
#
# def observer(query_points):
#     y = vlmop2(query_points)
#     return {OBJECTIVE: trieste.data.Dataset(query_points, y)}
#
#
# # Let's randomly sample some initial data from the observer ...
#
# num_initial_points = 10
# initial_query_points = search_space.sample(num_initial_points)
# initial_data = observer(initial_query_points)
#
# # ... and visualise those points in the design space.
#
# _, ax = plot_function_2d(vlmop2, mins, maxs, grid_density=100, contour=True, title=['Obj 1', 'Obj 2'])
# plot_bo_points(initial_query_points, ax=ax[0, 0], num_init=num_initial_points)
# plot_bo_points(initial_query_points, ax=ax[0, 1], num_init=num_initial_points)
# plt.show()
#
# # ... and in the objective space
#
# from util.plotting import plot_bo_points_in_obj_space
#
# plot_bo_points_in_obj_space(initial_data[OBJECTIVE])
# plt.show()
#
#
# # ## Modelling the two functions
# #
# # We'll model the different objective functions with their own Gaussian process regression models.
#
def create_bo_model(data, input_dim=2, l=1.0):
    variance = tf.math.reduce_variance(data.observations)
    lengthscale = l * np.ones(input_dim, dtype=gpflow.default_float())
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=lengthscale)
    jitter = gpflow.kernels.White(1e-12)
    gpr = gpflow.models.GPR(data.astuple(), kernel + jitter, noise_variance=1e-2)
    # gpflow.set_trainable(gpr.likelihood, False)
    return create_model({
        "model": gpr,
        "optimizer": gpflow.optimizers.Scipy(),
        "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)},
        },
    })
#
#
# objective_models = [(create_bo_model(Dataset(initial_data[OBJECTIVE].query_points,
#                                              tf.gather(initial_data[OBJECTIVE].observations, [i], axis=1))), 1) \
#                     for i in range(num_objective)]
#
# models = {OBJECTIVE: ModelStack(*objective_models)}
#
# # ## Define the acquisition process
# #
# # Here we utilize the `BatchMonteCarloHypervolumeExpectedImprovement` acquisition function proposed in
# # Daulton [1]:
#
# qehvi = BatchMonteCarloHypervolumeExpectedImprovement().using(OBJECTIVE)
# rule = trieste.acquisition.rule.BatchAcquisitionRule(num_query_points=3, builder=qehvi)
#
# # ## Run the optimization loop
# #
# # We can now run the optimization loop
#
# num_steps = 10
# bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
# result = bo.optimize(num_steps, initial_data, models, acquisition_rule=rule)
#
# # To conclude, we visualize the queried data in the design space
#
# # +
# datasets = result.try_get_final_datasets()
# data_query_points = datasets[OBJECTIVE].query_points
#
# _, ax = plot_function_2d(vlmop2, mins, maxs, grid_density=100, contour=True, title=['Obj 1', 'Obj 2'])
# plot_bo_points(data_query_points, ax=ax[0, 0], num_init=num_initial_points)
# plot_bo_points(data_query_points, ax=ax[0, 1], num_init=num_initial_points)
# plt.show()
# # -
#
# # ... and visulize in the objective space, orange dots denotes the nondominated points.
#
# plot_bo_points_in_obj_space(datasets[OBJECTIVE], num_init=num_initial_points)
# plt.show()


# ## Constraint Handling

# Sometimes, there are black-box constraint present in the problem, we can also handle them with the acquisition function:

# As an example, we use parallel Expected Hypervolume Improvement ($q$EHVI) [1]  acquisition functions to optimize a synthetic C2-DTLZ2 test function with $M=2$ objectives, $V=1$ constraint, and $d=12$ parameters. The two objectives are
# \begin{equation}
# f_1(\mathbf x) = (1+ g(\mathbf x_M))\cos\big(\frac{\pi}{2}x_1\big)
# \end{equation}
# $$f_2(\mathbf x) = (1+ g(\mathbf x_M))\sin\big(\frac{\pi}{2}x_1\big)$$
#
# where $g(\mathbf x) = \sum_{x_i \in \mathbf x_M} (x_i - 0.5)^2, \mathbf x \in [0,1]^d,$ and $\mathbf x_M$ represents the last $d - M +1$ elements of $\mathbf x$. Additionally, the C2-DTLZ2 problem uses the following constraint:
# $$c(\mathbf x) = - \min \bigg[\min_{i=1}^M\bigg((f_i(\mathbf x) -1 )^2 + \sum_{j=1, j\neq i}^M (f_j^2 - r^2) \bigg), \bigg(\sum_{i=1}^M \big((f_i(\mathbf x) - \frac{1}{\sqrt{M}})^2 - r^2\big)\bigg)\bigg]\geq 0$$
#
# where $\mathbf x \in [0,1]^d$ and $r=0.2$. 
#
# The goal here is to *minimize* both objectives.
# TODO: This has been checked numerically same as botorch version
class C2_DTLZ2:
    r = 0.2

    def __init__(self, input_dim, num_objectives):
        """
        C2-DTLZ2 refer E.2 of [1]
        """
        self.d = input_dim
        self.M = num_objectives

    def obj(self, x: TensorType) -> TensorType:
        def g(xM):
            z = (xM - 0.5) ** 2
            return tf.reduce_sum(z, axis=1, keepdims=True)

        def f(x):
            f = None
            for i in range(self.M):
                # (1 + g(xM)) part
                y = (1 + g(x[:, self.M - 1:]))
                # calc cos part
                for j in range(self.M - 1 - i):
                    y *= cos(pi / 2 * x[:, j, tf.newaxis])
                if i > 0:  # more than 1 obj
                    y *= sin(pi / 2 * x[:, self.M - 1 - i, tf.newaxis])
                f = y if f is None else tf.concat([f, y], 1)
            return f

        return f(x)

    def con(self, x: TensorType) -> TensorType:
        f_X = self.obj(x)
        term1 = (f_X - 1) ** 2
        # mask = ~(torch.eye(f_X.shape[-1], device=f_X.device).bool())
        # indices = torch.arange(f_X.shape[1], device=f_X.device).repeat(f_X.shape[1], 1)
        mask = ~tf.cast(tf.eye(self.M), dtype=tf.bool)
        indices = tf.tile(tf.constant(np.arange(self.M))[tf.newaxis, ...], [self.M, 1])
        indexer = tf.reshape(indices[mask], (self.M, self.M - 1)) # [M, M-1]
        # create an index without i
        # [..., M]->[..., M, M]
        term2_inner = tf.tile(f_X[:, tf.newaxis, :], [1, self.M, 1])
        # [..., M, M]&[M, M-1]->[..., M, M-1]
        # FIXME:
        term2_inner = tf.gather(term2_inner, indexer, axis=-1)[:, 0, :]
        # [..., M]
        term2 = tf.reduce_sum(term2_inner ** 2 - self.r ** 2, axis=-1)
        # get minimum across M: [...]
        min1 = tf.reduce_min(term1 + term2, axis=-1, keepdims=True)
        # checked
        min2 = tf.reduce_sum((f_X - 1 / sqrt(tf.cast(self.M, dtype=x.dtype))) ** 2 -
                             self.r ** 2, axis=1, keepdims=True)
        # outer min
        return tf.reduce_min(tf.concat([min1, min2], axis=1), keepdims=True, axis=1)


# We setup our acquisition function:

# +
input_dim = 12
mins = [0] * input_dim
maxs = [1] * input_dim
lower_bound = tf.cast(mins, gpflow.default_float())
upper_bound = tf.cast(maxs, gpflow.default_float())
search_space = trieste.space.Box(lower_bound, upper_bound)

num_objective = 2
num_constr = 1


def observer(query_points):
    benchmark = C2_DTLZ2(input_dim=input_dim, num_objectives=num_objective)
    return {OBJECTIVE: trieste.data.Dataset(query_points, benchmark.obj(query_points)),
            CONSTRAINT: trieste.data.Dataset(query_points, benchmark.con(query_points))}


num_initial_points = 2 * (input_dim + 1)
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

print(observer(0.2 * np.ones([1, 12])))

objective_models = [(create_bo_model(Dataset(initial_data[OBJECTIVE].query_points,
                                             tf.gather(initial_data[OBJECTIVE].observations, [i], axis=1)),
                                     input_dim=12), 1) for i in range(num_objective)]
constr_models = [(create_bo_model(Dataset(initial_data[CONSTRAINT].query_points,
                                             tf.gather(initial_data[CONSTRAINT].observations, [i], axis=1)),
                                  input_dim=12), 1) for i in range(num_constr)]

models = {OBJECTIVE: ModelStack(*objective_models), CONSTRAINT: ModelStack(*constr_models)}

# +
qehvi = BatchMonteCarloConstraintHypervolumeExpectedImprovement()
rule = trieste.acquisition.rule.BatchAcquisitionRule(num_query_points=2, builder=qehvi)

num_steps = 50
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_steps, initial_data, models, acquisition_rule=rule)
# -

# Now let's visualize the result:

# +
datasets = result.try_get_final_datasets()
mask_fail = tf.reduce_any(tf.greater(datasets[CONSTRAINT].observations, 0.0), axis=-1)

from util.plotting import plot_bo_points_in_obj_space
plot_bo_points_in_obj_space(datasets[OBJECTIVE].observations, num_init=num_initial_points, mask_fail=mask_fail)
plt.show()
# -

# [1] [Daulton, S., Balandat, M., & Bakshy, E. (2020). Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization. arXiv preprint arXiv:2006.05078.](https://arxiv.org/abs/2006.05078)

# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
