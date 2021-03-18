# -*- coding: utf-8 -*-
# # Multi-objective optimization: an Expected HyperVolume Improvement Approach

# +
import trieste
import gpflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from trieste.type import TensorType
from trieste.data import Dataset
from trieste.models import create_model

from trieste.acquisition.rule import OBJECTIVE, CONSTRAINT
from trieste.models.model_interfaces import ModelStack
from trieste.acquisition.qCEHVI import BatchMonteCarloConstraintHypervolumeExpectedImprovement

np.random.seed(1793)
tf.random.set_seed(1793)


def create_bo_model(data, input_dim=2, l=1.0):
    variance = tf.math.reduce_variance(data.observations)
    lengthscale = l * np.ones(input_dim, dtype=gpflow.default_float())
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=lengthscale)
    jitter = gpflow.kernels.White(1e-12)
    gpr = gpflow.models.GPR(data.astuple(), kernel + jitter, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)
    return create_model({
        "model": gpr,
        "optimizer": gpflow.optimizers.Scipy(),
        "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)},
        },
    })


class Constr_Ex:
    def __init__(self):
        self.d = 2
        self.M = 2

    @staticmethod
    def obj(x: TensorType) -> TensorType:
        _x = x[:, 0, tf.newaxis]
        _y = x[:, 1, tf.newaxis]
        return tf.concat([_x, (1+_y)/_x], axis=1)

    @staticmethod
    def con(x: TensorType) -> TensorType:
        _x = x[:, 0, tf.newaxis]
        _y = x[:, 1, tf.newaxis]
        con1 = 6 - _y - 9 * _x
        con2 = 1 + _y - 9 * _x
        return tf.concat([con1, con2], axis=1)


# We setup our acquisition function:

# +
input_dim = 2
mins = [0.1, 0.0]
maxs = [1.0, 5.0]
lower_bound = tf.cast(mins, gpflow.default_float())
upper_bound = tf.cast(maxs, gpflow.default_float())
search_space = trieste.space.Box(lower_bound, upper_bound)

num_objective = 2
num_constr = 2


def observer(query_points):
    benchmark = Constr_Ex()
    return {OBJECTIVE: trieste.data.Dataset(query_points, benchmark.obj(query_points)),
            CONSTRAINT: trieste.data.Dataset(query_points, benchmark.con(query_points))}

num_initial_points = 2 * (input_dim + 1)
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)


objective_models = [(create_bo_model(Dataset(initial_data[OBJECTIVE].query_points,
                                             tf.gather(initial_data[OBJECTIVE].observations, [i], axis=1)),
                                     input_dim=2), 1) for i in range(num_objective)]
constr_models = [(create_bo_model(Dataset(initial_data[CONSTRAINT].query_points,
                                             tf.gather(initial_data[CONSTRAINT].observations, [i], axis=1)),
                                  input_dim=2), 1) for i in range(num_constr)]

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
data_observations = datasets[OBJECTIVE].observations
mask_fail = tf.reduce_all(tf.greater(- data_observations[CONSTRAINT].observations), axis=-1)

from util.plotting import plot_bo_points_in_obj_space
plot_bo_points_in_obj_space(initial_data[OBJECTIVE], mask_fail=mask_fail)
plt.show()
# -

# [1] [Daulton, S., Balandat, M., & Bakshy, E. (2020). Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization. arXiv preprint arXiv:2006.05078.](https://arxiv.org/abs/2006.05078)

# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
