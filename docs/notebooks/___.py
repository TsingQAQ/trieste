from trieste.utils.parametric_gp_posterior import (
    gen_approx_posterior_through_rff_wsa,
    rff_approx_of_rbf_kernel,
)
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import gpflow
from trieste.space import Box
from trieste.utils.objectives import branin
from trieste.models import ModelStack

tf.random.set_seed(100)
tf.keras.backend.set_floatx("float64")

Xs_samples = Box([0.0, 0.0], [1.0, 1.0]).sample(23)
X = Xs_samples
Y = branin(X)
k = gpflow.kernels.RBF(lengthscales=[1.0, 1.0])
m_2d = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m_2d.training_loss, m_2d.trainable_variables, options=dict(maxiter=100))
m_2d2 = gpflow.models.GPR(data=(X, -Y), kernel=k, mean_function=None)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m_2d2.training_loss, m_2d2.trainable_variables, options=dict(maxiter=100))

from trieste.utils.mo_utils import sample_pareto_fronts_from_parametric_gp_posterior

m_stack = ModelStack((m_2d, 1), (m_2d2, 1))
f_samples = sample_pareto_fronts_from_parametric_gp_posterior(
    m_stack, 1, Box([0.0, 0.0], [1.0, 1.0])
)
