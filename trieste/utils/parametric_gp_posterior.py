from math import pi

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from trieste.type import TensorType
from ..models.model_interfaces import ProbabilisticModel


def rff_approx_of_rbf_kernel(
    kernel,
    input_dim: int,
    training_points: TensorType,
    num_features: int = 1000,
    seed: [int, None] = None,
):
    tf.debugging.assert_shapes([(training_points, ["N", "D"])])
    scaler_l = tf.linalg.diag(1 / tf.reshape(kernel.lengthscales, [-1]))  # [feature num, input dim]
    omega = tf.cast(
        tfd.Normal(loc=0, scale=1).sample([num_features, input_dim], seed=seed),
        dtype=training_points.dtype,
    )  # [feature num, input dim]
    b = (
        2
        * pi
        * tf.cast(tfd.Uniform(0, 1).sample(num_features, seed=seed), dtype=training_points.dtype)
    )  # b \sim U[0, 2pi]

    b_for_n_observations = tf.transpose(
        tf.tile(b[tf.newaxis, ...], [tf.shape(training_points)[0], 1])
    )  # Nx * b

    sigma2 = kernel.variance

    # z(x) = \sqrt(2a/m) cos(Wx + b)
    vector_Z_T = tf.sqrt(2 * sigma2 / num_features) * tf.cos(
        tf.matmul(omega, tf.matmul(scaler_l, tf.transpose(training_points))) + b_for_n_observations
    )
    return omega, b, vector_Z_T


def get_weighted_space_feature_param_posterior(
    model, vector_Z_T: TensorType, observations: TensorType
):
    """
    θ posterior sampler
    refer :cite:
    Hernández-Lobato, J. M., Hoffman, M. W., & Ghahramani, Z. (2014). Predictive entropy search for efficient global optimization of black-box functions. arXiv preprint arXiv:1406.2541.

    return: theta posterior sampler
    """
    num_features = tf.shape(vector_Z_T)[0]
    sigma_le_2 = model.likelihood.variance
    # A: φ(x)^T φ(x)  + σ_{le}^2I
    A = tf.matmul(vector_Z_T, vector_Z_T, transpose_b=True) + sigma_le_2 * tf.eye(
        num_features, dtype=vector_Z_T.dtype
    )

    # posterior of θ sample: θ|Dn ∼ N(A−1ΦTyn, σ2A−1), shape: (m_ftrs), shape checked correct
    jitter = tf.constant(0, dtype=A.dtype)
    while True:
        try:
            A_inverse = tf.linalg.inv(A + jitter * tf.eye(A.shape[0], dtype=A.dtype)).numpy()
            break
        except Exception as e:
            jitter += 1e-5
            if jitter > 1e-3:
                tf.print(e)
                raise ValueError

    # θ(A^{-1} φ(x)^T Y )
    mean_of_post_theta = tf.matmul(np.matmul(A_inverse, vector_Z_T), observations)
    mean_of_post_theta = tf.squeeze(mean_of_post_theta)
    variance_of_post_theta = sigma_le_2 * A_inverse

    jitter = tf.constant(0, dtype=A.dtype)
    while True:
        try:
            theta_sampler = tfd.MultivariateNormalTriL(
                mean_of_post_theta,
                scale_tril=tf.linalg.cholesky(
                    variance_of_post_theta
                    + jitter
                    * tf.eye(variance_of_post_theta.shape[0], dtype=variance_of_post_theta.dtype)
                ),
            )
            break
        except Exception as e:
            jitter += 1e-5
            if jitter > 1e-3:
                tf.print(e)
                raise ValueError

    return theta_sampler


def gen_approx_posterior_through_rff_wsa(
    model, sample_num: int, seed: [int, None] = None, num_features: int = 1000
) -> list:
    """
    Build Parametric approximation model posterior trajectory through RFF
    """
    input_dim = tf.shape(model.data[0])[1]
    training_points, training_obs = model.data
    omega, b, vector_Z_T = rff_approx_of_rbf_kernel(
        model.kernel,
        input_dim=input_dim,
        training_points=training_points,
        num_features=num_features,
        seed=seed,
    )
    # get theta posterior sampler
    theta_post_sampler = get_weighted_space_feature_param_posterior(model, vector_Z_T, training_obs)

    def makeFunc(kernel, theta):
        """
        :param kernel_var kernel variance
        :param m_ftrs RFF features, a.k.a value of D
        :param W spectral density sample
        :param b uniform sample from [0, 2pi]

        return a sampled approximated posterior trajectory of GP
        """
        kernel_var = kernel.variance
        scaler_l = tf.linalg.diag(1 / tf.reshape(kernel.lengthscales, [-1]))

        def trajectory(x):
            return tf.matmul(
                tf.sqrt(2 * kernel_var / num_features)
                * tf.transpose(
                    tf.cos(
                        tf.matmul(omega, tf.matmul(scaler_l, tf.transpose(x))) + b[..., tf.newaxis]
                    )
                ),
                theta[..., tf.newaxis],
            )

        return trajectory

    theta_samples = theta_post_sampler.sample(sample_num)
    gp_parametric_posterior = []
    for theta_sample in theta_samples:
        gp_parametric_posterior.append(makeFunc(model.kernel, theta_sample))

    return gp_parametric_posterior
