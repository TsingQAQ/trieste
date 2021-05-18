# -*- coding: utf-8 -*-
# # Parallel Pareto Frontier Entropy Search

# ## Related Work

# - [MESMO](https://par.nsf.gov/servlets/purl/10145801) paper, NIPS 2019, Belakaria et al
# - [PFES](http://proceedings.mlr.press/v119/suzuki20a.html), ICML 2020, Shinya Suzuki et al
#
#
# Multi-Fidelity Part
# - [MF-OSEMO](file:///C:/Users/Administrator/Downloads/6561-Article%20Text-9786-1-10-20200519.pdf) paper, 2020 AAAI, Belakaria et al
#
# Constraint Part
# - [MESMOC](https://arxiv.org/pdf/2009.01721.pdf) paper, NIPS 2020 Workshop, Belakaria et al
# - [MESMOC+](https://arxiv.org/pdf/2011.01150.pdf) paper, AISTATS, Daniel Fernández-Sánchez (Daniel Hernández-Lobato)
#
# Uncatogrized
# - [iMOCA](https://arxiv.org/pdf/2009.05700.pdf) paper, NIPS 2020 Workshop, Belakaria et al

# -----------

# ## Main

# \begin{equation}
# \begin{aligned}
# \alpha(x) &= H[PF\vert D] - \mathbb{E}_{f_x}H[PF \vert D, \{x, \boldsymbol{f}_x\}] \\& = H[\boldsymbol{f}_x\vert D] - \mathbb{E}_{PF}H[\boldsymbol{f}_x \vert D, x, PF]
# \end{aligned}
# \end{equation}

# MESMO approximation:

# \begin{equation}
# H[\boldsymbol{f}_x \vert D, x, PF] \approx \sum_{j=1}^K H[y^j \vert D, x, max\{z_1^j, ..., z_m^j\}]
# \end{equation}

# Where $\{z_1,..., z_m\}$ are sampled pareto front points

# ---------------

# Recall the definition of acquisition function in PFES paper [1]:
# \begin{equation}
# \alpha(\boldsymbol{x}) = H[p(\boldsymbol{f}_x) \vert D] - \frac{1}{|PF|} \Sigma_{\mathcal{F^*}\in PF} H[p(\boldsymbol{f}_x \vert D, \boldsymbol{f}_x \prec \mathcal{F^*})] \tag{1}
# \end{equation}

# Where $\mathcal{F^*}$ denotes the sampled pareto pront as a discrete approximation of pareto frontier. 

# $p(\boldsymbol{f}_x \vert D, \boldsymbol{f}_x \prec \mathcal{F^*})$ can be rewritten as:  
#
# \begin{equation}
# p(\boldsymbol{f}_x \vert D, \boldsymbol{f}_x \prec \mathcal{F^*}) = \left\{
# \begin{aligned}
# &\frac{1}{Z}p(\boldsymbol{f}_x \vert D)\quad  \{\boldsymbol{f}_x \in R^M: \boldsymbol{f}_x \prec \mathcal{F^*}\}\\
# &0 \quad \quad else\\
# \end{aligned}\tag{2}
# \right.
# \end{equation}  
# Where $Z = \int_\mathcal{F} p(\boldsymbol{f}_x) d\boldsymbol{f}_x$ is the normalization constant, $\mathcal{F}$ is defined as the dominated objective space.

# Assume the dominated space can been partitioned into $M$ cells, with statistical independence (for simplicity derivation, might not necessary) assumption on different obj, we have:  
# \begin{equation}
# \begin{aligned}
# Z & = \sum_{m=1}^{M}\left[\prod_{i=1}^{L}\left(\Phi(\frac{u_m^i - \mu_i(x)}{\sigma_i(x)}) - \Phi(\frac{l_m^i - \mu_i(x)}{\sigma_i(x)})\right)\right] \\& = \sum_{m=1}^{M} Z_m
# \end{aligned} \tag{3}
# \end{equation}
# and 
# \begin{equation}
# Z_m  = \prod_{i=1}^L Z_{m_i} \tag{4}
# \end{equation}
#
# where $u_m^i$, $l_m^i$ denotes the upper and lower bound of cell $m$ at dimension $i$. 

# -------

# ### Deriviation of the differential entropy based on conditional distribution 

# Thorem 3.1 of PFES paper reveals the calculation of $H[p(\boldsymbol{f}_x \vert D, \boldsymbol{f}_x \prec \mathcal{F^*})]$ for a single query points:

# \begin{equation}
# \begin{aligned}
# H[p(\boldsymbol{f}_x \vert D, \boldsymbol{f}_x \prec \mathcal{F^*})] &= - \int_\mathcal{F}  \frac{p(\boldsymbol{f}_x \vert D)}{Z} log \frac{p(\boldsymbol{f}_x \vert D)}{Z} d \boldsymbol{f}_x
# \\ &=  - \int_\mathcal{F}  \frac{p(\boldsymbol{f}_x \vert D)}{Z} log p(\boldsymbol{f}_x \vert D) d \boldsymbol{f}_x + \frac{log Z}{Z} \int_\mathcal{F}  p(\boldsymbol{f}_x \vert D)  d \boldsymbol{f}_x \\ &= -  \frac{1}{Z}\int_\mathcal{F}  p(\boldsymbol{f}_x \vert D) log p(\boldsymbol{f}_x \vert D) d \boldsymbol{f}_x + log Z \\&= -  \frac{1}{Z} \sum_{m=1}^M \int_{\mathcal{F}_m}p(\boldsymbol{f}_x \vert D) log p(\boldsymbol{f}_x \vert D) d \boldsymbol{f}_x + log Z
# \end{aligned} \tag{5}
# \end{equation}

# Define auxilary random variable (truncated normal)  $\boldsymbol{h}_{x_{\mathcal{F}_m}} := \boldsymbol{f}_{x} \cdot I\{\boldsymbol{f}_{x} \in \mathcal{F}_m\}$, then:

# \begin{equation}
# p(\boldsymbol{h}_{x_{\mathcal{F}_m}}) = \left\{
# \begin{aligned}
# &\frac{1}{Z_m}p(\boldsymbol{f}_x \vert D)\quad  \{\boldsymbol{f}_x \in \mathcal{F}_m\}\\
# &0 \quad else\\
# \end{aligned} \tag{6}
# \right.
# \end{equation}

# Then we could write its differential entropy as:
# \begin{equation}
# \begin{aligned}
# \mathbb{H}[\boldsymbol{h}_{x_{\mathcal{F}_m}}] &= -\int_{\mathcal{F}_m} p({h}_{x_{\mathcal{F}_m}})log p({h}_{x_{\mathcal{F}_m}}) d{h}_{x_{\mathcal{F}_m}} \\ &= - \int_{\mathcal{F}_m} \frac{1}{Z_m}p(\boldsymbol{f}_x \vert D)log [\frac{1}{Z_m}p(\boldsymbol{f}_x \vert D)] d\boldsymbol{f}_x  \\&= - \int_{\mathcal{F}_m} \frac{1}{Z_m} p(\boldsymbol{f}_x \vert D)log p(\boldsymbol{f}_x \vert D) d\boldsymbol{f}_x + \frac{1}{Z_m} log Z_m \int_{\mathcal{F}_m}  p(\boldsymbol{f}_x \vert D) d\boldsymbol{f}_x\\& = - \frac{1}{Z_m} \int_{\mathcal{F}_m} p(\boldsymbol{f}_x \vert D)log p(\boldsymbol{f}_x \vert D)d\boldsymbol{f}_x + log Z_m
# \end{aligned} \tag{7}
# \end{equation}
# So we have:
# \begin{equation}
# \int_{\mathcal{F}_m} p(\boldsymbol{f}_x \vert D)log p(\boldsymbol{f}_x \vert D)d\boldsymbol{f}_x = Z_mlog Z_m - Z_m\mathbb{H}[\boldsymbol{h}_{x_{\mathcal{F}_m}}]  \tag{8}
# \end{equation}

# We plug in $\boldsymbol{h}_x$ into the original differential entropy expression, i.e., substitute Eq. 8 into Eq. 5:

# \begin{equation}
# \begin{aligned}
# \mathbb{H}[p(\boldsymbol{f}_x \vert D, \boldsymbol{f}_x \prec \mathcal{F^*})] &= -  \frac{1}{Z} \sum_{m=1}^M \int_{\mathcal{F}_m}p(\boldsymbol{f}_x \vert D) log p(\boldsymbol{f}_x \vert D) d \boldsymbol{f}_x + log Z \\ &=   -  \frac{1}{Z} \sum_{m=1}^M \left[ Z_mlog Z_m - Z_m\mathbb{H}[\boldsymbol{h}_{x_{\mathcal{F}_m}}]  \right]+ log Z 
# \end{aligned}\tag{9}
# \end{equation}

# Where $M$ denotes the partitioned cell total number 

# ----------------------

# ### Single Query Point Case (PFES):

# \begin{equation}
# \begin{aligned}
# H[p(\boldsymbol{f}_x \vert D, \boldsymbol{f}_x \prec \mathcal{F^*})] & =  -  \frac{1}{Z} \sum_{m=1}^M \left[ Z_mlog Z_m - Z_m\mathbb{H}[\boldsymbol{h}_{x_{\mathcal{F}_m}}]  \right]+ log Z  \\ & = - \frac{1}{Z} \sum_{m=1}^M \left[ Z_mlog Z_m  + Z_m\int_{l1}^{u1}\int_{l2}^{u2}...\int_{lL}^{uL} p(\boldsymbol{h}_x)logp(\boldsymbol{h}_x)d\boldsymbol{h}_x\right] + log Z\\ & = - \frac{1}{Z} \sum_{m=1}^M \left[ Z_mlog Z_m  + Z_m \int_{l1}^{u1}\int_{l2}^{u2}...\int_{lL}^{uL} \prod_{i=1}^L p(\boldsymbol{h}_{x_i}) \sum_{i=1}^L logp(\boldsymbol{h}_{x_i})d\boldsymbol{h}_x\right] + log Z\\ & = - \frac{1}{Z} \sum_{m=1}^M \left[ Z_m log Z_m   - Z_m \sum_{i=1}^L \left(  \int_{lj}^{uj}\prod_{j\neq i}^{L}p(\boldsymbol{h}_{x_j}) d\boldsymbol{h}_{x_j}\cdot \underbrace{- \int_{li}^{ui}  p(\boldsymbol{h}_{x_i}) log p(\boldsymbol{h}_{x_i})d\boldsymbol{h}_{x_i}}_{\text{entropy of 1d truncated normal}}\right) \right] + log Z\\ & = - \frac{1}{Z} \sum_{m=1}^M \left[ Z_m log Z_m  - Z_m \sum_{i=1}^L \underbrace{\left(log(\sqrt{2 \pi e}\sigma_i Z_{mi}) + \frac{l_i \phi(l_i) - u_i \phi(u_i)}{2Z_{mi}} \right)}_{\text{1d truncated differential entropy}}  \right]+ log Z \\ & = - \frac{1}{Z} \sum_{m=1}^M \left[ Z_m log Z_m  - Z_m \sum_{i=1}^L log(\sqrt{2 \pi e}\sigma_i Z_{mi})  - Z_m \sum_{i=1}^L \frac{l_i \phi(l_i) - u_i \phi(u_i)}{2Z_{mi}} \right]+ log Z \\&=   \sum_{m=1}^M \left[ \frac{Z_m}{Z} \sum_{i=1}^L \frac{l_i \phi(l_i) - u_i \phi(u_i)}{2Z_{mi}} \right]+ log Z - \frac{1}{Z} \sum_{m=1}^M \left[ Z_m log Z_m  - Z_m log({2 \pi e}^\frac{L}{2} Z_m \prod_{i=1}^L \sigma_i )  \right] \\& = \sum_{m=1}^M \left[ \frac{Z_m}{Z} \sum_{i=1}^L \frac{l_i \phi(l_i) - u_i \phi(u_i)}{2Z_{mi}} \right]+ log Z + \frac{1}{Z} \sum_{m=1}^M \left[ Z_m  log ({2 \pi e})^\frac{L}{2} \prod_{i=1}^L \sigma_i  \right] \\& = \sum_{m=1}^M \left[ \frac{Z_m}{Z} \sum_{i=1}^L \frac{l_i \phi(l_i) - u_i \phi(u_i)}{2Z_{mi}} \right]+  log ({2 \pi e})^\frac{L}{2} Z \prod_{i=1}^L \sigma_i 
# \end{aligned} \tag{10}
# \end{equation}

# Where $L$ denotes the objective numbers, $Z_{mi} = \Phi(\frac{u_m^i - \mu_i(x)}{\sigma_i(x)}) - \Phi(\frac{l_m^i - \mu_i(x)}{\sigma_i(x)})$

# ------------------

# ### Batch Case by GIBBON

# In the most simple case, we assume noise free and single fidelity condition. i.e., $C_i$ = $A_i$. 

# \begin{equation}
# \begin{aligned}
# & H[p(\boldsymbol{f}_\boldsymbol{x} \vert D, \boldsymbol{f}_\boldsymbol{x} \prec \mathcal{F^*})] \\ & = -  \frac{1}{Z} \sum_{m=1}^M \left[ Z_mlog Z_m - Z_m\mathbb{H}[\boldsymbol{h}_{x_{\mathcal{F}_m}}]  \right]+ log Z \quad  (Eq. 9)\\& = -  \frac{1}{Z} \sum_{m=1}^M \left[ Z_mlog Z_m - Z_m \sum_{i=1}^L \mathbb{H}[\boldsymbol{f}_{\boldsymbol{x}}^i \vert \boldsymbol{f}_{\boldsymbol{x}}^i \in \mathcal{F}_m^i]  \right]+ log Z \\& = -  \frac{1}{Z} \sum_{m=1}^M (Z_mlog Z_m)+ log Z + \frac{Z_m}{Z}\sum_{m=1}^M \sum_{i=1}^L \mathbb{H}[\boldsymbol{f}_{\boldsymbol{x}}^i \vert \boldsymbol{f}_{\boldsymbol{x}}^i \in \mathcal{F}_m^i] \\ & \\ &\leq -  \frac{1}{Z} \sum_{m=1}^M (Z_mlog Z_m)+ log Z + \frac{Z_m}{Z}\sum_{m=1}^M \sum_{i=1}^L \sum_{j=1}^B \mathbb{H}[\boldsymbol{f}_{x_j}^i \vert \boldsymbol{f}_{\boldsymbol{x}}^i \in \mathcal{F}_m^i]  \quad{\text{information-theoretic inequality}}\\ & = -  \frac{1}{Z} \sum_{m=1}^M (Z_mlog Z_m)+ log Z + \frac{Z_m}{Z}\sum_{m=1}^M \sum_{i=1}^L \sum_{j=1}^B \mathbb{H}[\boldsymbol{f}_{x_j}^i \vert \boldsymbol{f}_{x_j}^i \in \mathcal{F}_m^i]  \quad {\text{conditional independence}} 
# \end{aligned}\tag{11}
# \end{equation}

# -------

# ## Plan
# - Implement PFES
# - Add GIBBON to evaluate the Batch Performance

# -----------------

# # Sample GP Posterior in Weight space approximation by RFF

# ## 1D case

# +
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import gpflow
import gpflux

from gpflow.config import default_float

from gpflux.layers.basis_functions.random_fourier_features import RandomFourierFeatures
from gpflux.sampling import KernelWithFeatureDecomposition
from gpflux.models.deep_gp import sample_dgp


tf.keras.backend.set_floatx("float64")
# -

data = np.genfromtxt("util/data/regression_1D.csv", delimiter=",")
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)
k = gpflow.kernels.RBF()
m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

# Here we manually do the RFF by Mahimi

# $$
# k(X, X^\prime) \approx \sum_{i=1}^I \phi_i(X) \phi_i(X^\prime),
# $$
# with $I$ Fourier features $\phi_i$  following Rahimi and Recht "Random features for large-scale kernel machines" (NeurIPS, 2007) defined as

# $$
# \phi_i(X) = \sqrt{\frac{2 \sigma^2}{l}} \cos(\theta_i X + \tau_i),
# $$
# where $\sigma^2$ refers to the kernel variance and $l$ to the kernel lengthscale. $\theta_i$ and $\tau_i$ are randomly drawn hyperparameters that determine each feature function $\phi_i$. The hyperparameter $\theta_i$ is randomly drawn from the kernel's spectral density. The spectral density of a stationary kernel is obtained by interpreting the kernel as a function of one argument only (i.e. the distance between $X$ and $X^\prime$) and performing a Fourier transform on that function, resulting in an unnormalised probability density (from which samples can be obtained). The hyperparameter $\tau_i$ is obtained by sampling from a uniform distribution $\tau_i \sim \mathcal{U}(0,2\pi)$. Note that both $\theta_i$ and $\tau_i$ are fixed and not optimised over. An interesting direction of future research is how to automatically identify those (but this is outside the scope of this notebook). If we drew infinitely many samples, i.e. $I \rightarrow \infty$, we would recover the true kernel perfectly.

# Below we show how the RFF can be conducted *manually*

# --------------

# $$
# k_{unit}(X, X^\prime) \approx \frac{1}{D}\sum_{i=1}^D z_{{\omega}_i}(X) z_{{\omega}_i}(X^\prime),
# $$
# with $D$ Fourier features $\phi_i$  following Rahimi and Recht "Random features for large-scale kernel machines" (NeurIPS, 2007) defined as

# $$
# z_{\omega_i}(X) = \sqrt{2} \cos(\omega_i' X + b_i),
# $$

# By considering scaling coefficient (i.e., output scaling: kernel variance $\sigma^2$ and input scaling: kernel lengthscale $l$), the augmented kernel could be represented by RFF as:

# \begin{equation}
# \begin{aligned}
# k(X, X^\prime) &\approx \frac{\sigma^2}{D}\sum_{i=1}^D z_{{\omega}_i}(X/l) z_{{\omega}_i}(X^\prime/l) \\& =  \sum_{i=1}^D z_{{\omega}_i}'(X) z_{{\omega}_i}'(X^\prime)
# \end{aligned}
# \end{equation}

# Where $z_{{\omega}_i}'$ is defined as:

# $$
# z_{\omega_i}'(X) = \sqrt{\frac{2\sigma^2}{D}} \cos(\omega' (diag(l)^{-1} X) + b_i)
# $$

# We begin with setting of feature number:

D = 1000

# +
scaler_l = 1/np.diag(np.atleast_1d(m.kernel.lengthscales.numpy())) # [feature num, input dim]

omega = np.random.randn(D, X.shape[1]) # [feature num, input dim]

b = 2 * np.pi * np.random.uniform(0, 1, D) # b \sim U[0, 2pi]

b_for_nObservations = np.array([b, ] * X.shape[0]).T  # Nx * b

sigma2 = m.kernel.variance.numpy()

# z(x) = \sqrt(2a/m) cos(Wx + b)
vector_Z_T = np.sqrt(2 * sigma2 / D) * np.cos(np.matmul(omega, np.matmul(scaler_l, X.T)) + b_for_nObservations)
# -

# Given the RFF approximation (a.k.a, `vector_z_T`), we seek a parametric form of the GP posterior, which has been discussed by Lobato [1] as follows:
# "The feature mapping $\Phi(x)$ allows us to approximate the Gaussian process prior for $f$ with a linear model $f(x) = \phi(x)^T\theta$ where $\theta ∼ \mathcal{N}(0, I)$ is a standard Gaussian. By conditioning on $D_n$, the posterior for $\theta$ is also multivariate
# Gaussian, $\theta|D_n ∼ N(A^{-1}\Phi^T y_n, \sigma_{le}^2 A^{−1})$ where $A = \Phi^T\Phi+ \sigma_{le}^2I$ and $\Phi^T = [\Phi(x_1) . . . \Phi(x_n)]$
# "
#
# Note: 
# 1. The $\Phi^T = [\Phi(x_1) . . . \Phi(x_n)]$ is the same as `vector_Z_T` here
# 2. $\sigma_{le}^2$ represents the likelihood variance

# +
sigma_le_2 = m.likelihood.variance.numpy()
# A: φ(x)^T φ(x)  + σ_{le}^2I
A = np.dot(vector_Z_T, vector_Z_T.T) + sigma_le_2 * np.eye(D)

# posterior of θ sample: θ|Dn ∼ N(A−1ΦTyn, σ2A−1), shape: (m_ftrs), shape checked correct
A_inverse = tf.linalg.inv(A).numpy()

# θ(A^{-1} φ(x)^T Y )
mean_of_post_theta = np.dot(np.dot(A_inverse, vector_Z_T), Y)
mean_of_post_theta = np.squeeze(np.asarray(mean_of_post_theta))
variance_of_post_theta = sigma_le_2 * A_inverse

theta_sample = np.random.multivariate_normal(mean_of_post_theta, variance_of_post_theta)


def makeFunc(kernel_var, m_ftrs, W, b, theta):
    """
    :param kernel_var kernel variance
    :param m_ftrs RFF features, a.k.a value of D
    :param W spectral density sample
    :param b uniform sample from [0, 2pi]
    
    return a sampled approximated posterior trajectory of GP
    """
    return lambda x: np.dot(np.sqrt(2 * kernel_var / D) * np.cos(np.dot(omega, np.matmul(scaler_l, np.atleast_2d(x).T)) 
                                                            + np.atleast_2d(b).T).T, theta)


# -

# Let's take a comparison between GP posterior sample in functional space and the sample from WSA + RFF

# +
## generate test points for prediction
xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)  # test points must be of shape (N, D)

## predict mean and variance of latent GP at test points
mean, var = m.predict_f(xx)

## generate 10 samples from posterior
tf.random.set_seed(1)  # for reproducibility
samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

    
## plot
plt.figure(figsize=(12, 6))
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, mean, "C0", lw=2)
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)

for _ in range(100):
    theta_sample = np.random.multivariate_normal(mean_of_post_theta, variance_of_post_theta)
    f_sample = makeFunc(sigma2, D, omega, b, theta_sample)
    yy = f_sample(xx)
    plt.plot(xx, yy, color="C1")


plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
_ = plt.xlim(-0.1, 1.1)

# -

# --------------

# ## 2D case

# +
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from trieste.utils.objectives import branin
from trieste.space import Box
from util.plotting import plot_gp_2d, plot_function_2d

import gpflow


tf.keras.backend.set_floatx("float64")
# -

Xs_samples = Box([0.0, 0.0], [1.0, 1.0]).sample(23)
X = Xs_samples
Y = branin(X)
k = gpflow.kernels.RBF(lengthscales=[1.0, 1.0])
m_2d = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m_2d.training_loss, m_2d.trainable_variables, options=dict(maxiter=100))

m_2d

plot_gp_2d(m_2d,[0.0, 0.0], [1.0, 1.0])

# -------------------------

D = 1000

# +
scaler_l = np.diag(1/np.atleast_1d(m_2d.kernel.lengthscales.numpy())) # [feature num, input dim]

omega = np.random.randn(D, X.shape[1]) # [feature num, input dim]

b = 2 * np.pi * np.random.uniform(0, 1, D) # b \sim U[0, 2pi]

b_for_nObservations = np.array([b, ] * X.shape[0]).T  # Nx * b

sigma2 = m_2d.kernel.variance.numpy()

# z(x) = \sqrt(2a/m) cos(Wx + b)
vector_Z_T = np.sqrt(2 * sigma2 / D) * np.cos(np.matmul(omega, np.matmul(scaler_l, tf.transpose(X))) + b_for_nObservations)

# +
sigma_le_2 = m_2d.likelihood.variance.numpy()
# A: φ(x)^T φ(x)  + σ_{le}^2I
A = np.dot(vector_Z_T, vector_Z_T.T) + sigma_le_2 * np.eye(D)

# posterior of θ sample: θ|Dn ∼ N(A−1ΦTyn, σ2A−1), shape: (m_ftrs), shape checked correct
A_inverse = tf.linalg.inv(A).numpy()

# θ(A^{-1} φ(x)^T Y )
mean_of_post_theta = np.dot(np.dot(A_inverse, vector_Z_T), Y)
mean_of_post_theta = np.squeeze(np.asarray(mean_of_post_theta))
variance_of_post_theta = sigma_le_2 * A_inverse

theta_sample = np.random.multivariate_normal(mean_of_post_theta, variance_of_post_theta)


def makeFunc(kernel_var, m_ftrs, W, b, theta):
    """
    :param kernel_var kernel variance
    :param m_ftrs RFF features, a.k.a value of D
    :param W spectral density sample
    :param b uniform sample from [0, 2pi]
    
    return a sampled approximated posterior trajectory of GP
    """
    return lambda x: np.dot(np.sqrt(2 * kernel_var / D) * np.cos(np.dot(omega, np.matmul(scaler_l, np.atleast_2d(x).T)) 
                                                            + np.atleast_2d(b).T).T, theta)


# -

for _ in range(3):
    theta_sample = np.random.multivariate_normal(mean_of_post_theta, variance_of_post_theta)
    f_sample = makeFunc(sigma2, D, omega, b, theta_sample)
    plot_function_2d(f_sample, [0.0, 0.0], [1.0, 1.0])
