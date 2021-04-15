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
# - [MESMOC+](https://arxiv.org/pdf/2011.01150.pdf) paper, Daniel Fernández-Sánchez (Daniel Hernández-Lobato)
#
# Uncatogrized
# - [iMOCA](https://arxiv.org/pdf/2009.05700.pdf) paper, NIPS 2020 Workshop, Belakaria et al

# -----------

# ## Main

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
