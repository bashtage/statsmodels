# coding: utf-8

# ## Markov switching autoregression models

# This notebook provides an example of the use of Markov switching models
# in Statsmodels to replicate a number of results presented in Kim and
# Nelson (1999). It applies the Hamilton (1989) filter the Kim (1994)
# smoother.
#
# This is tested against the Markov-switching models from E-views 8, which
# can be found at http://www.eviews.com/EViews8/ev8ecswitch_n.html#MarkovAR
# or the Markov-switching models of Stata 14 which can be found at
# http://www.stata.com/manuals14/tsmswitch.pdf.

# In[ ]:

# get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# NBER recessions
from pandas_datareader.data import DataReader
from datetime import datetime
usrec = DataReader(
    'USREC', 'fred', start=datetime(1947, 1, 1), end=datetime(2013, 4, 1))

# ### Hamilton (1989) switching model of GNP
#
# This replicates Hamilton's (1989) seminal paper introducing Markov-
# switching models. The model is an autoregressive model of order 4 in which
# the mean of the process switches between two regimes. It can be written:
#
# $$
# y_t = \mu_{S_t} + \phi_1 (y_{t-1} - \mu_{S_{t-1}}) + \phi_2 (y_{t-2} -
# \mu_{S_{t-2}}) + \phi_3 (y_{t-3} - \mu_{S_{t-3}}) + \phi_4 (y_{t-4} -
# \mu_{S_{t-4}}) + \varepsilon_t
# $$
#
# Each period, the regime transitions according to the following matrix of
# transition probabilities:
#
# $$ P(S_t = s_t | S_{t-1} = s_{t-1}) =
# \begin{bmatrix}
# p_{00} & p_{10} \\
# p_{01} & p_{11}
# \end{bmatrix}
# $$
#
# where $p_{ij}$ is the probability of transitioning *from* regime $i$,
# *to* regime $j$.
#
# The model class is `MarkovAutoregression` in the time-series part of
# `Statsmodels`. In order to create the model, we must specify the number of
# regimes with `k_regimes=2`, and the order of the autoregression with
# `order=4`. The default model also includes switching autoregressive
# coefficients, so here we also need to specify `switching_ar=False` to
# avoid that.
#
# After creation, the model is `fit` via maximum likelihood estimation.
# Under the hood, good starting parameters are found using a number of steps
# of the expectation maximization (EM) algorithm, and a quasi-Newton (BFGS)
# algorithm is applied to quickly find the maximum.

# In[ ]:

# Get the RGNP data to replicate Hamilton
dta = pd.read_stata('https://www.stata-press.com/data/r14/rgnp.dta').iloc[1:]
dta.index = pd.DatetimeIndex(dta.date, freq='QS')
dta_hamilton = dta.rgnp

# Plot the data
dta_hamilton.plot(title='Growth rate of Real GNP', figsize=(12, 3))

# Fit the model
mod_hamilton = sm.tsa.MarkovAutoregression(
    dta_hamilton, k_regimes=2, order=4, switching_ar=False)
res_hamilton = mod_hamilton.fit()

# In[ ]:

res_hamilton.summary()

# We plot the filtered and smoothed probabilities of a recession. Filtered
# refers to an estimate of the probability at time $t$ based on data up to
# and including time $t$ (but excluding time $t+1, ..., T$). Smoothed refers
# to an estimate of the probability at time $t$ using all the data in the
# sample.
#
# For reference, the shaded periods represent the NBER recessions.

# In[ ]:

fig, axes = plt.subplots(2, figsize=(7, 7))
ax = axes[0]
ax.plot(res_hamilton.filtered_marginal_probabilities[0])
ax.fill_between(
    usrec.index, 0, 1, where=usrec['USREC'].values, color='k', alpha=0.1)
ax.set_xlim(dta_hamilton.index[4], dta_hamilton.index[-1])
ax.set(title='Filtered probability of recession')

ax = axes[1]
ax.plot(res_hamilton.smoothed_marginal_probabilities[0])
ax.fill_between(
    usrec.index, 0, 1, where=usrec['USREC'].values, color='k', alpha=0.1)
ax.set_xlim(dta_hamilton.index[4], dta_hamilton.index[-1])
ax.set(title='Smoothed probability of recession')

fig.tight_layout()

# From the estimated transition matrix we can calculate the expected
# duration of a recession versus an expansion.

# In[ ]:

print(res_hamilton.expected_durations)

# In this case, it is expected that a recession will last about one year
# (4 quarters) and an expansion about two and a half years.

# ### Kim, Nelson, and Startz (1998) Three-state Variance Switching
#
# This model demonstrates estimation with regime heteroskedasticity
# (switching of variances) and no mean effect. The dataset can be reached at
# http://econ.korea.ac.kr/~cjkim/MARKOV/data/ew_excs.prn.
#
# The model in question is:
#
# $$
# \begin{align}
# y_t & = \varepsilon_t \\
# \varepsilon_t & \sim N(0, \sigma_{S_t}^2)
# \end{align}
# $$
#
# Since there is no autoregressive component, this model can be fit using
# the `MarkovRegression` class. Since there is no mean effect, we specify
# `trend='nc'`. There are hypotheized to be three regimes for the switching
# variances, so we specify `k_regimes=3` and `switching_variance=True` (by
# default, the variance is assumed to be the same across regimes).

# In[ ]:

# Get the dataset
ew_excs = requests.get(
    'http://econ.korea.ac.kr/~cjkim/MARKOV/data/ew_excs.prn').content
raw = pd.read_table(
    BytesIO(ew_excs), header=None, skipfooter=1, engine='python')
raw.index = pd.date_range('1926-01-01', '1995-12-01', freq='MS')

dta_kns = raw.loc[:'1986'] - raw.loc[:'1986'].mean()

# Plot the dataset
dta_kns[0].plot(title='Excess returns', figsize=(12, 3))

# Fit the model
mod_kns = sm.tsa.MarkovRegression(
    dta_kns, k_regimes=3, trend='nc', switching_variance=True)
res_kns = mod_kns.fit()

# In[ ]:

res_kns.summary()

# Below we plot the probabilities of being in each of the regimes; only in
# a few periods is a high-variance regime probable.

# In[ ]:

fig, axes = plt.subplots(3, figsize=(10, 7))

ax = axes[0]
ax.plot(res_kns.smoothed_marginal_probabilities[0])
ax.set(title='Smoothed probability of a low-variance regime for stock returns')

ax = axes[1]
ax.plot(res_kns.smoothed_marginal_probabilities[1])
ax.set(
    title='Smoothed probability of a medium-variance regime for stock returns')

ax = axes[2]
ax.plot(res_kns.smoothed_marginal_probabilities[2])
ax.set(
    title='Smoothed probability of a high-variance regime for stock returns')

fig.tight_layout()

# ### Filardo (1994) Time-Varying Transition Probabilities
#
# This model demonstrates estimation with time-varying transition
# probabilities. The dataset can be reached at
# http://econ.korea.ac.kr/~cjkim/MARKOV/data/filardo.prn.
#
# In the above models we have assumed that the transition probabilities
# are constant across time. Here we allow the probabilities to change with
# the state of the economy. Otherwise, the model is the same Markov
# autoregression of Hamilton (1989).
#
# Each period, the regime now transitions according to the following
# matrix of time-varying transition probabilities:
#
# $$ P(S_t = s_t | S_{t-1} = s_{t-1}) =
# \begin{bmatrix}
# p_{00,t} & p_{10,t} \\
# p_{01,t} & p_{11,t}
# \end{bmatrix}
# $$
#
# where $p_{ij,t}$ is the probability of transitioning *from* regime $i$,
# *to* regime $j$ in period $t$, and is defined to be:
#
# $$
# p_{ij,t} = \frac{\exp\{ x_{t-1}' \beta_{ij} \}}{1 + \exp\{ x_{t-1}'
# \beta_{ij} \}}
# $$
#
# Instead of estimating the transition probabilities as part of maximum
# likelihood, the regression coefficients $\beta_{ij}$ are estimated. These
# coefficients relate the transition probabilities to a vector of pre-
# determined or exogenous regressors $x_{t-1}$.

# In[ ]:

# Get the dataset
filardo = requests.get(
    'http://econ.korea.ac.kr/~cjkim/MARKOV/data/filardo.prn').content
dta_filardo = pd.read_table(
    BytesIO(filardo), sep=' +', header=None, skipfooter=1, engine='python')
dta_filardo.columns = ['month', 'ip', 'leading']
dta_filardo.index = pd.date_range('1948-01-01', '1991-04-01', freq='MS')

dta_filardo['dlip'] = np.log(dta_filardo['ip']).diff() * 100
# Deflated pre-1960 observations by ratio of std. devs.
# See hmt_tvp.opt or Filardo (1994) p. 302
std_ratio = dta_filardo['dlip']['1960-01-01':].std(
) / dta_filardo['dlip'][:'1959-12-01'].std()
dta_filardo[
    'dlip'][:'1959-12-01'] = dta_filardo['dlip'][:'1959-12-01'] * std_ratio

dta_filardo['dlleading'] = np.log(dta_filardo['leading']).diff() * 100
dta_filardo['dmdlleading'] = dta_filardo['dlleading'] - dta_filardo[
    'dlleading'].mean()

# Plot the data
dta_filardo['dlip'].plot(
    title='Standardized growth rate of industrial production', figsize=(13, 3))
plt.figure()
dta_filardo['dmdlleading'].plot(
    title='Leading indicator', figsize=(13, 3))

# The time-varying transition probabilities are specified by the
# `exog_tvtp` parameter.
#
# Here we demonstrate another feature of model fitting - the use of a
# random search for MLE starting parameters. Because Markov switching models
# are often characterized by many local maxima of the likelihood function,
# performing an initial optimization step can be helpful to find the best
# parameters.
#
# Below, we specify that 20 random perturbations from the starting
# parameter vector are examined and the best one used as the actual starting
# parameters. Because of the random nature of the search, we seed the random
# number generator beforehand to allow replication of the result.

# In[ ]:

mod_filardo = sm.tsa.MarkovAutoregression(
    dta_filardo.iloc[2:]['dlip'],
    k_regimes=2,
    order=4,
    switching_ar=False,
    exog_tvtp=sm.add_constant(dta_filardo.iloc[1:-1]['dmdlleading']))

np.random.seed(12345)
res_filardo = mod_filardo.fit(search_reps=20)

# In[ ]:

res_filardo.summary()

# Below we plot the smoothed probability of the economy operating in a
# low-production state, and again include the NBER recessions for
# comparison.

# In[ ]:

fig, ax = plt.subplots(figsize=(12, 3))

ax.plot(res_filardo.smoothed_marginal_probabilities[0])
ax.fill_between(
    usrec.index, 0, 1, where=usrec['USREC'].values, color='gray', alpha=0.2)
ax.set_xlim(dta_filardo.index[6], dta_filardo.index[-1])
ax.set(title='Smoothed probability of a low-production state')

# Using the time-varying transition probabilities, we can see how the
# expected duration of a low-production state changes over time:
#

# In[ ]:

res_filardo.expected_durations[0].plot(
    title='Expected duration of a low-production state', figsize=(12, 3))

# During recessions, the expected duration of a low-production state is
# much higher than in an expansion.
