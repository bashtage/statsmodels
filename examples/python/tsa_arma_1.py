# coding: utf-8

# # Autoregressive Moving Average (ARMA): Artificial data

# In[ ]:

# get_ipython().run_line_magic('matplotlib', 'inline')

from __future__ import print_function
import numpy as np
import statsmodels.api as sm
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(12345)

# Generate some data from an ARMA process:

# In[ ]:

arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])

# The conventions of the arma_generate function require that we specify a
# 1 for the zero-lag of the AR and MA parameters and that the AR parameters
# be negated.

# In[ ]:

arparams = np.r_[1, -arparams]
maparams = np.r_[1, maparams]
nobs = 250
y = arma_generate_sample(arparams, maparams, nobs)

#  Now, optionally, we can add some dates information. For this example,
# we'll use a pandas time series.

# In[ ]:

dates = sm.tsa.datetools.dates_from_range('1980m1', length=nobs)
y = pd.Series(y, index=dates)
arma_mod = sm.tsa.ARMA(y, order=(2, 2))
arma_res = arma_mod.fit(trend='nc', disp=-1)

# In[ ]:

print(arma_res.summary())

# In[ ]:

y.tail()

# In[ ]:

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
fig = arma_res.plot_predict(start='1999-06-30', end='2001-05-31', ax=ax)
legend = ax.legend(loc='upper left')
