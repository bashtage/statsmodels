# coding: utf-8

# # Autoregressive Moving Average (ARMA): Sunspots data

# This notebook replicates the existing ARMA notebook using the
# `statsmodels.tsa.statespace.SARIMAX` class rather than the
# `statsmodels.tsa.ARMA` class.

# In[ ]:

# get_ipython().run_line_magic('matplotlib', 'inline')

# In[ ]:

from __future__ import print_function
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

# In[ ]:

from statsmodels.graphics.api import qqplot

# ## Sunpots Data

# In[ ]:

print(sm.datasets.sunspots.NOTE)

# In[ ]:

dta = sm.datasets.sunspots.load_pandas().data

# In[ ]:

dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]

# In[ ]:

dta.plot(figsize=(12, 4))

# In[ ]:

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)

# In[ ]:

arma_mod20 = sm.tsa.statespace.SARIMAX(
    dta, order=(2, 0, 0), trend='c').fit(disp=False)
print(arma_mod20.params)

# In[ ]:

arma_mod30 = sm.tsa.statespace.SARIMAX(
    dta, order=(3, 0, 0), trend='c').fit(disp=False)

# In[ ]:

print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)

# In[ ]:

print(arma_mod30.params)

# In[ ]:

print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)

# * Does our model obey the theory?

# In[ ]:

sm.stats.durbin_watson(arma_mod30.resid)

# In[ ]:

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(111)
ax = plt.plot(arma_mod30.resid)

# In[ ]:

resid = arma_mod30.resid

# In[ ]:

stats.normaltest(resid)

# In[ ]:

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)

# In[ ]:

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

# In[ ]:

r, q, p = sm.tsa.acf(resid, qstat=True)
data = np.c_[range(1, 41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

# * This indicates a lack of fit.

# * In-sample dynamic prediction. How good does our model do?

# In[ ]:

predict_sunspots = arma_mod30.predict(start='1990', end='2012', dynamic=True)

# In[ ]:

fig, ax = plt.subplots(figsize=(12, 8))
dta.loc['1950':].plot(ax=ax)
predict_sunspots.plot(
    ax=ax, style='r')

# In[ ]:


def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()


# In[ ]:

mean_forecast_err(dta.SUNACTIVITY, predict_sunspots)
