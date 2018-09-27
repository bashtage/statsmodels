# coding: utf-8

# # Recursive least squares
#
# Recursive least squares is an expanding window version of ordinary least
# squares. In addition to availability of regression coefficients computed
# recursively, the recursively computed residuals the construction of
# statistics to investigate parameter instability.
#
# The `RecursiveLS` class allows computation of recursive residuals and
# computes CUSUM and CUSUM of squares statistics. Plotting these statistics
# along with reference lines denoting statistically significant deviations
# from the null hypothesis of stable parameters allows an easy visual
# indication of parameter stability.
#
# Finally, the `RecursiveLS` model allows imposing linear restrictions on
# the parameter vectors, and can be constructed using the formula interface.

# In[ ]:

# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader

np.set_printoptions(suppress=True)

# ## Example 1: Copper
#
# We first consider parameter stability in the copper dataset (description
# below).

# In[ ]:

print(sm.datasets.copper.DESCRLONG)

dta = sm.datasets.copper.load_pandas().data
dta.index = pd.date_range('1951-01-01', '1975-01-01', freq='AS')
endog = dta['WORLDCONSUMPTION']

# To the regressors in the dataset, we add a column of ones for an
# intercept
exog = sm.add_constant(
    dta[['COPPERPRICE', 'INCOMEINDEX', 'ALUMPRICE', 'INVENTORYINDEX']])

# First, construct and fir the model, and print a summary. Although the
# `RLS` model computes the regression parameters recursively, so there are
# as many estimates as there are datapoints, the summary table only presents
# the regression parameters estimated on the entire sample; except for small
# effects from initialization of the recursiions, these estimates are
# equivalent to OLS estimates.

# In[ ]:

mod = sm.RecursiveLS(endog, exog)
res = mod.fit()

print(res.summary())

# The recursive coefficients are available in the `recursive_coefficients`
# attribute. Alternatively, plots can generated using the
# `plot_recursive_coefficient` method.

# In[ ]:

print(res.recursive_coefficients.filtered[0])
res.plot_recursive_coefficient(
    range(mod.k_exog), alpha=None, figsize=(10, 6))

# The CUSUM statistic is available in the `cusum` attribute, but usually
# it is more convenient to visually check for parameter stability using the
# `plot_cusum` method. In the plot below, the CUSUM statistic does not move
# outside of the 5% significance bands, so we fail to reject the null
# hypothesis of stable parameters at the 5% level.

# In[ ]:

print(res.cusum)
fig = res.plot_cusum()

# Another related statistic is the CUSUM of squares. It is available in
# the `cusum_squares` attribute, but it is similarly more convenient to
# check it visually, using the `plot_cusum_squares` method. In the plot
# below, the CUSUM of squares statistic does not move outside of the 5%
# significance bands, so we fail to reject the null hypothesis of stable
# parameters at the 5% level.

# In[ ]:

res.plot_cusum_squares()

# # Example 2: Quantity theory of money
#
# The quantity theory of money suggests that "a given change in the rate
# of change in the quantity of money induces ... an equal change in the rate
# of price inflation" (Lucas, 1980). Following Lucas, we examine the
# relationship between double-sided exponentially weighted moving averages
# of money growth and CPI inflation. Although Lucas found the relationship
# between these variables to be stable, more recently it appears that the
# relationship is unstable; see e.g. Sargent and Surico (2010).

# In[ ]:

start = '1959-12-01'
end = '2015-01-01'
m2 = DataReader('M2SL', 'fred', start=start, end=end)
cpi = DataReader('CPIAUCSL', 'fred', start=start, end=end)

# In[ ]:


def ewma(series, beta, n_window):
    nobs = len(series)
    scalar = (1 - beta) / (1 + beta)
    ma = []
    k = np.arange(n_window, 0, -1)
    weights = np.r_[beta**k, 1, beta**k[::-1]]
    for t in range(n_window, nobs - n_window):
        window = series.iloc[t - n_window:t + n_window + 1].values
        ma.append(scalar * np.sum(weights * window))
    return pd.Series(
        ma, name=series.name, index=series.iloc[n_window:-n_window].index)


m2_ewma = ewma(
    np.log(m2['M2SL'].resample('QS').mean()).diff().iloc[1:], 0.95, 10 * 4)
cpi_ewma = ewma(
    np.log(cpi['CPIAUCSL'].resample('QS').mean()).diff().iloc[1:], 0.95,
    10 * 4)

# After constructing the moving averages using the $\beta = 0.95$ filter
# of Lucas (with a window of 10 years on either side), we plot each of the
# series below. Although they appear to move together prior for part of the
# sample, after 1990 they appear to diverge.

# In[ ]:

fig, ax = plt.subplots(figsize=(13, 3))

ax.plot(m2_ewma, label='M2 Growth (EWMA)')
ax.plot(cpi_ewma, label='CPI Inflation (EWMA)')
ax.legend()

# In[ ]:

endog = cpi_ewma
exog = sm.add_constant(m2_ewma)
exog.columns = ['const', 'M2']

mod = sm.RecursiveLS(endog, exog)
res = mod.fit()

print(res.summary())

# In[ ]:

res.plot_recursive_coefficient(
    1, alpha=None)

# The CUSUM plot now shows subtantial deviation at the 5% level,
# suggesting a rejection of the null hypothesis of parameter stability.

# In[ ]:

res.plot_cusum()

# Similarly, the CUSUM of squares shows subtantial deviation at the 5%
# level, also suggesting a rejection of the null hypothesis of parameter
# stability.

# In[ ]:

res.plot_cusum_squares()

# # Example 3: Linear restrictions and formulas

# ### Linear restrictions
#
# It is not hard to implement linear restrictions, using the `constraints`
# parameter in constructing the model.

# In[ ]:

endog = dta['WORLDCONSUMPTION']
exog = sm.add_constant(
    dta[['COPPERPRICE', 'INCOMEINDEX', 'ALUMPRICE', 'INVENTORYINDEX']])

mod = sm.RecursiveLS(endog, exog, constraints='COPPERPRICE = ALUMPRICE')
res = mod.fit()
print(res.summary())

# ### Formula
#
# One could fit the same model using the class method `from_formula`.

# In[ ]:

mod = sm.RecursiveLS.from_formula(
    'WORLDCONSUMPTION ~ COPPERPRICE + INCOMEINDEX + ALUMPRICE + INVENTORYINDEX',
    dta,
    constraints='COPPERPRICE = ALUMPRICE')
res = mod.fit()
print(res.summary())
