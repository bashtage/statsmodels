# coding: utf-8

# # Quantile regression

#
# This example page shows how to use ``statsmodels``' ``QuantReg`` class
# to replicate parts of the analysis published in
#
# * Koenker, Roger and Kevin F. Hallock. "Quantile Regressioin". Journal
# of Economic Perspectives, Volume 15, Number 4, Fall 2001, Pages 143–156
#
# We are interested in the relationship between income and expenditures on
# food for a sample of working class Belgian households in 1857 (the Engel
# data).
#
# ## Setup
#
# We first need to load some modules and to retrieve the data.
# Conveniently, the Engel dataset is shipped with ``statsmodels``.

# In[ ]:

# get_ipython().run_line_magic('matplotlib', 'inline')

from __future__ import print_function
import patsy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.regression.quantile_regression import QuantReg

data = sm.datasets.engel.load_pandas().data
data.head()

# ## Least Absolute Deviation
#
# The LAD model is a special case of quantile regression where q=0.5

# In[ ]:

mod = smf.quantreg('foodexp ~ income', data)
res = mod.fit(q=.5)
print(res.summary())

# ## Visualizing the results
#
# We estimate the quantile regression model for many quantiles between .05
# and .95, and compare best fit line from each of these models to Ordinary
# Least Squares results.

# ### Prepare data for plotting
#
# For convenience, we place the quantile regression results in a Pandas
# DataFrame, and the OLS results in a dictionary.

# In[ ]:

quantiles = np.arange(.05, .96, .1)


def fit_model(q):
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['income']
            ] + res.conf_int().loc['income'].tolist()


models = [fit_model(x) for x in quantiles]
models = pd.DataFrame(models, columns=['q', 'a', 'b', 'lb', 'ub'])

ols = smf.ols('foodexp ~ income', data).fit()
ols_ci = ols.conf_int().loc['income'].tolist()
ols = dict(
    a=ols.params['Intercept'],
    b=ols.params['income'],
    lb=ols_ci[0],
    ub=ols_ci[1])

print(models)
print(ols)

# ### First plot
#
# This plot compares best fit lines for 10 quantile regression models to
# the least squares fit. As Koenker and Hallock (2001) point out, we see
# that:
#
# 1. Food expenditure increases with income
# 2. The *dispersion* of food expenditure increases with income
# 3. The least squares estimates fit low income observations quite poorly
# (i.e. the OLS line passes over most low income households)

# In[ ]:

x = np.arange(data.income.min(), data.income.max(), 50)
get_y = lambda a, b: a + b * x

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(models.shape[0]):
    y = get_y(models.a[i], models.b[i])
    ax.plot(x, y, linestyle='dotted', color='grey')

y = get_y(ols['a'], ols['b'])

ax.plot(x, y, color='red', label='OLS')
ax.scatter(data.income, data.foodexp, alpha=.2)
ax.set_xlim((240, 3000))
ax.set_ylim((240, 2000))
legend = ax.legend()
ax.set_xlabel('Income', fontsize=16)
ax.set_ylabel(
    'Food expenditure', fontsize=16)

# ### Second plot
#
# The dotted black lines form 95% point-wise confidence band around 10
# quantile regression estimates (solid black line). The red lines represent
# OLS regression results along with their 95% confindence interval.
#
# In most cases, the quantile regression point estimates lie outside the
# OLS confidence interval, which suggests that the effect of income on food
# expenditure may not be constant across the distribution.

# In[ ]:

n = models.shape[0]
p1 = plt.plot(models.q, models.b, color='black', label='Quantile Reg.')
p2 = plt.plot(models.q, models.ub, linestyle='dotted', color='black')
p3 = plt.plot(models.q, models.lb, linestyle='dotted', color='black')
p4 = plt.plot(models.q, [ols['b']] * n, color='red', label='OLS')
p5 = plt.plot(models.q, [ols['lb']] * n, linestyle='dotted', color='red')
p6 = plt.plot(models.q, [ols['ub']] * n, linestyle='dotted', color='red')
plt.ylabel(r'$\beta_{income}$')
plt.xlabel('Quantiles of the conditional food expenditure distribution')
plt.legend()
plt.show()
