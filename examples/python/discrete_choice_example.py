# coding: utf-8

# # Discrete Choice Models

# ## Fair's Affair data

# A survey of women only was conducted in 1974 by *Redbook* asking about
# extramarital affairs.

# In[ ]:

# get_ipython().run_line_magic('matplotlib', 'inline')

from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import logit, probit, poisson, ols

# In[ ]:

print(sm.datasets.fair.SOURCE)

# In[ ]:

print(sm.datasets.fair.NOTE)

# In[ ]:

dta = sm.datasets.fair.load_pandas().data

# In[ ]:

dta['affair'] = (dta['affairs'] > 0).astype(float)
print(dta.head(10))

# In[ ]:

print(dta.describe())

# In[ ]:

affair_mod = logit(
    "affair ~ occupation + educ + occupation_husb"
    "+ rate_marriage + age + yrs_married + children"
    " + religious", dta).fit()

# In[ ]:

print(affair_mod.summary())

# How well are we predicting?

# In[ ]:

affair_mod.pred_table()

# The coefficients of the discrete choice model do not tell us much. What
# we're after is marginal effects.

# In[ ]:

mfx = affair_mod.get_margeff()
print(mfx.summary())

# In[ ]:

respondent1000 = dta.iloc[1000]
print(respondent1000)

# In[ ]:

resp = dict(
    zip(
        range(1, 9), respondent1000[[
            "occupation", "educ", "occupation_husb", "rate_marriage", "age",
            "yrs_married", "children", "religious"
        ]].tolist()))
resp.update({0: 1})
print(resp)

# In[ ]:

mfx = affair_mod.get_margeff(atexog=resp)
print(mfx.summary())

# `predict` expects a `DataFrame` since `patsy` is used to select columns.

# In[ ]:

respondent1000 = dta.iloc[[1000]]
affair_mod.predict(respondent1000)

# In[ ]:

affair_mod.fittedvalues[1000]

# In[ ]:

affair_mod.model.cdf(affair_mod.fittedvalues[1000])

# The "correct" model here is likely the Tobit model. We have an work in
# progress branch "tobit-model" on github, if anyone is interested in
# censored regression models.

# ### Exercise: Logit vs Probit

# In[ ]:

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
support = np.linspace(-6, 6, 1000)
ax.plot(support, stats.logistic.cdf(support), 'r-', label='Logistic')
ax.plot(support, stats.norm.cdf(support), label='Probit')
ax.legend()

# In[ ]:

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
support = np.linspace(-6, 6, 1000)
ax.plot(support, stats.logistic.pdf(support), 'r-', label='Logistic')
ax.plot(support, stats.norm.pdf(support), label='Probit')
ax.legend()

# Compare the estimates of the Logit Fair model above to a Probit model.
# Does the prediction table look better? Much difference in marginal
# effects?

# ### Genarlized Linear Model Example

# In[ ]:

print(sm.datasets.star98.SOURCE)

# In[ ]:

print(sm.datasets.star98.DESCRLONG)

# In[ ]:

print(sm.datasets.star98.NOTE)

# In[ ]:

dta = sm.datasets.star98.load_pandas().data
print(dta.columns)

# In[ ]:

print(dta[[
    'NABOVE', 'NBELOW', 'LOWINC', 'PERASIAN', 'PERBLACK', 'PERHISP', 'PERMINTE'
]].head(10))

# In[ ]:

print(dta[[
    'AVYRSEXP', 'AVSALK', 'PERSPENK', 'PTRATIO', 'PCTAF', 'PCTCHRT', 'PCTYRRND'
]].head(10))

# In[ ]:

formula = 'NABOVE + NBELOW ~ LOWINC + PERASIAN + PERBLACK + PERHISP + PCTCHRT '
formula += '+ PCTYRRND + PERMINTE*AVYRSEXP*AVSALK + PERSPENK*PTRATIO*PCTAF'

# #### Aside: Binomial distribution

# Toss a six-sided die 5 times, what's the probability of exactly 2 fours?

# In[ ]:

stats.binom(5, 1. / 6).pmf(2)

# In[ ]:

from scipy.misc import comb
comb(5, 2) * (1 / 6.)**2 * (5 / 6.)**3

# In[ ]:

from statsmodels.formula.api import glm
glm_mod = glm(formula, dta, family=sm.families.Binomial()).fit()

# In[ ]:

print(glm_mod.summary())

# The number of trials

# In[ ]:

glm_mod.model.data.orig_endog.sum(1)

# In[ ]:

glm_mod.fittedvalues * glm_mod.model.data.orig_endog.sum(1)

# First differences: We hold all explanatory variables constant at their
# means and manipulate the percentage of low income households to assess its
# impact
# on the response variables:

# In[ ]:

exog = glm_mod.model.data.orig_exog  # get the dataframe

# In[ ]:

means25 = exog.mean()
print(means25)

# In[ ]:

means25['LOWINC'] = exog['LOWINC'].quantile(.25)
print(means25)

# In[ ]:

means75 = exog.mean()
means75['LOWINC'] = exog['LOWINC'].quantile(.75)
print(means75)

# Again, `predict` expects a `DataFrame` since `patsy` is used to select
# columns.

# In[ ]:

resp25 = glm_mod.predict(pd.DataFrame(means25).T)
resp75 = glm_mod.predict(pd.DataFrame(means75).T)
diff = resp75 - resp25

# The interquartile first difference for the percentage of low income
# households in a school district is:

# In[ ]:

print("%2.4f%%" % (diff[0] * 100))

# In[ ]:

nobs = glm_mod.nobs
y = glm_mod.model.endog
yhat = glm_mod.mu

# In[ ]:

from statsmodels.graphics.api import abline_plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, ylabel='Observed Values', xlabel='Fitted Values')
ax.scatter(yhat, y)
y_vs_yhat = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
fig = abline_plot(model_results=y_vs_yhat, ax=ax)

# #### Plot fitted values vs Pearson residuals

# Pearson residuals are defined to be
#
# $$\frac{(y - \mu)}{\sqrt{(var(\mu))}}$$
#
# where var is typically determined by the family. E.g., binomial variance
# is $np(1 - p)$

# In[ ]:

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(
    111,
    title='Residual Dependence Plot',
    xlabel='Fitted Values',
    ylabel='Pearson Residuals')
ax.scatter(yhat, stats.zscore(glm_mod.resid_pearson))
ax.axis('tight')
ax.plot([0.0, 1.0], [0.0, 0.0], 'k-')

# #### Histogram of standardized deviance residuals with Kernel Density
# Estimate overlayed

# The definition of the deviance residuals depends on the family. For the
# Binomial distribution this is
#
# $$r_{dev} = sign\left(Y-\mu\right)*\sqrt{2n(Y\log\frac{Y}{\mu}+(1-Y)\log
# \frac{(1-Y)}{(1-\mu)}}$$
#
# They can be used to detect ill-fitting covariates

# In[ ]:

resid = glm_mod.resid_deviance
resid_std = stats.zscore(resid)
kde_resid = sm.nonparametric.KDEUnivariate(resid_std)
kde_resid.fit()

# In[ ]:

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, title="Standardized Deviance Residuals")
ax.hist(
    resid_std, bins=25, normed=True)
ax.plot(kde_resid.support, kde_resid.density, 'r')

# #### QQ-plot of deviance residuals

# In[ ]:

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
fig = sm.graphics.qqplot(resid, line='r', ax=ax)
