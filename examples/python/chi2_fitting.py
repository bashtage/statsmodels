# coding: utf-8

# # Least squares fitting of models to data

# This is a quick introduction to `statsmodels` for physical scientists
# (e.g. physicists, astronomers) or engineers.
#
# Why is this needed?
#
# Because most of `statsmodels` was written by statisticians and they use
# a different terminology and sometimes methods, making it hard to know
# which classes and functions are relevant and what their inputs and outputs
# mean.

# In[ ]:

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ## Linear models

# Assume you have data points with measurements `y` at positions `x` as
# well as measurement errors `y_err`.
#
# How can you use `statsmodels` to fit a straight line model to this data?
#
# For an extensive discussion see [Hogg et al. (2010), "Data analysis
# recipes: Fitting a model to data"](http://arxiv.org/abs/1008.4686) ...
# we'll use the example data given by them in Table 1.
#
# So the model is `f(x) = a * x + b` and on Figure 1 they print the result
# we want to reproduce ... the best-fit parameter and the parameter errors
# for a "standard weighted least-squares fit" for this data are:
# * `a = 2.24 +- 0.11`
# * `b = 34 +- 18`

# In[ ]:

data = """
  x   y y_err
201 592    61
244 401    25
 47 583    38
287 402    15
203 495    21
 58 173    15
210 479    27
202 504    14
198 510    30
158 416    16
165 393    14
201 442    25
157 317    52
131 311    16
166 400    34
160 337    31
186 423    42
125 334    26
218 533    16
146 344    22
"""
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
data = pd.read_csv(StringIO(data), delim_whitespace=True).astype(float)

# Note: for the results we compare with the paper here, they drop the
# first four points
data.head()

# To fit a straight line use the weighted least squares class [WLS](https:
# //www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.
# WLS.html) ... the parameters are called:
# * `exog` = `sm.add_constant(x)`
# * `endog` = `y`
# * `weights` = `1 / sqrt(y_err)`
#
# Note that `exog` must be a 2-dimensional array with `x` as a column and
# an extra column of ones. Adding this column of ones means you want to fit
# the model `y = a * x + b`, leaving it off means you want to fit the model
# `y = a * x`.
#
# And you have to use the option `cov_type='fixed scale'` to tell
# `statsmodels` that you really have measurement errors with an absolute
# scale. If you don't, `statsmodels` will treat the weights as relative
# weights between the data points and internally re-scale them so that the
# best-fit model will have `chi**2 / ndf = 1`.

# In[ ]:

exog = sm.add_constant(data['x'])
endog = data['y']
weights = 1. / (data['y_err']**2)
wls = sm.WLS(endog, exog, weights)
results = wls.fit(cov_type='fixed scale')
print(results.summary())

# ### Check against scipy.optimize.curve_fit

# In[ ]:

# You can use `scipy.optimize.curve_fit` to get the best-fit parameters
# and parameter errors.
from scipy.optimize import curve_fit


def f(x, a, b):
    return a * x + b


xdata = data['x']
ydata = data['y']
p0 = [0, 0]  # initial parameter estimate
sigma = data['y_err']
popt, pcov = curve_fit(f, xdata, ydata, p0, sigma, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
print('a = {0:10.3f} +- {1:10.3f}'.format(popt[0], perr[0]))
print('b = {0:10.3f} +- {1:10.3f}'.format(popt[1], perr[1]))

# ### Check against self-written cost function

# In[ ]:

# You can also use `scipy.optimize.minimize` and write your own cost
# function.
# This doesn't give you the parameter errors though ... you'd have
# to estimate the HESSE matrix separately ...
from scipy.optimize import minimize


def chi2(pars):
    """Cost function.
    """
    y_model = pars[0] * data['x'] + pars[1]
    chi = (data['y'] - y_model) / data['y_err']
    return np.sum(chi**2)


result = minimize(fun=chi2, x0=[0, 0])
popt = result.x
print('a = {0:10.3f}'.format(popt[0]))
print('b = {0:10.3f}'.format(popt[1]))

# ## Non-linear models

# In[ ]:

# TODO: we could use the examples from here:
# http://probfit.readthedocs.org/en/latest/api.html#probfit.costfunc.Chi2R
# egression
