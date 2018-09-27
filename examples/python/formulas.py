# coding: utf-8

# # Formulas: Fitting models using R-style formulas

# Since version 0.5.0, ``statsmodels`` allows users to fit statistical
# models using R-style formulas. Internally, ``statsmodels`` uses the
# [patsy](http://patsy.readthedocs.org/) package to convert formulas and
# data to the matrices that are used in model fitting. The formula framework
# is quite powerful; this tutorial only scratches the surface. A full
# description of the formula language can be found in the ``patsy`` docs:
#
# * [Patsy formula language description](http://patsy.readthedocs.org/)
#
# ## Loading modules and functions

# In[ ]:

from __future__ import print_function
import numpy as np
import statsmodels.api as sm

# #### Import convention

# You can import explicitly from statsmodels.formula.api

# In[ ]:

from statsmodels.formula.api import ols

# Alternatively, you can just use the `formula` namespace of the main
# `statsmodels.api`.

# In[ ]:

sm.formula.ols

# Or you can use the following conventioin

# In[ ]:

import statsmodels.formula.api as smf

# These names are just a convenient way to get access to each model's
# `from_formula` classmethod. See, for instance

# In[ ]:

sm.OLS.from_formula

# All of the lower case models accept ``formula`` and ``data`` arguments,
# whereas upper case ones take ``endog`` and ``exog`` design matrices.
# ``formula`` accepts a string which describes the model in terms of a
# ``patsy`` formula. ``data`` takes a [pandas](http://pandas.pydata.org/)
# data frame or any other data structure that defines a ``__getitem__`` for
# variable names like a structured array or a dictionary of variables.
#
# ``dir(sm.formula)`` will print a list of available models.
#
# Formula-compatible models have the following generic call signature:
# ``(formula, data, subset=None, *args, **kwargs)``

#
# ## OLS regression using formulas
#
# To begin, we fit the linear model described on the [Getting
# Started](gettingstarted.html) page. Download the data, subset columns, and
# list-wise delete to remove missing observations:

# In[ ]:

dta = sm.datasets.get_rdataset("Guerry", "HistData", cache=True)

# In[ ]:

df = dta.data[['Lottery', 'Literacy', 'Wealth', 'Region']].dropna()
df.head()

# Fit the model:

# In[ ]:

mod = ols(formula='Lottery ~ Literacy + Wealth + Region', data=df)
res = mod.fit()
print(res.summary())

# ## Categorical variables
#
# Looking at the summary printed above, notice that ``patsy`` determined
# that elements of *Region* were text strings, so it treated *Region* as a
# categorical variable. `patsy`'s default is also to include an intercept,
# so we automatically dropped one of the *Region* categories.
#
# If *Region* had been an integer variable that we wanted to treat
# explicitly as categorical, we could have done so by using the ``C()``
# operator:

# In[ ]:

res = ols(formula='Lottery ~ Literacy + Wealth + C(Region)', data=df).fit()
print(res.params)

# Patsy's mode advanced features for categorical variables are discussed
# in: [Patsy: Contrast Coding Systems for categorical
# variables](contrasts.html)

# ## Operators
#
# We have already seen that "~" separates the left-hand side of the model
# from the right-hand side, and that "+" adds new columns to the design
# matrix.
#
# ### Removing variables
#
# The "-" sign can be used to remove columns/variables. For instance, we
# can remove the intercept from a model by:

# In[ ]:

res = ols(formula='Lottery ~ Literacy + Wealth + C(Region) -1 ', data=df).fit()
print(res.params)

# ### Multiplicative interactions
#
# ":" adds a new column to the design matrix with the interaction of the
# other two columns. "*" will also include the individual columns that were
# multiplied together:

# In[ ]:

res1 = ols(formula='Lottery ~ Literacy : Wealth - 1', data=df).fit()
res2 = ols(formula='Lottery ~ Literacy * Wealth - 1', data=df).fit()
print(res1.params, '\n')
print(res2.params)

# Many other things are possible with operators. Please consult the [patsy
# docs](https://patsy.readthedocs.org/en/latest/formulas.html) to learn
# more.

# ## Functions
#
# You can apply vectorized functions to the variables in your model:

# In[ ]:

res = smf.ols(formula='Lottery ~ np.log(Literacy)', data=df).fit()
print(res.params)

# Define a custom function:

# In[ ]:


def log_plus_1(x):
    return np.log(x) + 1.


res = smf.ols(formula='Lottery ~ log_plus_1(Literacy)', data=df).fit()
print(res.params)

# Any function that is in the calling namespace is available to the
# formula.

# ## Using formulas with models that do not (yet) support them
#
# Even if a given `statsmodels` function does not support formulas, you
# can still use `patsy`'s formula language to produce design matrices. Those
# matrices
# can then be fed to the fitting function as `endog` and `exog` arguments.
#
# To generate ``numpy`` arrays:

# In[ ]:

import patsy
f = 'Lottery ~ Literacy * Wealth'
y, X = patsy.dmatrices(f, df, return_type='matrix')
print(y[:5])
print(X[:5])

# To generate pandas data frames:

# In[ ]:

f = 'Lottery ~ Literacy * Wealth'
y, X = patsy.dmatrices(f, df, return_type='dataframe')
print(y[:5])
print(X[:5])

# In[ ]:

print(sm.OLS(y, X).fit().summary())
