# coding: utf-8

# # Influence Measures for GLM Logit
#
#
# Based on draft version for GLMInfluence, which will also apply to
# discrete Logit, Probit and Poisson, and eventually be extended to cover
# most models outside of time series analysis.
#
# The example for logistic regression was used by Pregibon (1981)
# "Logistic Regression diagnostics" and is based on data by Finney (1947).
#
# GLMInfluence includes the basic influence measures but still misses some
# measures described in Pregibon (1981), for example those related to
# deviance and effects on confidence intervals.

# In[ ]:

import os.path
import numpy as np
import pandas as pd

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

# In[ ]:

import statsmodels.stats.tests.test_influence
test_module = statsmodels.stats.tests.test_influence.__file__
cur_dir = cur_dir = os.path.abspath(os.path.dirname(test_module))

file_name = 'binary_constrict.csv'
file_path = os.path.join(cur_dir, 'results', file_name)
df = pd.read_csv(file_path, index_col=0)

# In[ ]:

res = GLM(
    df['constrict'],
    df[['const', 'log_rate', 'log_volumne']],
    family=families.Binomial()).fit(
        attach_wls=True, atol=1e-10)
print(res.summary())

# ## get the influence measures
#
# GLMResults has a `get_influence` method similar to OLSResults, that
# returns and instance of the GLMInfluence class. This class has methods and
# (cached) attributes to inspect influence and outlier measures.
#
# This measures are based on a one-step approximation to the the results
# for deleting one observation. One-step approximations are usually accurate
# for small changes but underestimate the magnitude of large changes. Event
# though large changes are underestimated, they still show clearly the
# effect of influential observations
#
# In this example observation 4 and 18 have a large standardized residual
# and large Cook's distance, but not a large leverage. Observation 13 has
# the largest leverage but only small Cook's distance and not a large
# studentized residual.
#
# Only the two observations 4 and 18 have a large impact on the parameter
# estimates.

# In[ ]:

infl = res.get_influence(observed=False)

# In[ ]:

summ_df = infl.summary_frame()
summ_df.sort_values('cooks_d', ascending=False)[:10]

# In[ ]:

infl.plot_influence()

# In[ ]:

infl.plot_index(y_var='cooks', threshold=2 * infl.cooks_distance[0].mean())

# In[ ]:

infl.plot_index(y_var='resid', threshold=1)

# In[ ]:

infl.plot_index(y_var='dfbeta', idx=1, threshold=0.5)

# In[ ]:

infl.plot_index(y_var='dfbeta', idx=2, threshold=0.5)

# In[ ]:

infl.plot_index(y_var='dfbeta', idx=0, threshold=0.5)
