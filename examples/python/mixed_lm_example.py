# coding: utf-8

# # Linear Mixed Effects Models

# In[ ]:

# get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# In[ ]:

# get_ipython().run_line_magic('load_ext', 'rpy2.ipython')

# In[ ]:

# get_ipython().run_line_magic('R', 'library(lme4)')

# Comparing R lmer to Statsmodels MixedLM
# =======================================
#
# The Statsmodels imputation of linear mixed models (MixedLM) closely
# follows the approach outlined in Lindstrom and Bates (JASA 1988).  This is
# also the approach followed in the  R package LME4.  Other packages such as
# Stata, SAS, etc. should also be consistent with this approach, as the
# basic techniques in this area are mostly mature.
#
# Here we show how linear mixed models can be fit using the MixedLM
# procedure in Statsmodels.  Results from R (LME4) are included for
# comparison.
#
# Here are our import statements:

# ## Growth curves of pigs
#
# These are longitudinal data from a factorial experiment. The outcome
# variable is the weight of each pig, and the only predictor variable we
# will use here is "time".  First we fit a model that expresses the mean
# weight as a linear function of time, with a random intercept for each pig.
# The model is specified using formulas. Since the random effects structure
# is not specified, the default random effects structure (a random intercept
# for each group) is automatically used.

# In[ ]:

data = sm.datasets.get_rdataset('dietox', 'geepack').data
md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])
mdf = md.fit()
print(mdf.summary())

# Here is the same model fit in R using LMER:

# In[ ]:

# get_ipython().run_cell_magic('R', '', "data(dietox, package='geepack')")

# In[ ]:

# get_ipython().run_line_magic('R', "print(summary(lmer('Weight ~ Time +
# (1|Pig)', data=dietox)))")

# Note that in the Statsmodels summary of results, the fixed effects and
# random effects parameter estimates are shown in a single table.  The
# random effect for animal is labeled "Intercept RE" in the Statmodels
# output above.  In the LME4 output, this effect is the pig intercept under
# the random effects section.
#
# There has been a lot of debate about whether the standard errors for
# random effect variance and covariance parameters are useful.  In LME4,
# these standard errors are not displayed, because the authors of the
# package believe they are not very informative.  While there is good reason
# to question their utility, we elected to include the standard errors in
# the summary table, but do not show the corresponding Wald confidence
# intervals.
#
# Next we fit a model with two random effects for each animal: a random
# intercept, and a random slope (with respect to time).  This means that
# each pig may have a different baseline weight, as well as growing at a
# different rate. The formula specifies that "Time" is a covariate with a
# random coefficient.  By default, formulas always include an intercept
# (which could be suppressed here using "0 + Time" as the formula).

# In[ ]:

md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"], re_formula="~Time")
mdf = md.fit()
print(mdf.summary())

# Here is the same model fit using LMER in R:

# In[ ]:

# get_ipython().run_line_magic('R', 'print(summary(lmer("Weight ~ Time +
# (1 + Time | Pig)", data=dietox)))')

# The random intercept and random slope are only weakly correlated $(0.294
# / \sqrt{19.493 * 0.416} \approx 0.1)$.  So next we fit a model in which
# the two random effects are constrained to be uncorrelated:

# In[ ]:

.294 / (19.493 * .416)**.5

# In[ ]:

md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"], re_formula="~Time")
free = sm.regression.mixed_linear_model.MixedLMParams.from_components(
    np.ones(2), np.eye(2))

mdf = md.fit(free=free)
print(mdf.summary())

# The likelihood drops by 0.3 when we fix the correlation parameter to 0.
# Comparing 2 x 0.3 = 0.6 to the chi^2 1 df reference distribution suggests
# that the data are very consistent with a model in which this parameter is
# equal to 0.
#
# Here is the same model fit using LMER in R (note that here R is
# reporting the REML criterion instead of the likelihood, where the REML
# criterion is twice the log likeihood):

# In[ ]:

# get_ipython().run_line_magic('R', 'print(summary(lmer("Weight ~ Time +
# (1 | Pig) + (0 + Time | Pig)", data=dietox)))')

# ## Sitka growth data
#
# This is one of the example data sets provided in the LMER R library.
# The outcome variable is the size of the tree, and the covariate used here
# is a time value.  The data are grouped by tree.

# In[ ]:

data = sm.datasets.get_rdataset("Sitka", "MASS").data
endog = data["size"]
data["Intercept"] = 1
exog = data[["Intercept", "Time"]]

# Here is the statsmodels LME fit for a basic model with a random
# intercept.  We are passing the endog and exog data directly to the LME
# init function as arrays.  Also note that endog_re is specified explicitly
# in argument 4 as a random intercept (although this would also be the
# default if it were not specified).

# In[ ]:

md = sm.MixedLM(endog, exog, groups=data["tree"], exog_re=exog["Intercept"])
mdf = md.fit()
print(mdf.summary())

# Here is the same model fit in R using LMER:

# In[ ]:

# get_ipython().run_cell_magic('R', '', 'data(Sitka,
# package="MASS")\nprint(summary(lmer("size ~ Time + (1 | tree)",
# data=Sitka)))')

# We can now try to add a random slope.  We start with R this time.  From
# the code and output below we see that the REML estimate of the variance of
# the random slope is nearly zero.

# In[ ]:

# get_ipython().run_line_magic('R', 'print(summary(lmer("size ~ Time + (1
# + Time | tree)", data=Sitka)))')

# If we run this in statsmodels LME with defaults, we see that the
# variance estimate is indeed very small, which leads to a warning about the
# solution being on the boundary of the parameter space.  The regression
# slopes agree very well with R, but the likelihood value is much higher
# than that returned by R.

# In[ ]:

exog_re = exog.copy()
md = sm.MixedLM(endog, exog, data["tree"], exog_re)
mdf = md.fit()
print(mdf.summary())

# We can further explore the random effects struture by constructing plots
# of the profile likelihoods. We start with the random intercept, generating
# a plot of the profile likelihood from 0.1 units below to 0.1 units above
# the MLE. Since each optimization inside the profile likelihood generates a
# warning (due to the random slope variance being close to zero), we turn
# off the warnings here.

# In[ ]:

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    likev = mdf.profile_re(0, 're', dist_low=0.1, dist_high=0.1)

# Here is a plot of the profile likelihood function.  We multiply the log-
# likelihood difference by 2 to obtain the usual $\chi^2$ reference
# distribution with 1 degree of freedom.

# In[ ]:

import matplotlib.pyplot as plt

# In[ ]:

plt.figure(figsize=(10, 8))
plt.plot(likev[:, 0], 2 * likev[:, 1])
plt.xlabel("Variance of random slope", size=17)
plt.ylabel("-2 times profile log likelihood", size=17)

# Here is a plot of the profile likelihood function. The profile
# likelihood plot shows that the MLE of the random slope variance parameter
# is a very small positive number, and that there is low uncertainty in this
# estimate.

# In[ ]:

re = mdf.cov_re.iloc[1, 1]
likev = mdf.profile_re(1, 're', dist_low=.5 * re, dist_high=0.8 * re)

plt.figure(figsize=(10, 8))
plt.plot(likev[:, 0], 2 * likev[:, 1])
plt.xlabel("Variance of random slope", size=17)
plt.ylabel("-2 times profile log likelihood", size=17)
