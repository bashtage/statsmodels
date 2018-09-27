# coding: utf-8

# # Interactions and ANOVA

# Note: This script is based heavily on Jonathan Taylor's class notes
# http://www.stanford.edu/class/stats191/interactions.html
#
# Download and format data:

# In[ ]:

# get_ipython().run_line_magic('matplotlib', 'inline')

from __future__ import print_function
from statsmodels.compat import urlopen
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import statsmodels.api as sm
import pandas as pd
pd.set_option("display.width", 100)
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.graphics.api import interaction_plot, abline_plot
from statsmodels.stats.anova import anova_lm

try:
    salary_table = pd.read_csv('salary.table')
except:  # recent pandas can read URL without urlopen
    url = 'http://stats191.stanford.edu/data/salary.table'
    fh = urlopen(url)
    salary_table = pd.read_table(fh)
    salary_table.to_csv('salary.table')

E = salary_table.E
M = salary_table.M
X = salary_table.X
S = salary_table.S

# Take a look at the data:

# In[ ]:

plt.figure(figsize=(6, 6))
symbols = ['D', '^']
colors = ['r', 'g', 'blue']
factor_groups = salary_table.groupby(['E', 'M'])
for values, group in factor_groups:
    i, j = values
    plt.scatter(
        group['X'], group['S'], marker=symbols[j], color=colors[i - 1], s=144)
plt.xlabel('Experience')
plt.ylabel('Salary')

# Fit a linear model:

# In[ ]:

formula = 'S ~ C(E) + C(M) + X'
lm = ols(formula, salary_table).fit()
print(lm.summary())

# Have a look at the created design matrix:

# In[ ]:

lm.model.exog[:5]

# Or since we initially passed in a DataFrame, we have a DataFrame
# available in

# In[ ]:

lm.model.data.orig_exog[:5]

# We keep a reference to the original untouched data in

# In[ ]:

lm.model.data.frame[:5]

# Influence statistics

# In[ ]:

infl = lm.get_influence()
print(infl.summary_table())

# or get a dataframe

# In[ ]:

df_infl = infl.summary_frame()

# In[ ]:

df_infl[:5]

# Now plot the reiduals within the groups separately:

# In[ ]:

resid = lm.resid
plt.figure(figsize=(6, 6))
for values, group in factor_groups:
    i, j = values
    group_num = i * 2 + j - 1  # for plotting purposes
    x = [group_num] * len(group)
    plt.scatter(
        x,
        resid[group.index],
        marker=symbols[j],
        color=colors[i - 1],
        s=144,
        edgecolors='black')
plt.xlabel('Group')
plt.ylabel('Residuals')

# Now we will test some interactions using anova or f_test

# In[ ]:

interX_lm = ols("S ~ C(E) * X + C(M)", salary_table).fit()
print(interX_lm.summary())

# Do an ANOVA check

# In[ ]:

from statsmodels.stats.api import anova_lm

table1 = anova_lm(lm, interX_lm)
print(table1)

interM_lm = ols("S ~ X + C(E)*C(M)", data=salary_table).fit()
print(interM_lm.summary())

table2 = anova_lm(lm, interM_lm)
print(table2)

# The design matrix as a DataFrame

# In[ ]:

interM_lm.model.data.orig_exog[:5]

# The design matrix as an ndarray

# In[ ]:

interM_lm.model.exog
interM_lm.model.exog_names

# In[ ]:

infl = interM_lm.get_influence()
resid = infl.resid_studentized_internal
plt.figure(figsize=(6, 6))
for values, group in factor_groups:
    i, j = values
    idx = group.index
    plt.scatter(
        X[idx],
        resid[idx],
        marker=symbols[j],
        color=colors[i - 1],
        s=144,
        edgecolors='black')
plt.xlabel('X')
plt.ylabel('standardized resids')

# Looks like one observation is an outlier.

# In[ ]:

drop_idx = abs(resid).argmax()
print(drop_idx)  # zero-based index
idx = salary_table.index.drop(drop_idx)

lm32 = ols('S ~ C(E) + X + C(M)', data=salary_table, subset=idx).fit()

print(lm32.summary())
print('\n')

interX_lm32 = ols('S ~ C(E) * X + C(M)', data=salary_table, subset=idx).fit()

print(interX_lm32.summary())
print('\n')

table3 = anova_lm(lm32, interX_lm32)
print(table3)
print('\n')

interM_lm32 = ols('S ~ X + C(E) * C(M)', data=salary_table, subset=idx).fit()

table4 = anova_lm(lm32, interM_lm32)
print(table4)
print('\n')

#  Replot the residuals

# In[ ]:

try:
    resid = interM_lm32.get_influence().summary_frame()['standard_resid']
except:
    resid = interM_lm32.get_influence().summary_frame()['standard_resid']

plt.figure(figsize=(6, 6))
for values, group in factor_groups:
    i, j = values
    idx = group.index
    plt.scatter(
        X[idx],
        resid[idx],
        marker=symbols[j],
        color=colors[i - 1],
        s=144,
        edgecolors='black')
plt.xlabel('X[~[32]]')
plt.ylabel('standardized resids')

#  Plot the fitted values

# In[ ]:

lm_final = ols('S ~ X + C(E)*C(M)', data=salary_table.drop([drop_idx])).fit()
mf = lm_final.model.data.orig_exog
lstyle = ['-', '--']

plt.figure(figsize=(6, 6))
for values, group in factor_groups:
    i, j = values
    idx = group.index
    plt.scatter(
        X[idx],
        S[idx],
        marker=symbols[j],
        color=colors[i - 1],
        s=144,
        edgecolors='black')
    # drop NA because there is no idx 32 in the final model
    plt.plot(
        mf.X[idx].dropna(),
        lm_final.fittedvalues[idx].dropna(),
        ls=lstyle[j],
        color=colors[i - 1])
plt.xlabel('Experience')
plt.ylabel('Salary')

# From our first look at the data, the difference between Master's and PhD
# in the management group is different than in the non-management group.
# This is an interaction between the two qualitative variables management,M
# and education,E. We can visualize this by first removing the effect of
# experience, then plotting the means within each of the 6 groups using
# interaction.plot.

# In[ ]:

U = S - X * interX_lm32.params['X']

plt.figure(figsize=(6, 6))
interaction_plot(
    E,
    M,
    U,
    colors=['red', 'blue'],
    markers=['^', 'D'],
    markersize=10,
    ax=plt.gca())

# ## Minority Employment Data

# In[ ]:

try:
    jobtest_table = pd.read_table('jobtest.table')
except:  # don't have data already
    url = 'http://stats191.stanford.edu/data/jobtest.table'
    jobtest_table = pd.read_table(url)

factor_group = jobtest_table.groupby(['MINORITY'])

fig, ax = plt.subplots(figsize=(6, 6))
colors = ['purple', 'green']
markers = ['o', 'v']
for factor, group in factor_group:
    ax.scatter(
        group['TEST'],
        group['JPERF'],
        color=colors[factor],
        marker=markers[factor],
        s=12**2)
ax.set_xlabel('TEST')
ax.set_ylabel('JPERF')

# In[ ]:

min_lm = ols('JPERF ~ TEST', data=jobtest_table).fit()
print(min_lm.summary())

# In[ ]:

fig, ax = plt.subplots(figsize=(6, 6))
for factor, group in factor_group:
    ax.scatter(
        group['TEST'],
        group['JPERF'],
        color=colors[factor],
        marker=markers[factor],
        s=12**2)

ax.set_xlabel('TEST')
ax.set_ylabel('JPERF')
fig = abline_plot(model_results=min_lm, ax=ax)

# In[ ]:

min_lm2 = ols('JPERF ~ TEST + TEST:MINORITY', data=jobtest_table).fit()

print(min_lm2.summary())

# In[ ]:

fig, ax = plt.subplots(figsize=(6, 6))
for factor, group in factor_group:
    ax.scatter(
        group['TEST'],
        group['JPERF'],
        color=colors[factor],
        marker=markers[factor],
        s=12**2)

fig = abline_plot(
    intercept=min_lm2.params['Intercept'],
    slope=min_lm2.params['TEST'],
    ax=ax,
    color='purple')
fig = abline_plot(
    intercept=min_lm2.params['Intercept'],
    slope=min_lm2.params['TEST'] + min_lm2.params['TEST:MINORITY'],
    ax=ax,
    color='green')

# In[ ]:

min_lm3 = ols('JPERF ~ TEST + MINORITY', data=jobtest_table).fit()
print(min_lm3.summary())

# In[ ]:

fig, ax = plt.subplots(figsize=(6, 6))
for factor, group in factor_group:
    ax.scatter(
        group['TEST'],
        group['JPERF'],
        color=colors[factor],
        marker=markers[factor],
        s=12**2)

fig = abline_plot(
    intercept=min_lm3.params['Intercept'],
    slope=min_lm3.params['TEST'],
    ax=ax,
    color='purple')
fig = abline_plot(
    intercept=min_lm3.params['Intercept'] + min_lm3.params['MINORITY'],
    slope=min_lm3.params['TEST'],
    ax=ax,
    color='green')

# In[ ]:

min_lm4 = ols('JPERF ~ TEST * MINORITY', data=jobtest_table).fit()
print(min_lm4.summary())

# In[ ]:

fig, ax = plt.subplots(figsize=(8, 6))
for factor, group in factor_group:
    ax.scatter(
        group['TEST'],
        group['JPERF'],
        color=colors[factor],
        marker=markers[factor],
        s=12**2)

fig = abline_plot(
    intercept=min_lm4.params['Intercept'],
    slope=min_lm4.params['TEST'],
    ax=ax,
    color='purple')
fig = abline_plot(
    intercept=min_lm4.params['Intercept'] + min_lm4.params['MINORITY'],
    slope=min_lm4.params['TEST'] + min_lm4.params['TEST:MINORITY'],
    ax=ax,
    color='green')

# In[ ]:

# is there any effect of MINORITY on slope or intercept?
table5 = anova_lm(min_lm, min_lm4)
print(table5)

# In[ ]:

# is there any effect of MINORITY on intercept
table6 = anova_lm(min_lm, min_lm3)
print(table6)

# In[ ]:

# is there any effect of MINORITY on slope
table7 = anova_lm(min_lm, min_lm2)
print(table7)

# In[ ]:

# is it just the slope or both?
table8 = anova_lm(min_lm2, min_lm4)
print(table8)

# ## One-way ANOVA

# In[ ]:

try:
    rehab_table = pd.read_csv('rehab.table')
except:
    url = 'http://stats191.stanford.edu/data/rehab.csv'
    rehab_table = pd.read_table(url, delimiter=",")
    rehab_table.to_csv('rehab.table')

fig, ax = plt.subplots(figsize=(8, 6))
fig = rehab_table.boxplot('Time', 'Fitness', ax=ax, grid=False)

# In[ ]:

rehab_lm = ols('Time ~ C(Fitness)', data=rehab_table).fit()
table9 = anova_lm(rehab_lm)
print(table9)

print(rehab_lm.model.data.orig_exog)

# In[ ]:

print(rehab_lm.summary())

# ## Two-way ANOVA

# In[ ]:

try:
    kidney_table = pd.read_table('./kidney.table')
except:
    url = 'http://stats191.stanford.edu/data/kidney.table'
    kidney_table = pd.read_csv(url, delim_whitespace=True)

# Explore the dataset

# In[ ]:

kidney_table.head(10)

# Balanced panel

# In[ ]:

kt = kidney_table
plt.figure(figsize=(8, 6))
fig = interaction_plot(
    kt['Weight'],
    kt['Duration'],
    np.log(kt['Days'] + 1),
    colors=['red', 'blue'],
    markers=['D', '^'],
    ms=10,
    ax=plt.gca())

# You have things available in the calling namespace available in the
# formula evaluation namespace

# In[ ]:

kidney_lm = ols('np.log(Days+1) ~ C(Duration) * C(Weight)', data=kt).fit()

table10 = anova_lm(kidney_lm)

print(
    anova_lm(
        ols('np.log(Days+1) ~ C(Duration) + C(Weight)', data=kt).fit(),
        kidney_lm))
print(
    anova_lm(
        ols('np.log(Days+1) ~ C(Duration)', data=kt).fit(),
        ols('np.log(Days+1) ~ C(Duration) + C(Weight, Sum)', data=kt).fit()))
print(
    anova_lm(
        ols('np.log(Days+1) ~ C(Weight)', data=kt).fit(),
        ols('np.log(Days+1) ~ C(Duration) + C(Weight, Sum)', data=kt).fit()))

# ## Sum of squares
#
#  Illustrates the use of different types of sums of squares (I,II,II)
#  and how the Sum contrast can be used to produce the same output between
#  the 3.
#
#  Types I and II are equivalent under a balanced design.
#
#  Don't use Type III with non-orthogonal contrast - ie., Treatment

# In[ ]:

sum_lm = ols(
    'np.log(Days+1) ~ C(Duration, Sum) * C(Weight, Sum)', data=kt).fit()

print(anova_lm(sum_lm))
print(anova_lm(sum_lm, typ=2))
print(anova_lm(sum_lm, typ=3))

# In[ ]:

nosum_lm = ols(
    'np.log(Days+1) ~ C(Duration, Treatment) * C(Weight, Treatment)',
    data=kt).fit()
print(anova_lm(nosum_lm))
print(anova_lm(nosum_lm, typ=2))
print(anova_lm(nosum_lm, typ=3))
