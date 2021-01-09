# =============================================================================
# How to visualize 1-dimentional data?
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1 = np.loadtxt('example_1.txt')
df2 = np.loadtxt('example_2.txt')

print(df1.shape, df2.shape)

# =============================================================================
# HISTOGRAMS
# =============================================================================
plt.hist(df1, label='df1', bins=30)
plt.hist(df2, label='df2', bins=30)
plt.legend(); # use semi-colon to supress the output of previous print outs
# add labels
plt.ylabel('Count')

######
# when data at same scale then you're looking distributions and consequently probabilities
# with probability area under graph has to sum to 1, 1 is the total probability

# lets establish bins for one plt, and use that for plots
# use numpy linespace, and find minimum and maximum of both datasets
bins = np.linspace(min(df1.min(), df2.min()), max(df1.max(), df2.max()))# start, stop, bins
# if density is equal to true the count will be converted to probabilty
# make additional adjustments by adding, histtype, linestyle, and linewidth 
plt.hist(df1, bins=bins, label='df1', density = True, histtype='step', lw = 3)
plt.hist(df2, bins=bins, label='df2', density = True, histtype='step', ls = ':')
plt.legend(); 
plt.ylabel('Count')

# alternative of using plt, with sns
sns.histplot(df1, bins=bins)
sns.histplot(df2, bins=bins, color='r')

# alternative way with less code
sns.distplot(df1)
sns.distplot(df2)


## can add multiple datasets to the plot function which will build a stack plot
## the stacked area shows over lap of two sets
plt.hist([df1, df2], bins=bins, label='stacked', density = True, histtype= 'barstacked', alpha = 0.5)
plt.hist(df1, bins=bins, label='df1', density = True, histtype='step', lw = 1)
plt.hist(df2, bins=bins, label='df2', density = True, histtype='step', ls = ':')
plt.legend();

# =============================================================================
# BEE-SWARMS
# =============================================================================
# read np arrays into dataframe
# concat arrary and index them as 0 and 1 to identify which is which
dataset = pd.DataFrame({'value': np.concatenate((df1, df2)), 
                        'type': np.concatenate((np.zeros(df1.shape), (np.ones(df2.shape))))
                        })
# check dataframe
dataset.info()

# plot visual
# each circle is a data point
sns.swarmplot(dataset.value)

# view individual data point, and fit plot
# in this visual, data tails are highly visable and peaks can also be identified
## utility of this plot increases as the data becomes more **Categorical**
sns.swarmplot(x='type', y='value', data=dataset, size=2.5);

# =============================================================================
# BOX Plots
# =============================================================================
# boxplot is good to see bunch of distributions
# quartiles define how data is distributed
# interquartile range of boxl=plot by default is 1.5x, 
# sns by default will assume anything outside this range will by outlier
# this can be changed by using the defaultg argument **whis and setting the range manually
sns.boxplot(x='type', y='value', data=dataset, whis=2)
# overlay a swarmplot and see data distribution
sns.swarmplot(x='type', y='value', data=dataset, size=3, color='k', alpha=0.3);

# =============================================================================
# VIOLIN Plots
# =============================================================================
###########Use violin plots vs. Box###########
# you can see density estimate vs. a plain box!
sns.violinplot(x='type', y='value', data=dataset)
# overlay a swarmplot and see data distribution
sns.swarmplot(x='type', y='value', data=dataset, size=2, color='k', alpha=0.3);
 
# can view interquartile ranges via argument inner
sns.violinplot(x='type', y='value', data=dataset, inner='quartile')

# the plot is very smooth, however, we don't have as many data points
# this can be modified via the bandwidth function
# bw is automatically calculated by scipy
####Note:
    # if you under smooth data you'll just get noise
    # if you over smooth, you can lose interesting features
sns.violinplot(x='type', y='value', data=dataset, inner='quartile', bw=0.2)

# =============================================================================
# EMPIRICAL CUMULATIVE DISTRIBUTION FUNCTION (CDF)
# =============================================================================
# can get the informtion out of a histogram plot without the weakness of bining!
# lets sort datapoints and plot them with their respective histograms
sdf1 = np.sort(df1)
sdf2 = np.sort(df2)
# CDF values range from 0-1, can obtain them from numpy line-space
# 1/df1.size, defines however much each data point contributes to thw percentile
cdf = np.linspace(start=1/df1.size, stop=1, num=df1.size)

# lets plot them, superimpose with the corresponding histplot
# areas with tight distributions, you get steep inclines
# areas with long tails you get platu at the end
plt.plot(sdf1, cdf, label='df1 CDF')
plt.plot(sdf2, cdf, label='df2 CDF')
plt.hist(df1, density=True, histtype='step', alpha=0.3)
plt.hist(df2, density=True, histtype='step', alpha=0.3)
plt.legend();






