import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt

# =============================================================================
# Load dataset
# =============================================================================
df = np.loadtxt('dataset.txt')

# view data distribution
plt.hist(df, bins=100);

# =============================================================================
# CENTRALITY
# the center of the data (MEAN/AVG, MEDIAN, MODE)
# =============================================================================
# note: if you need weighted average use np.average()
my_mean = sum(df)/len(df)
print(my_mean, df.mean(), np.mean(df), np.average(df))

# median, sorting all data and taking the middle element
# means are SENSETIVE to outliers, but medians are not!
data_median = np.median(df)
print(data_median)

# TESTING outlier impact on dataset
# comparing Mean vs. Median
outlier = np.insert(df, 0, 5000)
plt.hist(df, label='Data', bins=100)
plt.axvline(np.mean(df), ls='--', label='Mean Data')
plt.axvline(np.median(df), ls=':', label='Median Data')
plt.axvline(np.mean(outlier), c='r', ls='--', label='Mean Outlier', alpha=0.7)
plt.axvline(np.median(outlier), c='r', ls=':', label='Median Outlier', alpha=0.7)
plt.legend()
plt.xlim(0,20)

# MODE - returns the most common value in a dataset 
#####MODE is normally the point of maximum likelihood
# 5 with 9 repeats
mode = sc.mode(df)
print(mode)

# visualizing MODE of data using Kernal Density
kde = sc.gaussian_kde(df)
x_values = np.linspace(df.min(), df.max(), 1000)
y_values = kde(x_values)
mode = x_values[y_values.argmax()]

plt.hist(df, bins=100, density=True, label='Data hist', histtype='step')
plt.plot(x_values, y_values, label='kde')
plt.axvline(mode, label='mode')
plt.legend();

# =============================================================================
# visualizing central values
# =============================================================================
plt.hist(df, bins=100, label='Dataset', alpha=0.3)
plt.axvline(np.mean(df), label='Mean', ls='--', c='r')
plt.axvline(np.median(df), label='Median', ls=':', c='b')
plt.axvline(mode, label='Mode', ls='dashdot', c='g')
plt.legend()
plt.show();