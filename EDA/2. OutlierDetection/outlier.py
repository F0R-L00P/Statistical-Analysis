# Basics
# The most basic and most-common way of manually doing outlier pruning on data distributions is to:

# 1- Model your data as some analytic distribution
# 2- Find all points below a certain probability
# 3- Remove them
# 4- Refit the distributions, and potentially run again from Step 1.

# =============================================================================
# Finding local outliers in 1-dimensional data
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

df = np.loadtext("outlier_1d")

mean, std = np.mean(df), np.std(df)
z_score = np.abs((df - mean) / std)
threshold = 3
good = z_score < threshold

print(f"Rejection {(~good).sum()} points")
from scipy.stats import norm
print(f"z-score of 3 corresponds to a prob of {100 * 2 * norm.sf(threshold):0.2f}%")

visual_scatter = np.random.normal(size=df.size)
plt.scatter(df[good], visual_scatter[good], s=2, label="Normal Data", color="#4CAF50")
plt.scatter(df[~good], visual_scatter[~good], s=8, label="Outlier", color="#F44336")
plt.legend();

# =============================================================================
# Finding local outliers in 2-dimensional data
# =============================================================================
df = np.loadtxt("outlier_2d.txt")

from scipy.stats import multivariate_normal as mn
mean, cov = np.mean(df, axis=0), np.cov(df.T)
good = mn(mean, cov).pdf(df) > 0.01 / 100 #can change point to see outlier removal, 0.25 vs 0.01 

plt.scatter(df[good, 0], df[good, 1], s=2, label="Good", color="#4CAF50")
plt.scatter(df[~good, 0], df[~good, 1], s=8, label="Bad", color="#F44336")
plt.legend();


# =============================================================================
# Finding outliers linear/curve
# =============================================================================
# load data with np.loadtext
# if a distribution is not available, however, outliers are presented
# data can be modeled using polynomial method, this method can easily identify outliers that are within 3 std 
# of the polynomial, and can be removed.

df = np.loadtxt("outlier_curve.txt")

# fit x and y using numpy poly
xs, ys = df.T
p = np.polyfit(xs, ys, deg=5)
ps = np.polyval(p, xs)
plt.plot(xs, ys, ".", label="Data", ms=1)
plt.plot(xs, ps, label="Bad poly fit")
plt.legend();

# fit data based on desired number of iterations 
x, y = xs.copy(), ys.copy()
for i in range(5):
    p = np.polyfit(x, y, deg=5)
    ps = np.polyval(p, x)
    good = y - ps < 3  # only remove positive outliers
    
    x_bad, y_bad = x[~good], y[~good]
    x, y = x[good], y[good]            # variables x and y stored data without the outliers, 
    
    plt.plot(x, y, ".", label="Used Data", ms=1)
    plt.plot(x, np.polyval(p, x), label=f"Poly fit {i}")
    plt.plot(x_bad, y_bad, ".", label="Not used Data", ms=5, c="r")
    plt.legend()
    plt.show()
    
    if (~good).sum() == 0:
        break

# =============================================================================
# Sklearn
# =============================================================================
# an alternative to manual outlier detection, is the uber powerful Sklearn package
from sklearn.neighbors import LocalOutlierFactor

df = np.loadtext("outlier_2d")

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.005)
good = lof.fit_predict(df) == 1
plt.scatter(df[good, 0], df[good, 1], s=2, label="Good", color="#4CAF50")
plt.scatter(df[~good, 0], df[~good, 1], s=8, label="Bad", color="#F44336")
plt.legend();