import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# Load data
# =============================================================================
df_diabetes = pd.read_csv('Diabetes.csv')
df_hw = pd.read_csv('height_weight.csv')

# exclude 2 columns from dataset before 0 imputation and removal
cols = [i for i in df_diabetes.columns if i not in ['Pregnancies', 'Outcome']]    

df1 = df_diabetes.copy()
# replace 0 with Not a Nomber (NaN) and drop them
df1[cols] = df1[cols].replace(0, np.nan)

# =============================================================================
# Scatter Mtrix
# =============================================================================
# the band data (col1 of plot) is integer data
pd.plotting.scatter_matrix(df1, figsize=(10, 10))
# OR sns.pairplot(df1, corner=True) --> Corner will render half of plots as its identical copy but rotated

# visualiz data with overlaying colours
# you can see correlation based on outcome of a single marker (i.e. BMI, Glucose)
df2 = df1.dropna()
colours = df2['Outcome'].map(lambda x: 'b' if x else 'r')
pd.plotting.scatter_matrix(df2, figsize=(10, 10), color=colours);

# =============================================================================
# HEAT MAPS & CORRELATION PLOTS
# =============================================================================
# simply show correlation between variables
# the data is identical between the diagonal line, can focus on upper or lower triangle
# 1 represents total correlation, highly correlated
# 0 represents no correlation - they are completely in-dependent
# -1 represents total inverse correlation - move opposite to each other
sns.heatmap(df1.corr())

# make heatmap, annotate data, and change colour, annotation format will be at 2 floating point
# for list of colour palets, visit https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
sns.heatmap(df1.corr(), annot=True, cmap='viridis', fmt='0.2f')
# you can easily look at the last column of the matrix and find correlation between dependent/independent vars

## =============================================================================
# ### Aadvance visual correlation model-Triangle correlation
# =============================================================================
# define plot size
plt.figure(figsize=(15, 12))
# define the mask mato set the values in the upper triangle to True
mask = np.triu(np.ones_like(df1.corr(), dtype=np.bool))
# use vmin/max to define data spread for correlation -1 to 1
heatmap = sns.heatmap(df1.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='bone', fmt='0.2f')
# title and change fontsize = padding
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=20);

# =============================================================================
# ### Advance visual correlation model-Independent with Dependent variable
# =============================================================================
# setup figure size
plt.figure(figsize=(8, 12))
# set column correlation to target - in this case 'Outcome
heatmap = sns.heatmap(df1.corr()[['Outcome']].sort_values(by='Outcome', ascending=False), 
                      vmin=-1, vmax=1, annot=True, cmap='copper', fmt='0.2f')
# title, font and padding
heatmap.set_title('Features Correlating with Sales Price', fontdict={'fontsize':18}, pad=16);

# =============================================================================
# 2D HISTOGRAMS
# =============================================================================
df2 = pd.read_csv('height_weight.csv')
df2.describe()

# can view positive correlation between hight and weight
# this relationship can be better viewed using a contour plot
plt.hist2d(x=df2.height, y=df2.weight, bins=20, cmap='magma')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show();

# =============================================================================
# CONTOUR PLOT
# =============================================================================
# hist: count, or number of data points that fall into each bin
# x/y is the edges between the x and y bins
hist, x_edge, y_edge = np.histogram2d(x=df2.height, y=df2.weight, bins=20)
# mostly care about the center of the bins
# from first to end(not including)
# getting two edges and dividing them by 2
x_center = 0.5 * (x_edge[1:] + x_edge[:-1])
y_center = 0.5 * (y_edge[1:] + y_edge[:-1])

# can change smoothness based on data if we have alot od data, this can be done by level argument
# if you dont have extra data using KDE
plt.contour(x_center, y_center, hist)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show();

# =============================================================================
# KERNAL DENSITY ESTIMATION - KDE PLOTS
# KDE - allows us to estimate the probability density of the finite dataset
# =============================================================================
# KDE - make a distribution for each dataset
# once the distributions are added, you get a density distribution
# this distribution can be based on, for example, gaussian distribution
####This density can be viewed one-dimensionally 
sns.kdeplot(df2.weight, shade=True)

# check density of BIVARIATE density and look with histogram
# can use bandwdith argument bw, to control kernel size.
    # smaller bw = point format distribution
    # larger bw = kernel overlap and smooth surface
sns.kdeplot(x=df2.height, y=df2.weight, cmap='magma', bw=0.025)
plt.hist2d(x=df2.height, y=df2.weight, bins=20, cmap='magma', alpha=0.3)

# making plot with shading
sns.kdeplot(x=df2.height, y=df2.weight, cmap='magma', shade=True)

# can impose scatter and KDE for density estimation view
sns.kdeplot(x=df2.height, y=df2.weight, cmap='magma')
plt.scatter(x=df2.height, y=df2.weight, cmap='magma', alpha=0.2)


# =============================================================================
# Sample Plot
# =============================================================================
mask = df2['sex'] == 1
plt.scatter(x=df2.loc[mask, 'height'], y=df2.loc[mask, 'weight'], c='b', s=1, label='Male')
# ~ inverse
plt.scatter(x=df2.loc[~mask, 'height'], y=df2.loc[~mask, 'weight'], c='r', s=1, label='Female')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend(loc=2);


######Can view joint plot using density and distribution#####
sns.jointplot(data=df2, x='height', y='weight', 
              hue='sex', kind='kde');