# =============================================================================
# Preparing Dataset
# =============================================================================
# How does dataset handel invalid values?
# what to do with null values
# should we summerize, group or filter data?
# =============================================================================
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

df = pd.read_csv('Diabetes.csv')

df.info()
df.head()

# can use dropna and sepcify rows to drop with howmany null values in the
# lets specify a df
df2 = df[['Glucose', 'BMI', 'Age', 'Outcome']]

# check summary statistics
# glucose has a jump from 0 at min to 99 at 25%
df2.describe()

# lets check for zero values with boolean mask
# droping values on the horizontal axis (1)
(df2[df2.columns[:-1]] == 0).any(axis = 1)

# lets extract from df2, everything that is FALSE, which is non-zero
# to inverse th elogic can use tilda ~
df3 = df2.loc[~(df2[df2.columns[:-1]] == 0).any(axis = 1)]
# see how many data points were dropped
df3.info()
df3.describe()

# lets aggregate and view data
# can use another methods on top of group by or he object is printed
df3.groupby('Outcome').mean()

# can use aggregate function, can pass a dictionary that links columns names
# to the aggregate method you want to use
# can use multiple statistical measure for each column
df3.groupby('Outcome').agg({"Glucose":"mean", "BMI":"median", "Age":"sum"})

# can also see multiple aggregates per column
df3.groupby("Outcome").agg({"mean", "median", "std"})

# filter data bu specific criteria, i.e. outcome.
# lets see how many people have diabetes, how many are non-diabetic
diabetic = df3[df3['Outcome'] == 1]
non_diabetic = df3[df3['Outcome'] == 0]
print(diabetic.shape, non_diabetic.shape)

# writing data to file, drop index
df3.to_csv('clean_diabetes.csv', index = False)
