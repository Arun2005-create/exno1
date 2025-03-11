import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
df.isnull().sum()
df.isnull().any()
df.dropna()
df.fillna(0)
df.fillna(method='ffill')
df.fillna(method='bfill')
df_dropped = df.dropna()
df_dropped
ir=pd.read_csv("/content/iris.csv")
ir
ir.describe()
import seaborn as sns
sns.boxplot(x='sepal_width',data=ir)
rid=ir[((ir.sepal_width<(q1-1.5*iq))|(ir.sepal_width>(q3+1.5*iq)))]
rid['sepal_width']
delid=ir[~((ir.sepal_width<(q1-1.5*iq))|(ir.sepal_width>(q3+1.5*iq)))]
delid
sns.boxplot(x='sepal_width',data=delid)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats


dataset=pd.read_csv("heights.csv")
dataset
df = pd.read_csv("heights.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)


iqr = q3-q1
iqr
low = q1- 1.5*iqr
low
high = q3 + 1.5*iqr
high
df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
z = np.abs(stats.zscore(df['height']))
z
df1 = df[z<3]
df1