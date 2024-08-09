import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error

"""
[Source 1](https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/)

[Source 2](https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b)
"""

df = pd.read_csv('REE_profiles.csv')
date = []
for k in range(0, len(df)):
    date.append(pd.Timestamp(int(df.date[k][0:4]), int(df.date[k][5:7]), int(df.date[k][8:10]), int(df.date[k][11:13])))
df = df.drop(['date', 'season'], axis=1)
df['date'] = date
df = df.set_index('date')
df = df.resample('D').mean()  # resize to daily basis instead of hour basis
print(df.head())
print(df.tail())
print(df.dtypes)

ts1 = df.coef_a
ts2 = df.coef_b
ts3 = df.coef_c
ts4 = df.coef_d

fig = plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(ts1, 'k')
ax1.legend(('ts1',), fontsize=12)

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(ts2, 'k')
ax2.legend(('ts2',), fontsize=12)

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(ts3, 'k')
ax3.legend(('ts3',), fontsize=12)

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(ts4, 'k')
ax4.legend(('ts4',), fontsize=12)

plt.show()

train = ts1['2013-01-01':'2018-01-01']
test = ts1['2018-01-01':'2018-10-31']

print(ts1.index[0])
print(ts1.index[len(ts1) - 1])
