import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
plt.rcParams.update({'font.size': 12})
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times New Roman']})
rc('text', usetex=True)
import numpy as np


df = pd.read_csv('results_forecast_demand_val.csv')
df = df.rename(index=str, columns={"Unnamed: 0": "date"})
df = df.set_index('date')
df = np.multiply(df,1/800000)

df1 = pd.read_csv('results_forecast_irradiance.csv')
df1 = df1.rename(index=str, columns={"Unnamed: 0": "date"})
df1 = df1.set_index('date')
df1  = np.multiply(df1,1/(0.015*800000))

df3 = pd.read_csv('results_forecast_temperature.csv')
df3 = df3.rename(index=str, columns={"Unnamed: 0": "date"})
df3 = df3.set_index('date')

fig = plt.figure(num=None, figsize=(16,12), dpi=120, facecolor='w', edgecolor='k')

ax = fig.add_subplot(2, 1, 1)
ax.plot(df.loc[:,'Pd_obs'].dropna().index, df.loc[:, 'Pd_obs'].dropna().get_values(),'k-')
ax.plot(df.loc[:,'Pd_hat'].dropna().index, df.loc[:, 'Pd_hat'].dropna().get_values(),'r-')
ax.fill_between(df.loc[:,'Pd_ucl'].dropna().index,
                df.loc[:, 'Pd_ucl'].dropna().get_values(),
                df.loc[:, 'Pd_lcl'].dropna().get_values(), color='gray', alpha=.2)
plt.xticks(np.arange(0,24*6,24),['2018-04-26','2018-04-27','2018-04-28','2018-04-29','2018-04-30'], fontsize=10)
plt.title('a)', fontsize=18)
ax.set_ylabel('Load Demand $\sum_k \sum_p \hat{p}^{p}_{k,t} $ $(pu)$', fontsize=18)
#ax.set_xlabel('Time t', fontsize=16)
plt.legend(('Observed','Forecast','Confidence Intervals',), fontsize=16, loc='lower left')
ax.tick_params(axis='both', labelsize=18)

ax = fig.add_subplot(2, 1, 2)
ax.plot(df1.loc[:,'G_obs'].dropna().index, df1.loc[:, 'G_obs'].dropna().get_values(),'k-')
ax.plot(df1.loc[:,'G_hat'].dropna().index, df1.loc[:, 'G_hat'].dropna().get_values(),'r-')
ax.fill_between(df1.loc[:, 'G_ucl'].dropna().index,
                df1.loc[:, 'G_ucl'].dropna().get_values(),
                df1.loc[:, 'G_lcl'].dropna().get_values(), color='gray', alpha=.2)
plt.xticks(np.arange(0,24*6,24),['2018-04-26','2018-04-27','2018-04-28','2018-04-29','2018-04-30'], fontsize=10)
plt.title('b)', fontsize=18)
ax.set_ylabel('PV Generation $\sum_k \sum_p \hat{p}^{p}_{g,k,t}$ $(pu)$', fontsize=18)
#ax.set_xlabel('Time t', fontsize=16)
plt.legend(('Observed','Forecast','Confidence Intervals',), fontsize=16, loc='lower left')
ax.tick_params(axis='both', labelsize=18)


plt.savefig('fig_forecast_val.eps')
plt.show()
plt.close()
