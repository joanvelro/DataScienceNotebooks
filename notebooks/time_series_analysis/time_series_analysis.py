import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-paper')
plt.rcParams.update({'font.size': 12})
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

df = pd.read_csv('REE_time_series.csv')
df2 = pd.read_csv('weather_data.csv')

# demand data
# tsa = pd.Series(data=df.iloc[:, 2].values*3000, index=df.iloc[:, 0])
tsa = pd.Series(data=df.iloc[:, 2].values * 3000,
                index=pd.date_range(df.iloc[0, 0], periods=df.iloc[:, 2].size, freq='H'))
tsb = pd.Series(data=df.iloc[:, 3].values * 3000, index=df.iloc[:, 0])
tsc = pd.Series(data=df.iloc[:, 4].values * 3000, index=df.iloc[:, 0])
tsd = pd.Series(data=df.iloc[:, 5].values * 3000, index=df.iloc[:, 0])

# generation data
tsg = pd.Series(data=df2.iloc[:, 0].values * (2 / 3000),
                index=pd.date_range(df2.index[0], periods=df2.index.size, freq='1min'))
tsgd = pd.Series(data=df2.iloc[:, 1].values * (2 / 3000),
                 index=pd.date_range(df2.index[0], periods=df2.index.size, freq='1min'))

if 0 == 1:
    fig = plt.figure(num=1, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(tsg.resample('D').mean(), 'k-')
    ax1.plot(tsgd.resample('D').mean(), 'k--')
    plt.show()
    plt.savefig('daily_irradiance_mean_value.pdf')
    plt.close()

if 0 == 1:
    fig = plt.figure(num=1, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(tsg['2015-01-01 00:01:00':'2015-01-15 00:01:00'], 'k-')
    ax1.plot(tsgd['2015-01-01 00:01:00':'2015-01-15 00:01:00'], 'k--')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(tsg['2015-07-01 00:01:00':'2015-07-15 00:01:00'], 'r-')
    plt.show()
    plt.savefig('weeks_irradiance.pdf')
    plt.close()

if 1 == 1:
    fig = plt.figure(num=1, figsize=(8, 8), dpi=120, facecolor='w', edgecolor='k')

    ax1 = fig.add_subplot(2, 1, 1)
    mu = tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values.mean()
    sigma = tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values.std()
    ax1.plot(np.arange(0, 1441, 1), np.abs(0.5 * tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values +
                                           np.random.normal(mu * 0.3, sigma * 0.1, 1441)), color='grey',
             linestyle='dashed')
    ax1.plot(np.arange(0, 1441, 1), np.abs(0.5 * tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values +
                                           np.random.normal(mu * 0.1, sigma * 0.1, 1441)), color='grey',
             linestyle='dashed')
    ax1.plot(np.arange(0, 1441, 1), 0.6 * tsg['2015-01-02 00:01:00':'2015-01-03 00:01:00'].values, 'k-', linewidth=2)

    ax1.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    ax2 = fig.add_subplot(2, 1, 2)
    mu = tsgd['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values.mean()
    sigma = tsgd['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values.std()
    ax2.plot(np.arange(0, 1441, 1), np.abs(0.5 * tsgd['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values +
                                           np.random.normal(mu * 0.3, sigma * 0.1, 1441)), color='grey',
             linestyle='dashed')
    ax2.plot(np.arange(0, 1441, 1), np.abs(0.5 * tsgd['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values +
                                           np.random.normal(mu * 0.1, sigma * 0.1, 1441)), color='grey',
             linestyle='dashed')
    ax2.plot(np.arange(0, 1441, 1), 0.6 * tsgd['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values, 'k-', linewidth=2)

    ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    # ax2.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    plt.show()
    plt.savefig('plot.pdf')
    plt.close()

if 1 == 1:
    fig = plt.figure(num=1, figsize=(10, 14), dpi=120, facecolor='w', edgecolor='k')

    # Plot summmer + working day profile
    ax1 = fig.add_subplot(4, 1, 1)
    sd = '2015-01-01 00:00:00'
    ed = '2015-01-01 23:59:00'
    ax1.plot(np.arange(0, 1440 - 1, 1), tsg[sd:ed].values, 'k-', linewidth=2)
    sd = '2015-02-01 00:00:00'
    ed = '2015-02-01 23:59:00'
    ax1.plot(np.arange(0, 1440, 1), tsg[sd:ed].values, 'k-', linewidth=2)
    sd = '2015-03-01 00:00:00'
    ed = '2015-03-01 23:59:00'
    ax1.plot(np.arange(0, 1440, 1), tsg[sd:ed].values, 'k-', linewidth=2)
    sd = '2015-04-01 00:00:00'
    ed = '2015-04-01 23:59:00'
    ax1.plot(np.arange(0, 1440, 1), tsg[sd:ed].values, 'k-', linewidth=2)

    # ax1.set_xticks(np.arange(0, 1440, 1))
    # ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax1.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax1.set_ylabel('Aggregated Demand (pu)', fontsize=14)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_title('a) Summer + Working days', fontsize=16)

    plt.show()
    plt.savefig('plot.pdf')
    plt.close()

# Demand Reference Patterns
if 1 == 1:
    fig = plt.figure(num=1, figsize=(40, 14), dpi=120, facecolor='w', edgecolor='k')

    # Plot summmer + working day profile
    ax1 = fig.add_subplot(4, 1, 1)
    sd = '2018-08-18 00:00:00'
    ed = '2018-08-18 23:00:00'
    ax1.plot(np.arange(0, 24, 1), tsc[sd:ed].values, 'k-', linewidth=2)
    sd = '2013-08-10 00:00:00'
    ed = '2013-08-10 23:00:00'
    ax1.plot(np.arange(0, 24, 1), tsc[sd:ed].values, color='grey', linestyle='dashed')
    sd = '2014-08-03 00:00:00'
    ed = '2014-08-03 23:00:00'
    ax1.plot(np.arange(0, 24, 1), tsc[sd:ed].values, color='grey', linestyle='dashed')
    sd = '2015-08-05 00:00:00'
    ed = '2015-08-05 23:00:00'
    ax1.plot(np.arange(0, 24, 1), tsc[sd:ed].values, color='grey', linestyle='dashed')
    sd = '2016-08-15 00:00:00'
    ed = '2016-08-15 23:00:00'
    ax1.plot(np.arange(0, 24, 1), tsc[sd:ed].values, color='grey', linestyle='dashed')
    sd = '2017-08-16 00:00:00'
    ed = '2017-08-16 23:00:00'
    ax1.plot(np.arange(0, 24, 1), tsc[sd:ed].values, color='grey', linestyle='dashed')

    ax1.set_xticks(np.arange(0, 24, 1))
    ax1.set_xticklabels(np.arange(1, 25, 1))
    ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax1.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax1.set_ylabel('Aggregated Demand (pu)', fontsize=10)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_title('a) Summer + Working days', fontsize=16)
    ax1.legend(('$p^{su+wd}_{d,r}(t)$', 'Scenarios',), fontsize=14, loc='upper left')

    # Plot summmer + nowork days profile
    ax2 = fig.add_subplot(4, 1, 2)
    sd = '2018-04-05 00:00:00'
    ed = '2018-04-05 23:00:00'
    ax2.plot(np.arange(0, 24, 1), tsc[sd:ed].values - 0.1, 'k-', linewidth=2)
    sd = '2013-05-05 00:00:00'
    ed = '2013-05-05 23:00:00'
    ax2.plot(np.arange(0, 24, 1), tsc[sd:ed].values, color='grey', linestyle='dashed')
    sd = '2014-05-05 00:00:00'
    ed = '2014-05-05 23:00:00'
    ax2.plot(np.arange(0, 24, 1), tsc[sd:ed].values, color='grey', linestyle='dashed')
    sd = '2015-05-05 00:00:00'
    ed = '2015-05-05 23:00:00'
    ax2.plot(np.arange(0, 24, 1), tsc[sd:ed].values, color='grey', linestyle='dashed')
    sd = '2016-05-05 00:00:00'
    ed = '2016-05-05 23:00:00'
    ax2.plot(np.arange(0, 24, 1), tsc[sd:ed].values, color='grey', linestyle='dashed')
    sd = '2017-05-05 00:00:00'
    ed = '2017-05-05 23:00:00'
    ax2.plot(np.arange(0, 24, 1), tsc[sd:ed].values, color='grey', linestyle='dashed')

    ax2.set_xticks(np.arange(0, 24, 1))
    ax2.set_xticklabels(np.arange(1, 25, 1))
    ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax2.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax2.set_ylabel('Aggregated Demand (pu)', fontsize=10)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_title('b) Summer + Non-Working days', fontsize=16)
    ax2.legend(('$p^{su+nwd}_{d,r}(t)$', 'Scenarios',), fontsize=14, loc='upper left')

    # Plot winter + holidays profile
    ax3 = fig.add_subplot(4, 1, 3)
    sd = '2015-12-26 00:00:00'
    ed = '2015-12-26 23:00:00'
    ax3.plot(np.arange(0, 24, 1), tsa[sd:ed].values, 'k-', linewidth=2)
    sd = '2013-12-26 00:00:00'
    ed = '2013-12-26 23:00:00'
    ax3.plot(np.arange(0, 24, 1), tsa[sd:ed].values, color='grey', linestyle='dashed')
    sd = '2014-12-26 00:00:00'
    ed = '2014-12-26 23:00:00'
    ax3.plot(np.arange(0, 24, 1), tsa[sd:ed].values, color='grey', linestyle='dashed')
    sd = '2016-12-26 00:00:00'
    ed = '2016-12-26 23:00:00'
    ax3.plot(np.arange(0, 24, 1), tsa[sd:ed].values, color='grey', linestyle='dashed')
    sd = '2017-12-26 00:00:00'
    ed = '2017-12-26 23:00:00'
    ax3.plot(np.arange(0, 24, 1), tsa[sd:ed].values, color='grey', linestyle='dashed')
    sd = '2018-10-31 00:00:00'
    ed = '2018-10-31 23:00:00'
    ax3.plot(np.arange(0, 24, 1), tsa[sd:ed].values, color='grey', linestyle='dashed')

    ax3.set_xticks(np.arange(0, 24, 1))
    ax3.set_xticklabels(np.arange(1, 25, 1))
    ax3.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax3.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax3.set_ylabel('Aggregated Demand (pu)', fontsize=10)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.set_title('c) Winter + Working days', fontsize=16)
    ax3.legend(('$p^{wi+wd}_{d,r}(t)$', 'Scenarios',), fontsize=14, loc='upper left')

    # Plot winter + work days profile
    ax4 = fig.add_subplot(4, 1, 4)
    sd = '2017-02-14 00:00:00'
    ed = '2017-02-14 23:00:00'
    ax4.plot(np.arange(0, 24, 1), tsd[sd:ed].values, 'k-', linewidth=2)
    sd = '2013-04-08 00:00:00'
    ed = '2013-04-08 23:00:00'
    # ax4.plot(np.arange(0,24,1),tsd[sd:ed].values,'k-')
    sd = '2014-02-07 00:00:00'
    ed = '2014-02-07 23:00:00'
    # ax4.plot(np.arange(0, 24, 1),tsd[sd:ed].values,'k--')
    sd = '2015-02-05 00:00:00'
    ed = '2015-02-05 23:00:00'
    ax4.plot(np.arange(0, 24, 1), tsd[sd:ed].values, color='grey', linestyle='dashed')
    sd = '2016-01-01 00:00:00'
    ed = '2016-01-01 23:00:00'
    ax4.plot(np.arange(0, 24, 1), tsd[sd:ed].values, color='grey', linestyle='dashed')
    sd = '2017-12-24 00:00:00'
    ed = '2017-12-24 23:00:00'
    ax4.plot(np.arange(0, 24, 1), tsd[sd:ed].values, color='grey', linestyle='dashed')

    # [] {}
    ax4.set_xticks(np.arange(0, 24, 1))
    ax4.set_xticklabels(np.arange(1, 25, 1))
    ax4.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax4.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax4.set_xlabel('Time $t$ (h)', fontsize=14)
    ax4.set_ylabel('Aggregated Demand (pu)', fontsize=10)
    ax4.tick_params(axis='both', labelsize=14)
    ax4.set_title('d) Winter + Non-working days', fontsize=16)
    ax4.legend(('$p^{wi+nwd}_{d,r}(t)$', 'Scenarios',), fontsize=14, loc='upper left')

    plt.subplots_adjust(wspace=0.0, hspace=0.3)
    plt.show()
    plt.savefig('Reference_patters_demand.pdf')
    plt.close()

# Generation Reference Patterns
if 1 == 1:
    fig = plt.figure(num=1, figsize=(10, 12), dpi=120, facecolor='w', edgecolor='k')

    ax1 = fig.add_subplot(2, 1, 1)
    mu = tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values.mean()
    sigma = tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values.std()
    ax1.plot(np.arange(0, 1441, 1), np.abs(0.6 * tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values +
                                           np.random.normal(mu * 0.2, sigma * 0.1, 1441)), color='grey',
             linestyle='dashed')
    ax1.plot(np.arange(0, 1441, 1), 0.7 * tsg['2015-01-02 00:01:00':'2015-01-03 00:01:00'].values, 'k-', linewidth=2)
    # ax1.plot(np.arange(0, 1441, 1), np.abs(0.5*tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values +
    #         np.random.normal(mu * 0.1, sigma * 0.1, 1441)), color='grey', linestyle='dashed')

    ax1.set_xlabel('Time (h)', fontsize=14)
    ax1.set_ylabel('Aggregated Generation (pu)', fontsize=10)
    ax1.set_xticks(np.arange(0, 60 * 24, 60))
    ax1.set_xticklabels(np.arange(0, 24, 1))
    ax1.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4])
    ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8])
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_title('a) Summer', fontsize=16)
    ax1.legend(('Scenarios', '$p^{su}_{g,r}(t)$',), fontsize=14, loc='upper left')

    ax2 = fig.add_subplot(2, 1, 2)
    mu = tsgd['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values.mean()
    sigma = tsgd['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values.std()
    ax2.plot(np.arange(0, 1441, 1), np.abs(0.5 * tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values +
                                           np.random.normal(mu * 0.1, sigma * 0.1, 1441)), color='grey',
             linestyle='dashed')
    ax2.plot(np.arange(0, 1441, 1), 0.5 * tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values, 'k-', linewidth=2)
    # ax2.plot(np.arange(0, 1441, 1), np.abs(0.5 * tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values +
    #         np.random.normal(mu * 0.1, sigma * 0.1, 1441)), color='grey', linestyle='dashed')

    ax2.set_xlabel('Time $t$ (h)', fontsize=14)
    ax2.set_ylabel('Aggregated Generation (pu)', fontsize=10)
    ax2.set_xticks(np.arange(0, 60 * 24, 60))
    ax2.set_xticklabels(np.arange(0, 24, 1))
    ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_title('b) Winter', fontsize=16)
    ax2.legend(('Scenarios', '$p^{wi}_{g,r}(t)$',), fontsize=14, loc='upper left')

    plt.subplots_adjust(wspace=0.0, hspace=0.6)
    plt.show()
    plt.savefig('Reference_patterns_generation.pdf')
    plt.close()

