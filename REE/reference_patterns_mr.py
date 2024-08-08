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


P_ctd = 546  # MW
P_dem = 0.6 * P_ctd
P_gen = 0.6 * P_ctd
"""=========================="""
"""Demand Reference Patterns"""
"""=========================="""
if 1 == 1:
    width = 100
    height = 12
    fig = plt.figure(num=1, figsize=(width, height), dpi=60, facecolor='w', edgecolor='k')

    # Plot summmer + working day profile
    ax1 = fig.add_subplot(4, 2, 1)
    sd = '2018-08-18 00:00:00'
    ed = '2018-08-18 23:00:00'
    ax1.plot(np.arange(0, 24, 1), tsc[sd:ed].values * P_ctd, 'k-', linewidth=2)
    sd = '2013-08-10 00:00:00'
    ed = '2013-08-10 23:00:00'
    ax1.plot(np.arange(0, 24, 1), tsc[sd:ed].values * P_ctd, color='grey', linestyle='dashed')
    sd = '2014-08-03 00:00:00'
    ed = '2014-08-03 23:00:00'
    ax1.plot(np.arange(0, 24, 1), tsc[sd:ed].values * P_ctd, color='grey', linestyle='dashed')
    sd = '2015-08-05 00:00:00'
    ed = '2015-08-05 23:00:00'
    ax1.plot(np.arange(0, 24, 1), tsc[sd:ed].values * P_ctd, color='grey', linestyle='dashed')
    sd = '2016-08-15 00:00:00'
    ed = '2016-08-15 23:00:00'
    ax1.plot(np.arange(0, 24, 1), tsc[sd:ed].values * P_ctd, color='grey', linestyle='dashed')
    sd = '2017-08-16 00:00:00'
    ed = '2017-08-16 23:00:00'
    ax1.plot(np.arange(0, 24, 1), tsc[sd:ed].values * P_ctd, color='grey', linestyle='dashed')

    ax1.set_xticks(np.arange(0, 24, 1))
    ax1.set_xticklabels(np.arange(1, 25, 1))
    ax1.set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
    ax1.set_yticklabels([0, 50, 100, 150, 200, 250, 300, 350])
    ax1.set_ylabel('Aggregated Demand (MW)', fontsize=14)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_title('a) $D_1$: Demand Summer + Working days', fontsize=16)
    ax1.legend(('$p^{su+wd}_{d,r}(t)$', 'Scenarios',), fontsize=14, loc='upper left')

    # Plot summmer + nowork days profile
    ax2 = fig.add_subplot(4, 2, 2)
    sd = '2018-04-05 00:00:00'
    ed = '2018-04-05 23:00:00'
    ax2.plot(np.arange(0, 24, 1), (tsc[sd:ed].values - 0.1) * P_ctd, 'k-', linewidth=2)
    sd = '2013-05-05 00:00:00'
    ed = '2013-05-05 23:00:00'
    ax2.plot(np.arange(0, 24, 1), tsc[sd:ed].values * P_ctd, color='grey', linestyle='dashed')
    sd = '2014-05-05 00:00:00'
    ed = '2014-05-05 23:00:00'
    ax2.plot(np.arange(0, 24, 1), tsc[sd:ed].values * P_ctd, color='grey', linestyle='dashed')
    sd = '2015-05-05 00:00:00'
    ed = '2015-05-05 23:00:00'
    ax2.plot(np.arange(0, 24, 1), tsc[sd:ed].values * P_ctd, color='grey', linestyle='dashed')
    sd = '2016-05-05 00:00:00'
    ed = '2016-05-05 23:00:00'
    ax2.plot(np.arange(0, 24, 1), tsc[sd:ed].values * P_ctd, color='grey', linestyle='dashed')
    sd = '2017-05-05 00:00:00'
    ed = '2017-05-05 23:00:00'
    ax2.plot(np.arange(0, 24, 1), tsc[sd:ed].values * P_ctd, color='grey', linestyle='dashed')

    ax2.set_xticks(np.arange(0, 24, 1))
    ax2.set_xticklabels(np.arange(1, 25, 1))
    ax2.set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
    ax2.set_yticklabels([0, 50, 100, 150, 200, 250, 300, 350])
    ax2.set_ylabel('Aggregated Demand (MW)', fontsize=14)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_title('b) $D_2$: Demand Summer + Non-Working days', fontsize=16)
    ax2.legend(('$p^{su+nwd}_{d,r}(t)$', 'Scenarios',), fontsize=14, loc='upper left')

    # Plot winter + holidays profile
    ax3 = fig.add_subplot(4, 2, 3)
    sd = '2015-12-26 00:00:00'
    ed = '2015-12-26 23:00:00'
    ax3.plot(np.arange(0, 24, 1), tsa[sd:ed].values * P_ctd, 'k-', linewidth=2)
    sd = '2013-12-26 00:00:00'
    ed = '2013-12-26 23:00:00'
    ax3.plot(np.arange(0, 24, 1), tsa[sd:ed].values * P_ctd, color='grey', linestyle='dashed')
    sd = '2014-12-26 00:00:00'
    ed = '2014-12-26 23:00:00'
    ax3.plot(np.arange(0, 24, 1), tsa[sd:ed].values * P_ctd, color='grey', linestyle='dashed')
    sd = '2016-12-26 00:00:00'
    ed = '2016-12-26 23:00:00'
    ax3.plot(np.arange(0, 24, 1), tsa[sd:ed].values * P_ctd, color='grey', linestyle='dashed')
    sd = '2017-12-26 00:00:00'
    ed = '2017-12-26 23:00:00'
    ax3.plot(np.arange(0, 24, 1), tsa[sd:ed].values * P_ctd, color='grey', linestyle='dashed')
    sd = '2018-10-31 00:00:00'
    ed = '2018-10-31 23:00:00'
    ax3.plot(np.arange(0, 24, 1), tsa[sd:ed].values * P_ctd, color='grey', linestyle='dashed')

    ax3.set_xticks(np.arange(0, 24, 1))
    ax3.set_xticklabels(np.arange(1, 25, 1))
    ax3.set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
    ax3.set_yticklabels([0, 50, 100, 150, 200, 250, 300, 350])
    ax3.set_ylabel('Aggregated Demand (MW)', fontsize=14)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.set_title('c) $D_3$: Demand Winter + Working days', fontsize=16)
    ax3.legend(('$p^{wi+wd}_{d,r}(t)$', 'Scenarios',), fontsize=14, loc='upper left')

    # Plot winter + work days profile
    ax4 = fig.add_subplot(4, 2, 4)
    sd = '2017-02-14 00:00:00'
    ed = '2017-02-14 23:00:00'
    ax4.plot(np.arange(0, 24, 1), tsd[sd:ed].values * P_ctd, 'k-', linewidth=2)
    sd = '2013-04-08 00:00:00'
    ed = '2013-04-08 23:00:00'
    # ax4.plot(np.arange(0,24,1),tsd[sd:ed].values,'k-')
    sd = '2014-02-07 00:00:00'
    ed = '2014-02-07 23:00:00'
    # ax4.plot(np.arange(0, 24, 1),tsd[sd:ed].values,'k--')
    sd = '2015-02-05 00:00:00'
    ed = '2015-02-05 23:00:00'
    ax4.plot(np.arange(0, 24, 1), tsd[sd:ed].values * P_ctd, color='grey', linestyle='dashed')
    sd = '2016-01-01 00:00:00'
    ed = '2016-01-01 23:00:00'
    ax4.plot(np.arange(0, 24, 1), tsd[sd:ed].values * P_ctd, color='grey', linestyle='dashed')
    sd = '2017-12-24 00:00:00'
    ed = '2017-12-24 23:00:00'
    ax4.plot(np.arange(0, 24, 1), tsd[sd:ed].values * P_ctd, color='grey', linestyle='dashed')

    # [] {}
    ax4.set_xticks(np.arange(0, 24, 1))
    ax4.set_xticklabels(np.arange(1, 25, 1))
    ax4.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax4.set_yticklabels([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax4.set_ylabel('Aggregated Demand (MW)', fontsize=14)
    ax4.tick_params(axis='both', labelsize=14)
    ax4.set_title('d) $D_4$: Demand: Winter + Non-working days', fontsize=16)
    ax4.legend(('$p^{wi+nwd}_{d,r}(t)$', 'Scenarios',), fontsize=14, loc='lower left')

    """GENERATION WINTER"""
    ax5 = fig.add_subplot(4, 2, 5)
    mu = tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values.mean()
    sigma = tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values.std()
    ax5.plot(np.arange(0, 1441, 1), P_gen * np.abs(0.6 * tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values +
                                                   np.random.normal(mu * 0.2, sigma * 0.1, 1441)), color='grey',
             linestyle='dashed')
    ax5.plot(np.arange(0, 1441, 1), P_gen * 0.7 * tsg['2015-01-02 00:01:00':'2015-01-03 00:01:00'].values, 'k-',
             linewidth=2)
    # ax1.plot(np.arange(0, 1441, 1), np.abs(0.5*tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values +
    #         np.random.normal(mu * 0.1, sigma * 0.1, 1441)), color='grey', linestyle='dashed')

    # ax5.set_xlabel('Time (h)', fontsize=14)
    # ax5.set_xticks(np.arange(0, 24, 1))
    # ax5.set_xticklabels(np.arange(1, 25, 1))
    ax5.set_ylabel('Aggregated Generation (MW)', fontsize=14)
    ax5.set_xticks(np.arange(0, 61 * 24, 60))
    ax5.set_xticklabels(np.arange(0, 25, 1))
    ax5.set_yticks([0, 50, 100, 150, 200])
    ax5.set_yticklabels([0, 25, 50, 75, 100])
    ax5.tick_params(axis='both', labelsize=14)
    ax5.set_title('e) $G_1$: Generation Winter', fontsize=16)
    ax5.legend(('Scenarios', '$p^{wi}_{g,r}(t)$',), fontsize=14, loc='upper left')

    """GENERATION SUMMER"""
    ax6 = fig.add_subplot(4, 2, 6)
    mu = tsgd['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values.mean()
    sigma = tsgd['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values.std()
    ax6.plot(np.arange(0, 1441, 1), P_gen * np.abs(0.5 * tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values +
                                                   np.random.normal(mu * 0.1, sigma * 0.1, 1441)), color='grey',
             linestyle='dashed')
    ax6.plot(np.arange(0, 1441, 1), P_gen * 0.5 * tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values, 'k-',
             linewidth=2)
    # ax2.plot(np.arange(0, 1441, 1), np.abs(0.5 * tsg['2015-01-01 00:01:00':'2015-01-02 00:01:00'].values +
    #         np.random.normal(mu * 0.1, sigma * 0.1, 1441)), color='grey', linestyle='dashed')

    # ax6.set_xlabel('Time $t$ (h)', fontsize=14)
    ax6.set_ylabel('Aggregated generation (MW)', fontsize=14)
    ax6.set_xticks(np.arange(0, 61 * 24, 60))
    ax6.set_xticklabels(np.arange(0, 25, 1))
    # ax6.set_xticks(np.arange(0, 24, 1))
    # ax6.set_xticklabels(np.arange(1, 25, 1))
    ax6.set_yticks([0, 25, 50, 75, 100])
    ax6.set_yticklabels([0, 25, 50, 75, 100])
    ax6.tick_params(axis='both', labelsize=14)
    ax6.set_title('f) $G_2$: Generation Summer', fontsize=16)
    ax6.legend(('Scenarios', '$p^{su}_{g,r}(t)$',), fontsize=14, loc='upper left')

    """======LOSSES COMBINATION 1======="""
    ax7 = fig.add_subplot(4, 2, 7)
    sd = '2018-08-18 00:00:00'
    ed = '2018-08-18 23:00:00'
    ax7.plot(np.arange(0, 24, 1),
             np.abs(tsc[sd:ed].values - tsg['2015-01-01 00:01:00':'2015-01-01 23:01:00'].resample('H').mean()), 'k-',
             linewidth=1)
    sd = '2017-02-14 00:00:00'
    ed = '2017-02-14 23:00:00'
    ax7.plot(np.arange(0, 24, 1),
             np.abs(tsd[sd:ed].values - tsg['2015-01-01 00:01:00':'2015-01-01 23:01:00'].resample('H').mean()), 'k--',
             linewidth=1)
    sd = '2016-04-15 00:00:00'
    ed = '2016-04-15 23:00:00'
    ax7.plot(np.arange(0, 24, 1),
             np.abs(tsc[sd:ed].values - tsg['2015-01-01 00:01:00':'2015-01-01 23:01:00'].resample('H').mean()), 'k-',
             linewidth=2)
    sd = '2016-03-15 00:00:00'
    ed = '2016-03-15 23:00:00'
    ax7.plot(np.arange(0, 24, 1),
             np.abs(tsa[sd:ed].values - tsg['2015-01-01 00:01:00':'2015-01-01 23:01:00'].resample('H').mean()), 'k--',
             linewidth=2)

    ax7.set_xlabel('Time $t$ (h)', fontsize=14)
    ax7.set_ylabel('Active Power losses $(MW)$', fontsize=14)
    # ax7.set_xticks(np.arange(0, 24, 1))
    # ax7.set_xticklabels(np.arange(0, 24, 1))
    ax7.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax7.set_yticklabels([0.0, 2.0, 4.0, 6.0, 8.0])
    ax7.set_xticks(np.arange(0, 24, 1))
    ax7.set_xticklabels(np.arange(1, 25, 1))

    ax7.tick_params(axis='both', labelsize=14)
    ax7.set_title('g) Losses combination $(D_1 + D_2) - (G_1 + G_2)$ ', fontsize=16)
    ax7.legend(('$D_1-G_1$', '$D_1-G_2$', '$D_2-G_1$', '$D_2-G_1$'), fontsize=14, loc='upper left')

    """======LOSSES COMBINATION 2======="""
    ax8 = fig.add_subplot(4, 2, 8)
    sd = '2015-11-18 00:00:00'
    ed = '2015-11-18 23:00:00'
    ax8.plot(np.arange(0, 24, 1),
             np.abs(tsa[sd:ed].values - tsg['2015-01-01 00:01:00':'2015-01-01 23:01:00'].resample('H').mean()), 'k-',
             linewidth=1)
    sd = '2017-01-14 00:00:00'
    ed = '2017-01-14 23:00:00'
    ax8.plot(np.arange(0, 24, 1),
             np.abs(tsd[sd:ed].values - tsg['2015-01-01 00:01:00':'2015-01-01 23:01:00'].resample('H').mean()), 'k--',
             linewidth=1)
    sd = '2016-02-15 00:00:00'
    ed = '2016-02-15 23:00:00'
    ax8.plot(np.arange(0, 24, 1),
             np.abs(tsd[sd:ed].values - tsg['2015-01-01 00:01:00':'2015-01-01 23:01:00'].resample('H').mean()), 'k-',
             linewidth=2)
    sd = '2016-06-15 00:00:00'
    ed = '2016-06-15 23:00:00'
    ax8.plot(np.arange(0, 24, 1),
             np.abs(tsd[sd:ed].values - tsg['2015-01-01 00:01:00':'2015-01-01 23:01:00'].resample('H').mean()), 'k--',
             linewidth=2)

    ax8.set_xlabel('Time $t$ (h)', fontsize=14)
    ax8.set_ylabel('Active Power losses $(MW)$', fontsize=14)
    # ax8.set_xticks(np.arange(0, 24, 1))
    # ax8.set_xticklabels(np.arange(0, 24, 1))
    ax8.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax8.set_yticklabels([0.0, 2.0, 4.0, 6.0, 8.0])

    ax8.set_xticks(np.arange(0, 24, 1))
    ax8.set_xticklabels(np.arange(1, 25, 1))

    ax8.tick_params(axis='both', labelsize=14)
    ax8.set_title('h) Losses combination $(D_3 + D_4) - (G_1 + G_2)$ ', fontsize=16)
    ax8.legend(('$D_3-G_1$', '$D_3-G_2$', '$D_4-G_1$', '$D_4-G_1$'), fontsize=14, loc='upper left')

    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    plt.show()
    plt.savefig('0.0-reference_patterns_mr.pdf')
    plt.close()
