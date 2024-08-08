import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Read original date
if 1 == 1:
    df_val = pd.read_csv('./data/data_loads_dem_original_pyomo.csv')

# Create time series
if 1 == 1:
    Pdem1 = np.zeros(70000)
    k = 0
    for t in df_val.data_time.unique():
        Pdem1[k] = df_val[df_val.data_time == t].PDem.sum()
        k += 1
    Pdem1 = Pdem1[0:k]
    Pdem2 = np.multiply(Pdem1, 1.02)
    Pdem3 = np.multiply(Pdem1, 1.04)
    Pdem4 = np.multiply(Pdem1, 1.06)
    Pdem = np.concatenate((Pdem1, Pdem2, Pdem3, Pdem4), axis=0)
    start_date = '2013-01-01 00:00:00'
    end_date = '2013-01-05 00:00:00'
    dates = pd.date_range(start_date, end_date, freq='10min')
    dates = dates[0:len(dates) - 1]
    df = pd.DataFrame({'Pdem': Pdem, 'dates': dates})
    df = df.set_index('dates')
    # y = df.Pdem
    y = df.Pdem.resample('H').mean()

# Plot demand
if 0 == 1:
    fig = plt.figure(num=None, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(y, 'k-')
    plt.show()

# Fitting an ARIMA Time Series Model
if 1 == 1:
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])
    results.plot_diagnostics(figsize=(10, 8))
    plt.show()

# Validating forecast
if 1 == 1:
    date = '2013-01-04 00:00:00'
    pred = results.get_prediction(start=pd.to_datetime(date), dynamic=False)
    pred_ci = pred.conf_int()
    ax = y['2013':].plot(color='black', label='observed', figsize=(10, 8))
    pred.predicted_mean.plot(color='red', ax=ax, label='One-step ahead Forecast', alpha=.7)
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='gray', alpha=.2)
    plt.legend(('Observed', 'Forecast'), fontsize=18)
    plt.show()

# Validating forecast dynamic
if 0 == 1:
    date = '2013-01-04 00:00:00'
    pred_dynamic = results.get_prediction(start=pd.to_datetime(date), dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()
    ax = y['2013':].plot(color='black', label='observed', figsize=(10, 8))
    pred_dynamic.predicted_mean.plot(color='red', label='Dynamic Forecast', ax=ax)
    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='gray', alpha=.6)
    # ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(date), y.index[-1],
    #                 alpha=.1, zorder=-1)
    ax.set_ylabel('Load Demand $\hat{p}^p_{k,t}$ $(pu)$', fontsize=18)
    ax.set_xlabel('Time t', fontsize=18)
    plt.legend(('Observed', 'Forecast'), fontsize=18)
    plt.show()

# Forecast one day ahead
if 1 == 1:
    pred_uc = results.get_forecast(steps=24)
    pred_ci = pred_uc.conf_int()  # confidence intervals
    fig = plt.figure(num=None, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(y, 'k-', label='observed')
    ax.plot(pred_uc.predicted_mean, 'r', label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='gray', alpha=.2)
    ax.set_ylabel('Load Demand $\hat{p}^p_{k,t}$ $(pu)$', fontsize=18)
    ax.set_xlabel('Time t', fontsize=18)
    plt.legend(('Observed', 'Forecast'), fontsize=18)
    plt.show()

if 1 == 1:
    Pd_hat = pred_uc.predicted_mean
    Pd_obs = y
    Pd_ucl = pred_ci.iloc[:, 0]
    Pd_lcl = pred_ci.iloc[:, 1]
    df_demand_results = pd.DataFrame({'Pd_hat': Pd_hat, 'Pd_obs': Pd_obs, 'Pd_ucl': Pd_ucl, 'Pd_lcl': Pd_lcl})
    df_demand_results.to_csv("./results/results_forecast_demand_val.csv")
