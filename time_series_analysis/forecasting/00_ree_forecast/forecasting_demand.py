import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('fivethirtyeight')
import itertools
import warnings
import statsmodels.api as sm

# from pandas.core import datetools

if 1 == 1:
    df = pd.read_csv('./data/REE_profiles.csv')
    date = []
    for k in range(0, len(df)):
        date.append(
            pd.Timestamp(int(df.date[k][0:4]), int(df.date[k][5:7]), int(df.date[k][8:10]), int(df.date[k][11:13])))
    df = df.drop(['date', 'season'], axis=1)
    df['date'] = date
    df = df.set_index('date')
    df = df * 1000
    ts = df.coef_a
    # ts = ts.resample('D').mean()
    ts = ts.fillna(ts.bfill())
    print(ts.head())
    print(ts.tail())
    print(len(ts))
    print(type(ts))
    start_date = '2013-01-01 00:00:00'
    end_date = '2013-01-05 00:00:00'
    date = '2013-01-03 00:00:00'
    y = ts[start_date:end_date]
    print(y.index.min())
    print(y.index.max())

""" plot demand
"""
if 1 == 1:
    fig = plt.figure(num=None, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(y, 'k-')
    plt.ylabel('Load Demand $\hat{p}^p_{k,t}$ $(pu)$', fontsize=18)
    plt.xlabel('Time t', fontsize=18)
    plt.legend(('$\hat{p}^p_{k,t}$ $(pu)$',), fontsize=18)
    plt.savefig('./results/fig_demand_1_data.pdf')
    plt.show()

""" Parameter Selection for the ARIMA Time Series Model
"""
if 1 == 1:
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    print('')
    p = np.arange(100000)
    d = np.arange(100000)
    q = np.arange(100000)
    S1 = np.arange(100000)
    S2 = np.arange(100000)
    S3 = np.arange(100000)
    S4 = np.arange(100000)
    AIC = np.arange(100000)
    warnings.filterwarnings("ignore")  # specify to ignore warning messages
    warnings.simplefilter("ignore")
    k = 0
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                p[k] = param[0]
                d[k] = param[1]
                q[k] = param[2]
                S1[k] = param_seasonal[0]
                S2[k] = param_seasonal[1]
                S3[k] = param_seasonal[2]
                S4[k] = param_seasonal[3]
                AIC[k] = results.aic
                k += 1
            except:
                continue
    p = p[0:k]
    d = d[0:k]
    q = q[0:k]
    S1 = S1[0:k]
    S2 = S2[0:k]
    S3 = S3[0:k]
    S4 = S4[0:k]
    AIC = AIC[0:k]
    df = pd.DataFrame({'p': p, 'd': d, 'q': q, 'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4, 'AIC': AIC}).sort_values(
        by="AIC", ascending=True)
    df = df.reset_index(drop=True)
    print(
        'ARIMA{}x{} - AIC:{}'.format((df.p[0], df.d[0], df.q[0]), (df.S1[0], df.S2[0], df.S3[0], df.S4[0]), df.AIC[0]))

""" Fitting an ARIMA Time Series Model
"""
if 1 == 1:
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])
    results.plot_diagnostics(figsize=(8, 6))
    plt.savefig('./results/fig_demand_2_arima_model.pdf')
    plt.show()

""" Validating forecast
"""
if 1 == 1:
    pred = results.get_prediction(start=pd.to_datetime(date), dynamic=False)
    y_hat = pred.predicted_mean

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.plot(y['2013':], color='black', label='observed', )
    ax.plot(y_hat, label='One-step ahead Forecast', alpha=.7, color='red')

    pred_ci = pred.conf_int()
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='gray', alpha=.2)
    plt.ylabel('Load Demand $\hat{p}^p_{k,t}$ $(pu)$', fontsize=18)
    plt.xlabel('Time t', fontsize=18)
    plt.legend(('Observed', 'Forecast'), fontsize=18)
    plt.show()
    plt.savefig('./results/fig_demand_3_training.pdf')

""" Calculate MSE of forecast
"""
if 1 == 1:
    y_truth = y[date:]
    rmse = ((y_hat - y_truth) ** 2).mean()
    print(' RMSE forecast is {}'.format(round(rmse, 6)))

""" Forecast  
"""
if 1 == 1:
    forecast = results.get_forecast(steps=48)
    y_hat = forecast.predicted_mean

    fig = plt.figure(num=None, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(y, 'k-', label='observed')
    ax.plot(y_hat, 'r', label='Forecast')

    pred_ci = forecast.conf_int()  # confidence intervals
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='gray', alpha=.2)
    ax.set_ylabel('Load Demand $\hat{p}^p_{k,t}$ $(pu)$', fontsize=18)
    ax.set_xlabel('Time t', fontsize=18)
    plt.legend(('Observed', 'Forecast'), fontsize=18)
    plt.savefig('./results/fig_demand_5_forecast.pdf')
    plt.show()


""" Save results  
"""
if 1 == 1:
    Pd_hat = pred_uc.predicted_mean
    Pd_obs = y
    Pd_ucl = pred_ci.iloc[:, 0]
    Pd_lcl = pred_ci.iloc[:, 1]
    df_demand_results = pd.DataFrame({'Pd_hat': pred_uc.predicted_mean, 'Pd_obs': y, 'Pd_ucl': pred_ci.iloc[:, 0],
                                      'Pd_lcl': pred_ci.iloc[:, 1]})

    df_demand_results.to_csv("./results/results_forecast_demand.csv")
