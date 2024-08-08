import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')
import itertools
import warnings
import statsmodels.api as sm
#from pandas.core import datetools

df0 = pd.read_csv('weather_data.csv',sep=',') # 1-min resolution
date = []
for k in range(0,len(df0)):
    date.append(pd.Timestamp(int(df0.index[k][0:4]), int(df0.index[k][5:7]), int(df0.index[k][8:10]), int(df0.index[k][11:13]), int(df0.index[k][14:16])))
df0['date'] = date
df0 = df0.set_index('date')

df0 = df0.fillna(df0.bfill())


ts = df0['Air Temperature [deg C]']

ts = ts.resample('H').mean()


print(ts.head())
print(ts.tail())
print(len(ts))
print(type(ts))


start_date = '2015-01-01 00:00:00'
end_date   = '2015-01-05 00:00:00'
date       = '2015-01-03 00:00:00'
y = ts[start_date:end_date]

fig = plt.figure(num=None, figsize=(10,8), dpi=120, facecolor='w', edgecolor='k')
ax = fig.add_subplot(1, 1, 1)
ax.plot(y,'k-')
plt.ylabel('Ambient Temperature $T_{amb}$ $(ºC)$', fontsize=18)
plt.xlabel('Time t', fontsize=18)
plt.legend(('$T_{amb}$ $(ºC)$',), fontsize=16)
plt.savefig('fig_1_temperature_data.eps')
plt.show()
plt.close()



# Parameter Selection for the ARIMA Time Series Model
if 0==1:
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
    k=0
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
                k+=1
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
    df = pd.DataFrame({'p':p,'d':d,'q':q,'S1':S1,'S2':S2,'S3':S3,'S4':S4,'AIC':AIC}).sort_values(by="AIC",ascending=True)
    df = df.reset_index(drop=True)
    print('ARIMA{}x{} - AIC:{}'.format((df.p[0],df.d[0],df.q[0]), (df.S1[0],df.S2[0],df.S3[0],df.S4[0]), df.AIC[0]))



# Fitting an ARIMA Time Series Model
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(12, 8))
plt.savefig('fig_2_temperature_arima_model.eps')
plt.show()
plt.close()



# Validating forecast
pred = results.get_prediction(start=pd.to_datetime(date), dynamic=False)
pred_ci = pred.conf_int()


ax = y['2015':].plot(color='black',label='observed', figsize=(10, 8))
pred.predicted_mean.plot(color='red',ax=ax, label='One-step ahead Forecast', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='gray', alpha=.2)
ax.set_ylabel('Ambient Temperature $T_{amb}$ $(ºC)$', fontsize=18)
plt.xlabel('Time t', fontsize=18)
plt.legend(('Observed','Forecast'),fontsize=18)
plt.savefig('fig_3_temperature_training.eps')
plt.show()
plt.close()

# Compute the mean square error
y_forecasted = pred.predicted_mean
y_truth = y[date:]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 4)))




# Validating forecast dynamic
pred_dynamic = results.get_prediction(start=pd.to_datetime(date), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()
ax = y['2013':].plot(color='black',label='observed', figsize=(10, 8))
pred_dynamic.predicted_mean.plot(color='red',label='Dynamic Forecast', ax=ax)
ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='gray', alpha=.6)
#ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(date), y.index[-1],
#                 alpha=.1, zorder=-1)
ax.set_ylabel('Ambient Temperature $T_{amb}$ $(ºC)$', fontsize=18)
ax.set_xlabel('Time t', fontsize=18)
plt.legend(('Observed','Forecast'),fontsize=18)
plt.savefig('fig_4_temperature_training_dynamic.eps')
plt.show()
plt.close()


# Compute the mean square error
y_forecasted = pred_dynamic.predicted_mean
y_truth = y[date:]
# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {} (Dynamic)'.format(round(mse, 2)))



# Forecast one day ahead
pred_uc = results.get_forecast(steps=24)
pred_ci = pred_uc.conf_int() # confidence intervals

fig = plt.figure(num=None, figsize=(10,8), dpi=120, facecolor='w', edgecolor='k')
ax = fig.add_subplot(1, 1, 1)
ax.plot(y,'k-',label='observed')
ax.plot(pred_uc.predicted_mean,'r--',label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='gray', alpha=.25)
ax.set_ylabel('Ambient Temperature $T_{amb}$ $(ºC)$', fontsize=18)
ax.set_xlabel('Time t', fontsize=18)
plt.legend(('Observed','Forecast'),fontsize=18)
plt.savefig('fig_5_temperature_forecast.eps')
plt.show()
plt.close()


T_hat = pred_uc.predicted_mean
T_obs = y
T_ucl = pred_ci.iloc[:, 0]
T_lcl = pred_ci.iloc[:, 1]
df_temperature_results = pd.DataFrame({'T_hat':T_hat,'T_obs':T_obs,'T_ucl':T_ucl,'T_lcl':T_lcl})
df_temperature_results.to_csv("results_forecast_temperature.csv")





