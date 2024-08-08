from sqlalchemy.engine import create_engine
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from forecasting_functions import getting_sarimax_parameters
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
# from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.tools as smt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import r2_score

""" ============ CONNECT TO DB ==========
"""
if 1 == 1:
    engine = create_engine("postgresql://gpadmin:pivotal@10.0.2.6:5432/gpadmin")

""" ============ SQL WORKOUT ==========
"""
if 1 == 1:
    engine.execute("""drop table if exists madlib.ventas_timeseries;""")

    engine.execute("""create table madlib.ventas_timeseries as (
    SELECT count(*) as ventas, date_trunc('day', ventas."fecha_venta")::timestamp as fecha
    from interbus.ventas
    group by fecha
    order by fecha);
    """)

    # a = engine.execute("""select count(*) from madlib.ventas_timeseries""").fetchall()
    # print(a[0][0])

""" ============ OBTAIN DATA FRAME ========================
"""
if 1 == 1:
    sql = """SELECT * FROM madlib.ventas_timeseries;"""

    df = pd.read_sql_query(sql, engine)
    df.sort_values(by='fecha', inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.set_index(['fecha'], inplace=True)
    print(df.head())

""" ================== PLOT TIME SERIES ====================
"""
if 1 == 1:
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    t0 = df.index
    y0 = df.ventas
    ax1.plot(t0, y0)
    ax2 = fig.add_subplot(2, 1, 2)
    # t = df.loc[(df.index > '2019-01-01') & (df.index < '2019-05-01'), 'ventas'].index
    # y = df.loc[(df.index > '2019-01-01') & (df.index < '2019-05-01'), 'ventas']
    t = df.index
    y = df.ventas
    ax2.plot(t, y)
    plt.show()

""" ========= AUTOCORRELATION PLOT ===========
"""
if 0 == 1:
    fig = plt.figure(figsize=(12, 8))
    autocorrelation_plot(y)
    plt.show()

""" ====== TIME SERIES DECOMPOSITION =====
Before applying any statistical model on a time series, we want to ensure it’s stationary.
Mean, varaince and convariance betwen n ith and i+m th term of the time series shoud not be function of the time 
"""
if 1 == 1:
    result = seasonal_decompose(y, model='additive')
    print(result.trend)
    print(result.seasonal)
    print(result.resid)
    print(result.observed)

    result.plot()
    plt.show()

    if 1 == 1:
        y_log = np.log(y)  # more stationaty
        result = seasonal_decompose(y_log, model='additive')
        print(result.trend)
        print(result.seasonal)
        print(result.resid)
        print(result.observed)

        result.plot()
        plt.show()

""" ===== ROLLING STATISTICS ====
the rolling mean and rolling standard deviation increase with time.
Therefore, we can conclude that the time series is not stationary.
"""
if 1 == 1:
    print('plot rolling statistics')
    rolling_mean = y.rolling(window=12).mean()
    rolling_std = y.rolling(window=12).std()
    plt.plot(df, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Rolling Standard Deviation')
    plt.show()


""" ===== SDF TEST =======
The ADF Statistic is far from the critical values and the p-value is greater than the threshold (0.05). 
Thus, we can conclude that the time series is not stationary.
"""
if 1 == 1:
    print('SDF test')
    result = adfuller(y)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


    def get_stationarity(timeseries, title):
        """
        This function a function  run the two tests which determine
         whether a given time series is stationary.
        """
        # rolling statistics
        rolling_mean = timeseries.rolling(window=12).mean()
        rolling_std = timeseries.rolling(window=12).std()

        # rolling statistics plot
        original = plt.plot(timeseries, color='blue', label='Original')
        mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
        std = plt.plot(rolling_std, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation' + ' - ' + title)
        plt.show(block=False)

        # Dickey–Fuller test:
        result = adfuller(timeseries)
        print(' === ADF Statistic: {} ===='.format(result[0]))
        print(' === p-value: {} === '.format(result[1]))
        print(' === Critical Values: ===')
        for key, value in result[4].items():
            print('\t{}: {}'.format(key, value))
        print('\n')

""" ==== MAKING STATIONARY THE TIME SERIES ====
"""
if 1 == 1:
    """
    Taking the log of the dependent variable is as simple way of lowering the rate at which rolling 
    mean increases.
    """
    print('taking log of the time series')
    y_log = np.log(y)
    plt.plot(y_log)
    plt.show()

    """
    ARIMA model adds differencing to an ARMA model. Differencing subtracts the current value
    from the previous and can be used to transform a time series into one that’s stationary
    first-order differencing addresses linear trends, and employs the transformation zi = yi — yi-1.
    Second-order differencing addresses quadratic trends and employs a first-order difference of a
    first-order difference, namely zi = (yi — yi-1) — (yi-1 — yi-2), and so on.
    """
    """https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving
    -average-model-arima-c1005347b0d7 """
    print('differencing time series')
    yp = y - y.shift(periods=-1)
    ypp = (y - y.shift(periods=-1)) - (y.shift(periods=-1) - y.shift(periods=-2))
    yp.dropna(inplace=True)
    ypp.dropna(inplace=True)
    ax = plt.subplot(3, 1, 1)
    ax.plot(y, 'r')
    ax = plt.subplot(3, 1, 2)
    ax.plot(yp, 'b')
    ax = plt.subplot(3, 1, 3)
    ax.plot(ypp, 'k')
    plt.show()

    print('checking stationarity of differened time series')
    get_stationarity(yp, 'first order differencing')  # better to use one order of differencing (d=1)
    get_stationarity(ypp, 'second order differencing')

    """
    Taking the log of the dependent variable is as simple way of lowering the rate at which rolling 
    mean increases.
    """
    print('taking log and differencing with the rolling mean')
    rolling_mean = y_log.rolling(window=12).mean()
    y_log_minus_mean = y_log - rolling_mean
    y_log_minus_mean.dropna(inplace=True)
    title = 'Logaritmic'
    get_stationarity(y_log_minus_mean, title)

    """
    Applying exponential decay is another way of transforming a time series such that it is stationary.
    """
    print('applying exponential decay and log')
    rolling_mean_exp_decay = y_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
    y_log_exp_decay = y_log - rolling_mean_exp_decay
    y_log_exp_decay.dropna(inplace=True)
    title = 'exponential decay'
    get_stationarity(y_log_exp_decay, title)

    """
    time shifting subtract each  point by the one that preceded it.
    """
    print('applying time shifting and log')
    y_log_shift = y_log - y_log.shift()
    y_log_shift.dropna(inplace=True)
    title = 'time shifting'
    get_stationarity(y_log_shift, title)

"""  ==========  ACF and PACF PLOT ==========

The correlation between the observations at the current point in time and the observations
at all previous points in time. We can use ACF to determine the optimal number of MA terms.
The number of terms determines the order of the model.
Partial Auto Correlation Function (PACF)
As the name implies, PACF is a subset of ACF. PACF expresses the correlation between observations
made at two points in time while accounting for any influence from other data points. 
We can use PACF to determine the optimal number of terms to use in the AR model. 
The number of terms determines the order of the model.
"""
if 1 == 1:

    print('ploting ACF')
    plot_acf(y_log_minus_mean, lags=100)
    plt.show()

    print('ploting PACF')
    plot_pacf(y_log_minus_mean, lags=100)
    plt.show()

""" ================== EXPONENTIAL SMOOTHING ====================
"""
if 0 == 1:
    def exp_smooth(alpha, y):
        ft = 0
        forecast = []
        for i in range(0, len(y.values)):
            if ft == 0:
                ft = y.values[i]
                forecast.append(ft)
            else:
                ft = alpha * y.values[i] + (1 - alpha) * ft
                forecast.append(ft)


    y_forecasted_es1 = exp_smooth(0.2, y)
    y_forecasted_es2 = exp_smooth(0.4, y)
    y_forecasted_es3 = exp_smooth(0.6, y)
    y_forecasted_es4 = exp_smooth(0.8, y)

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(y)
    ax1.plot(y_forecasted_es1)
    ax1.plot(y_forecasted_es2)
    ax1.plot(y_forecasted_es3)
    ax1.plot(y_forecasted_es4)
    plt.show()

""" ========== FORECASTING ARIMA WITH STATIONARY VERSION OF DATA =====
"""
if 1 == 1:
    print('Forecasting with ARIMA')
    # fit model
    y_log.rename('sales', inplace=True)
    model = ARIMA(y_log, order=(5, 1, 2))
    results = model.fit(disp=-1)
    # we compare the differenced version of the log time series
    # because d=1 in ARIMA model

    plt.figure()
    plt.plot(y_log_shift, color='blue', label='data')
    plt.plot(results.fittedvalues, color='red', label='ARIMA')
    plt.legend()
    plt.show()

    # calculate metrics R2, APE and MAPE

    APE = np.abs(y_log_shift - results.fittedvalues) / y_log_shift
    plt.figure
    plt.plot(APE)
    plt.show()

    r2 = r2_score(y_log_shift, results.fittedvalues)
    MAPE = APE.mean()
    print('r2:', r2)
    print('MAPE:', MAPE)

    # we can see how the model compares to the original time series.
    predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(y_log.iloc[0], index=y_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    predictions_ARIMA = np.exp(predictions_ARIMA_log)

    plt.figure()
    plt.plot(y, 'r', label='real data')
    plt.plot(predictions_ARIMA, 'b', label='ARIMA forecast')
    plt.legend()
    plt.show()

    # plot predictions
    results.plot_predict(1, 364)
    plt.show()




def getting_sarimax_parameters(y):
    """ Parameter Selection for the ARIMA Time Series Model
    """
    import itertools
    import pandas as pd
    import numpy as np
    import warnings
    import statsmodels.api as sm
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
    print(df.head())
    print(
        'ARIMA{}x{} - AIC:{}'.format((df.p[0], df.d[0], df.q[0]), (df.S1[0], df.S2[0], df.S3[0], df.S4[0]), df.AIC[0]))


""" ================== OBTAIN SARIMAX PARAMETERS ==============
"""
if 0 == 1:
    getting_sarimax_parameters(y)

""" ==================== FIT SARIMAX MODEL ========================
"""
if 0 == 1:
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=(1, 1, 0),
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    sarimax_fitted = mod.fit()
    print(sarimax_fitted.summary().tables[1])
    sarimax_fitted.plot_diagnostics(figsize=(8, 6))
    plt.show()

""" ==================== TEST SARIMAX MODEL  ========================
"""
if 0 == 1:
    pred = sarimax_fitted.get_prediction(start=0, dynamic=False)
    y_hat = pred.predicted_mean

    fig = plt.figure(num=None, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(y, 'k-', label='observed')
    ax.plot(y_hat, 'r', label='Forecast')

""" ==================== FORECAST WITH SARIMAX ========================
"""
if 0 == 1:
    forecast = sarimax_fitted.get_forecast(steps=24 * 7)
    y_hat = forecast.predicted_mean

    fig = plt.figure(num=None, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(y, 'k-', label='observed')
    ax.plot(y_hat, 'r', label='Forecast')

""" ====================== FIT ARIMA MODEL ======================
"""
if 0 == 1:
    arima_model = ARIMA(y, order=(1, 1, 1)).fit(disp=0)
    print(arima_model.summary())

""" ====================== ROLLING FORECAST ARIMA MODEL WITH TEST/TRAIN SPLITTING ======================
"""
if 0 == 1:
    Y = y.values
    size = int(len(Y) * 0.70)
    train, test = Y[0:size], Y[size:len(Y)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 1))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f, error=%f' % (yhat, obs, 100 * abs(obs - yhat) / obs))
    r2 = r2_score(test, predictions)
    print('Test R2: %.3f' % r2)

    # plot
    df_results = pd.DataFrame({'test': test, 'predictions': np.asarray(predictions).ravel()})

    plt.figure()
    plt.plot(df_results['test'], color='red', label='real')
    plt.plot(df_results['predictions'], color='blue', label='prediction')
    plt.legend()
    plt.show()

""" =================== ROLLING FORECAST ARIMA MODEL ALL DATA  ========================
"""
if 0 == 1:
    n_periods = 30 * 2
    Y = y.values
    Yp = Y.tolist()  # data to be updated
    predictions = list()
    for t in range(n_periods):
        # fit arima model
        model = ARIMA(Yp, order=(5, 1, 2))
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat[0])
        Yp.append(yhat[0])
        print('|', end=' ')

    idx0 = pd.date_range(start=y.index[len(y) - 1], periods=n_periods + 1)
    idx1 = pd.concat([pd.DataFrame(index=y.index), pd.DataFrame(index=idx0[1:])], axis=1).index
    df_results1 = pd.DataFrame(data=Yp, index=idx1, columns=['data'])

    idx = pd.date_range(start=y.index[len(y) - 1], periods=n_periods + 1)
    df_results2 = pd.DataFrame(data=predictions, index=idx[1:], columns=['predictions'])

    plt.figure()
    plt.plot(df_results1['data'], color='red', label='real')
    plt.plot(df_results2['predictions'], color='blue', label='prediction')
    plt.legend()
    plt.show()
