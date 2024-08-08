





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
