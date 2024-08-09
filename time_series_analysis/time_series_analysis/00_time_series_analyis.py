"""source:
 https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b"""

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pylab import rcParams

rcParams['figure.figsize'] = 18, 8

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

df = pd.read_excel(".data/sample_superstore.xls")

furniture = df.loc[df['Category'] == 'Furniture']
print('initial date:', furniture['Order Date'].min())
print('end date:', furniture['Order Date'].max())

cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode',
        'Customer ID', 'Customer Name', 'Segment',
        'Country', 'City', 'State', 'Postal Code',
        'Region', 'Product ID', 'Category', 'Sub-Category',
        'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')
print('null values:\n', furniture.isnull().sum())

furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

furniture = furniture.set_index('Order Date')

y = furniture['Sales'].resample('MS').mean()

print(y['2017':])

if 0 == 1:
    y.plot(figsize=(8, 4))
    plt.show()


if 1 == 1:
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    print('observed nan values:', np.isnan(decomposition.observed).sum())
    print('trend nan values:', np.isnan(decomposition.trend).sum())
    print('seasonal nan values:', np.isnan(decomposition.seasonal).sum())
    print('residual nan values:', np.isnan(decomposition.resid).sum())

if 1 == 1:
    x_axes = pd.date_range(y.index.min(), y.index.max(), freq='M')
    fig2 = plt.figure(num=1, figsize=(12, 6), dpi=80, facecolor='k', edgecolor='k')
    ax1 = fig2.add_subplot(4, 1, 1)
    ax1.plot(y, color='black', linestyle='-', linewidth=2)
    ax1.plot(decomposition.observed, color='black', linestyle='-.', linewidth=2)
    ax1.set_ylabel("Observed", fontsize=12)
    ax1.set_xticks(x_axes)
    ax1.set_xticklabels(())
    ax1.grid(True)
    ax2 = fig2.add_subplot(4, 1, 2)
    ax2.plot(decomposition.trend.dropna(), color='black', linestyle='--', linewidth=2)
    ax2.set_ylabel("Trend", fontsize=12)
    ax2.set_xticks(x_axes)
    ax2.set_xticklabels(())
    ax2.grid(True)
    ax3 = fig2.add_subplot(4, 1, 3)
    ax3.plot(decomposition.seasonal.dropna(), color='black', linestyle='--', linewidth=3)
    ax3.set_ylabel("Seasonal", fontsize=12)
    ax3.set_xticks(x_axes)
    ax3.set_xticklabels(())
    ax3.grid(True)
    ax4 = fig2.add_subplot(4, 1, 4)
    ax4.plot(decomposition.resid.dropna(), color='red', linestyle=':', linewidth=2)
    ax4.set_ylabel("Residuals", fontsize=12)
    ax4.set_xticks(x_axes)
    ax4.grid(True)
    ax4.set_xticklabels([i for i in range(0, len(x_axes))])
    #fig2.savefig('decomposition.png', dpi=500)
    #plt.close()

""" Parameter Selection using grid search"""
if 1 == 0:
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue

"""fitting the ARIMA model"""
if 1 == 1:
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])

"""model diagnostic"""
if 0 == 1:
    results.plot_diagnostics(figsize=(16, 8))
    #plt.savefig('diagnostic.png')
    #plt.close()

"""Validating forecast"""
""" forecasts to start at 2017–01–01 to the end of the data."""
if 1 == 1:
    pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=True)
    pred_ci = pred.conf_int()
    ax = y['2014':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Furniture Sales')
    plt.legend()
    plt.show()
    #plt.savefig('validation.png')
    #plt.close

    y_forecasted = pred.predicted_mean
    y_truth = y['2017-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

"""Producing and visualizing forecasts"""
if 0 == 0:
    pred_uc = results.get_forecast(steps=100)
    """confidence interval"""
    pred_ci = pred_uc.conf_int()
    ax2 = y.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax2.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Furniture Sales')
    plt.legend()
    plt.show()
    #plt.savefig('forecast.png')
    #plt.close()


if 0 == 0:
    furniture = df.loc[df['Category'] == 'Furniture']
    office = df.loc[df['Category'] == 'Office Supplies']
    print(furniture.shape)
    print(office.shape)

    cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country',
            'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name',
            'Quantity', 'Discount', 'Profit']
    furniture.drop(cols, axis=1, inplace=True)
    office.drop(cols, axis=1, inplace=True)
    furniture = furniture.sort_values('Order Date')
    office = office.sort_values('Order Date')
    furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
    office = office.groupby('Order Date')['Sales'].sum().reset_index()
    furniture = furniture.set_index('Order Date')
    office = office.set_index('Order Date')
    y_furniture = furniture['Sales'].resample('MS').mean()
    y_office = office['Sales'].resample('MS').mean()
    furniture = pd.DataFrame({'Order Date': y_furniture.index, 'Sales': y_furniture.values})
    office = pd.DataFrame({'Order Date': y_office.index, 'Sales': y_office.values})
    store = furniture.merge(office, how='inner', on='Order Date')
    store.rename(columns={'Sales_x': 'furniture_sales', 'Sales_y': 'office_sales'}, inplace=True)
    store.head()

    plt.figure(figsize=(20, 8))
    plt.plot(store['Order Date'], store['furniture_sales'], 'b-', label='furniture')
    plt.plot(store['Order Date'], store['office_sales'], 'r-', label='office supplies')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Sales of Furniture and Office Supplies')
    plt.legend()
    plt.show()

    first_date = store.ix[np.min(list(np.where(store['office_sales'] >
                                               store['furniture_sales'])[0])), 'Order Date']
    print("Office supplies first time produced higher sales than furniture is {}.".format(first_date.date()))

