import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from datetime import datetime, date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

# Load PV Data
store = pd.HDFStore('pv.hd5')
data = store['pv']
store.close()

# Drop other columns
ts = data.loc[:, ['dataPowerOut', 'dataTemp', 'Insolation']]

ts.plot(subplots=True)

ts['2014-10-04':'2014-10-12'].plot(subplots=True)

"""Create feature vector
The feature vector is made up from lagged active energy, calendar information and meteorological forecasts

"""


def get_train_dataset(ts, train_date, steps=12, freq='5min'):  # Lagged Active

    # Select rows up to the train_date
    target = ts.asfreq(freq).loc[:train_date, :]

    for n in np.arange(1, steps + 1):
        target['lagged' + str(n)] = target['dataPowerOut'].shift(n)

    # Time
    target['day'] = pd.Series(target.index.day, target.index)
    target['hour'] = pd.Series(target.index.hour, target.index)
    target['weekday'] = pd.Series(target.index.weekday, target.index)
    target['week'] = pd.Series(target.index.week, target.index)

    # Drop Nans: Train only with complete feature vectors
    target = target.dropna()

    # Split feature vector / target
    train_x = target.drop('dataPowerOut', axis=1)  # Feature vector
    train_y = target.ix[:, 'dataPowerOut']  # Actual measurements

    # Scale vectors: substract mean of every column and normalize to a std=1
    scaler_x = StandardScaler().fit(train_x.values)
    scaler_y = StandardScaler().fit(train_y.values)
    train_X = scaler_x.transform(train_x.copy())
    train_Y = scaler_y.transform(train_y.copy())

    return scaler_x, scaler_y, train_X, train_Y, target

"""Use the available measurements up to a certain date to fit a model"""

train_date = '2015-02-01'
scaler_x, scaler_y, train_X, train_Y, target = get_train_dataset(ts, train_date, steps=12, freq='5min')

clf = linear_model.LinearRegression()
clf.fit(train_X, train_Y)

"""Load Profiles
In case measurements are missing from the database, the algorithm falls back to load profiles, which are the average 
values of the train dataset grouped by hour of the day and minute of the hour.
"""

# Select train dataset
target = ts.asfreq('5min').loc[:train_date, :]

# Group measurements based on calendar information
grouped = target.loc[:, 'dataPowerOut'].astype(float).groupby([lambda x: x.month, lambda x: x.hour, lambda x: x.minute])

# Calculate the average values
profile = grouped.mean().copy()

# Nans are zeros
profile = profile.fillna(value=0.0)

"""Test model
The forecast function outputs the the next forecasted values of active energy. 
The number of values it returns is set by the variable *horizon*. 
"""


def forecast(test_date, retrain=False, horizon=3, freq='5min', lagged_measurement=12 * 12):
    # Define global variables
    global scaler_x, scaler_y, train_X, train_Y, clf

    # Dataframe containing output
    forecasts = pd.DataFrame()

    # Range of dates of previous measurements
    drange = pd.date_range(end=test_date, periods=lagged_measurement, freq=freq)

    for h in range(horizon):
        f_date = drange[-1] + timedelta(minutes=5 * (1 + h))

        # Get Meteo information
        met = test_filled.loc[f_date, ['dataTemp', 'Insolation']].values.squeeze()

        # Get previous measurements
        if h == 0:
            lagged_meas = test_filled.loc[drange[0]:drange[-1], 'dataPowerOut'].values.squeeze()
        else:
            lagged_meas = np.roll(lagged_meas, -1)
            lagged_meas[-1] = forecast

        # Append to feature vector meteo and date information
        date_info = np.array([f_date.day, f_date.hour, f_date.weekday(), f_date.week])

        # Create feature vector
        feature_vector = met.copy()
        feature_vector = np.hstack((feature_vector, lagged_meas.squeeze()))
        feature_vector = np.hstack((feature_vector, date_info.squeeze()))

        # If the feature vector contains nans fall back to the load profiles
        if np.isnan(feature_vector).sum():
            forecast = profile[f_date.month][f_date.hour][f_date.minute]
        else:
            # Normalize
            feature_vector = scaler_x.transform(feature_vector)

            # Predict
            forecast = clf.predict(feature_vector)

            # Denormalize
            forecast = scaler_y.inverse_transform(forecast).item()

        # Forecasts of the next h values
        forecasts.loc[drange[-1] + timedelta(minutes=5), 'f_' + str(h)] = forecast

        # If measurement is empty, fill with prediction
        if h == 0 and np.isnan(test.loc[f_date, 'dataPowerOut']):
            test_filled.loc[f_date, 'dataPowerOut'] = forecasts.loc[f_date, 'f_0']

    # Optional: retrain the model using the latest measurements
    if retrain:
        scaler_x, scaler_y, train_X, train_Y = get_train_dataset(ts, f_date, steps=lagged_measurement, freq=freq)
        # clf.fit(train_X, train_Y)

    return forecasts

test = ts.loc[train_date:, :].copy()
test = test.asfreq('5min')

# Copy of the test database. If a measurement is missing it is filled with a forecast
test_filled = test.copy()

# First date to be forecasted
test_date = datetime(2015, 2, 2, 0, 0, 0)

# Horizon: number of forecasts to produce at each step
horizon = 3

# Empty dataframe that will contain the forecasts
forecasts = pd.DataFrame()

# Simulate the process of forecasting. From a given date it outputs h forecasts for every hour.
for i in range(1500):
    test_date = test_date + timedelta(minutes=5)
    forecasts = pd.concat([forecasts, forecast(test_date.strftime('%Y-%m-%d %H:%M:%S'),
                                               retrain=False, horizon=3, freq='5min', lagged_measurement=12)])

# Plot
plt.plot(test.loc[:, 'dataPowerOut']) # Actual measurement
plt.plot(forecasts.loc[:, 'f_0']) # Hour ahead forecasts

# Dataframe containing forecasts
forecasts.head()