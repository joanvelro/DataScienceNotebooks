import pandas as pd
import matplotlib.pyplot as plt

"""Load Meteo Data
Hourly resolution

* Cloud Cover Fraction
* Dew Point
* Humidity
* Pressure
*Temperature
 *WindSpeed

2013-11-01 --> 2014-11-30
"""
df = pd.read_json("C:\\Users\\javelascor\\INDRA\\00_data_weather\\meteo.json")

df.plot(subplots=True)

df.index.min()
df.index.max()