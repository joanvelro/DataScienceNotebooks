import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
plt.rcParams.update({'font.size': 12})
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Times New Roman']})
rc('text', usetex=True)

if 1 == 1:
    df = pd.read_csv('REE_profiles.csv')
    # format data as time series
    date = []
    if 1 == 1:
        for k in range(0, len(df)):
            for minute in np.arange(0,60,10):
                date.append(pd.Timestamp(int(df.date[k][0:4]),
                                         int(df.date[k][5:7]),
                                         int(df.date[k][8:10]),
                                         int(df.date[k][11:13]), minute))

                #print('year:',int(df.date[k][0:4]),'month:',int(df.date[k][5:7]),'day:',int(df.date[k][8:10]),'hour:',int(df.date[k][11:13]),'min:',min)
    #df = df.drop(['date','season'],axis=1)
    #df['date'] = date
    #df = df.set_index('date')

    df.to_csv('REE_time_series.csv', index=False)


