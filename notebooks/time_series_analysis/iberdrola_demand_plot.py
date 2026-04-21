import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio

pio.renderers.default = "browser"

if 1 == 1:
    df = pd.read_csv("iberdrola_demand.csv", sep=";", dtype={'time': str, 'demand': np.int})
    df['demand_wh'] = df[' demand_wh'].astype(int)
    df['date'] = pd.to_datetime(df['time'], format='%d/%m/%Y %H:%M:%S')
    df.drop(['time', ' demand_wh'], axis=1, inplace=True)
    df.set_index(['date'], drop=True, inplace=True)
    # df.rename(columns={' demand_wh': 'demand_wh'}, inplace=True)
    df['demand_kwh'] = df['demand_wh'].apply(lambda x: x * 1 / 1000)

    df['energy_cumulated_kwh'] = np.cumsum(df['demand_kwh'])

    ts = df['demand_kwh'].resample('D').sum()

    # plot all daily energy demand
    if 0 == 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index,
                                 y=ts.values,
                                 name='Daily energy demand (kWh)')
                      )
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title="Daily Energy Demand",
                          yaxis_title='Daily Energy Demand (kWh)',
                          xaxis_title='Days')
        fig.show()

    # plot all hourly power demand
    if 0 == 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index,
                                 y=df['demand_kwh'],
                                 name='Demand (Wh)')
                      )
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title="Hourly Power Demand",
                          yaxis_title='Active Power (kW)',
                          xaxis_title='Time (h)')
        fig.show()

    # plot all hourly power demand
    if 0 == 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pd.date_range('2019-12-26', '2019-12-27', freq='H'),
                                 y=df.loc['2019-12-26':, ['demand_kwh']].values.reshape(-1),
                                 name='Demand (Wh)')
                      )
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title="Hourly Power Demand",
                          yaxis_title='Active Power (kW)',
                          xaxis_title='Time (h)')
        fig.show()

    # plot cumulated energy consuption
    if 0 == 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index,
                                 y=df['energy_cumulated_kwh'],
                                 name='Energy')
                      )
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title="Energy Cumulated",
                          yaxis_title='Energy consuption (kWh)',
                          xaxis_title='Time (h)')
        fig.show()

    # hourly demand and cumulated demand
    if 0 == 0:
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x=df.index,
                       y=df['energy_cumulated_kwh'],
                       name="Cumulated Energy Consumption"),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(x=ts.index,
                       y=ts.values,
                       name="Daily Energy Demand"),
            secondary_y=False,
        )
        # Set x-axis title
        fig.update_xaxes(title_text="Time (h)")

        # Set y-axes titles
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"),
                          title_text="Household energy demand",
                          legend=dict(x=0.5, y=1.2))
        fig.update_yaxes(title_text="<b>Cumulated Energy Consumption (kWh)</b>",
                         secondary_y=True,
                         color="blue")
        fig.update_yaxes(title_text="<b>Daily Energy Demand (kWh)</b>",
                         secondary_y=False,
                         color="red")

        fig.show()

    # plot weekly demand pattern
    if 1 == 0:
        start_date = '2019-10-07'
        end_date = '2019-10-13'
        df_aux = df[start_date:end_date]
        if 0 == 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_aux.index,
                                     y=df_aux['demand_kwh'],
                                     name='Demand (Wh)')
                          )
            fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
            fig.update_layout(title="Hourly Power Demand",
                              yaxis_title='Active Power (kW)',
                              xaxis_title='Time (h)')
            fig.show()

        # plot weekly daily energy demand
        if 0 == 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts[start_date:end_date].index,
                                     y=ts[start_date:end_date].values,
                                     name='Daily energy demand (kWh)')
                          )
            fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
            fig.update_layout(title="Daily Energy Demand",
                              yaxis_title='Daily Energy Demand (kWh)',
                              xaxis_title='Days')
            fig.show()
