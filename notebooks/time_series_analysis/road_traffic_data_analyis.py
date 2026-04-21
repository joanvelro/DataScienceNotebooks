import pandas as pd
import numpy as np
import plotly.express as px
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#import psutil

default_colors = {
    'muted_blue': '#1f77b4',
    'safety_orange': '#ff7f0e',
    'cooked_asparagus_green': '#2ca02c',
    'brick_red': '#d62728',
    'muted_purple': '#9467bd',
    'chestnut_brown': '#8c564b',
    'raspberry_yogurt_pink': '#e377c2',
    'middle_gray': '#7f7f7f',
    'curry_yellow_green': '#bcbd22',
    'blue_teal': '#17becf'
}

css_colors = ['aliceblue',
              'antiquewhite',
              'aqua',
              'aquamarine',
              'azure',
              'beige',
              'bisque',
              'black',
              'blanchedalmond',
              'blue',
              'blueviolet',
              'brown',
              'burlywood',
              'cadetblue',
              'chartreuse',
              'chocolate',
              'coral',
              'cornflowerblue',
              'cornsilk',
              'crimson',
              'cyan',
              'darkblue',
              'darkcyan',
              'darkgoldenrod',
              'darkgray',
              'darkgrey',
              'darkgreen',
              'darkkhaki',
              'darkmagenta',
              'darkolivegreen',
              'darkorange',
              'darkorchid',
              'darkred',
              'darksalmon',
              'darkseagreen',
              'darkslateblue',
              'darkslategray',
              'darkslategrey',
              'darkturquoise',
              'darkviolet',
              'deeppink',
              'deepskyblue',
              'dimgray',
              'dimgrey',
              'dodgerblue',
              'firebrick',
              'floralwhite',
              'forestgreen',
              'fuchsia',
              'gainsboro',
              'ghostwhite',
              'gold',
              'goldenrod',
              'gray',
              'grey',
              'green',
              'greenyellow',
              'honeydew',
              'hotpink',
              'indianred',
              'indigo',
              'ivory',
              'khaki',
              'lavender',
              'lavenderblush',
              'lawngreen',
              'lemonchiffon',
              'lightblue',
              'lightcoral',
              'lightcyan',
              'lightgoldenrodyellow',
              'lightgray',
              'lightgrey',
              'lightgreen',
              'lightpink',
              'lightsalmon',
              'lightseagreen',
              'lightskyblue',
              'lightslategray',
              'lightslategrey',
              'lightsteelblue',
              'lightyellow',
              'lime',
              'limegreen',
              'linen',
              'magenta',
              'maroon',
              'mediumaquamarine',
              'mediumblue',
              'mediumorchid',
              'mediumpurple',
              'mediumseagreen',
              'mediumslateblue',
              'mediumspringgreen',
              'mediumturquoise',
              'mediumvioletred',
              'midnightblue',
              'mintcream',
              'mistyrose',
              'moccasin',
              'navajowhite',
              'navy',
              'oldlace',
              'olive',
              'olivedrab',
              'orange',
              'orangered',
              'orchid',
              'palegoldenrod',
              'palegreen',
              'paleturquoise',
              'palevioletred',
              'papayawhip',
              'peachpuff',
              'peru',
              'pink',
              'plum',
              'powderblue',
              'purple',
              'red',
              'rosybrown',
              'royalblue',
              'saddlebrown',
              'salmon',
              'sandybrown',
              'seagreen',
              'seashell',
              'sienna',
              'silver',
              'skyblue',
              'slateblue',
              'slategray',
              'slategrey',
              'snow',
              'springgreen',
              'steelblue',
              'tan',
              'teal',
              'thistle',
              'tomato',
              'turquoise',
              'violet',
              'wheat',
              'white',
              'whitesmoke',
              'yellow',
              'yellowgreen']

import plotly.io as pio
pio.renderers.default = "browser"
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as plotly
np.random.seed(1)

""" 
COMMENTS
lgvs: Light good vehicles
hgvs: heavy goods vehhicles


"""

df = pd.read_csv("./data/region_traffic.csv")
df1 = pd.read_csv("./data/local_authority_traffic.csv")
df2 = pd.read_csv("./data/dft_traffic_counts_aadf.csv", low_memory=False)
df3 = pd.read_csv("./data/dft_rawcount.csv", low_memory=False)
df.drop('total_link_length_miles', axis=1, inplace=True)

""""
https://plot.ly/python/
"""

"""PRE-PROCESSING"""
if 1 == 1:
    cols = df.columns.values
    df.dropna
    """make to vary within the smae range (0,1)"""
    Scaler = prep.MinMaxScaler()
    df[cols[[5, 6, 7, 8, 9, 10, 11, 12]]] = Scaler.fit_transform(df[cols[[5, 6, 7, 8, 9, 10, 11, 12]]])

"""BOXPLOTS 1"""
if 1 == 1:
    fig = px.box(df, y='all_motor_vehicles', x='year', points="all")
    fig.update_layout(
        title='All motor vehicles - Box Plot - by year',
        xaxis_nticks=36)
    fig.show()

"""BOXPLOTS 2"""
if 1 == 1:
    fig = px.box(df, y='all_motor_vehicles', x='name', points="all")
    fig.update_layout(
        title='All motor vehicles - Box Plot - by name',
        xaxis_nticks=36)
    fig.show()
    #plotly.io.write_image(fig, file='my_figure_file.png', format='png')

"""BOXPLOTS 3"""
if 1 == 1:
    fig = go.Figure()
    fig.add_trace(go.Box(x=df.pedal_cycles,
                         name='pedal_cycles',
                         marker_color='indianred'))
    fig.add_trace(go.Box(x=df.two_wheeled_motor_vehicles,
                         name='two_wheeled_motor_vehicles',
                         marker_color='mediumvioletred'))
    fig.add_trace(go.Box(x=df.all_motor_vehicles,
                         name='All motor vehicles',
                         marker_color='hotpink'))
    fig.add_trace(go.Box(x=df.cars_and_taxis,
                         name='cars_and_taxis',
                         marker_color='lime'))
    fig.add_trace(go.Box(x=df.buses_and_coaches,
                         name='buses_and_coaches',
                         marker_color='burlywood'))
    fig.add_trace(go.Box(x=df.lgvs,
                         name='lgvs',
                         marker_color='orange'))
    fig.add_trace(go.Box(x=df.all_hgvs,
                         name='all_hgvs',
                         marker_color=css_colors[34]))
    fig.update_layout(
        title='All motor vehicles - Box Plot - by type',
        xaxis_nticks=36)
    fig.show()

"""BOXPLOT 4"""
if 1 == 1:
    fig = go.Figure()
    y0 = df.lgvs
    fig.add_trace(go.Box(
        y=y0,
        name="All Points",
        jitter=0.3,
        pointpos=-1.8,
        boxpoints='all',  # represent all points
        marker_color='rgb(7,40,89)',
        line_color='rgb(7,40,89)'
    ))
    fig.add_trace(go.Box(
        y=y0,
        name="with Whiskers",
        boxpoints=False,  # no data points
        marker_color='rgb(9,56,125)',
        line_color='rgb(9,56,125)'
    ))
    fig.add_trace(go.Box(
        y=y0,
        name="Suspected Outliers",
        boxpoints='suspectedoutliers',  # only suspected outliers
        marker=dict(
            color='rgb(8,81,156)',
            outliercolor='rgba(219, 64, 82, 0.6)',
            line=dict(
                outliercolor='rgba(219, 64, 82, 0.6)',
                outlierwidth=2)),
        line_color='rgb(8,81,156)'
    ))
    fig.add_trace(go.Box(
        y=y0,
        name="Whiskers and Outliers",
        boxpoints='outliers',  # only outliers
        marker_color='rgb(107,174,214)',
        line_color='rgb(107,174,214)'
    ))
    fig.update_layout(title_text="LGVs")
    fig.update_layout(
        title='All motor vehicles - Boxplot Global',
        xaxis_nticks=36)
    fig.show()

""" BOXPLOT 5"""
if 1 == 1:
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df.cars_and_taxis,
        x=df.name,
        name='cars_and_taxis',
        marker_color='#3D9970'
    ))
    fig.add_trace(go.Box(
        y=df.buses_and_coaches,
        x=df.name,
        name='buses_and_coaches',
        marker_color='#FF4136'
    ))
    fig.add_trace(go.Box(
        y=df.lgvs,
        x=df.name,
        name='lgvs',
        marker_color='#FF851B'
    ))
    fig.update_layout(
        yaxis_title='normalized moisture',
        boxmode='group'  # group together boxes of the different traces for each value of x
    )
    fig.update_layout(
        title='All motor vehicles - Boxplot - by location',
        xaxis_nticks=36)
    fig.show()

"""BOXPLOT 6"""
if 0 == 1:
    years = df.year.unique()
    N = len(years)
    c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N)]

    fig = go.Figure(data=[go.Box(
        y=df.loc[df['year'] == years[i], :].cars_and_taxis.values,
        marker_color=c[i]
    ) for i in np.arange(0, N)])
    """
    format the layout
    """
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(zeroline=False, gridcolor='white'),
        paper_bgcolor='rgb(233,233,233)',
        plot_bgcolor='rgb(233,233,233)',
    )
    fig.show()

"""BOXPLOT 7"""
if 1 == 1:
    x_data = df.name.unique()
    N = 50
    y_data = [df.cars_and_taxis,
              df.two_wheeled_motor_vehicles,
              df.buses_and_coaches,
              df.lgvs,
              df.all_hgvs,
              df.all_motor_vehicles]
    colors = ['rgba(93, 164, 214, 0.5)',
              'rgba(255, 144, 14, 0.5)',
              'rgba(44, 160, 101, 0.5)',
              'rgba(255, 65, 54, 0.5)',
              'rgba(207, 114, 255, 0.5)',
              'rgba(127, 96, 0, 0.5)']

    fig = go.Figure()
    for xd, yd, cls in zip(x_data, y_data, colors):
        fig.add_trace(go.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker_size=2,
            line_width=1)
        )
    fig.update_layout(
        title='Points Scored by the Top 9 Scoring NBA Players in 2012',
        yaxis=dict(
            autorange=True,
            showgrid=True,
            zeroline=True,
            dtick=5,
            gridcolor='rgb(255, 255, 255)',
            gridwidth=1,
            zerolinecolor='rgb(255, 255, 255)',
            zerolinewidth=2,
        ),
        margin=dict(
            l=40,
            r=30,
            b=80,
            t=100,
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        showlegend=False
    )
    fig.update_layout(
        title='All motor vehicles - Boxplot - by location',
        xaxis_nticks=36)
    fig.show()

"""SCATER MATRIX PLOT 1"""
if 1 == 1:
    fig = px.scatter_matrix(df,
                            dimensions=["cars_and_taxis",
                                        "pedal_cycles",
                                        "two_wheeled_motor_vehicles"],
                            color="road_category_id")
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(
        title='All motor vehicles - Scatter Plot Matrix',
        xaxis_nticks=36)
    fig.show()

""" HEAT MAP 1 """
if 1 == 1:
    fig = go.Figure(data=go.Heatmap(
        z=df.all_hgvs,
        x=df.name,
        y=df.year,
        colorscale='Viridis'))
    fig.update_layout(
        title='Heat Map all vehicles year vs location',
        xaxis_nticks=36)
    fig.show()

""" HEAT MAP """
if 1 == 1:
    fig = go.Figure(data=go.Heatmap(
        z=df.pedal_cycles,
        x=df.name,
        y=df.year,
        colorscale='Viridis'))
    fig.update_layout(
        title='Heat Map Pedal Cycles year vs location',
        xaxis_nticks=36)
    fig.show()

""" CONTOUR PLOTS """
if 1 == 1:
    names = df.name.unique()
    years = df.year.unique()
    fig = go.Figure(data=
    go.Contour(
        z=[df[(df.road_category_id == 1) & (df.year == years[23])].all_motor_vehicles.values,
           df[(df.road_category_id == 4) & (df.year == years[23])].all_motor_vehicles.values,
           df[(df.road_category_id == 5) & (df.year == years[23])].all_motor_vehicles.values,
           df[(df.road_category_id == 6) & (df.year == years[23])].all_motor_vehicles.values,
           ],
        colorscale='Electric',
        line_smoothing=1),
    )
    fig.update_layout(
        title=go.layout.Title(
            text="All motor vehicles 2003",
            xref="paper",
            x=0
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="Location",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f"
                )
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="Category",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f"
                )
            )
        )
    )
    fig.update_layout(
        title='Heat Map Vehicles Category vs location',
        xaxis_nticks=36)
    fig.show()

"""2D HISTOGRAMS"""
if 1 == 1:
    x = df.cars_and_taxis
    y = df.buses_and_coaches
    fig = go.Figure()
    fig.add_trace(go.Histogram2dContour(
        x=x,
        y=y,
        colorscale='Blues',
        reversescale=True,
        xaxis='x',
        yaxis='y'
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        xaxis='x',
        yaxis='y',
        mode='markers',
        marker=dict(
            color='rgba(0,0,0,0.3)',
            size=3
        )
    ))
    fig.add_trace(go.Histogram(
        y=y,
        xaxis='x2',
        marker=dict(
            color='rgba(0,0,0,1)'
        )
    ))
    fig.add_trace(go.Histogram(
        x=x,
        yaxis='y2',
        marker=dict(
            color='rgba(0,0,0,1)'
        )
    ))

    fig.update_layout(
        autosize=False,
        xaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showgrid=False
        ),
        yaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showgrid=False
        ),
        xaxis2=dict(
            zeroline=False,
            domain=[0.85, 1],
            showgrid=False
        ),
        yaxis2=dict(
            zeroline=False,
            domain=[0.85, 1],
            showgrid=False
        ),
        height=600,
        width=600,
        bargap=0,
        hovermode='closest',
        showlegend=False
    )
    fig.update_layout(
        title='Histogram Private Vs Public transport',
        xaxis_nticks=36)
    fig.show()


"""TIME SERIES """
if 1 == 1:
    x0 = df3.hour.unique()
    dates = df3.count_date.unique()
    y1 = df3[df3['count_date'] == dates[0]].cars_and_taxis.values
    y2 = df3[df3['count_date'] == dates[0]].buses_and_coaches
    y3 = df3[df3['count_date'] == dates[0]].lgvs
    y4 = df3[df3['count_date'] == dates[0]].all_hgvs
    y5 = df3[df3['count_date'] == dates[0]].all_motor_vehicles
    y6 = df3[df3['count_date'] == dates[0]].two_wheeled_motor_vehicles
    fig = go.Figure()
    """Create and style traces """
    fig.add_trace(go.Scatter(x=x0, y=y1, name='Cars and taxis',
                             line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=x0, y=y2, name='Buses and coaches',
                             line=dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(x=x0, y=y3, name='LGVS',
                             line=dict(color='firebrick', width=4, dash='dash')))
    fig.add_trace(go.Scatter(x=x0, y=y4, name='ALL HGVS',
                             line=dict(color='royalblue', width=4, dash='dash')))
    fig.add_trace(go.Scatter(x=x0, y=y5, name='ALL Motor Vehicles',
                             line=dict(color='firebrick', width=4, dash='dot')))
    fig.add_trace(go.Scatter(x=x0, y=y6, name='LTwo-Wheeled motor vehicles',
                             line=dict(color='royalblue', width=4, dash='dot')))
    """Edit the layout"""
    fig.update_layout(title='Vehicles Hourly Time Series',
                      xaxis_title='Time (Hours)',
                      yaxis_title='No. Vehicles')
    fig.show()

""" TIME SERIES 2 """
if 1 == 1:
    x0 = df3.hour.unique()
    dates = df3.count_date.unique()
    road_variables = df3.columns.values[17:30]
    y1 = df3.loc[df3['count_date'] == dates[0], road_variables[2]]
    y2 = df3.loc[df3['count_date'] == dates[1], road_variables[2]]
    y3 = df3.loc[df3['count_date'] == dates[2], road_variables[2]]
    y4 = df3.loc[df3['count_date'] == dates[3], road_variables[2]]
    fig = go.Figure()
    """Create and style traces """
    fig.add_trace(go.Scatter(x=x0, y=y1, name='2013-10-24',
                             line=dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(x=x0, y=y2, name='2009-06-05',
                             line=dict(color='royalblue', width=4, dash='dash')))
    fig.add_trace(go.Scatter(x=x0, y=y3, name='2006-06-29',
                             line=dict(color='blue', width=4)))
    fig.add_trace(go.Scatter(x=x0, y=y4, name='2003-03-26',
                             line=dict(color='blue', width=4, dash='dash')))
    """Edit the layout"""
    fig.update_layout(title="Cars & Taxis Hourly time series by dates ",
                      xaxis_title='Time (Hours)',
                      yaxis_title='No. Vehicles')
    fig.show()

""" TIME SERIES 3 """
if 1 == 1:
    x0 = df2.year.unique()
    y1 = df2.groupby('year')['all_motor_vehicles'].sum()
    y2 = df2.groupby('year')['cars_and_taxis'].sum()
    fig = go.Figure()
    """Create and style traces """
    fig.add_trace(go.Scatter(x=x0, y=y1, name='all_motor_vehicles',
                             line=dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(x=x0, y=y2, name='cars_and_taxis',
                             line=dict(color='royalblue', width=4, dash='dash')))
    """Edit the layout"""
    fig.update_layout(title="All motor vehciels Vs Cars & Taxis Yearly Time series ",
                      xaxis_title='Time (Years)',
                      yaxis_title='No. Vehicles')
    fig.show()


