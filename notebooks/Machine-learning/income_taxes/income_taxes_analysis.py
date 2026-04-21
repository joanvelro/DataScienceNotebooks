import pandas as pd
import plotly.express as px
import numpy as np

# https://www.finect.com/usuario/Josetrecet/articulos/como-funciona-renta-tramos-irpf

df = pd.read_csv('..data/income_taxes_spain_single_no_child.csv')

df.rename(columns={'Bruto (Euros)': 'yearly_gross_salary',
                   'Neto (Euros)': 'yearly_net_salary',
                   'Bruto mes (14 pagas)': 'monthly_gross_salary',
                   'Retencion (%)': 'Taxes',
                   'Neto mes (14 pagas)': 'monthly_net_salary'}, inplace=True)


for i in df.index.values:
    df.loc[i, ['Taxes']] = float(df['Taxes'][i][0:5])

df['Taxes'] = df['Taxes'].astype(float)

series = pd.Series(df['yearly_gross_salary']).str.split(',', expand=True)
df['yearly_gross_salary'] = series[0].astype(int) * 1000

series = pd.Series(df['yearly_net_salary']).str.split(',', expand=True)
df['yearly_net_salary'] = series[0].astype(int) * 1000

series = pd.Series(df['monthly_gross_salary']).str.split(',', expand=True)
for i in range(0, len(series)):
    df.loc[i, ['monthly_gross_salary']] = float(series[0][i] + series[1][i])

series = pd.Series(df['monthly_net_salary']).str.split(',', expand=True)
for i in range(0, len(series)):
    df.loc[i, ['monthly_net_salary']] = float(series[0][i] + series[1][i])

df.drop(df[df['monthly_gross_salary'] == 2354].index, inplace=True)

df = df[df['yearly_gross_salary'] <= 70000]
df.loc[:, ['monthly_gross_salary']] = df['monthly_gross_salary'].astype(float)
df.loc[:, ['monthly_net_salary']] = df['monthly_net_salary'].astype(float)

df.dtypes


def plot():
    fig = px.scatter(df,
                     x='yearly_gross_salary',
                     y='monthly_net_salary',
                     color='Taxes',
                     size='Taxes',
                     text="monthly_net_salary",
                     )

    fig.update_xaxes(tickvals=np.arange(0, 120000, 2000),
                     ticktext=np.arange(0, 120000, 2000))

    fig.update_layout(font=dict(family="Courier New, monospace", size=14, color="#7f7f7f"))
    fig.update_layout(title_text="Income taxes analysis -  Spain  - (2019)")
    fig.update_xaxes(title_text="Yearly Gross Salary (k€)",
                     tickangle=45,
                     tickfont=dict(size=18),
                     ticktext=np.arange(18, 74, 2),
                     tickvals=np.arange(18, 74, 2) * 1000)
    fig.update_yaxes(title_text="Monthly Net Salary (€)",
                     tickfont=dict(size=16),
                     ticktext=np.arange(1, 5, 0.25).astype(float) * 1000,
                     tickvals=np.arange(1, 5, 0.25).astype(float) * 1000)
    fig.update_traces(textposition='top center')
    fig.write_html('../figures/income_taxes_analysis.html')
    fig.show()


plot()
