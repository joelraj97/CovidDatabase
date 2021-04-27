import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


from dash.dependencies import Output,Input

app=dash.Dash(__name__,external_stylesheets=[dbc.themes.CYBORG])
df=pd.read_csv("https://raw.githubusercontent.com/joelraj97/CovidLine/main/owid-covid-data(1).csv")


#App Layout

app.layout=html.Div([
    dbc.Button("Success",color="success",className="mr-1"),


    html.H1("Covid-19 Coronavirus Pandemic",style={"text-align":"center"}),


    dcc.Dropdown(id="my_option",
                 options=[{'label':i,'value':i}
                          for i in df["location"].unique()],
                 value="Afghanistan",
                 style={'size': 3, "offset": 2, 'order': 3,"color":"Red","width":"50%"},


                 ),
    html.Br(),
    html.Div(id="dateid",style={"text-align":"left","font-size":50,"color":"Blue"}),
    html.Br(),
    html.H2("Total Cases",style={"text-align":"center"}),
    html.Div(id="totalcases",style={'size': 3, "offset": 2, 'order': 3,"color":"Red","text-align":"center","font-size":90}),
    html.Br(),
    html.H2("Deaths",style={"text-align":"center"}),
    html.Div(id="deathno",title="Deaths",draggable="true",style={'size': 6, "offset": 2, 'order': 3,"color":"Red","text-align":"center","font-size":90}),
    html.Br(),
    dcc.Graph(id="linegraph2",figure={})

])

#call back

@app.callback(
     [Output(component_id="dateid",component_property="children"),
      Output(component_id="totalcases",component_property="children"),
     Output(component_id="deathno",component_property="children"),
     Output(component_id="linegraph2",component_property="figure")],
    Input(component_id="my_option",component_property="value")
)


def update_graph(option_slctd):
    filterdata=df[df["location"]==option_slctd]
    totalcases=int(filterdata["new_cases"].sum())
    deaths=int(filterdata["new_deaths"].sum())
    dates=filterdata["date"].tail(1)
    index=dates.index.values
    print(dates)
    strings = [str(integer) for integer in index]
    a_string = "".join(strings)
    keyvalue = int(a_string)
    print(keyvalue)
    date=filterdata.loc[keyvalue,"date"]
    print(date)
    #formatted_df = pd.to_datetime(date,format='%dd%mm%YYYY')
    #print(formatted_df)
    fig2=px.line(filterdata,x="date",y="new_cases",title="New Cases With Date",animation_group="new_cases_smoothed")
    fig2.update_layout(title_font_color="black",title_font_size=50)
    fig2.update_layout(title={'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    return "Data Upto: "+dates, totalcases,deaths,fig2

if __name__ == '__main__':
    app.run_server(debug=True)