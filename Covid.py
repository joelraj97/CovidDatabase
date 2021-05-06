import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output,Input
import CovidPred  as studying_pred
app=dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
df=pd.read_csv("owid-covid-data.csv")   #raed data from csv

app.head = html.Link(rel='stylesheet', href='./static/stylesheet.css'),
#App Layout
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}
sidebar = html.Div(
    [
dcc.Dropdown(
                 id="my_option",       # drop down menu
                 #className="left_menu",
                 options=[{'label':i,'value':i}
                          for i in df["location"].unique()

                          ],   #selecting options from the list of locations
                 value="India",   #initila value of the dropdown

                 #style={"width" :'100%',
                            #"display" : 'inline-block',
                           # "verticalAlign" : "middle"},#style features of the dropdown
                 multi=False,
)
         ],
        style=SIDEBAR_STYLE,
)
#content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)
app.layout=html.Div([


    html.H1("Covid-19 Coronavirus Pandemic",style={"text-align":"center"}), #heading of the application
    #dcc.Dropdown(
     #            id="my_option",       # drop down menu
      #           className="columns",
       #          options=[{'label':i,'value':i}
        #                  for i in df["location"].unique()

         #                 ],   #selecting options from the list of locations
          #       value="Afghanistan",   #initila value of the dropdown
           #      #className="app-margin",
            #     style={"width" :'100%',
             #               "display" : 'inline-block',
              #              "verticalAlign" : "middle"},#style features of the dropdown
               #  multi=False,   #wheather multiple values allowed in the dropdown
                 #className="left_menu",


                #),
    sidebar,
   # content,
    html.Br(),
   dbc.Container( html.Div(id="dateid",style={"text-align":"left","font-size":20,"color":"Blue","width":"50%","size":4,"offset":"7"})),  #latest date of the available data
    html.Br(),
    dbc.Container([
    dbc.Row(    #Row1
     [
            dbc.Col(   [    #first column of Row 1
            dbc.Alert([
                  html.H2("Total Cases",style={"text-align":"center"}),
                 html.Div(id="totalcases",style={'size': 3, "offset": 2, 'order': 3,"color":"Red","text-align":"center","font-size":40})])],
                 width={'size': 3, "offset": 0, 'order': 3}
        ),

           dbc.Col(   [    #first column of Row 1
            dbc.Alert([
                  html.H2("Total Cases Per ",style={"text-align":"center"}),
                 html.Div(id="totalcasesper",style={'size': 4, "offset": 2, 'order': 3,"color":"Red","text-align":"center","font-size":40})])],
                 width={'size': 4, "offset": 0, 'order': 3}
        ),

            dbc.Col( [     #second column of Row 1
              dbc.Alert([
                html.H2("Deaths",style={"text-align":"center"}),
                html.Div(id="deathno",title="Deaths",draggable="true",style={'size': 3, "offset": 2, 'order': 3,"color":"Red","text-align":"center","font-size":40})])],
                width={'size': 3, "offset": 0, 'order': 3}
            )
            ]) ,

    html.Br(),
    dbc.Row( [    #Row 2
        dbc.Col(    #First column of row 2

    dcc.Graph(id="linegraph2",figure={},style={'size': 2, "offset": 0, 'order': 2,"width":"20%","height": "50%"})    ,
    width={'size': 1, "offset": 0, 'order': 2,"max-width":"20%","height": "50%"}
            

        ) ,
        dbc.Col(       #Second Column of Row 2
          dcc.Graph(id="piechart",figure={},style={'size': 2, "offset": 0, 'order': 2,"width":"20%","height":"50%"})    ,
            width={'size': 1, "offset": 6, 'order': 2,"max-width":"20%","height": "50%"}
        )    ,


   ] )      ,
    html.Br(),
    # dcc.Graph(id="fig_LarsReg",figure={}),
    dcc.Graph(id="fig_PolyReg",figure={}),
    # dcc.Graph(id="fig_Holt",figure={}),
    # dcc.Graph(id="fig_LagPred",figure={}),
]),

])

#call back

@app.callback(
     [  # Output("page-content", "children"),
         Output(component_id="dateid",component_property="children"),
      Output(component_id="totalcases",component_property="children"),
      Output(component_id="totalcasesper",component_property="children"),
     Output(component_id="deathno",component_property="children"),
     Output(component_id="linegraph2",component_property="figure") ,
      Output(component_id="piechart",component_property="figure") ,

    # Output(component_id="fig_LarsReg",component_property="figure") ,
    Output(component_id="fig_PolyReg",component_property="figure") ,
    # Output(component_id="fig_Holt",component_property="figure") ,
    # Output(component_id="fig_LagPred",component_property="figure") ,
      ],
    Input(component_id="my_option",component_property="value")
)


def update_graph(option_slctd):

    print("Opted location is ",option_slctd)	
    filterdata=df[df["location"]==option_slctd]    #to filter out data for the selected country
    totalcases=int(filterdata["new_cases"].sum())         #to find the total cases in the selected country
    totalcasesper=int(filterdata["total_cases_per_million"].sum())
    deaths=int(filterdata["new_deaths"].sum())   #to find the deaths in the selsected country
    dates=filterdata["date"].tail(1)     #to return the latest value of date in the selected country
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
    fig2.update_layout(
        margin=dict(l=1, r=1, t=1, b=1),
        width=600,
        height=300,
        paper_bgcolor="LightSteelBlue",
    )

    piegraph=px.pie(filterdata,names=['Total case','Deaths'],values=[totalcases,deaths])
    piegraph.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        width=500,
        height=300,
        paper_bgcolor="LightSteelBlue",

    )
    pie=px.line(filterdata,x="total_cases",y="new_cases_smoothed")
    # fig_LarsReg_ret, fig_PolyReg_ret, fig_Holt_ret, fig_LagPred_ret = studying_pred.dt_process(df,option_slctd)  # returns the figures to show
    fig_PolyReg_ret = studying_pred.dt_process(df, option_slctd)  # returns the figures to show
    # return "Data Upto: "+dates, totalcases,totalcasesper,deaths,fig2,piegraph, fig_LarsReg_ret, fig_PolyReg_ret, fig_Holt_ret, fig_LagPred_ret
    return "Data Upto: " + dates, totalcases, totalcasesper, deaths, fig2, piegraph, fig_PolyReg_ret

if __name__ == '__main__':
    app.run_server(debug=True)         
