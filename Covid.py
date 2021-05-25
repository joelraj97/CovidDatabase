import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output,Input
import CovidPred  as studying_pred
app=dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
df=pd.read_csv("owid-covid-data.csv")   #read the covid data from file
non_z_loc_list = []
for locn in df["location"].unique():
    fildata = df[df["location"] == locn]  # to filter out data for the selected country
    totcases = int(fildata["new_cases"].sum())  # to find the total cases in the selected country
    if totcases!=0:
        non_z_loc_list.append(locn)

dff=df.copy()
dfff=df.copy()
dff.drop(dff.columns.difference(['location','new_cases']), 1, inplace=True)
dfff.drop(dfff.columns.difference(['location','population']), 1, inplace=True)
#dff.sum(axis=1)
newdf=dff.groupby(['location'])['new_cases'].sum().reset_index()
contdf=dfff.groupby(['location'])['population'].sum().reset_index()
print(contdf)
africaindex=contdf[contdf["location"]=="Africa"].index
index1=newdf[newdf['location']=='Africa'].index
africa=int(contdf.iloc[africaindex,1])
print(africa)
index2=newdf[newdf['location']=='World'].index
Europeindex=contdf[contdf["location"]=="Europe"].index
index3=newdf[newdf['location']=='Europe'].index
Europe=int(contdf.iloc[Europeindex,1])
Asiaindex=contdf[contdf["location"]=="Asia"].index
index4=newdf[newdf['location']=='Asia'].index
Asia=int(contdf.iloc[Asiaindex,1])
NAindex=contdf[contdf["location"]=="North America"].index
index5=newdf[newdf['location']=='North America'].index
NorthAmerica=int(contdf.iloc[NAindex,1])
SAindex=contdf[contdf["location"]=="South America"].index
index6=newdf[newdf['location']=='European Union'].index
index7=newdf[newdf['location']=='South America'].index
SouthAmerica=int(contdf.iloc[SAindex,1])
newdf.drop(index7 , inplace=True)
newdf.drop(index3 , inplace=True)
newdf.drop(index4 , inplace=True)
newdf.drop(index5 , inplace=True)
newdf.drop(index6 , inplace=True)
newdf.drop(index1 , inplace=True)
newdf.drop(index2 , inplace=True)
newdf.rename(columns = {'new_cases':'Total_Cases'}, inplace = True)
descendingdf=newdf.sort_values('Total_Cases',ascending=False).reset_index()
descendingdf.drop("index",inplace=True,axis=1)
#descendingdf.to_csv('file2.csv', header=False, index=False,mode='a')
print(descendingdf)
top5 = descendingdf.head(5)
barg = px.bar(top5, y="Total_Cases", x="location",color=[descendingdf.iloc[0,1],descendingdf.iloc[1,1],descendingdf.iloc[2,1],descendingdf.iloc[3,1],descendingdf.iloc[4,1]])
barg.update_layout \
         (
         margin=dict(l=20, r=20, t=20, b=20),
         width=500,
         height=190,
         paper_bgcolor="LightSteelBlue",
        )


# app.head = html.Link(rel='stylesheet', href='./static/stylesheet.css')

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    #"width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "width": "13%",
    " position" : "absolute",
    "height": "absolute",
    "z-index": 999,
    "background": "#2A3F54",
}


'''
wins commented the unused code 
# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}
'''
# Implementing the country selection box
sidebar = html.Div(
    [html.H5("Select Country",style={"color":"yellow"}),
    dcc.Dropdown(
                 id="my_option",       # drop down menu
                 options=[{'label':i,'value':i}
                          for i in non_z_loc_list   #df["location"].unique()
                          ],   #selecting options from the list of locations
                 value="India",   #initial value set in the dropdown
                 multi=False,
                 style={"height":"20px"},
                 clearable=False,
               ),
     html.Br(),
     dbc.Container( html.Div(id="dateid",style={"text-align":"left","font-size":13,"color":"Yellow"})),
    ],

      style=SIDEBAR_STYLE,

)


#App Layout
app.layout=html.Div([

    html.H1("Covid-19 Coronavirus Pandemic",style={"text-align":"center","font-size":30}), #heading of the application
    sidebar,
      #latest date of the available data
    dbc.Container([
    dbc.Row(    #Row1
     [
            dbc.Col(   [    #first column of Row 1
            dbc.Alert([
                  #html.H4("Total Cases",style={"text-align":"center","font-size":20}),
                 html.Div(id="totalcases",style={'size': 1, "offset": 2, 'order': 3,"color":"Red","text-align":"center","font-size":15})])],
                 width={'size': 3, "offset": 0, 'order': 3}
        ),

        #   dbc.Col(   [    #first column of Row 1
          #  dbc.Alert([
            #      html.H2("People Vaccinated ",style={"text-align":"center"}),
             #    html.Div(id="vaccination",style={'size': 4, "offset": 2, 'order': 3,"color":"Red","text-align":"center","font-size":40})])],
            #     width={'size': 4, "offset": 0, 'order': 3}
        #),

            dbc.Col( [     #second column of Row 1
              dbc.Alert([
               # html.H4("Deaths",style={"text-align":"center","font-size":20}),
                html.Div(id="deathno",title="Deaths",draggable="true",style={'size': 3, "offset": 2, 'order': 3,"color":"Red","text-align":"center","font-size":15})])],
                width={'size': 3, "offset": 0, 'order': 3}
            )
            ]) ,

    dbc.Tab([
    dbc.Row( [    #Row 2
        dbc.Col( [   #First column of row 2
        html.H4("New Cases With Date",style={"text-align":"left","size":4}),
    dcc.Graph(id="linegraph2",figure={},style={'size': 2, "offset": 0, 'order': 2,"width":"20%","height": "30%"}) ]   ,
    width={'size': 4,"offset": 0, 'order': 2,"max-width":"20%","height": "50%"},


        ) ,
        dbc.Col(  [    # Second Column of Row 2
                        html.H4("Percentage of Deaths", style={ "size": 3}),
        dcc.Graph(id="piechart",figure={},style={'size': 2, "offset": 0, 'order': 2,"width":"20%","height":"30%"})   ] ,
            width={'size':5,"offset": 3, 'order': 2,"max-width":"20%","height": "50%"}
        )    ,

   ] )      ,
  ]),
       dbc.Tab([
            dbc.Row([
               dbc.Col([
                  html.H3("Top 5 Countries Affected By Covid",style={"text-align":"left","font-size":20}),
                   dcc.Graph(id="barg",figure={})],
                   width={'size': 4,"offset": 0, 'order': 2,"max-width":"20%","height": "50%"},
                 ),
                dbc.Col([
                 html.H3("Testing based on Continents",style={"text-align":"left","font-size":20}),
                 dcc.Graph(id="continent",figure={})],
                 width={'size':5,"offset": 3, 'order': 2,"max-width":"20%","height": "50%"}
                ),


            ]),
            ]),
    dbc.Tab([
      dbc.Row([
        dbc.Col([
    dcc.Graph(id="fig_PolyReg",figure={})],
    #width={'size': 4,"offset": 0, 'order': 2,"max-width":"20%","height": "50%"},

        ),
    ]),
]),
]),
])

#call back

@app.callback(
     [
      Output(component_id="dateid",component_property="children"),
      Output(component_id="totalcases",component_property="children"),
     # Output(component_id="vaccination",component_property="children"),
      Output(component_id="deathno",component_property="children"),
      Output(component_id="linegraph2",component_property="figure") ,
      Output(component_id="piechart",component_property="figure") ,
      Output(component_id="fig_PolyReg",component_property="figure") ,
     Output(component_id="barg",component_property="figure"),
     Output(component_id="continent",component_property="figure"),
      ],

      Input(component_id="my_option",component_property="value"),

)

def update_graph(option_slctd):

  #  print("Opted location is ",option_slctd)
    filterdata=df[df["location"]==option_slctd]    #to filter out data for the selected country
    totalcases=int(filterdata["new_cases"].sum())         #to find the total cases in the selected country
    vacci=int(filterdata["new_vaccinations"].sum())
    deaths=int(filterdata["new_deaths"].sum())   #to find the deaths in the selsected country
    deaths=str(deaths)
    dates=filterdata["date"].tail(1)     #to return the latest value of date in the selected country
    index=dates.index.values
   # print(dates)
    strings = [str(integer) for integer in index]
    a_string = "".join(strings)
    keyvalue = int(a_string)
  #  print(keyvalue)
    date=filterdata.loc[keyvalue,"date"]
   # print(date)
    totalcases=str(totalcases)

    fig2=px.line(filterdata,x="date",y="new_cases")

    fig2.update_layout(
        margin=dict(l=1, r=1, t=1, b=1),
        width=500,
        height=190,
        paper_bgcolor="LightSteelBlue",
    )
   # fig2.show()

    piegraph=px.pie(filterdata,names=['Total case','Deaths'],values=[totalcases,deaths])
    piegraph.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        width=500,
        height=190,
        paper_bgcolor="LightSteelBlue",

    )
    pie=px.line(filterdata,x="total_cases",y="new_cases_smoothed")
    fig_PolyReg_ret = studying_pred.dt_process(df, option_slctd)  # returns the figures to show
    continentpie=px.pie(contdf,names=["Africa","Europe","Asia","North America","South America"],values=[africa,Europe,Asia,NorthAmerica,SouthAmerica])
    continentpie.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        width=500,
        height=190,
        paper_bgcolor="LightSteelBlue",
    )



    return "Data Upto: " + dates,"Total cases:"+ totalcases,"Deaths:"+ deaths,fig2, piegraph, fig_PolyReg_ret,barg,continentpie




if __name__ == '__main__':
    app.run_server(debug=True)         
