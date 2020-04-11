import os
import pathlib

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_table
import plotly.graph_objs as go
import dash_daq as daq
import sqlite3
from sqlite3 import Error

import pandas as pd
import glob
from os import walk

app  = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.config["suppress_callback_exceptions"] = True


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        #print(sqlite3.version)
    except Error as e:
        print(e)
    return conn

conn = create_connection(r"./database.db")
cur = conn.cursor()
cur.execute("SELECT simu FROM parameters GROUP BY simu")
dff = list(cur.fetchall())
cur.close()
conn.close()
simulationName = dff[0][0]


params = {"_height_forgiveness" : 0.005,
          "_height_weight" : 0.0,
          "_direction_forgiveness" : 0.005, 
          "_direction_weight" : 0.0, 
          "_orientation_forgiveness": 0.0005,
          "_orientation_weight": 0.0,
          "_mimic_forgiveness": 0.5,
          "_mimic_weight": 0.0,
          "_speed_forgiveness": 1.0,
          "_speed_weight": 0.0,
          "_ideal_speed": 0.01
         }
minmax = {"_height_forgiveness" : [0.005, 0.5],
          "_height_weight" : [0.0, 20.0],
          "_direction_forgiveness" : [0.005, 0.5], 
          "_direction_weight" : [0.0, 20.0], 
          "_orientation_forgiveness":[0.0005, 0.05],
          "_orientation_weight": [0.0, 20.0],
          "_mimic_forgiveness": [0.5, 5.0],
          "_mimic_weight": [0.0, 20.0],
          "_speed_forgiveness": [1.0, 10.0],
          "_speed_weight": [0.0, 20.0],
          "_ideal_speed": [0.01, 0.1],
         }



def init_params(simu):
    conn = create_connection(r"./database.db")
    cur = conn.cursor()
    for param in params:
        cur.execute("SELECT value FROM parameters WHERE type = '{0}' AND simu='{1}' ORDER BY step DESC".format(param, simu))
        d = list(cur.fetchall())
        if(len(d) > 0):
            params[param] = d[0][0]
            
    cur.close()
    conn.close()
    

def send_params(simu):
    conn = create_connection(r"./database.db")
    cur = conn.cursor()
    for param, value in params.items():
        #id, type, simu, steps, value
        sql = ''' INSERT INTO parameters(simu, type, step, value) VALUES('{0}','{1}',-1,{2}) '''.format(simu, param, float(value))
        cur.execute(sql)
        conn.commit()
    cur.close()
    conn.close()
    
def build_sliders():
    children = []
    for param, value in params.items():
        text = html.Div(id='{}_output'.format(param), style={'color':'black'}, children="{} = {}".format(param, value))

        
        slider = dcc.Slider(
            id = param,
            min = minmax[param][0],
            max = minmax[param][1],
            step = (minmax[param][1]-minmax[param][0])/100.0,
            value = value,
        )
        
        obj = html.Div(id = '{}_output'.format(param)+"_container", 
                    className = "container", 
                    children=[text, slider],
                    style={"margin":"15px", 
                           'padding':'15px', 
                           'background-color':"#f0f0f0",
                           'border-radius': '25px'})
        children.append(obj)
    return html.Div(id = "sliders", 
                    className = "container", 
                    children=children)

def create_callback(param):
    def callback(value):
        params[param]=value
        return '{} = {}'.format(param, value)
    return callback

for param in params:
    callback=create_callback(param)
    
    app.callback(Output("{}_output".format(param), "children"), [Input(param, "value")])(callback)


def build_rewards_graph(simu):
    conn = create_connection(r"./database.db")
    cur = conn.cursor()
    cur.execute("SELECT type FROM output GROUP BY type")
    types = list(cur.fetchall())
    nb_graph = len(types)
    ret = {}
    finalData = []
    for type in types:
        if(type[0] != "total_reward"):
            x_array = []
            y_array = []
            cur.execute("SELECT * FROM output WHERE type='"+ type[0]+"' AND simu='"+simu+"'")
            dff = cur.fetchall()
            for point in list(dff):
                x_array.append(point[3])
                y_array.append(point[4])
            finalData += [{"x":x_array,
                           "y":y_array,
                           "name":type[0],
                           'line':{"width":3}
                          }]
    cur.close()
    conn.close()
    finalData = {
                   'data':finalData,
                   'layout':dict(margin=dict(t=40),
                        hovermode="closest",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        legend={"font": {"color": "white"}},
                        font={"color": "white"},
                        title= "Average separated rewards"),
                }
    return finalData

def build_total_reward_graph(simu):
    conn = create_connection(r"./database.db")
    cur = conn.cursor()
    finalData = []
    cur.execute("SELECT * FROM output WHERE type='total_reward' AND simu='"+simu+"'")
    dff = cur.fetchall()
    x_array = []
    y_array = []
    for point in list(dff):
        x_array.append(point[3])
        y_array.append(point[4])
    finalData += [{"x":x_array,
                   "y":y_array,
                   'line':{"color":"lime", "width":3},
                  }]
    cur.close()
    conn.close()
    finalData = {
                   'data':finalData,
                   'layout':dict(margin=dict(t=40),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        legend={"font": {"color": "white"}},
                        font={"color": "white"},
                        title= "Total reward"),
                }
    return finalData
      
def build_banner(simu):
    conn = create_connection(r"./database.db")
    cur = conn.cursor()
    cur.execute("SELECT simu FROM parameters GROUP BY simu")
    dff = list(cur.fetchall())
    cur.close()
    conn.close()
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(id="hidden-div",style={"display":"none"}),
            html.Div(
                id="banner-text",
                children=[
                    html.H5("AIDA gym env control panel"),
                    html.H6("Settings update and data visualization"),
                ],
            ),
            html.Div(
                id="banner-logo",
                children=[
                     dcc.Dropdown(
                                   id='dropdown-simulation',
                                   options=[{'label': i[0], 'value': i[0]} for i in dff],
                                   value=simu,
                                   searchable=False,
                                   clearable=False,
                                   style={'width':'150px', 'z-index':'300'},
                                ),   
                    html.Img(id="logo", src=app.get_asset_url("dash-logo-new.png")),
                    dcc.Interval(
                        id='interval-component',
                        interval=1*10000
                    ),
                ],
            ),
        ],
    )

def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)

def build_chart_panel(simu):
    return html.Div(
        id="control-chart-container",
        className = "twelve columns",
        children=[
            generate_section_banner("Graphs"),
            html.Div(
                style={'background-color':'#0e4b6b', 'border-radius': '25px', 'margin':'25px',
                       'border-color': 'white','border-width':'1px','border-style': 'solid'},
                children=[dcc.Graph(
                    id="rewards-chart",
                    figure=go.Figure(build_rewards_graph(simu)),
                    animate=True
                        ),
                    ]
            ),
            html.Div(
                style={'background-color':'#0e4b6b', 'border-radius': '25px', 'margin':'25px',
                       'border-color': 'white','border-width':'1px','border-style': 'solid'},
                children=[dcc.Graph(
                    id="total-reward-chart",
                    figure=go.Figure(build_total_reward_graph(simu)),
                    animate=True
                        ),
                    ]
            )
            ]
        )



def build_cards(simu):

    files = glob.glob("./assets/*.gif")
    
    fileNames = []
    for f in files:
        fileName = f[9:]
        if(fileName.startswith(simu+"_")):
            fileNames += [fileName]
    cards = []
    
    for name in fileNames:
        card = dbc.Card(
        [
            dbc.CardImg(src=app.get_asset_url(name), top=True),
            dbc.CardBody(
                [
                    html.H4(
                        name,
                        className="card-title",
                        style={'color': 'black'}
                    ),
                ]
            ),
        ],
        className="w-75 mb-3")
        cards +=[ card ]
    out = []
    for i in range(0,len(cards),3):
        out += [dbc.Row( [ dbc.Col(cards[len(cards)-y-1], md=4) for y in range(i, i+3) if y < len(cards)]  )]
    return out


def build_tabs(simu):
    init_params(simu)
    tab1_content = build_chart_panel(simu)
    tab2_content = html.Div(children= build_cards(simu))
    
    tab3_content = dbc.Card(
                    dbc.CardBody(
                    [
                    build_sliders(),
                    html.Div(className="row", children=[
                        html.Div(html.Button('Submit', id='submit-val', n_clicks=0)),
                        html.Div(id="spin_pending", style={'visibility':'none'},children=[dbc.Spinner(color="warning", type="grow")])], style={"align-content":"center"}),
                    ]
                ),
                className="mt-3",
            )
    
    tabs = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Graphiques"),
        dbc.Tab(tab2_content, label="Vidéos"),
        dbc.Tab(tab3_content, label="Configuration"),
    ]
    )
    return html.Div([tabs])

@app.callback([Output('app-container', 'children'),Output('dropdown-simulation', 'options') ],
              [Input('dropdown-simulation', 'value')])
def build_page(simu):
    simulationName = simu
    conn = create_connection(r"./database.db")
    cur = conn.cursor()
    cur.execute("SELECT simu FROM parameters GROUP BY simu")
    dff = list(cur.fetchall())
    cur.close()
    conn.close()
    return [ [
               build_tabs(simu)
               #html.Div(id="app-content"),
            ],[{'label': i[0], 'value': i[0]} for i in dff]
           ]

def build_initial_page():
    return [
        build_banner(simulationName),
        html.Div(
            id="app-container",
            children = [
                build_tabs(simulationName)
                #html.Div(id="app-content"),
            ])
           ]
           
app.layout = html.Div(
    id="big-app-container",
    children=build_initial_page()
     )   


@app.callback([Output('rewards-chart', 'figure'),
              Output('total-reward-chart', 'figure'),Output('spin_pending', 'style')],
              [Input('interval-component', 'n_intervals'),
              Input('dropdown-simulation', 'value')])
def update_graph_bar(interval, simu):
    conn = create_connection(r"./database.db")
    cur = conn.cursor()
    cur.execute("SELECT simu FROM parameters WHERE step=-1")
    dff = list(cur.fetchall())
    if(len(dff)==0):
        style = {'visibility':'none'}
    else:
        style = {'visibility':'visible'}
    cur.close()
    conn.close()
    return go.Figure(build_rewards_graph(simu)), go.Figure(build_total_reward_graph(simu)),style

@app.callback(
    Output('hidden-div', 'children'),
    [Input('submit-val', 'n_clicks')]
)
def update_params(n_clicks):
    if(n_clicks>0):
        send_params(simulationName)
    return ""
                                
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
    
    