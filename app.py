from Model import *
from plotly.tools import mpl_to_plotly
import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import io
import base64
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

def get_factors_list(factors):
    return factors
  
app.layout = html.Div(children=[
    dbc.Alert(
            "Error: Must Choose Atleast One Input for Decision Tree",
            id="alert-auto",
            is_open=False,
            duration=4000,
            color="danger"
        ),
    dbc.Alert(
        "Updating Model: This May Take a Few Seconds",
        id = 'alert-update',
        is_open=False,
        duration=2000,
        color='success'
    ),
    html.H1(children='Hired/Not Hired Model'),
    html.Div(children=[
        html.Div(children=[
        html.Img(id='example'),
        html.Div(id='accuracy')
        ]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Div(children=[
        dcc.Checklist(
            id = 'factors',
            labelStyle={'display': 'block'},
            options=[
                {'label': 'Category', 'value': 'Cleaning'},
                {'label': u'Rating', 'value': 'Rating given'},
                {'label': 'Cost Estimate', 'value': 'Estimated Cost of work'},
                {'label': 'Result Position', 'value': 'Result on Search Table'},
                {'label': 'Service page viewed?', 'value': 'If service page was seen'},
                {'label': 'Hired', 'value': 'Hired'}
            ],
            value=['Cleaning', 'Rating Given', 'Estimated Cost of work', 'Result on Search Table', 
            'If service page was seen', 'Hired']
        ),
        html.Label('Tree Depth'),
        dcc.Dropdown(
            id = 'depth',
            options = [
                {'label': '1', 'value': 1},
                {'label': '2', 'value': 2},
                {'label': '3', 'value': 3}
            ],
            value = 2,
            clearable = False
           )
         
        ])
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    dcc.Graph(id='importance')
])

# Output Tree Diagram
@app.callback(
    dash.dependencies.Output('example', 'src'),
    dash.dependencies.Output('importance', 'figure'),
    dash.dependencies.Output('accuracy', 'children'),
    dash.dependencies.Input('factors', 'value'),
    dash.dependencies.Input('depth', 'value')
)
def update_figure(factors, depth):
    # Output Tree Diagram
    buf = io.BytesIO() # in-memory files
    factors = get_factors_list(factors)
    model_info = create_model(factors, depth)
    dtree = model_info[0]
    cols = model_info[1]
    accuracy = 'Model Accuracy: ' + str(model_info[2])
    visualize_tree(dtree, cols)
    plt.savefig(buf, format = "png",dpi=300, bbox_inches = "tight") # save to the above file object
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements

    # Importance bar chart
    importance = find_importance(dtree, cols)
    imp_graph = px.bar(importance, x = 'Factor', y = 'Importance Level')
    imp_graph.update_layout(title_text='Feature Importance of Decision Tree', title_x=0.5)
    return "data:image/png;base64,{}".format(data), imp_graph, accuracy

@app.callback(
    dash.dependencies.Output("alert-auto", "is_open"),
    dash.dependencies.Input('factors', 'value'),
    [dash.dependencies.State("alert-auto", "is_open")]
)
# Toggle alert if no factors are chosen
def toggle_alert(factors, is_open):
    if len(factors) <= 0:
        return not is_open
    return is_open

@app.callback(
    dash.dependencies.Output("alert-update", "is_open"),
    dash.dependencies.Input('factors', 'value'),
    dash.dependencies.Input('depth', 'value'),
    [dash.dependencies.State("alert-update", "is_open")]
)
# Toggle alert to show model is updating and may take a few seconds
def toggle_alert_update(factors, depth, is_open):
    if len(factors) > 0:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server()
