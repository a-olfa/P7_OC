import plotly.express as px
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import dash_daq as daq
import pandas as pd

df = pd.read_csv('final_df.csv')
model_file = '/model/model.pkl'
def drawBarPlot(df, column1, column2):
	return  html.Div(children=[
		html.H1(children=feature,style={'textAlign': 'center', 'color': '#7FDBFF'}),
		dcc.Graph(
            figure = px.bar(df, x=column1, y=column2, color='DAYS_CREDIT_UPDATE',  barmode="group")
			.update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    ),
                    config={
                        'displayModeBar': False
                    }
		)
	])	

# bar figure
def drawFigure(df, x, y):
    return  html.Div([      
                dcc.Graph(
                    figure=px.bar(
                        df, x=x, y=y, color='DAYS_CREDIT_UPDATE'
                    ).update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    ),
                    config={
                        'displayModeBar': False
                    }
                )            
    ])
# scatter definition
def draw_scatter(df, x, y):
	return html.Div([
		dcc.Graph(
			id='life-exp-vs-gdp',
			figure = px.scatter(df, x=x, y=y,
					 size="population", color='DAYS_CREDIT_UPDATE',  hover_name='DAYS_CREDIT_UPDATE',
					 log_x=True, size_max=30).update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    ),
                    config={
                        'displayModeBar': False
                    }
		)
	])

#draw gauge figure 
def drawGaugeFigure(score):
    return  html.Div([
					dbc.Row([
						dbc.Col([
							daq.Gauge(
							color={"gradient":True,"ranges":{"red":[0,60],"yellow":[60,80],"green":[80,100]}},
							value=score,
							label='Payment Default Risk Score',
							max=100,
							min=0,
							),							 
						], width=3),
						dbc.Col([
							html.H4("Score explanation"),
							html.P("This conversion happens behind the scenes by Dash's JavaScript front-end")
						], width=6)	               			
					])				
    ])
def get_summary_dataframe(df, id, columns):
	means = df[columns].mean(axis=0).value
	mins = df[columns].min(axis=0).value
	maxs = df[columns].max(axis=0).value
	customer_values = df.loc[id].value
	dataframe = pd.DataFrame([customer_values, means, mins, maxs], columns=columns)
	return dataframe
	


#Input text field
def drawInputtext():
	return html.Div([
		
					html.I("Customer identifier : "),
					html.Br(),       
					dcc.Dropdown(
							id="customer_id",
							#type="number",
							#style={'marginRight':'10px'},
							#placeholder="id",
							#autoComplete="on",
							options = [{'label': i, 'value': i}
								for i in df.index.unique()]
						)
					,
					html.Button('Submit', id='button_customer_id'),	
				
				
	])


# dropdown component
def drawDropdown(list_of_features):
 	return html.Div([		
                html.Div([
				dcc.Dropdown(
						id='features-dropdown',
						options=[{"label": i, "value": i} for i in list_of_features],
						value=list_of_features[0],
					),
					html.Div(id='dd-output-container')	
				])		
	])	
# update_output
def update_output(n_clicks, value):
    if n_clicks > 0:
        return 'You have entered: \n{}'.format(value)

# Text field
def drawText():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2("Risk Assessment Dashboard"),
                ], style={'textAlign': 'center'}) 
            ])
        ),
    ])




# Build App
app = JupyterDash(external_stylesheets=[dbc.themes.SLATE])

app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
						html.Div([
							html.H2("Risk Assessment Dashboard"),
						], style={'textAlign': 'center'})           
					])
                ]),
            ], align='center'), 
            html.Br(),
			html.Div([
				dbc.Card(
					dbc.CardBody([			
						dbc.Row([						
							dbc.Col([
								dbc.Row([
									dbc.Col([
										drawInputtext() ,
										html.Br(),
										drawFigure(df, "CODE_GENDER_F", "CREDIT_ENDDATE_PERCENTAGE"), 
										#(df, "CODE_GENDER_F", "CREDIT_ENDDATE_PERCENTAGE")
									]),
								]),
										
							], width=3),
							
							dbc.Col([
								dbc.Row([
									dbc.Row([										
										#drawGaugeFigure() 										
									]),	
								]),
								html.Br(), 
								dbc.Row([
									drawFigure(df, "CODE_GENDER_F", "CREDIT_ENDDATE_PERCENTAGE")
								]),
								html.Br(), 
								dbc.Row([
									dbc.Col([
										html.Br(), 
										#drawInputtext()
										#drawDropdown() ,
										#html.Br(), 
										#generate_table(df, max_rows=5)
									]),
									html.Br(), 
									dbc.Col([
										#drawInputtext()
										#drawFigure() 
									]),
								]),				
							], width=9),
						]), 
					])
				),  
			])
		]),  color = 'dark'
	)
])
# Run app and display result inline in the notebook
app.run_server(mode='external')