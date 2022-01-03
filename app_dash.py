import numpy as np
import random 
import sys
import pandas as pd
import matplotlib.pyplot as plt
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Output,Input, State
from dash import callback_context
import plotly.graph_objects as go
import dash_daq as daq
import plotly.express as px
import pickle

app = dash.Dash('Interactive dashboard', external_stylesheets=[dbc.themes.DARKLY])
app.title = 'Risk assessment dashboard'

df = pd.read_csv('final_customer_df.csv')

# dataframe on which model is applied
test_dataframe= pd.read_csv('final_df_trans.csv')
liste = ['SK_ID_CURR', 'TARGET']
elements_1 = [ele for ele in df.columns if ele not in liste]
targets = []
for i in df['TARGET'].values :
	targets.append(int(i))
filename = 'finalized_model.sav'
#loaded_model = pickle.load(open(filename, 'rb'))
#loaded_model.fit(df[elements_1], df['TARGET'])
#options = test_dataframe['SK_ID_CURR'].values

loaded_model = pickle.load(open(filename, 'rb'))
#customer dropdown options
options = df['SK_ID_CURR'].values
liste = ['SK_ID_CURR', 'TARGET']
elements_1 = [ele for ele in df.columns if ele not in liste]
loaded_model.fit(df[elements_1], df['TARGET'])
feature_list_labels = ['Percentage of active loans', 'Percentage of ended credits',  'Percentage of debt' ,  'Days before decison', 
						'Credit payment number of months','Paymenent period difference(nb days)', 'Total income amount', 'Difference between credit and goods price', 'Region rating']
feature_list_values= ['ACTIVE_LOANS_PERCENTAGE', 'CREDIT_ENDDATE_PERCENTAGE',  'DEBT_CREDIT_RATIO' ,  'DAYS_DECISION_y', 
						'MIN_MONTH','DAYS_DIFF', 'AMT_INCOME_TOTAL', 'DIFF_CRE_GOODS_x', 'REGION_RATING_CLIENT']
qualit_feature_list_labels = ['Gender', 'Education level', 'Own a house/flat ?', 'Own a car ?']
qualit_feature_list_values= ['CODE_GENDER', 'NAME_EDUCATION_TYPE',  'FLAG_OWN_REALTY', 'FLAG_OWN_CAR']


# layout definition as a grid of rows and columns 
app.layout = html.Div([
dbc.Card
	(
        dbc.CardBody
		([
            dbc.Row
			([		
			dbc.Card
				(
					dbc.CardBody
					([			
							html.Div([
								html.Br(),	
										html.H1("Risk Assessment Dashboard",
										style={'font-family':'Georgia', 'textAlign': 'center'})	,
										html.Br(),							
							]),
					])											
				)
			]),
			html.Br(),				
			dbc.Row
			([				
				dbc.Col([
					dbc.Card
				(
					dbc.CardBody
					([	
					html.Div(className='div-for-dropdown',
					children=[
						html.Div(children="Introduce customer identifier : "),
						html.Br(),
						html.Div(className='gap2'),
						html.Div([
						dcc.Dropdown(
							id='customer-dropdown',
							className='dropclass',
							options=[{"label":x,"value":x} for x in options],
							value='',	
							style= { 'color': '#212121'}							
						),
						html.Br(),							
						html.Div(id='div-for-details',
							children=[
								html.Br(),
								html.H5("Customer Informations ", style={'font-family':'Georgia'})	,
								html.Br(),	html.Label('Gender :'), html.Br(),
								html.Div(id = "genre" , children =""),
								html.Br(), html.Label('Age :'), html.Br(),
								html.Div(id = "age", children =""),
								html.Br(), html.Label('Revenues :'), html.Br(),
								html.Div(id = "rev", children =""),
								html.Br(), html.Label('Revenue sources :'), html.Br(),
								html.Div(id = "rev_src", children =""),
								html.Br(), html.Label('Marital situation :'), html.Br(),
								html.Div(id = "stat", children =""),
								html.Br(), html.Label('Number of children :'), html.Br(),
								html.Div(id = "child_nb", children =""),
								html.Br(),	 html.Label('Working domain :'), html.Br(),
								html.Div(id = "dom", children =""), html.Br()							
							], style= {'display': 'none'}),
						])
				
					]),
					])
				,style={'background-color': '#666666'}),
				], width = 2),
				
		
				dbc.Col([
					dbc.Card
					(
						dbc.CardBody
						([	html.H4("Score computing : ",
										style={'font-family':'Georgia', 'textAlign': 'left'})	,
										html.Br(),
							html.Div(id = 'gauge_div', children = [	
								dbc.Row([
									dbc.Col([							
										html.Div([
										daq.Gauge(id = "score_gauge", 
											color = {"gradient":True,"ranges":{"red":[0,60],"yellow":[60,80],"green":[80,100]}},
											value = 0,
											label = 'Payment Default Risk Score',
											max=100,
											min=0,
											),							 
										]),								
									], width=3),
									dbc.Col([
									html.Div([
										html.H5("Score explanation"),
										html.Div(id = "score_explanation", children = "")
										])	           			
									], width=4)	,
									dbc.Col([
										html.Div( 
										children = [
											html.H5("Important features in score calculation "),
											html.Div(
											dcc.Graph(id = "graph_feature_importance")
											)
										]),
									], width=5),										
							
								])									
							], style= {'display': 'none'}),
						])
					),
					html.Br(),					
					dbc.Card
					(
						dbc.CardBody
						([		
							dbc.Row([		
								dbc.Col([	
								html.Div([
									html.H5("Select a feature"),
								]),
								dcc.Dropdown(
									id='feature-dropdown',
									className='dropclass',
									options= [{"label":feature_list_labels[x],"value":feature_list_values[x]} for x in range(0, len(feature_list_values))],
#									and  for  y in  feature_list_labels],
									value='',	
									style=
                                    { 'color': '#212121'
                                    }
								),
								html.Br(),
								#html.Button('Submit', id='button_feature_id', n_clicks=0),
														
								], width=4),
								dbc.Col([
									html.Div(id = "feature_div1",
									children =[
									html.H5("Customer position among other groups"),
									dcc.Graph(id = "one_feature_graph")
								], style= {'display': 'none' }),
								], width=8)							
							])
						])
					),
					html.Br(),					
					dbc.Card
					(
						dbc.CardBody
						([		
							dbc.Row([		
								dbc.Col([	
								html.Div([
									html.H5("Select two features"),
								]),
								html.Div(
								[
								dcc.Store(id="session", storage_type="session"),
								dcc.Dropdown(
									id = 'feature-dropdown_1',
									className = 'dropclass',
									options = [{"label":feature_list_labels[x],"value":feature_list_values[x]} for x in range(0, len(feature_list_values))],
									value='',	
									style={ 'color': '#212121' }
								),
								html.Br(),
								dcc.Dropdown(
									id = 'feature-dropdown_2',
									className ='dropclass',
									options = [{"label":qualit_feature_list_labels[x],"value":qualit_feature_list_values[x]} for x in range(0, len(qualit_feature_list_labels))],
									value = qualit_feature_list_values[0],	
									style = { 'color': '#212121'}
								),	
								html.Br(),
								])
								], width=4),
								dbc.Col([
									html.Div(id = "feature_div2",
									children =[
									html.H5("Customer position among other groups (2-dimensional analysis)"),
									dcc.Graph(id = "two_features_graph")
								], style= {'display': 'none'}),
								], width=8)							
							]),
							html.Br(),
							dbc.Row([
								dbc.Col([
									html.Div(id ="div_fig1", 
									children = [
										html.H5("Variable ditribution"),
										dcc.Graph(id ="fig1")
									], style= {'display': 'none'})
								], width=6),
								dbc.Col([
									html.Div(id ="div_fig2", 
									children = [
										html.H5("Heatmap matrix"),
										dcc.Graph(id ="fig2")
									], style= {'display': 'none'})							
								], width=6),
								
							])
						])
					)												
				], width = 10),
																	
			])	
		])
	, style={'background-color': '#bfbfbf'})
])	



#callback detail section
@app.callback(Output(component_id='div-for-details', component_property='style'),
	Output('genre', 'children'),
	Output('age', 'children'),Output('rev', 'children'),Output('rev_src', 'children'),
	Output('stat', 'children'),Output('child_nb', 'children'),Output('dom', 'children'),
	Output('score_gauge', 'value'),
	Output('gauge_div', 'style'),
	Output('score_explanation', 'children'),
	Output('graph_feature_importance', 'figure'),
	Input('customer-dropdown', 'value'), prevent_initial_call=True
   )
def get_customer_details_score(value): 
	if value is not None :	
		#details filling
		global customer_1
		customer = test_dataframe[test_dataframe['SK_ID_CURR']==value]	
		nb_child = customer['CNT_CHILDREN']
		gender = customer['CODE_GENDER']
		income = customer['AMT_INCOME_TOTAL']
		inc_t = customer['NAME_INCOME_TYPE']
		f_stat =customer['NAME_FAMILY_STATUS']
		age = abs(customer['DAYS_BIRTH']) // 365
		org = customer['ORGANIZATION_TYPE']
		#customer_1 = test_dataframe.loc[test_dataframe['SK_ID_CURR'] == value]
		# Calculate the Score 
		#customer_1 = df[df['SK_ID_CURR'] == value]
		customer_1 = df[df['SK_ID_CURR'] == float(value)]
		target = df[df['SK_ID_CURR'] == value]['TARGET']
		score = loaded_model.predict_proba(customer_1[elements_1].values)[0]*100
		print(score[0])
		#score = loaded_model.predict_proba(customer[elements_1].values)[0]*100
		#sorted_idx = loaded_model.feature_importances_.argsort()
		# plt.barh(df.feature_names[sorted_idx], loaded_model.feature_importances_[sorted_idx])
		feature_imp = pd.DataFrame(sorted(zip(loaded_model.feature_importances_,test_dataframe.columns)), columns=['Value','Feature'])
					  
		fig = go.Figure()
		fig.add_trace(go.Bar(
			x=data['Feature'],
			y=data['Value'],
			xhoverformat="Q%q",
			hovertemplate="%{y}%{_xother}"
		))
		fig.update_layout(hovermode="x unified")

		dv_style = {'display': 'block'}
		if score[0] >60 :
			explanation = "Credit request accepted"
		elif score[0] >40 and score[0] <= 60 :
			explanation = "Credit request refused : minor risk"
		else :
			explanation = "Credit request refused : important risk"
		return dv_style, gender, age, income, inc_t, f_stat, nb_child, org, score[0], dv_style,  explanation



	
@app.callback(Output('feature_div1', 'style'),
	Output('one_feature_graph', 'figure'),
	Input('feature-dropdown', 'value'), prevent_initial_call=True
	) 
def get_graph_from_one_feature(feature_value):	
	if  feature_value is not None:	
		v= customer_1[feature_value].values [0]
		fig =  px.histogram(test_dataframe, x=feature_value, nbins=10, color='TARGET')
		fig.add_vline(x=v, line_dash = 'dash', line_color = 'firebrick')
		dv_style = {'display': 'block' , "height": "60%", "width": "100%"}
		return dv_style , fig
		
@app.callback(Output('feature_div2', 'style'),
	Output('div_fig1', 'style'),
	Output('div_fig2', 'style'),
	Output('two_features_graph', 'figure'),	
	Output('fig1', 'figure'),
	Output('fig2', 'figure'),	
	[Input('feature-dropdown_1', 'value'),	
	Input('feature-dropdown_2', 'value')],
	State("session", "data"),
	prevent_initial_call=True
	)
def get_graph_from_two_features(feature_value_1, feature_value_2, data):
	if feature_value_1 != None:	
		y = feature_value_1
	if feature_value_2 != None :
		x = feature_value_2
	v= customer_1[feature_value_1].values [0]
	fig = px.histogram(test_dataframe, 
			x=x,  y=y, 
			color='TARGET')
	fig.add_vline(x=v, line_dash = 'dash', line_color = 'firebrick')
	
	dv_style = {'display': 'block' }
	fig1 = px.pie(test_dataframe, values='TARGET', names=x)
	
	fig2 = px.density_heatmap(test_dataframe, x=y, y='TARGET')
	
	return dv_style,dv_style, dv_style, fig, fig1, fig2
			
	
if __name__=='__main__':
    app.run_server(debug = "on")