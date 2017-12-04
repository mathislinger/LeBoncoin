import dash
from dash.dependencies import Input, Output, State, Event
import dash_core_components as dcc
import dash_html_components as html

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import base64

# For the PCA
import unidecode
from string import digits
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from sklearn.decomposition import PCA

stop = set(stopwords.words('french'))
stop = set(pd.Series(np.array(list(stop))).str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))

# Open PCA transformation and list of columns to transform
list_pca_learning = pickle.load(open('list_pca_learning.sav', 'rb'))
pca_ = pickle.load(open('pca_transfo.sav', 'rb'))

# Open lasso model
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))


# Image
image_filename = 'billets.jpg' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app = dash.Dash('Hello World')

app.layout = html.Div(style={'backgroundColor': 'lavender'}, children=[
    html.H1(
        children='Hi! I am here to help you find your dreaming appartment :)',
        style={'textAlign': 'center','color': 'mediumblue'}),

    html.Div(children='Please fill the recquired information.', style={'textAlign': 'center','color': 'mediumblue'}),

    html.Div([
   
    html.Label('Number of pictures', style={'textAlign': 'center', 'margin-top': '30px', 'fontSize': 20}),
    dcc.Slider(
        id='nbphoto-dropdown',
        min=0,
        max=10,
        marks={i: str(i) for i in range(0, 11)},
        value=5,
    ),


    html.Label('Number of rooms', style={'textAlign': 'center', 'margin-top': '110px', 'fontSize': 20}),
    dcc.Slider(
        id='pieces-dropdown',
        min=1,
        max=5,
        marks={i: str(i) for i in range(1, 6)},
        value=3,
    ),


    html.Div(id='updatemode-output-container', style={'textAlign': 'center', 'margin-top': '110px', 'fontSize': 20}),
    dcc.Slider(
    id='surface-dropdown',
    min=0,
    max=250,
    step=1,
    value=12,
)],style={'width': '45%', 'float': 'right', 'display': 'inline-block', 'margin-right': '30px'}),

    html.Div([
        
    html.Label('District', style={'textAlign': 'center', 'margin-top': '20px', 'fontSize': 20}),
    dcc.Dropdown(
        id='district-dropdown',
        options=[
            {'label': '75001', 'value': 0},
            {'label': '75002', 'value': 1},
            {'label': '75003', 'value': 2},
            {'label': '75004', 'value': 3},
            {'label': '75005', 'value': 4},
            {'label': '75006', 'value': 5},
            {'label': '75007', 'value': 6},
            {'label': '75008', 'value': 7},
            {'label': '75009', 'value': 8},
            {'label': '75010', 'value': 9},
            {'label': '75011', 'value': 10},
            {'label': '75012', 'value': 11},
            {'label': '75013', 'value': 12},
            {'label': '75014', 'value': 13},
            {'label': '75015', 'value': 14},
            {'label': '75016', 'value': 15},
            {'label': '75017', 'value': 16},
            {'label': '75018', 'value': 17},
            {'label': '75019', 'value': 18},
            {'label': '75020', 'value': 19}
        ],
        value=1,
        multi=False,
    ),

    html.Div([
    html.Label('GES', style={'textAlign': 'center', 'margin-top': '15px', 'fontSize': 20}),
    dcc.RadioItems(
        id='ges-dropdown',
        options=[{'label': 'Not defined', 'value': 0},{'label': 'a', 'value': 1},{'label': 'b', 'value': 2},{'label': 'c', 'value': 3},
        {'label': 'd', 'value': 4},{'label': 'e', 'value': 5},{'label': 'f', 'value': 6},{'label': 'g', 'value': 7}
        ],
        value=0,
        labelStyle={},
        style={'margin-left': '20px', 'textAlign': 'center','columnCount': 3}
    ),

    html.Label('NRJ', style={'textAlign': 'center', 'margin-top': '15px', 'fontSize': 20}),
    dcc.RadioItems(
        id='nrj-dropdown',
        options=[{'label': 'Not defined', 'value': 0},{'label': 'a', 'value': 1},{'label': 'b', 'value': 2},{'label': 'c', 'value': 3},
        {'label': 'd', 'value': 4},{'label': 'e', 'value': 5},{'label': 'f', 'value': 6},{'label': 'g', 'value': 7}
        ],
        value=0,
        labelStyle={},
        style={'margin-left': '20px', 'textAlign': 'center','columnCount': 3}
    ),

    html.Label('Professional', style={'textAlign': 'center', 'margin-top': '15px', 'fontSize': 20}),
    dcc.RadioItems(
        id='pro-dropdown',
        options=[{'label': 'Professional', 'value': 1},{'label': 'Individual', 'value': 0}
        ],
        value=1,
        labelStyle={},
        style={'margin-left': '20px', 'textAlign': 'center', 'columnCount': 2}
    )],style={})]
    , style={'width': '45%', 'display': 'inline-block', 'margin-left': '30px'}),



    html.Label('Announce description', style={'textAlign': 'center', 'fontSize': 20}),
    html.Div(id='desc_treatment'),
    dcc.Input(id='desc-input', type='text', value='Enter the announce description', style={'width': '90%', 'margin-left': '20px'}),
    html.Button(id='submit', type='submit', children='ok'),

    html.Hr(),
    html.Div(style={'textAlign': 'center', 'fontSize': 25}, id='desc_treatment'),
    html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))], style={'textAlign': 'center'}),


    html.Hr(),
])
#, style={'columnCount': 2}


@app.callback(Output('updatemode-output-container', 'children'),[Input('surface-dropdown', 'value')])
def display_value(value):
    return 'Surface area: {} square meters'.format(value)

@app.callback(Output('desc_treatment', 'children'), [Input('nbphoto-dropdown', 'value'), Input('pieces-dropdown', 'value'), Input('surface-dropdown', 'value'),
Input('district-dropdown', 'value'), Input('ges-dropdown', 'value'), Input('nrj-dropdown', 'value'), Input('pro-dropdown', 'value')], [State('desc-input', 'value')], [Event('submit', 'click')])
def callback(nbphoto, pieces, surface, district, ges, nrj, pro, description):
    # Clean description
    unaccented_string = unidecode.unidecode(description).lower()
    removed_punct_digit = ''.join([i for i in unaccented_string if not i.isdigit() and i not in frozenset(string.punctuation)])
    cleaned_desc = ' '.join([j for j in removed_punct_digit.split(' ') if j not in stop])
    stemmer = FrenchStemmer()
    stemmed_desc = ' '.join([stemmer.stem(j) for j in cleaned_desc.split(' ')])
    # Get same array structure as the one used for the training part of the PCA
    my_desc = pd.DataFrame(data=np.repeat(1, len(stemmed_desc.split(' ')), axis=0).reshape(1,-1), columns = stemmed_desc.split(' '))
    desc_to_reduce = pd.DataFrame()
    for i, col in enumerate(list_pca_learning):
        if col in list(my_desc.columns):
            desc_to_reduce.loc[1,col] = 1
        else:
            desc_to_reduce.loc[1,col] = 0
    # Do projection on the correponding component and get features
    desc_vec = list(pca_.transform(desc_to_reduce.as_matrix())[0])
    # Get variables related to other features
    district_vec = [0] * district + [1] + [0] * (19-district)
    ges_vec = [0] * ges + [1] + [0] * (7-ges)
    nrj_vec = [0] * nrj + [1] + [0] * (7-nrj)
    # Join features to create the vector to predict
    features = [nbphoto] + [pieces] + [surface] + district_vec[1:] + ges_vec[1:] + nrj_vec[1:] + [pro] + [pieces * pieces] + [pieces * surface] + desc_vec
    poly = PolynomialFeatures(interaction_only=True)
    features_ = poly.fit_transform(np.array(features).reshape(1,-1))
    return 'If you want a such appartment, be ready to spend {} euros!'.format(int(np.around(loaded_model.predict(np.array(features_).reshape(1, -1))[0], decimals=0)))


app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})


if __name__ == '__main__':
    app.run_server()
