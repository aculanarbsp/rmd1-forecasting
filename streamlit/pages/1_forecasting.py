import pickle

import streamlit as st

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import statsmodels.api as sm
# import tensorflow as tf
# from datetime import timedelta

import time

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import keras._tf_keras.keras.initializers
from keras._tf_keras.keras.layers import Dense, Layer, LSTM, GRU, SimpleRNN, RNN
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.regularizers import l1, l2

from src.functions import plot_forecast
import pickle

# Open 2-year models

models_path = "streamlit/pages/models/batch360_stateFalse"

with open(f"{models_path}/2yr_models_rnn_bayes.pkl", "rb") as file:
     data_2yr_rnn = pickle.load(file)

with open(f"{models_path}/2yr_models_gru_bayes.pkl", "rb") as file:
     data_2yr_gru = pickle.load(file)

with open(f"{models_path}/2yr_models_lstm_bayes.pkl", "rb") as file:
     data_2yr_lstm = pickle.load(file)

data_2yr = {**data_2yr_rnn, **data_2yr_gru, **data_2yr_lstm}

st.write(data_2yr)

with open(f"{models_path}/10yr_models_rnn_bayes.pkl", "rb") as file:
     data_10yr_rnn = pickle.load(file)

with open(f"{models_path}/10yr_models_gru_bayes.pkl", "rb") as file:
     data_10yr_gru = pickle.load(file)

with open(f"{models_path}/10yr_models_lstm_bayes.pkl", "rb") as file:
     data_10yr_lstm = pickle.load(file)

data_10yr = {**data_10yr_rnn, **data_10yr_gru, **data_10yr_lstm}

# #########################################################

train_mean = data_2yr['rnn']['scaler_train_mean']
train_scale = data_2yr['rnn']['scaler_train_std']

st.write("### Upload data here:")

# Take input data from user
uploaded_csv_2yr = st.file_uploader("Input .csv file of 2-year yields here:", type="csv")

uploaded_csv_10yr = st.file_uploader("Input .csv file of 10-year yields here:", type="csv")

if uploaded_csv_2yr is not None:
    df_2yr = pd.read_csv(uploaded_csv_2yr)
    df_2yr['date'] = pd.to_datetime(df_2yr['date'])
    df_2yr.index = pd.to_datetime(df_2yr['date'])
    df_2yr.drop(columns=['date'], inplace=True)
    
    scaler_2yr = StandardScaler()
    scaler_2yr.mean_ = df_2yr['rnn']['scaler_train_mean']
    scaler_2yr.scale_ = df_2yr['rnn']['scaler_train_std']
    
    df_2yr_scaled_ = scaler_2yr.transform(np.array(df_2yr['yield']).reshape(-1,1))
    for_input_2yr = df_2yr_scaled_.reshape(1, len(df_2yr), 1)

    models = ['rnn', 'gru', 'lstm']

    rnn_pred = scaler_2yr.inverse_transform(data_2yr['rnn']['model'].predict(for_input_2yr))
    gru_pred = scaler_2yr.inverse_transform(data_2yr['gru']['model'].predict(for_input_2yr))
    lstm_pred = scaler_2yr.inverse_transform(data_2yr['lstm']['model'].predict(for_input_2yr))

    st.write("### Forecast plot (2yr UST):")
    plot_forecast(input_dates=df_2yr.index,
                  input_data=df_2yr,
                  forecast_rnn=rnn_pred,
                  forecast_gru=gru_pred,
                  forecast_lstm=lstm_pred,
                  dataset_="2yr")
     

if uploaded_csv_10yr is not None:
    df_10yr = pd.read_csv(uploaded_csv_10yr)
    df_10yr['date'] = pd.to_datetime(df_10yr['date'])
    df_10yr.index = pd.to_datetime(df_10yr['date'])
    df_10yr.drop(columns=['date'], inplace=True)
    
    scaler_10yr = StandardScaler()
    scaler_10yr.mean_ = df_10yr['rnn']['scaler_train_mean']
    scaler_10yr.scale_ = df_10yr['rnn']['scaler_train_std']
    
    df_10yr_scaled_ = scaler_10yr.transform(np.array(df_10yr['yield']).reshape(-1,1))
    for_input_10yr = df_10yr_scaled_.reshape(1, len(df_10yr), 1)

    models = ['rnn', 'gru', 'lstm']

    rnn_pred = scaler_10yr.inverse_transform(data_10yr['rnn']['model'].predict(for_input_10yr))
    gru_pred = scaler_10yr.inverse_transform(data_10yr['gru']['model'].predict(for_input_10yr))
    lstm_pred = scaler_10yr.inverse_transform(data_10yr['lstm']['model'].predict(for_input_10yr))

    st.write("### Forecast plot (10yr UST):")
    plot_forecast(input_dates=df_10yr.index,
                  input_data=df_10yr,
                  forecast_rnn=rnn_pred,
                  forecast_gru=gru_pred,
                  forecast_lstm=lstm_pred,
                  dataset_="10yr")