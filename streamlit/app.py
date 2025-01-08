import pickle
import tensorflow

import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tensorflow as tf
from datetime import timedelta

import time

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import keras._tf_keras.keras.initializers
from keras._tf_keras.keras.layers import Dense, Layer, LSTM, GRU, SimpleRNN, RNN
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.regularizers import l1, l2

st.write(f"Running on Tensorflow version: {tensorflow.__version__}")

with open('streamlit/pages/styles.css') as css: 
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.markdown("# Welcome to RMD1 Machine Learning Portal!")

st.markdown("Make use of recurrent neural network (RNN), gated recurrent unit (GRU), and long-short term memory (LSTM) for bond yiend time series forecast.")

# ##############################################################

# # Open 2-year models

models_path = "streamlit/pages/models/batch360_stateFalse"

with open(f"{models_path}/2yr_models_rnn_bayes.pkl", "rb") as file:
     data_2yr_rnn = pickle.load(file)

with open(f"{models_path}/2yr_models_gru_bayes.pkl", "rb") as file:
     data_2yr_gru = pickle.load(file)

with open(f"{models_path}/2yr_models_lstm__bayes.pkl", "rb") as file:
     data_2yr_lstm = pickle.load(file)

data_2yr = {**data_2yr_rnn, **data_2yr_gru, **data_2yr_lstm}

# # Open 10-year models

with open(f"{models_path}/10yr_models_rnn_bayes.pkl", "rb") as file:
     data_10yr_rnn = pickle.load(file)

with open(f"{models_path}/10yr_models_gru_bayes.pkl", "rb") as file:
     data_10yr_gru = pickle.load(file)

with open(f"{models_path}/10yr_models_lstm_bayes.pkl", "rb") as file:
     data_10yr_lstm = pickle.load(file)

data_10yr = {**data_10yr_rnn, **data_10yr_gru, **data_10yr_lstm}


st.markdown("#### Best parameters for each model")

st.markdown(""" <style> .center-text { text-align: center; font-weight: bold; font-size: 20px; } </style> """, unsafe_allow_html=True)
st.markdown('<p class="center-text">2-year UST models:</p>', unsafe_allow_html=True)

table_models_2yr = pd.DataFrame.from_dict(data_2yr, orient='index')
table_models_2yr.drop(
    columns=["model", "function", "label", "color", "dataset", "forecast_horizon",
             "pred_train", "pred_test", "pred_train_scaled", "MSE_train", 'MSE_test', 'MSE_val',
             "pred_test_scaled", "y_train_scaled", "y_test_scaled", "pred_val",
             "pred_val_scaled", "y_val_scaled",
          #    'scaler_train_mean', 'scaler_train_scale',
          #    'scaler_val_mean', 'scaler_val_scale', 'scaler_test_mean', 'scaler_test_scale',
             'MSE_train_scaled', 'MSE_val_scaled', 'MSE_test_scaled',
          #    'MAE_train_scaled', 'MAE_test_scaled', 'MAE_val_scaled',
             'R2_train_scaled', 'R2_test_scaled', 'R2_val_scaled',
             'scaler_train_mean', 'scaler_train_std',
             'scaler_test_mean', 'scaler_test_std',
             'scaler_val_mean', 'scaler_val_std', 'cv_results'
             ], 
    inplace=True)
table_models_2yr['cv_time'] = table_models_2yr['cv_time'].apply(lambda x: (x/60))
table_models_2yr['train_time'] = table_models_2yr['train_time'].apply(lambda x: x/60)
table_models_2yr.rename(columns={'cv_time': 'Cross-val time (mins)',
                     'train_time': 'Training time (mins)',
                     'l1_reg': 'L1 Regularization',
                     'H': 'Nodes',
                    #  'MSE_train': 'MSE Train',
                    #  'MSE_test':'MSE Test',
                    #  'MSE_val': 'MSE Val',
                     'MAE_train_scaled': 'MAE Train',
                     'MAE_val_scaled': 'MAE Val',
                     'MAE_test_scaled': 'MAE Test'}, inplace=True)
st.table(table_models_2yr)

st.markdown(""" <style> .center-text { text-align: center; font-weight: bold; font-size: 20px; } </style> """, unsafe_allow_html=True)
st.markdown('<p class="center-text">10-year UST models:</p>', unsafe_allow_html=True)

table_models_10yr = pd.DataFrame.from_dict(data_10yr, orient='index')
table_models_10yr.drop(
    columns=["model", "function", "label", "color", "dataset", "forecast_horizon",
             "pred_train", "pred_test", "pred_train_scaled", "MSE_train", 'MSE_test', 'MSE_val',
             "pred_test_scaled", "y_train_scaled", "y_test_scaled", "pred_val",
             "pred_val_scaled", "y_val_scaled",
          #    'scaler_train_mean', 'scaler_train_scale',
          #    'scaler_val_mean', 'scaler_val_scale', 'scaler_test_mean', 'scaler_test_scale',
             'MSE_train_scaled', 'MSE_val_scaled', 'MSE_test_scaled',
          #    'MAE_train_scaled', 'MAE_test_scaled', 'MAE_val_scaled',
             'R2_train_scaled', 'R2_test_scaled', 'R2_val_scaled',
             'scaler_train_mean', 'scaler_train_std',
             'scaler_test_mean', 'scaler_test_std',
             'scaler_val_mean', 'scaler_val_std', 'cv_results'
             ], 
    inplace=True)
table_models_10yr['cv_time'] = table_models_10yr['cv_time'].apply(lambda x: (x/60))
table_models_10yr['train_time'] = table_models_10yr['train_time'].apply(lambda x: x/60)
table_models_10yr.rename(columns={'cv_time': 'Cross-val time (mins)',
                     'train_time': 'Training time (mins)',
                     'l1_reg': 'L1 Regularization',
                     'H': 'Nodes',
                    #  'MSE_train': 'MSE Train',
                    #  'MSE_test':'MSE Test',
                    #  'MSE_val': 'MSE Val',
                     'MAE_train_scaled': 'MAE Train',
                     'MAE_val_scaled': 'MAE Val',
                     'MAE_test_scaled': 'MAE Test'}, inplace=True)
st.table(table_models_10yr)