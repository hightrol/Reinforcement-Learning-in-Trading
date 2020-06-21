# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:19:17 2020

@author: haiming
"""

import pandas as pd
import talib as talib
import tensorflow as tf
from env.env import TradingEnv
from util.util import train, evaluate
from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy, MlpLstmPolicy, LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, A2C, DQN

# Load Data
df = pd.read_csv('data/OHLC_data.csv')
df = df.sort_values('date')

# technical factors
df2 = df.copy(deep=True)
df2['mfi'] = talib.MFI(df2['high'], df2['low'], df2['close'], df2['volume'], timeperiod=24)
df2['mom'] = talib.MOM(df2['close'], timeperiod=24)
df2['rsi'] = talib.RSI(df2['close'], timeperiod=24)
macd, macdsignal, macdhist = talib.MACD(df2['close'])
df2['macd'] = macd
df2['ma0-6'] = df2['close'] / talib.MA(df2['close'], timeperiod = 6).fillna(method='bfill') - 1
df2['ma0-24'] = df2['close'] / talib.MA(df2['close'], timeperiod = 24).fillna(method='bfill') - 1
df2['ma0-120'] = df2['close'] / talib.MA(df2['close'], timeperiod = 120).fillna(method='bfill') - 1
df2['slowk'], df2['slowj'] = talib.STOCH(df2['high'], df2['low'], df2['close'])
tech_factors = [ea for ea in df2.columns if ea not in df.columns]
# drop first 168 observation as they have inaccurate values
df2 = df2.iloc[120:, :].reset_index(drop=True)
print(df2.shape)
print(tech_factors)

# Train Test data split
train_df = df2.iloc[:15000].reset_index(drop=True)
test_df = df2.iloc[15000:].reset_index(drop=True)
print(train_df.shape, test_df.shape)

# Train
## A2C+FeedForward
n = train_df.shape[0]
ff_env = DummyVecEnv([lambda: TradingEnv(train_df, start=0, end=n-3, features=tech_factors)])
policy_kwargs = {'layers':[16, 16, 16], 'feature_extraction':'mlp', 'act_fun':tf.nn.sigmoid}
ff_model = A2C(FeedForwardPolicy, ff_env, verbose=False, policy_kwargs = policy_kwargs, 
            learning_rate = 0.0005, gamma=1, tensorboard_log='./tensorboard/')
ff_model = train(ff_model, n-3, 30)

## A2C+LSTM
n = train_df.shape[0]
lstm_env = DummyVecEnv([lambda: TradingEnv(train_df, start=0, end=n-3, features=tech_factors)])
policy_kwargs = {'n_lstm':16, 'feature_extraction':'mlp', 'net_arch':['lstm']}
lstm_model = A2C(LstmPolicy, lstm_env, verbose=False, policy_kwargs = policy_kwargs, 
            learning_rate = 0.0005, gamma=1, tensorboard_log='./tensorboard/')
lstm_model = train(lstm_model, n-3, 30)

# Test
evaluate(ff_model, test_df)
evaluate(lstm_model, test_df)


