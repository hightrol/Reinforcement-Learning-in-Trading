# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:53:16 2020

@author: haiming
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines.common.base_class import BaseRLModel

def train(model, timesteps, epochs):
    assert isinstance(model, BaseRLModel)
    training_perf = []
    for i in range(epochs):
        print('training for {}-th iteration'.format(i))
        model.env.reset()
        trading_envs = [ea for ea in model.env.envs]
        model.learn(total_timesteps=timesteps)
        gross_pnl, net_pnl = sum([ea.gross_pnl for ea in trading_envs]), sum([ea.net_pnl for ea in trading_envs])
        training_perf.append([gross_pnl, net_pnl])
        print('gross pnl: {}, net pnl: {}, transaction cost: {}'.format(
            gross_pnl, net_pnl, gross_pnl-net_pnl))
    training_res = pd.DataFrame(training_perf, columns = ['gross total pnl', 'net total pnl'])
    training_res.index.name = 'epoch'
    training_res.plot(figsize=(20,20), color=['orange', 'green'])
    plt.title('reward and pnl in different training epoch')
    plt.show()   
    return model


def evaluate(model, test_data):
    assert isinstance(model, BaseRLModel)
    model.env.envs[0].training = False
    model.env.envs[0].df = test_data
    obs = model.env.reset()
    done = False
    for i in range(test_data.shape[0]-24):
        action, _states = model.predict(obs)
        obs, rewards, done, info = model.env.step(action)
    pnl_df = pd.DataFrame(model.env.envs[0].ret_history, columns = ['gross', 'net']).cumsum()
    pnl_df['buy_and_hold'] = test_data['close'] - test_data['close'].iloc[0]
    pnl_df.plot(figsize=(20,20), color=['orange', 'green', 'blue'])
    plt.show()