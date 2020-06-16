# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:17:47 2020

@author: haiming
"""

import random
import gym
from gym import spaces
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class TradingEnv(gym.Env):
    """A crypotrading environment using Gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, training = True, start = None, end = None, features = None):
        super(TradingEnv, self).__init__()
        self.df = df
        self.training = training
        # Actions: have stock (1) or doesn't have stock (0) etc
        self.action_space = spaces.Discrete(2)
        # Prices contains the OHCL values for the last five prices
        self.features = features if features else df.columns
        self.observation_space = spaces.Box(low=-4, high=4, shape=(len(self.features)+1, ), dtype=np.float32)
        self.scaler = StandardScaler()
        self.start = start if start else 0
        self.end = end if end else self.df.shape[0]-3
        self.transaction_cost = 0.0001
        
    def reset(self):
        self.shares_held = 0
        # pnl with no transaction cost
        self.gross_pnl = 0
        # pnl with transaction cost
        self.net_pnl = 0
        self.current_step = 0 if self.training else 0
        self._scale_data()
        # list of tuples containing gross_ret and net_ret
        self.ret_history = []
        return self._next_observation()
    
    def _scale_data(self):        
        # normalize data
        if self.training:
            self.scaler = self.scaler.fit(self.df[self.features])
        df_tech = self.scaler.transform(self.df[self.features])
        for ix, col in enumerate(self.features):
            self.df[col] = df_tech[:,ix]        

    def _next_observation(self):
        # Get the signals
        signals = self.df.loc[self.current_step, self.features].values
        signals = np.append(signals, self.shares_held)
        return signals

    def _take_action(self, action):
        ret = self.shares_held * (self.df.loc[self.current_step, "close"] - self.df.loc[self.current_step-1, "close"]) if self.current_step > 0 else 0
        net_ret = gross_ret = ret
        if action != self.shares_held:
            net_ret -= self.transaction_cost*self.df.loc[self.current_step, "close"]
        self.shares_held = action
        self.gross_pnl += gross_ret
        self.net_pnl += net_ret
        self.ret_history.append((gross_ret, net_ret))

    def step(self, action):
        # Execute one time step within the environment
        done = False                  
        self._take_action(action)
        
        # balance between gross_ret and net_ret
        reward = self.ret_history[-1][-1]
        
        if self.current_step < self.end:
            self.current_step += 1
        else:
            done = True

        obs = self._next_observation()

        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print('Step: {}'.format(self.current_step))
        print('Cash: {}'.format(self.cash))
        print('Shares held: {}'.format(self.shares_held))
        print('Current Total Pnl: {}'.format(self.pnl))
