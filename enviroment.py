# -*- coding: utf-8 -*-
import numpy as np
import itertools

class Env():
    """
    3-stock enviroment
    State vector: #shares of stock1, 
        #shares of stock2,
        #shares of stock3,
        closing price 1,
        closing price 2,
        closing price 3,
        unused cash
    Action space: categorical, 3 stocks with buy/sell/hold = 3^3 = 27 possibilities
    """
    def __init__(self, data, initial_investment=80e3):
        self.stock_history = data
        self.n_step, self.n_stock = data.shape
        self.initial_investment = initial_investment
        self.current_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_at_hand = None
        self.action_space = np.arange(3**self.n_stock)
        # 0-sell, 1-hold, 2-buy, permutations of all possible actions
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        self.state_dim = self.n_stock*2 + 1
        self.reset()
        
    def reset(self):
        # point at first day in dataset
        self.current_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_history[self.current_step]
        self.cash_at_hand = self.initial_investment
        return self.get_obs()
    
    def step(self, action):
        assert action in self.action_space
        
        # get current value
        prev_val = self.get_val()
        # update price, go to the next day
        self.current_step += 1
        self.stock_price = self.stock_history[self.current_step]
        
        # perform the trade
        self.trade(action)
        
        # get next value after performing the action
        cur_val = self.get_val()
        # reward is the increase of portfolio value
        reward = cur_val - prev_val
        done = self.current_step == self.n_step-1
        
        info = {'current value of portfolio': cur_val}
        # next state, rewardm done, info
        return self.get_obs(), reward, action, done, info
    
    def get_obs(self):
        obs = np.empty(self.state_dim)
        # first fill stocks owned
        obs[:self.n_stock] = self.stock_owned
        # next are prices
        obs[self.n_stock:self.n_stock+3] = self.stock_price
        obs[-1] = self.cash_at_hand
        return obs
    
    def get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_at_hand
    
    def trade(self, action):
        action_vect = self.action_list[action]
        
        sell_index= []
        buy_index = []
        for i, a in enumerate(action_vect):
            if a==0:
                sell_index.append(i)
            if a==2:
                buy_index.append(i)
        # first sell all what we want to sell
        if sell_index:
            # Simplification: sell all shares
            for i in sell_index:
                self.cash_at_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            # buy one stock at one time until we run out of cash
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_at_hand > self.stock_price[i]:
                        self.cash_at_hand -= self.stock_price[i]
                        self.stock_owned[i] += 1
                    else:
                        can_buy = False
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        