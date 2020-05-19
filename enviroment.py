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
        self.action_space = np.arrange(3**self.n_stock)
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
        return self._get_obs()