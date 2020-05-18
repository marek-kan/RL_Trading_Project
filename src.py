
import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

from datetime import datetime
import interpools
import argparse
import rs
import os
import pickle


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_data():
    """
    Returns stock prices of Apple (column 0), Motorola (1), Starbucks (2)
    """
    df = pd.read_csv('data/stock_prices.csv')
    return df.values

def get_scaler(env):
    """
    Plays episodes randomly and stores states for the scaler
    Returns scaler object for the states
    env - enviroment object
    """
    states = []
    for i in range(env.n_step):
        action = np.random.choice(env.action_space)
        s, r, done, info = env.step(action)
        states.append(s)
        if done:
            break
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler

class ReplayBuffer():
    def __init__(self, obs_dim, act_dim, size):
        # states
        self.obs1_buf = np.zeros(shape=(size, obs_dim), dtype=np.float32)
        # next states
        self.obs2_buf = np.zeros(shape=(size, obs_dim), dtype=np.float32)
        # actions, represented by integers
        self.acts_buf = np.zeros(shape=(size), dtype=np.uint8)
        # rewards
        self.rew_buf = np.zeros(shape=(size), dtype=np.float32)
        # done flag, 0/1
        self.done_buf = np.zeros(shape=(size), dtype=np.uint8)
        # pointer - start, current sizem max size
        self.ptr, self.size, self.max_size = 0, 0, size
        
    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        # circular pointer
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s = self.obs1_buf[idxs],
                    s2 = self.obs2_buf[idxs],
                    a = self.acts_buf[idxs],
                    r = self.rew_buf[idxs],
                    done = self.done_buf[idxs]
                )