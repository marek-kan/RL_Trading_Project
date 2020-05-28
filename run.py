# -*- coding: utf-8 -*-

from src import *
from enviroment import Env
from agent import Agent
import pickle
import time
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

mode = 'train' # either train or test
model_folder = 'models'
rewards_folder = 'rewards'
n_episodes = 1
batch_size = 32
initial_investment = 20e3
window_size = 7

check_dir(model_folder)
check_dir(rewards_folder)

data = get_data()
returns, prices = process_data(data, window=window_size)
n_timesteps, n_stocks = prices.shape
n_train = int(0.7*n_timesteps)

train_prices = prices[:n_train]
train_returns = returns[:n_train]
test_prices = prices[n_train:]
test_returns = returns[n_train:]

env = Env(data=train_returns, prices=train_prices, window=window_size, initial_investment=initial_investment)
state_size = env.state_dim
action_size = len(env.action_space)

agent = Agent(state_size, action_size, batch_size)
scaler = get_scaler(env)

portfolio_values = []

if mode=='test':
    with open(f'{model_folder}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    env = Env(data=test_returns, prices=test_prices, window=window_size, initial_investment=initial_investment)
    agent.epsilon = 0.01
    # agent.epsilon = 1
    agent.load(f'{model_folder}/dqn.h5')
    
    for e in range(n_episodes):
        t0 = time.time()
        val = play_one_episode(scaler, agent, env, mode)
        dt = time.time() - t0
        print(f'Episode {e+1}/{n_episodes}; Episode end value {val:.2f}; duration {dt}s')
        portfolio_values.append(val)
    
if mode=='train':
    val = 0
    e = 0
    while val < 35000:
        t0 = time.time()
        val = play_one_episode(scaler, agent, env, mode)
        dt = time.time() - t0
        print(f'Episode {e+1}/{n_episodes}; Episode end value {val:.2f}; duration {dt}s')
        portfolio_values.append(val)
    agent.save(f'{model_folder}/dqn.h5')
    pickle.dump(scaler, open(f'{model_folder}/scaler.pkl', 'wb'))
        
    np.save(f'{rewards_folder}/{mode}.npy', portfolio_values)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    