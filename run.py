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
n_episodes = 5
batch_size = 32
initial_investment = 20e3

check_dir(model_folder)
check_dir(rewards_folder)

data = get_data()
n_timesteps, n_stocks = data.shape
n_train = n_timesteps//2

train = data[:n_train]
test = data[n_train:]

env = Env(train, initial_investment)
state_size = env.state_dim
action_size = len(env.action_space)

agent = Agent(state_size, action_size, batch_size)
scaler = get_scaler(env)

portfolio_values = []

if mode=='test':
    with open(f'{model_folder}/scaler.pkl') as f:
        scaler = pickle.load(f)
    
    env = Env(test_data, initial_investment)
    agent.epsilon = 0.01
    agent.load(f'{model_folder}/dqn.h5')
    
for e in range(n_episodes):
    t0 = time.time()
    val = play_one_episode(scaler, agent, env, mode)
    dt = time.time() - t0
    print(f'Episode {e+1}/{n_episodes}; Episode end value {val:.2f}; duration {dt}s')
    portfolio_values.append(val)
    
if mode=='train':
    agent.save(f'{model_folder}/dqn.h5')
    pickle.dump(scaler, open(f'{model_folder}/scaler.pkl', 'wb'))
        
np.save(f'{rewards_folder}/{mode}.npy', portfolio_values)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    