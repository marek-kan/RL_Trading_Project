# -*- coding: utf-8 -*-
import src
import numpy as np

class Agent():
    def __init__(self, state_size, action_size):
        self.s_size = state_size
        self.a_size = action_size
        self.memory = src.ReplayBuffer(state_size, action_size, size=500)
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = src.create_model(state_size, action_size)

    def update_replay_buffer(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
        
    def act(self, state):
#        uses epsilon-greedy
        if np.random.rand()<=self.epsilon:
            return np.random.choice(self.a_size)
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
        
    