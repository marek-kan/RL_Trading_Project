# -*- coding: utf-8 -*-
import src

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
