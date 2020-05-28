# -*- coding: utf-8 -*-
import src
import numpy as np

class Agent():
    def __init__(self, state_size, action_size, batch_size=32):
        self.s_size = state_size
        self.a_size = action_size
        self.batch_size = batch_size
        self.memory = src.ReplayBuffer(state_size, action_size, size=500)
        self.gamma = 0.95 # discount rate 
        self.epsilon = 1.0 # exploration rate, 1 - pure exploration, 0 - deterministic
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = src.create_model(state_size, action_size)

    def update_replay_buffer(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
        
    def act(self, state):
#        uses epsilon-greedy
        if np.random.rand()<=self.epsilon:
            return np.random.choice(self.a_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
        
    def replay(self):
        if self.memory.size < self.batch_size:
            return print('Not enough data in replay buffer!')
        
        batch = self.memory.sample_batch(self.batch_size)
        states = batch['s']
        actions = batch['a']
        rewards = batch['r']
        next_states = batch['s2']
        done = batch['done']
        
        # Calculate the tentative target Q(s', a)
        target = rewards + (1 - done) * self.gamma * np.amax(self.model.predict(next_states), axis=1)
        """
        We need target to be same shape as predictions. However, we only need
        to update DNN for actions which were taken. So set target equal to 
        the predictions for all values. Then only change targets for action
        taken. 
        """
#            Calculate prediction for each state and action
        target_full = self.model.predict(states)
#            We want to change this array for actions where we have actual target
#            Error on  actions which werent taken will be 0
        target_full[np.arange(self.batch_size), actions] = target
        
        self.model.train_on_batch(states, target_full)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
                
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save_weights(name)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            