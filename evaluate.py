# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:22:48 2020

@author: Marek
"""

import matplotlib.pyplot as plt
import numpy as np

random = np.load('rewards/test_1.npy')
model = np.load('rewards/test_0.01.npy')

plt.hist(random, bins=20, label='Random trading')
# plt.title('Random trading')
# plt.show()
# plt.close()

plt.hist(model, bins=20, label='RL Agent')
# plt.title('RL Agent')
plt.xlabel('Portfolio value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print(f'Random trading avg: {random.mean()}','\n',
      f'RL Agent avg: {model.mean()}')

pooled_var = ((len(model)-1)*model.std()**2 + (len(random)-1)*random.std()**2)/(len(model)+len(random)-2)
standard_error = np.sqrt(pooled_var/len(model) + pooled_var/len(random))
z_score = (model.mean() - random.mean()) / standard_error
print(f'Z Score: {z_score}')


