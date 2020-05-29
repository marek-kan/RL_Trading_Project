# Reinforcement Learning Trading Project
In this repositary I will explore usage of Reinforcement Learning (Q-Learning) to Stock Market. I use couple of simplifications:
  * Ignore trading fees
  * If agent wants to sell -> sell all shares of given stock
  * If agent wants to buy -> buy as many as possible (in case of multiple buy signals it buys in rounds)
  * Sell before buy
So any action follows specified order.

State is represented by past stock returns, I used 7 days window, number of shares we own, and how much cash we have at hand. Model used is simple regression neural network. I want to explore usage of RNNs in future. 

## Results


## Description
 * run.py - manages training and testing
 * evaluate.py - draws histograms (random vs agent) and computes statistical test
 * agent.py - contains Agent class
 * enviroment.py - contains Enviroment class
 * src.py - contains Replay Buffer class and some general functions
 * data folder - contains two datasets:
   * stock_prices.csv - stocks for apple, starbucks and motorola; small developing dataset
   * sugar_corn_gold.csv - data from 1.1.2000 to 29.5.2020, dataset on which I performed reported test
