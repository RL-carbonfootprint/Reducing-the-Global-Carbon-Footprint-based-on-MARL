#importing dependencies
import numpy as np
import os, random
import pandas as pd
import matplotlib.pyplot as plt
from RL_function import RL_loop
plt.style.use('seaborn')

def evaluate_RL(condition): 
    if condition == 'certain':
        #extracting results from both simulations
        global_state_per_epoch, immediate_rewards_per_epoch, cumulative_reward_per_epoch, Q_LT, Q_MT, Q_ST, Q_LT_per_epoch, Q_MT_per_epoch, Q_ST_per_epoch, LT_action_space, MT_action_space, ST_action_space, Population_LT, Population_ST, Population_MT, Population_array, Q_LT_average, Q_MT_average, Q_ST_average = RL_loop('certain')
    elif condition == 'uncertain':
        #extracting results from both simulations
        global_state_per_epoch, immediate_rewards_per_epoch, cumulative_reward_per_epoch, Q_LT, Q_MT, Q_ST, Q_LT_per_epoch, Q_MT_per_epoch, Q_ST_per_epoch, LT_action_space, MT_action_space, ST_action_space, Population_LT, Population_ST, Population_MT, Population_array, Q_LT_average, Q_MT_average, Q_ST_average = RL_loop('uncertain')

    #evaluating the trained model
    print('Lowest Global State Achived During this Game: ',np.min(global_state_per_epoch)) #print the lowest global state per epoch. Maybe average over epochs?
    print('\n')
    print("Lowest Immediate Reward for LT: ",np.min(immediate_rewards_per_epoch[:, 0])) #print the learning outcome of 30 years of interacting with the environment
    print("Lowest Immediate Reward for MT: ",np.min(immediate_rewards_per_epoch[:, 1]))
    print("Lowest Immediate Reward for ST: ",np.min(immediate_rewards_per_epoch[:, 2]))

    #print Q-tables
    print('Q_LT: \n')
    print(pd.DataFrame(Q_LT).head())
    print('Q_MT: \n')
    print(pd.DataFrame(Q_MT).head())
    print('Q_ST: \n')
    print(pd.DataFrame(Q_ST).head())

    #Synergistic Policy
    Q_LT_Best = Q_LT_per_epoch[np.argmin(cumulative_reward_per_epoch)] # return the Q-table indices with the lowest cumulative reward per epoch 
    Q_MT_Best = Q_MT_per_epoch[np.argmin(cumulative_reward_per_epoch)]
    Q_ST_Best = Q_ST_per_epoch[np.argmin(cumulative_reward_per_epoch)]


    LT_Synergistic = [LT_action_space[i] for i in Q_LT_Best[1:]] #get all the actions frm the best Q-values regarding cumulative reward (disregarding the first year)
    MT_Synergistic = [MT_action_space[i] for i in Q_MT_Best[1:]]
    ST_Synergistic = [ST_action_space[i] for i in Q_ST_Best[1:]]

    print("LT's Strategy to achieve Lowest Cumulative Reward:\n",LT_Synergistic)
    print("\nMT's Strategy to achieve Lowest Cumulative Reward:\n",MT_Synergistic)
    print("\nST's Strategy to achieve Lowest Cumulative Reward:\n",ST_Synergistic)

    #Selfish Planning Policy
    Q_LT_Immediate_Best = Q_LT_per_epoch[np.argmin(immediate_rewards_per_epoch[:, 0])] # return the Q-table indices with the lowest immediate reward per epoch per agent (LT, MT, ST)
    Q_MT_Immediate_Best = Q_MT_per_epoch[np.argmin(immediate_rewards_per_epoch[:, 1])]
    Q_ST_Immediate_Best = Q_ST_per_epoch[np.argmin(immediate_rewards_per_epoch[:, 2])]

    LT_Selfish_Plan = [LT_action_space[i] for i in Q_LT_Immediate_Best[1:]]
    MT_Selfish_Plan = [MT_action_space[i] for i in Q_MT_Immediate_Best[1:]]
    ST_Selfish_Plan = [ST_action_space[i] for i in Q_ST_Immediate_Best[1:]]

    print("LT's Strategy to achieve Lowest Immediate Reward:\n",LT_Selfish_Plan)
    print("\nMT's Strategy to achieve Lowest Immediate Reward:\n",MT_Selfish_Plan)
    print("\nST's Strategy to achieve Lowest Immediate Reward:\n",ST_Selfish_Plan)

    #Greedy Policy
    LT_Greedy = [LT_action_space[i] for i in np.argmin(Q_LT, axis=1)] #return the action of the indice of the smallest value per year in the Q-table from the last epoch 
    MT_Greedy = [MT_action_space[i] for i in np.argmin(Q_MT, axis=1)]
    ST_Greedy = [ST_action_space[i] for i in np.argmin(Q_ST, axis=1)]
    print("\nGreedy Policy of LT, based on LT's Final Q-Table:\n",LT_Greedy)
    print("\nGreedy Policy of MT, based on MT's Final Q-Table:\n",MT_Greedy)
    print("\nGreedy Policy of ST, based on ST's Final Q-Table:\n",ST_Greedy)

    return Population_LT, Population_ST, Population_MT, Population_array, LT_Greedy,MT_Greedy,ST_Greedy, LT_Selfish_Plan,MT_Selfish_Plan,ST_Selfish_Plan, LT_Synergistic,MT_Synergistic,ST_Synergistic, Q_LT_average, Q_MT_average, Q_ST_average
