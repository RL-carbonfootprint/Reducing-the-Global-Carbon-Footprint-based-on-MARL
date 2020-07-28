#importing dependencies
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#loading dataFrames
LT_df = pd.read_csv('./metadata/LT_db.csv', index_col=0)
MT_df = pd.read_csv('./metadata/MT_db.csv', index_col=0)
ST_df = pd.read_csv('./metadata/ST_db.csv', index_col=0)

years_array=np.arange(int(LT_df.columns[0])-1,int(LT_df.columns[-1])+1)
global_trend_Real = [4.025, 4.074, 4.124, 4.152, 4.227, 4.224, 4.194, 4.173, 4.068, 4.002,4.011, 4.036, 4.071, 4.082, 4.05, 3.968, 4.038, 4.081, 4.088, 4.258, 4.414,  4.528, 4.636, 4.671, 4.762, 4.662, 4.835, 4.975, 5.005, 4.998, 4.981]
years = len(LT_df.columns)
#initializing all global variables
#initializing variable that counts the state periods
epochs = 32
number_of_agents = 3

#for graphical purposes; showing the action space in the first row of the Q-table
action_count = ("     -0.2 -0.16 -0.12 -0.08 -0.04  0    0.04  0.08  0.12  0.16  0.2") 

global_state = 4.97 #current CO2 emissions are at 4.97 metric tons per capita world-wide. Source: World Bank

#creating a list to store the global state for each epoch
global_state_per_epoch = []

#number of different actions that a single agent can take in a given state
size_of_action_space = 10
cost_of_action = 10 #defining cost to reduce CO2 emissions per metric ton

#Q-tables
Q_LT = np.zeros((years + 1, size_of_action_space))
Q_MT = np.zeros((years + 1, size_of_action_space))
Q_ST = np.zeros((years + 1, size_of_action_space))

#Q-tables for every epoch that stores all max values from
Q_LT_per_epoch = []
Q_MT_per_epoch = []
Q_ST_per_epoch = []

#defining the weight factors of immediate rewards 
LT_reward_factor = 0.4
MT_reward_factor = 0.5
ST_reward_factor = 0.6 
cumulative_reward = 0 #initializing cumulative reward, which is 0 to start with

#creating a list to store the cumulative reward for each epoch
cumulative_reward_per_epoch = []

#creating a list to store the immediate rewards for each epoch
immediate_rewards_per_epoch = []

LT_epsilon_min = 0.1 #defining minimal epsilon for LT
LT_epsilon_decay = 0.999 #defining decay rate of LT's epsilon
MT_epsilon_min = 0.06
MT_epsilon_decay = 0.995
ST_epsilon_min = 0.03 
ST_epsilon_decay = 0.990 
alpha = 0.1 #initializing the learning rate of the Q-values
alpha_min = 0.01 #initializing minimal learning rate after decay
alpha_decay = 0.999 #initializing decay of learning rate
gamma = 0.7  #<=>reward discount
LT_values = LT_df.values[:-1].T
Population_LT = LT_df.values[-1]

LT_ravel = LT_values.ravel()
LT_avg = sum(LT_ravel) / len(LT_ravel)

LT_action_space = np.round(
    np.concatenate(
        (np.linspace(min(LT_ravel), LT_avg, size_of_action_space // 2),
         np.linspace(LT_avg, max(LT_ravel),
                     size_of_action_space - (size_of_action_space // 2)))), 3)
MT_values = MT_df.values[:-1].T
Population_MT = MT_df.values[-1]

MT_ravel = MT_values.ravel()
MT_avg = sum(MT_ravel) / len(MT_ravel)

MT_action_space = np.round(
    np.concatenate(
        (np.linspace(min(MT_ravel), MT_avg, size_of_action_space // 2),
         np.linspace(MT_avg, max(MT_ravel),
                     size_of_action_space - (size_of_action_space // 2)))), 3)

ST_values = ST_df.values[:-1].T
Population_ST = ST_df.values[-1]
Population_array = Population_LT + Population_MT + Population_ST

ST_ravel = ST_values.ravel()
ST_avg = sum(ST_ravel) / len(ST_ravel)

ST_action_space = np.round(
    np.concatenate(
        (np.linspace(min(ST_ravel), ST_avg, size_of_action_space // 2),
         np.linspace(ST_avg, max(ST_ravel),
                     size_of_action_space - (size_of_action_space // 2)))), 3)

#defining best actions from action space
Min_Q_LT = 0
Min_Q_MT = 0
Min_Q_ST = 0

for epoch in range(1, epochs + 1):

    #initalize variables
    global_state_of_co2_emission = 4.025  #CO2 emmision in 1984

    cumulative_reward = 0  #intially 0

    alpha = 0.1  #learning rate
    
    #exploration vs explotation
    LT_epsilon = 0.9
    MT_epsilon = 0.8
    ST_epsilon = 0.7

    for year in range(0, years):

        Min_Q_LT = np.argmin(Q_LT[year])
        Min_Q_MT = np.argmin(Q_MT[year])
        Min_Q_ST = np.argmin(Q_ST[year])

        #LT
        if np.random.rand() <= LT_epsilon:
            LT_action = random.choice(LT_values[year])
        else:
            LT_action = LT_action_space[Min_Q_LT]

        LT_immediate_reward = LT_action * (LT_reward_factor + cost_of_action)
        
        Q_LT[year, abs(LT_action_space - LT_action).argmin()] = round(
            (1 - alpha) * Q_LT[year, Min_Q_LT] + alpha *
            (LT_immediate_reward + gamma * np.amin(Q_LT[year + 1, :])), 3)

        #MT
        if np.random.rand() <= MT_epsilon:
            MT_action = random.choice(MT_values[year])
        else:
            MT_action = MT_action_space[Min_Q_MT]

        MT_immediate_reward = MT_action * (MT_reward_factor + cost_of_action)
        
        Q_MT[year, abs(MT_action_space - MT_action).argmin()] = round(
            (1 - alpha) * Q_MT[year, Min_Q_MT] + alpha *
            (MT_immediate_reward + gamma * np.amin(Q_MT[year + 1, :])), 3)

        #ST
        if np.random.rand() <= ST_epsilon:
            ST_action = random.choice(ST_values[year])
        else:
            ST_action = ST_action_space[Min_Q_ST]

        ST_immediate_reward = ST_action * (ST_reward_factor + cost_of_action)
        
        Q_ST[year, abs(ST_action_space - ST_action).argmin()] = round(
            (1 - alpha) * Q_ST[year, Min_Q_ST] + alpha *
            (ST_immediate_reward + gamma * np.amin(Q_ST[year + 1, :])), 3)

        cumulative_reward += LT_immediate_reward + MT_immediate_reward + ST_immediate_reward - cost_of_action * (
            LT_action_space[np.argmin(Q_LT[year, :])] -
            MT_action_space[np.argmin(Q_MT[year, :])] -
            ST_action_space[np.argmin(Q_ST[year, :])])

        alpha = alpha * alpha_decay if (alpha > alpha_min) else alpha
        
        LT_epsilon = LT_epsilon * LT_epsilon_decay if (
            LT_epsilon > LT_epsilon_min) else LT_epsilon_min
        
        MT_epsilon = MT_epsilon * MT_epsilon_decay if (
            MT_epsilon > MT_epsilon_min) else MT_epsilon_min
        
        ST_epsilon = ST_epsilon * ST_epsilon_decay if (
            ST_epsilon > ST_epsilon_min) else ST_epsilon_min

        global_state_of_co2_emission += (
            (LT_action * Population_LT[year]) +
            (MT_action * Population_MT[year]) +
            (ST_action * Population_ST[year])) / Population_array[year]

    Q_LT_per_epoch.append(np.argmin(Q_LT, axis=1).tolist())
    Q_MT_per_epoch.append(np.argmin(Q_MT, axis=1).tolist())
    Q_ST_per_epoch.append(np.argmin(Q_ST, axis=1).tolist())

    cumulative_reward_per_epoch.append(cumulative_reward)
    immediate_rewards_per_epoch.append(
        [LT_immediate_reward, MT_immediate_reward, ST_immediate_reward])
    global_state_per_epoch.append(global_state_of_co2_emission)
    
immediate_rewards_per_epoch = np.array(immediate_rewards_per_epoch, copy=False)
#evaluating the trained model
print('Lowest Global State Achived During this Game: ',np.min(global_state_per_epoch))
print('\n')
print("Lowest Immediate Reward for LT: ",np.min(immediate_rewards_per_epoch[:, 0]))
print("Lowest Immediate Reward for MT: ",np.min(immediate_rewards_per_epoch[:, 1]))
print("Lowest Immediate Reward for ST: ",np.min(immediate_rewards_per_epoch[:, 2]))
print('Q_LT: \n')
print(pd.DataFrame(Q_LT).head())
print('Q_MT: \n')
print(pd.DataFrame(Q_MT).head())
print('Q_ST: \n')
print(pd.DataFrame(Q_ST).head())

#Synergistic Policy
Q_LT_Best = Q_LT_per_epoch[np.argmin(cumulative_reward_per_epoch)]
Q_MT_Best = Q_MT_per_epoch[np.argmin(cumulative_reward_per_epoch)]
Q_ST_Best = Q_ST_per_epoch[np.argmin(cumulative_reward_per_epoch)]

LT_Synergistic = [LT_action_space[i] for i in Q_LT_Best[1:]]
MT_Synergistic = [MT_action_space[i] for i in Q_MT_Best[1:]]
ST_Synergistic = [ST_action_space[i] for i in Q_ST_Best[1:]]

print("LT's Strategy to achieve Lowest Cumulative Reward:\n",LT_Synergistic)
print("\nMT's Strategy to achieve Lowest Cumulative Reward:\n",MT_Synergistic)
print("\nST's Strategy to achieve Lowest Cumulative Reward:\n",ST_Synergistic)

#Selfish Planning Policy
Q_LT_Immediate_Best = Q_LT_per_epoch[np.argmin(immediate_rewards_per_epoch[:, 0])]
Q_MT_Immediate_Best = Q_MT_per_epoch[np.argmin(immediate_rewards_per_epoch[:, 1])]
Q_ST_Immediate_Best = Q_ST_per_epoch[np.argmin(immediate_rewards_per_epoch[:, 2])]


LT_Selfish_Plan = [LT_action_space[i] for i in Q_LT_Immediate_Best[1:]]
MT_Selfish_Plan = [MT_action_space[i] for i in Q_MT_Immediate_Best[1:]]
ST_Selfish_Plan = [ST_action_space[i] for i in Q_ST_Immediate_Best[1:]]

print("LT's Strategy to achieve Lowest Immediate Reward:\n",LT_Selfish_Plan)
print("\nMT's Strategy to achieve Lowest Immediate Reward:\n",MT_Selfish_Plan)
print("\nST's Strategy to achieve Lowest Immediate Reward:\n",ST_Selfish_Plan)

#Greedy Policy
LT_Greedy = [LT_action_space[i] for i in np.argmin(Q_LT, axis=1)]
MT_Greedy = [MT_action_space[i] for i in np.argmin(Q_MT, axis=1)]
ST_Greedy = [ST_action_space[i] for i in np.argmin(Q_ST, axis=1)]

print("\nGreedy Policy of LT, based on LT's Final Q-Table:\n",LT_Greedy)
print("\nGreedy Policy of MT, based on MT's Final Q-Table:\n",MT_Greedy)
print("\nGreedy Policy of ST, based on ST's Final Q-Table:\n",ST_Greedy)

def get_trend(Q_LT,Q_MT,Q_ST,):
    global_state_of_co2_emission = 4.025
    global_trend = [4.025]
    for year in range(0, years):
        #actions based on Policies
        LT_action = Q_LT[year]
        MT_action = Q_MT[year]
        ST_action = Q_ST[year]

        global_state_of_co2_emission += (
            (LT_action * Population_LT[year]) + (MT_action * Population_MT[year]) +
            (ST_action * Population_ST[year])) / Population_array[year]

        global_trend.append(np.round(global_state_of_co2_emission,3))
    return global_trend

#ALL Policies
plt.figure(figsize=(10,8))
plt.plot(years_array,global_trend_Greedy)
plt.scatter(years_array,global_trend_Greedy)
plt.plot(years_array,global_trend_Selfish)
plt.scatter(years_array,global_trend_Selfish)
plt.plot(years_array,global_trend_Synergistic)
plt.scatter(years_array,global_trend_Synergistic)
plt.plot(years_array,global_trend_Real)
plt.scatter(years_array,global_trend_Real)
plt.xlabel('Years')
plt.ylabel('CO2 Emission')
plt.legend(['Greedy','Selfish','Synergistic','Real'])
plt.show()