#importing dependencies
import numpy as np
import os, random
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def rand_skew_norm(fAlpha, fLocation, fScale):
    sigma = fAlpha / np.sqrt(1.0 + fAlpha**2) 

    afRN = np.random.randn(2)
    u0 = afRN[0]
    v = afRN[1]
    u1 = sigma*u0 + np.sqrt(1.0 -sigma**2) * v 

    if u0 >= 0:
        return u1*fScale + fLocation 
    return (-u1)*fScale + fLocation 

def randn_skew(N, skew):
    return [rand_skew_norm(4, skew, 0.025) for x in range(N)]


def RL_loop(condition):
    #random.seed(10)

    #loading dataFrames of empirical data for all years for all countries
    LT_df = pd.read_csv(os.path.join('Reducing-the-Global-Carbon-Footprint-based-on-MARL', 'metadata','LT_db.csv'), index_col=0)
    MT_df = pd.read_csv(os.path.join('Reducing-the-Global-Carbon-Footprint-based-on-MARL','metadata','MT_db.csv'), index_col=0)
    ST_df = pd.read_csv(os.path.join('Reducing-the-Global-Carbon-Footprint-based-on-MARL','metadata','ST_db.csv'), index_col=0)

    #initializing all global variables
    years = len(LT_df.columns)
    epochs = 40
    Q_LT_average, Q_MT_average, Q_ST_average = [], [], []

    #creating a list to store the global state for each epoch
    global_state_per_epoch = []

    #number of different actions that a single agent can take in a given state
    size_of_action_space = 10
    cost_of_action = 0.1 #defining cost to reduce CO2 emissions per metric ton

    #Q-tables
    Q_LT = np.zeros((years + 1, size_of_action_space))
    Q_MT = np.zeros((years + 1, size_of_action_space))
    Q_ST = np.zeros((years + 1, size_of_action_space))

    #Q-tables for every epoch that stores all max values from
    Q_LT_per_epoch = []
    Q_MT_per_epoch = []
    Q_ST_per_epoch = []

    #creating a list to store the cumulative reward for each epoch
    cumulative_reward_per_epoch = []

    #creating a list to store the immediate rewards for each epoch
    immediate_rewards_per_epoch = []

    #if condition == 'certain':
        #defining the weight factors of immediate rewards 
    LT_reward_factor = 0.7
    MT_reward_factor = 0.8
    ST_reward_factor = 0.9 

    LT_epsilon_min = 0.1 #defining minimal epsilon for LT
    LT_epsilon_decay = 0.999 #defining decay rate of LT's epsilon
    MT_epsilon_min = 0.06
    MT_epsilon_decay = 0.995
    ST_epsilon_min = 0.03 
    ST_epsilon_decay = 0.990 
    alpha_min = 0.01 #initializing minimal learning rate after decay
    alpha_decay = 0.999 #initializing decay of learning rate
    gamma = 0.7  #<=>reward discount


    #defining action space for LT
    LT_values = LT_df.values[:-1].T #chose all rows with co2 emission (for all years and countries)
    Population_LT = LT_df.values[-1] #save summarised population

    LT_ravel = LT_values.ravel() #flatten array
    LT_avg = sum(LT_ravel) / len(LT_ravel) #how much they emit on average in a year (1985-2014)

    LT_action_space = np.round(
        np.concatenate(
            (np.linspace(min(LT_ravel), LT_avg, size_of_action_space // 2), #generate 5 evenly spaced samples between minimum and average co2 emission floor
            np.linspace(LT_avg, max(LT_ravel), #generate 5 evenly spaced sample between average and maximum
                        size_of_action_space - (size_of_action_space // 2)))), 3) #round the integer to 3 decimals


    #define action space for MT
    MT_values = MT_df.values[:-1].T
    Population_MT = MT_df.values[-1]

    MT_ravel = MT_values.ravel()
    MT_avg = sum(MT_ravel) / len(MT_ravel)

    MT_action_space = np.round(
        np.concatenate(
            (np.linspace(min(MT_ravel), MT_avg, size_of_action_space // 2),
            np.linspace(MT_avg, max(MT_ravel),
                        size_of_action_space - (size_of_action_space // 2)))), 3)

    #Define action space for ST
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
        #sigma = 0.25
        
        #exploration vs explotation
        LT_epsilon = 0.9
        MT_epsilon = 0.8
        ST_epsilon = 0.7

        for year in range(0, years):
            #if condition == 'uncertain':
                #defining the weight factors of immediate rewards 
                #LT_reward_factor = np.random.normal(0.7, sigma, 1)[0] 
                #MT_reward_factor = np.random.normal(0.8, sigma, 1)[0] 
                #ST_reward_factor = np.random.normal(0.9, sigma, 1)[0]  
                #LT_reward_factor = np.random.choice(randn_skew(100, 0.7), size=1)[0]
                #MT_reward_factor = np.random.choice(randn_skew(100, 0.8), size=1)[0]
                #ST_reward_factor = np.random.choice(randn_skew(100, 0.9), size=1)[0]
                
            Min_Q_LT = np.argmin(Q_LT[year]) #find the position of the lowest Q-value for a given year
            Min_Q_MT = np.argmin(Q_MT[year])
            Min_Q_ST = np.argmin(Q_ST[year])

            #Define LT-action
            if np.random.rand() <= LT_epsilon: #the smaller epsilon gets, the less likely the agent is to take a random action form the action space 
                LT_action = random.choice(LT_values[year])
            else:
                LT_action = LT_action_space[Min_Q_LT] #if epsilon is smaller than random value, then choose the best action from the Q-table (the greedy choice)

            #calculate immediate consequences of your actions (we want the smallest possible reward, i.e. co2 emission)
            LT_immediate_reward = LT_action * (LT_reward_factor - cost_of_action)
            #LT_immediate_reward = np.random.normal(LT_immediate_reward, sigma, 1)[0]
            if condition == 'uncertain':
                LT_immediate_reward = np.random.choice(randn_skew(100, LT_immediate_reward), size=1)[0]
            
            #calculate Q-value
            Q_LT[year, abs(LT_action_space - LT_action).argmin()] = round( #in the Q-table in the position of the [year, the position of the selected action in the action space]
                (1 - alpha) * Q_LT[year, Min_Q_LT] + alpha *
                (LT_immediate_reward + gamma * np.amin(Q_LT[year + 1, :])), 3)

            #Define MT action and Q-value
            if np.random.rand() <= MT_epsilon:
                MT_action = random.choice(MT_values[year])
            else:
                MT_action = MT_action_space[Min_Q_MT]

            MT_immediate_reward = MT_action * (MT_reward_factor - cost_of_action)
            #MT_immediate_reward = np.random.normal(MT_immediate_reward, sigma, 1)[0]
            if condition == 'uncertain':
                MT_immediate_reward = np.random.choice(randn_skew(100, MT_immediate_reward), size=1)[0]
            
            Q_MT[year, abs(MT_action_space - MT_action).argmin()] = round(
                (1 - alpha) * Q_MT[year, Min_Q_MT] + alpha *
                (MT_immediate_reward + gamma * np.amin(Q_MT[year + 1, :])), 3)

            #Define ST action and Q-value
            if np.random.rand() <= ST_epsilon:
                ST_action = random.choice(ST_values[year])
            else:
                ST_action = ST_action_space[Min_Q_ST]

            ST_immediate_reward = ST_action * (ST_reward_factor - cost_of_action)
            #ST_immediate_reward = np.random.normal(ST_immediate_reward, sigma, 1)[0]
            if condition == 'uncertain':
                ST_immediate_reward = np.random.choice(randn_skew(100, ST_immediate_reward), size=1)[0]

            Q_ST[year, abs(ST_action_space - ST_action).argmin()] = round(
                (1 - alpha) * Q_ST[year, Min_Q_ST] + alpha *
                (ST_immediate_reward + gamma * np.amin(Q_ST[year + 1, :])), 3)


            #calculate cumulative reward (the lowest cumulative reward is the best, i.e. lowest co2 emission)
            cumulative_reward += LT_immediate_reward + MT_immediate_reward + ST_immediate_reward - cost_of_action * (
                LT_action_space[np.argmin(Q_LT[year, :])] -
                MT_action_space[np.argmin(Q_MT[year, :])] -
                ST_action_space[np.argmin(Q_ST[year, :])]) #remove the cost of an action from the imeediate rewards to get the updated status of the earth and the environment

            #update alpha to decay over time untill alpha min
            alpha = alpha * alpha_decay if (alpha > alpha_min) else alpha
            
            #update epsilon to decay over time untill epsilon min for all agents
            LT_epsilon = LT_epsilon * LT_epsilon_decay if (
                LT_epsilon > LT_epsilon_min) else LT_epsilon_min
            
            MT_epsilon = MT_epsilon * MT_epsilon_decay if (
                MT_epsilon > MT_epsilon_min) else MT_epsilon_min
            
            ST_epsilon = ST_epsilon * ST_epsilon_decay if (
                ST_epsilon > ST_epsilon_min) else ST_epsilon_min

            #Calculate global state of C02 emission , i.e. co2 emission per capita times the populatoin per agent group (LT, MT, ST) divided by overall world population
            global_state_of_co2_emission += (
                (LT_action * Population_LT[year]) +
                (MT_action * Population_MT[year]) +
                (ST_action * Population_ST[year])) / Population_array[year]


        #append the best Q-value for action per year (per epoch)
        Q_LT_per_epoch.append(np.argmin(Q_LT, axis=1).tolist()) #return indices for the lowest Q-value each year. 
        Q_MT_per_epoch.append(np.argmin(Q_MT, axis=1).tolist())
        Q_ST_per_epoch.append(np.argmin(Q_ST, axis=1).tolist())

        #calculate average Q-value for all states and actions
        Q_LT_average.append(np.mean(Q_LT))
        Q_MT_average.append(np.mean(Q_MT))
        Q_ST_average.append(np.mean(Q_ST))

        #save cumulative reward, immediate reward and global state per epoch
        cumulative_reward_per_epoch.append(cumulative_reward)
        immediate_rewards_per_epoch.append(
            [LT_immediate_reward, MT_immediate_reward, ST_immediate_reward])
        global_state_per_epoch.append(global_state_of_co2_emission)
        

    #convert to array
    immediate_rewards_per_epoch = np.array(immediate_rewards_per_epoch, copy=False)

    return global_state_per_epoch, immediate_rewards_per_epoch, cumulative_reward_per_epoch, Q_LT, Q_MT, Q_ST, Q_LT_per_epoch, Q_MT_per_epoch, Q_ST_per_epoch, LT_action_space, MT_action_space, ST_action_space, Population_LT, Population_ST, Population_MT, Population_array, Q_LT_average, Q_MT_average, Q_ST_average
