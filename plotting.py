#importing dependencies
import numpy as np
import os, random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from evaluate import evaluate_RL
from scipy import stats
plt.style.use('seaborn')


#evaluate RL across several runs with 40 learning epochs
test_runs = 1000
LT_Greedy_list,MT_Greedy_list,ST_Greedy_list, LT_Selfish_Plan_list,MT_Selfish_Plan_list,ST_Selfish_Plan_list, LT_Synergistic_list,MT_Synergistic_list,ST_Synergistic_list, Q_LT_average_list, Q_MT_average_list, Q_ST_average_list = [],[],[],[],[],[],[],[],[],[],[],[]
LT_Greedy_u_list,MT_Greedy_u_list,ST_Greedy_u_list, LT_Selfish_Plan_u_list,MT_Selfish_Plan_u_list,ST_Selfish_Plan_u_list, LT_Synergistic_u_list,MT_Synergistic_u_list,ST_Synergistic_u_list, Q_LT_average_u_list, Q_MT_average_u_list, Q_ST_average_u_list = [],[],[],[],[],[],[],[],[],[],[],[]


for i in range(0, test_runs): 
    print(i)
    #save certain runs
    _, _, _, _, LT_Greedy, MT_Greedy,ST_Greedy, LT_Selfish_Plan,MT_Selfish_Plan,ST_Selfish_Plan, LT_Synergistic,MT_Synergistic,ST_Synergistic, Q_LT_average, Q_MT_average, Q_ST_average = evaluate_RL('certain')
    LT_Greedy_list.append(LT_Greedy)
    MT_Greedy_list.append(MT_Greedy)
    ST_Greedy_list.append(ST_Greedy)
    LT_Selfish_Plan_list.append(LT_Selfish_Plan)
    MT_Selfish_Plan_list.append(MT_Selfish_Plan)
    ST_Selfish_Plan_list.append(ST_Selfish_Plan)
    LT_Synergistic_list.append(LT_Synergistic)
    MT_Synergistic_list.append(MT_Synergistic)
    ST_Synergistic_list.append(ST_Synergistic)
    Q_LT_average_list.append(Q_LT_average)
    Q_MT_average_list.append(Q_MT_average)
    Q_ST_average_list.append(Q_ST_average)
    
    #save uncertain runs
    _, _, _, _, LT_Greedy, MT_Greedy,ST_Greedy, LT_Selfish_Plan,MT_Selfish_Plan,ST_Selfish_Plan, LT_Synergistic,MT_Synergistic,ST_Synergistic, Q_LT_average, Q_MT_average, Q_ST_average = evaluate_RL('uncertain')
    LT_Greedy_u_list.append(LT_Greedy)
    MT_Greedy_u_list.append(MT_Greedy)
    ST_Greedy_u_list.append(ST_Greedy)
    LT_Selfish_Plan_u_list.append(LT_Selfish_Plan)
    MT_Selfish_Plan_u_list.append(MT_Selfish_Plan)
    ST_Selfish_Plan_u_list.append(ST_Selfish_Plan)
    LT_Synergistic_u_list.append(LT_Synergistic)
    MT_Synergistic_u_list.append(MT_Synergistic)
    ST_Synergistic_u_list.append(ST_Synergistic)
    Q_LT_average_u_list.append(Q_LT_average)
    Q_MT_average_u_list.append(Q_MT_average)
    Q_ST_average_u_list.append(Q_ST_average)
   
    
#average Q-values 
Q_LT_average_mean = np.average(Q_LT_average_list, axis=0)
Q_MT_average_mean = np.average(Q_MT_average_list, axis=0)
Q_ST_average_mean = np.average(Q_ST_average_list, axis=0)

Q_LT_average_u_mean = np.average(Q_LT_average_u_list, axis=0)
Q_MT_average_u_mean = np.average(Q_MT_average_u_list, axis=0)
Q_ST_average_u_mean = np.average(Q_ST_average_u_list, axis=0)

#average policies per agent across test runs
LT_Greedy_mean = np.average(LT_Greedy_list, axis=0)
MT_Greedy_mean = np.average(MT_Greedy_list, axis=0)
ST_Greedy_mean = np.average(ST_Greedy_list, axis=0)
LT_Selfish_Plan_mean = np.average(LT_Selfish_Plan_list, axis=0)
MT_Selfish_Plan_mean = np.average(MT_Selfish_Plan_list, axis=0)
ST_Selfish_Plan_mean = np.average(ST_Selfish_Plan_list, axis=0)
LT_Synergistic_mean = np.average(LT_Synergistic_list, axis=0)
MT_Synergistic_mean = np.average(MT_Synergistic_list, axis=0)
ST_Synergistic_mean = np.average(ST_Synergistic_list, axis=0)

LT_Greedy_u_mean = np.average(LT_Greedy_u_list, axis=0)
MT_Greedy_u_mean = np.average(MT_Greedy_u_list, axis=0)
ST_Greedy_u_mean = np.average(ST_Greedy_u_list, axis=0)
LT_Selfish_Plan_u_mean = np.average(LT_Selfish_Plan_u_list, axis=0)
MT_Selfish_Plan_u_mean = np.average(MT_Selfish_Plan_u_list, axis=0)
ST_Selfish_Plan_u_mean = np.average(ST_Selfish_Plan_u_list, axis=0)
LT_Synergistic_u_mean = np.average(LT_Synergistic_u_list, axis=0)
MT_Synergistic_u_mean = np.average(MT_Synergistic_u_list, axis=0)
ST_Synergistic_u_mean = np.average(ST_Synergistic_u_list, axis=0)

#calculate sd per policies per agent across test runs
LT_Greedy_std = np.std(LT_Greedy_list, axis=0)
MT_Greedy_std = np.std(MT_Greedy_list, axis=0)
ST_Greedy_std = np.std(ST_Greedy_list, axis=0)

#calculate std for policies
#greedy = np.array([LT_Greedy_list[0][0:len(LT_Greedy_list)],MT_Greedy_list[0][0:len(MT_Greedy_list)], ST_Greedy_list[0][0:len(ST_Greedy_list)]])
#print("greeeeeeeeeeeeeeeed", greedy)
#global_trend_Greedy_std = np.std(greedy, axis=0)
#print("greedy certain std:", global_trend_Greedy_std)
#u_greedy = np.array([LT_Greedy_u_list,MT_Greedy_u_list, ST_Greedy_u_list])
#global_trend_Greedy_u_std = np.std(u_greedy, axis=0)
#print("greedy uncertain std:", global_trend_Greedy_u_std)

#selfish = np.array([LT_Selfish_Plan_list,MT_Selfish_Plan_list, ST_Selfish_Plan_list])
#global_trend_Selfish_std = np.std(selfish, axis=0)
#print("selfish certain std:", global_trend_Selfish_std)
#u_selfish = np.array([LT_Selfish_Plan_u_list,MT_Selfish_Plan_u_list, ST_Selfish_Plan_u_list])
#global_trend_Selfish_u_std = np.std(u_selfish, axis=0)
#print("selfish uncertain std:", global_trend_Selfish_u_std)

#util = np.array([LT_Synergistic_list,MT_Synergistic_list, ST_Synergistic_list])
#global_trend_Synergistic_std = np.std(util, axis=0)
#print("util certain std:", global_trend_Synergistic_std)
#u_util = np.array([LT_Synergistic_u_list,MT_Synergistic_u_list, ST_Synergistic_u_list])
#global_trend_Synergistic_u_std = np.std(u_util, axis=0)
#print("util uncertain std:", global_trend_Synergistic_u_std)


LT_Selfish_Plan_std = np.std(LT_Selfish_Plan_list, axis=0)
MT_Selfish_Plan_std = np.std(MT_Selfish_Plan_list, axis=0)
ST_Selfish_Plan_std = np.std(ST_Selfish_Plan_list, axis=0)
LT_Synergistic_std = np.std(LT_Synergistic_list, axis=0)
MT_Synergistic_std = np.std(MT_Synergistic_list, axis=0)
ST_Synergistic_std = np.std(ST_Synergistic_list, axis=0)

LT_Greedy_u_std = np.std(LT_Greedy_u_list, axis=0)
MT_Greedy_u_std = np.std(MT_Greedy_u_list, axis=0)
ST_Greedy_u_std = np.std(ST_Greedy_u_list, axis=0)
LT_Selfish_Plan_u_std = np.std(LT_Selfish_Plan_u_list, axis=0)
MT_Selfish_Plan_u_std = np.std(MT_Selfish_Plan_u_list, axis=0)
ST_Selfish_Plan_u_std = np.std(ST_Selfish_Plan_u_list, axis=0)
LT_Synergistic_u_std = np.std(LT_Synergistic_u_list, axis=0)
MT_Synergistic_u_std = np.std(MT_Synergistic_u_list, axis=0)
ST_Synergistic_u_std = np.std(ST_Synergistic_u_list, axis=0)

#average std across agents
data = np.array([LT_Greedy_std, MT_Greedy_std, ST_Greedy_std])
global_trend_Greedy_std = np.average(data, axis=0)
data = np.array([LT_Selfish_Plan_std,MT_Selfish_Plan_std,ST_Selfish_Plan_std])
global_trend_Selfish_std = np.average(data, axis=0)
data = np.array([LT_Synergistic_std,MT_Synergistic_std,ST_Synergistic_std])
global_trend_Synergistic_std = np.average(data, axis=0)

data = np.array([LT_Greedy_u_std, MT_Greedy_u_std, ST_Greedy_u_std])
global_trend_Greedy_u_std = np.average(data, axis=0)
data = np.array([LT_Selfish_Plan_u_std,MT_Selfish_Plan_u_std,ST_Selfish_Plan_u_std])
global_trend_Selfish_u_std = np.average(data, axis=0)
data = np.array([LT_Synergistic_u_std,MT_Synergistic_u_std,ST_Synergistic_u_std,])
global_trend_Synergistic_u_std = np.average(data, axis=0)


#Uncertainty: find sd between policies across agents and runs
Greedy_u_data = np.array([LT_Greedy_u_mean, ST_Greedy_u_mean, MT_Greedy_u_mean])
Selfish_u_data = np.array([LT_Selfish_Plan_u_mean, ST_Selfish_Plan_u_mean, MT_Selfish_Plan_u_mean])
Synergistic_u_data = np.array([LT_Synergistic_u_mean, ST_Synergistic_u_mean, MT_Synergistic_u_mean])

##calculate sd
all_policies_uncertainty_std = np.std([np.average(Greedy_u_data, axis = 0), np.insert(np.average(Selfish_u_data, axis = 0), 0, 0), np.insert(np.average(Synergistic_u_data, axis = 0), 0, 0)], axis = 0)
print("all_policies_uncertainty_std", all_policies_uncertainty_std)

#Certainty: find sd between policies across agents and runs
Greedy_data = np.array([LT_Greedy_mean, ST_Greedy_mean, MT_Greedy_mean])
Selfish_data = np.array([LT_Selfish_Plan_mean, ST_Selfish_Plan_mean, MT_Selfish_Plan_mean])
Synergistic_data = np.array([LT_Synergistic_mean, ST_Synergistic_mean, MT_Synergistic_mean])

##calculate sd
all_policies_certainty_std = np.std([np.average(Greedy_data, axis = 0), np.insert(np.average(Selfish_data, axis = 0), 0, 0), np.insert(np.average(Synergistic_data, axis = 0), 0, 0)], axis = 0)
print("all_policies_certainty_std", all_policies_certainty_std)


#save population arrays
Population_LT, Population_ST, Population_MT, Population_array, _,_,_, _,_,_, _,_,_, _,_,_, = evaluate_RL('uncertain')

#loading dataFrames of empirical data for all years for all countries
LT_df = pd.read_csv(os.path.join('Reducing-the-Global-Carbon-Footprint-based-on-MARL', 'metadata','LT_db.csv'), index_col=0)
MT_df = pd.read_csv(os.path.join('Reducing-the-Global-Carbon-Footprint-based-on-MARL','metadata','MT_db.csv'), index_col=0)
ST_df = pd.read_csv(os.path.join('Reducing-the-Global-Carbon-Footprint-based-on-MARL','metadata','ST_db.csv'), index_col=0)

years_array=np.arange(int(LT_df.columns[0])-1,int(LT_df.columns[-1])+1) #an array of all years, which has empirical data 
global_trend_Real = [4.025, 4.074, 4.124, 4.152, 4.227, 4.224, 4.194, 4.173, 4.068, 4.002,4.011, 4.036, 4.071, 4.082, 4.05, 3.968, 4.038, 4.081, 4.088, 4.258, 4.414,  4.528, 4.636, 4.671, 4.762, 4.662, 4.835, 4.975, 5.005, 4.998, 4.981] #Source World Bank. 
years = len(LT_df.columns)


def get_trend(Q_LT,Q_MT,Q_ST):
    global_state_of_co2_emission = 4.025 #as of 1985
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

#get global trends from different policies
global_trend_Greedy = get_trend(LT_Greedy_mean,MT_Greedy_mean,ST_Greedy_mean)
global_trend_Selfish = get_trend(LT_Selfish_Plan_mean,MT_Selfish_Plan_mean,ST_Selfish_Plan_mean)
global_trend_Synergistic = get_trend(LT_Synergistic_mean,MT_Synergistic_mean,ST_Synergistic_mean)

global_trend_Greedy_u = get_trend(LT_Greedy_u_mean,MT_Greedy_u_mean,ST_Greedy_u_mean)
global_trend_Selfish_u = get_trend(LT_Selfish_Plan_u_mean,MT_Selfish_Plan_u_mean,ST_Selfish_Plan_u_mean)
global_trend_Synergistic_u = get_trend(LT_Synergistic_u_mean,MT_Synergistic_u_mean,ST_Synergistic_u_mean)

##report SD and mean
#calculate SD
print("greedy certain std:", global_trend_Greedy_std[-1])
print("selfish certain std:", global_trend_Selfish_std[-1])
print("util certain std:", global_trend_Synergistic_std[-1])

print("greedy uncertain std:", global_trend_Greedy_u_std[-1])
print("selfish ununcertain std:", global_trend_Selfish_u_std[-1])
print("util uncertain std:", global_trend_Synergistic_std[-1])

#calculate mean
print("certain greedy:", global_trend_Greedy[-1])
print("certain selfish:", global_trend_Selfish[-1])
print("certain util:", global_trend_Synergistic[-1])

print("uncertain greedy:", global_trend_Greedy_u[-1])
print("uncertain selfish:", global_trend_Selfish_u[-1])
print("uncertain util:", global_trend_Synergistic_u[-1])

#ALL Policies certain
plt.figure(figsize=(10,8))
plt.plot(years_array,global_trend_Greedy)
plt.plot(years_array,global_trend_Selfish)
plt.plot(years_array,global_trend_Synergistic)
plt.plot(years_array,global_trend_Real)

plt.xlabel('Years', fontsize=17)
plt.ylabel('CO2 Emission', fontsize=17)
plt.legend(['Greedy','Selfish','Utilitarian','Empirical data'], fontsize=15)
plt.title("Certain environment", fontsize=20)
plt.savefig('Reducing-the-Global-Carbon-Footprint-based-on-MARL/plots/environment_certain.png')
#plt.show()

#ALL Policies uncertain
plt.figure(figsize=(10,8))
plt.plot(years_array,global_trend_Greedy_u)
plt.plot(years_array,global_trend_Selfish_u)
plt.plot(years_array,global_trend_Synergistic_u)
plt.plot(years_array,global_trend_Real)

plt.xlabel('Years', fontsize=17)
plt.ylabel('CO2 Emission', fontsize=17)
plt.legend(['Greedy','Selfish','Utilitarian','Empirical data'],fontsize=15)
plt.title("Uncertain environment", fontsize=20)
plt.savefig('Reducing-the-Global-Carbon-Footprint-based-on-MARL/plots/environment_uncertain.png')

#Plotting the mean of uncertainty and certainty
data_certain = np.array([global_trend_Greedy, global_trend_Selfish, global_trend_Synergistic])
all_policies_certain = np.average(data_certain, axis=0)
print("average certain",all_policies_certain[-1])

data_uncertain = np.array([global_trend_Greedy_u, global_trend_Selfish_u, global_trend_Synergistic_u])
all_policies_uncertain = np.average(data_uncertain, axis=0)
print("average uncertain",all_policies_uncertain[-1])

plt.figure(figsize=(10,8))
plt.plot(years_array,all_policies_uncertain)
plt.plot(years_array,all_policies_certain)
plt.plot(years_array,global_trend_Real)
plt.xlabel('Years', fontsize=17)
plt.ylabel('CO2 Emission', fontsize=17)
plt.legend(['Uncertain (average)', 'Certain (average)','Empirical data'], fontsize=15)
plt.title("Comparison of conditions", fontsize=20)
plt.savefig('Reducing-the-Global-Carbon-Footprint-based-on-MARL/plots/comparison_of_conditions.png')

##Certain learning
#plotting average Q-value over epochs
plt.figure(figsize=(10,8))
epochs = range(0,len(Q_LT_average_mean))
plt.axis([0, 40, 0, 0.003])
plt.plot(epochs, Q_LT_average_mean, color='purple')
plt.plot(epochs, Q_MT_average_mean, color='orange')
plt.plot(epochs, Q_ST_average_mean, color='green')
plt.xlabel('Epochs', fontsize=17)
plt.ylabel('Average Q-value', fontsize=17)
plt.legend(['Average Q-values for LT', 'Average Q-values for MT','Average Q-values for ST'], fontsize=15)
plt.title("Learning over epochs, Certain environment", fontsize=20)
plt.savefig('Reducing-the-Global-Carbon-Footprint-based-on-MARL/plots/learning_over_epochs_certain.png')

##Uncertain learning
#plotting average Q-value over epochs
plt.figure(figsize=(10,8))
epochs = range(0,len(Q_LT_average_mean))
plt.plot(epochs, Q_LT_average_u_mean, color = 'purple')
plt.plot(epochs, Q_MT_average_u_mean, color = 'orange')
plt.plot(epochs, Q_ST_average_u_mean, color = 'green')
plt.axis([0, 40, 0, 0.003])
plt.xlabel('Epochs', fontsize=17)
plt.ylabel('Average Q-value', fontsize=17)
plt.legend(['Average Q-values for LT', 'Average Q-values for MT','Average Q-values for ST'], fontsize=15)
plt.title("Learning over epochs, Uncertain environment", fontsize=20)
plt.savefig('Reducing-the-Global-Carbon-Footprint-based-on-MARL/plots/learning_over_epochs_uncertain.png')


#######################T-TEST###########################

#Greedy
Greedy_data_trend = [get_trend(i[0], i[1], i[2]) for i in zip(LT_Greedy_list, MT_Greedy_list, ST_Greedy_list)]
global_state_greedy = np.array([i[-1] for i in Greedy_data_trend])
log_global_state_greedy = np.log(global_state_greedy)


Greedy_data_trend_u = [get_trend(i[0], i[1], i[2]) for i in zip(LT_Greedy_u_list, MT_Greedy_u_list, ST_Greedy_u_list)]
global_state_greedy_u = np.array([i[-1] for i in Greedy_data_trend_u])
log_global_state_greedy_u = np.log(global_state_greedy_u)

t_test_greedy = stats.ttest_ind(log_global_state_greedy, log_global_state_greedy_u)
print("t-test greedy", t_test_greedy)

#Selfish
Selfish_data_trend = [get_trend(i[0], i[1], i[2]) for i in zip(LT_Selfish_Plan_list, MT_Selfish_Plan_list, ST_Selfish_Plan_list)]
global_state_selfish = np.array([i[-1] for i in Selfish_data_trend])
log_global_state_selfish = np.log(global_state_selfish)


Selfish_data_trend_u = [get_trend(i[0], i[1], i[2]) for i in zip(LT_Selfish_Plan_u_list, MT_Selfish_Plan_u_list, ST_Selfish_Plan_u_list)]
global_state_selfish_u = np.array([i[-1] for i in Selfish_data_trend_u])
log_global_state_selfish_u = np.log(global_state_selfish_u)


t_test_selfish = stats.ttest_ind(log_global_state_selfish, log_global_state_selfish_u)
print("t-test selfish", t_test_selfish)

#Synergistic
Synergistic_data_trend = [get_trend(i[0], i[1], i[2]) for i in zip(LT_Synergistic_list, MT_Synergistic_list, ST_Synergistic_list)]
global_state_Synergistic = np.array([i[-1] for i in Synergistic_data_trend])
log_global_state_Synergistic = np.log(global_state_Synergistic)


Synergistic_data_trend_u = [get_trend(i[0], i[1], i[2]) for i in zip(LT_Synergistic_u_list, MT_Synergistic_u_list, ST_Synergistic_u_list)]
global_state_Synergistic_u = np.array([i[-1] for i in Synergistic_data_trend_u])
log_global_state_Synergistic_u = np.log(global_state_Synergistic_u)


t_test_util = stats.ttest_ind(log_global_state_Synergistic, log_global_state_Synergistic_u)
print("t-test util", t_test_util)

#Overall
all_policies_certain = np.array([Greedy_data_trend, Selfish_data_trend, Synergistic_data_trend])
all_policies_certain_mean = np.average(all_policies_certain, axis=0)
all_policies_certain_trend = np.array([i[-1] for i in all_policies_certain_mean])
log_all_policies_certain_trend = np.log(all_policies_certain_trend)

all_policies_uncertain = np.array([Greedy_data_trend_u, Selfish_data_trend_u, Synergistic_data_trend_u])
all_policies_uncertain_mean = np.average(all_policies_uncertain, axis=0)
all_policies_uncertain_trend = np.array([i[-1] for i in all_policies_uncertain_mean])
log_all_policies_uncertain_trend = np.log(all_policies_uncertain_trend)

t_test_certainty = stats.ttest_ind(log_all_policies_certain_trend, log_all_policies_uncertain_trend)
print("ttest certainty", t_test_certainty)

#Histogram: Overall Uncertainty/Certainty 
plt.figure(figsize=(10,8))
plt.hist(log_all_policies_certain_trend)
plt.title("Histogram of Global End-States, Certain environment: Overall", fontsize=20)
plt.savefig('Reducing-the-Global-Carbon-Footprint-based-on-MARL/plots/hist_global_states_certain.png')

plt.figure(figsize=(10,8))
plt.hist(log_all_policies_uncertain_trend)
plt.title("Histogram of Global End-States, Uncertain environment: Overall", fontsize=20)
plt.savefig('Reducing-the-Global-Carbon-Footprint-based-on-MARL/plots/hist_global_states_uncertain.png')

#Histogram: Greed
plt.figure(figsize=(10,8))
plt.hist(log_global_state_greedy)
plt.title("Histogram of Global End-States, Certain environment: Greedy Policy", fontsize=20)
plt.savefig('Reducing-the-Global-Carbon-Footprint-based-on-MARL/plots/hist_global_states_certain_greed.png')

plt.figure(figsize=(10,8))
plt.hist(log_global_state_greedy_u)
plt.title("Histogram of Global End-States, Uncertain environment: Greedy Policy", fontsize=20)
plt.savefig('Reducing-the-Global-Carbon-Footprint-based-on-MARL/plots/hist_global_states_uncertain_greed.png')

#Histogram: Synergistic
plt.figure(figsize=(10,8))
plt.hist(log_global_state_Synergistic)
plt.title("Histogram of Global End-States, Certain environment: Utilitarian Policy", fontsize=20)
plt.savefig('Reducing-the-Global-Carbon-Footprint-based-on-MARL/plots/hist_global_states_certain_utilitarian.png')

plt.figure(figsize=(10,8))
plt.hist(log_global_state_Synergistic_u)
plt.title("Histogram of Global End-States, Uncertain environment: Utilitarian Policy", fontsize=20)
plt.savefig('Reducing-the-Global-Carbon-Footprint-based-on-MARL/plots/hist_global_states_uncertain_utilitarian.png')

#Histogram: Selfish Planning
plt.figure(figsize=(10,8))
plt.hist(log_global_state_selfish)
plt.title("Histogram of Global End-States, Certain environment: Selfish Planning Policy", fontsize=20)
plt.savefig('Reducing-the-Global-Carbon-Footprint-based-on-MARL/plots/hist_global_states_certain_selfish.png')

plt.figure(figsize=(10,8))
plt.hist(log_global_state_selfish_u)
plt.title("Histogram of Global End-States, Uncertain environment: Selfish Planning Policy", fontsize=20)
plt.savefig('Reducing-the-Global-Carbon-Footprint-based-on-MARL/plots/hist_global_states_uncertain_selfish.png')



