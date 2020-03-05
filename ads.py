import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Returns the action that needs to be taken
def get_action(epsi, table):
    # Declaring the action value
    action = None
    # Defining what choices can be made and their respective probabilites 
    choices = ['Exploit', 'Explore']
    exploit_pr = 1 - epsi
    explore_pr = epsi
    choice = np.random.choice(choices, p=[exploit_pr,explore_pr])
    if choice == "Exploit":
        max_value = max(table.values())
        max_keys = [k for k, v in table.items() if v == max_value]
        # Even if the array with the max_keys has only one key,
        # then if that one key is chosen at random it would not be an issue.
        # If there is more than one key that has the same max_value,
        # then this will break the tie by choosing one randomly
        action = random.choice(max_keys)
    else:
        action = random.choice([1,2,3,4,5,6,7,8,9,10])

    return action

# Returns the reward for the given action.
# Looks into the table at the specified row to return a 0 or 1
def bandit(action, table, row):
    return table.values[row, action]

#Importing the dataset using pandas
dataset = pd.read_csv('Ads_Optimisation.csv')
num_experiments = 100
num_steps = len(dataset)

# Declaring the reward output for group of experiements
# After each experiement, we will receive a total reward and we will
# append it to this list
rewards = []

# Performs 5 different experiments and then plots the results of the experiment

for i in range(num_experiments):
    epsilon = .04
    q_table = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    q_table_freq = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    total_reward = 0
    for j in range(num_steps):
        action = get_action(epsilon, q_table)
        #doing minus one since the values are indexed from 0-9 rather than 1-10
        reward = bandit(action - 1, dataset, j)
        #Updating the frequency of the chosen action
        q_table_freq[action] = q_table_freq[action] + 1
        #Updating the value of the chosen action
        q_table[action] = q_table[action] + ((1/q_table_freq[action]) * (reward - q_table[action]))
        total_reward = total_reward + reward
    rewards.append(total_reward)
reward_average = sum(rewards) / len(rewards)

rewards_2 = []
for i in range(num_experiments):
    epsilon = .08
    q_table = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    q_table_freq = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    total_reward = 0
    for j in range(num_steps):
        action = get_action(epsilon, q_table)
        #doing minus one since the values are indexed from 0-9 rather than 1-10
        reward = bandit(action - 1, dataset, j)
        #Updating the frequency of the chosen action
        q_table_freq[action] = q_table_freq[action] + 1
        #Updating the value of the chosen action
        q_table[action] = q_table[action] + ((1/q_table_freq[action]) * (reward - q_table[action]))
        total_reward = total_reward + reward
    rewards_2.append(total_reward)
reward_average_2 = sum(rewards_2) / len(rewards_2)

rewards_3 = []
for i in range(num_experiments):
    epsilon = .1
    q_table = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    q_table_freq = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    total_reward = 0
    for j in range(num_steps):
        action = get_action(epsilon, q_table)
        #doing minus one since the values are indexed from 0-9 rather than 1-10
        reward = bandit(action - 1, dataset, j)
        #Updating the frequency of the chosen action
        q_table_freq[action] = q_table_freq[action] + 1
        #Updating the value of the chosen action
        q_table[action] = q_table[action] + ((1/q_table_freq[action]) * (reward - q_table[action]))
        total_reward = total_reward + reward
    rewards_3.append(total_reward)
reward_average_3 = sum(rewards_3) / len(rewards_3)

rewards_4 = []
for i in range(num_experiments):
    epsilon = .5
    q_table = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    q_table_freq = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    total_reward = 0
    for j in range(num_steps):
        action = get_action(epsilon, q_table)
        #doing minus one since the values are indexed from 0-9 rather than 1-10
        reward = bandit(action - 1, dataset, j)
        #Updating the frequency of the chosen action
        q_table_freq[action] = q_table_freq[action] + 1
        #Updating the value of the chosen action
        q_table[action] = q_table[action] + ((1/q_table_freq[action]) * (reward - q_table[action]))
        total_reward = total_reward + reward
    rewards_4.append(total_reward)
reward_average_4 = sum(rewards_4) / len(rewards_4)

rewards_5 = []
for i in range(num_experiments):
    epsilon = 0
    q_table = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    q_table_freq = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    total_reward = 0
    for j in range(num_steps):
        action = get_action(epsilon, q_table)
        #doing minus one since the values are indexed from 0-9 rather than 1-10
        reward = bandit(action - 1, dataset, j)
        #Updating the frequency of the chosen action
        q_table_freq[action] = q_table_freq[action] + 1
        #Updating the value of the chosen action
        q_table[action] = q_table[action] + ((1/q_table_freq[action]) * (reward - q_table[action]))
        total_reward = total_reward + reward
    rewards_5.append(total_reward)
reward_average_5 = sum(rewards_5) / len(rewards_5)



plot_rewards = [reward_average, reward_average_2, reward_average_3, reward_average_4, reward_average_5]
print(plot_rewards)
x = np.arange(5)
plt.bar(x, plot_rewards)
plt.xticks(x, ('.04 epsilon', '.08 epsilon', '.1 epsilon', '.5 epsilon', '0 epsilon'))
plt.show()
