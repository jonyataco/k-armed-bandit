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

if __name__ == "__main__":
    #Importing the dataset using pandas
    dataset = pd.read_csv('Ads_Optimisation.csv')
    num_experiments = 100
    num_steps = len(dataset)

    # Declaring the reward output for group of experiements
    # After each experiement, we will receive a total reward and we will
    # append it to this list
    rewards = []

    # List of epsilon values.
    epsilon_values = [.04, .08, .1, .5, 0]

    # List of expriment rewards
    plot_rewards = []
   
    for epsilon in epsilon_values: 
        for i in range(num_experiments):
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
        plot_rewards.append(reward_average)
    
print(plot_rewards)
x = np.arange(5)
plt.bar(x, plot_rewards)
plt.xticks(x, ('.04 epsilon', '.08 epsilon', '.1 epsilon', '.5 epsilon', '0 epsilon'))
plt.show()
