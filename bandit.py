import random
import numpy as np
import matplotlib.pyplot as plt

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

# Returns the reward for the given action
def bandit(action):
    estimated_means = {1: .25,
                       2: -1,
                       3: 2.45,
                       4: .65,
                       5: 2,
                       6: -1.85,
                       7: -.2,
                       8: -.95,
                       9: 1.1,
                       10: -.2}
    # Getting the mean from the estimated_means table and
    # defining sd as unit variance (1)
    mean = estimated_means[action]
    sd = 1
    reward = np.random.normal(mean,sd)
    return reward

# Going through the bandit algorithm for 2000 experiements with 1000 time steps
num_experiments = 2000
num_steps = 1000

# Declaring the reward output for group of experiements with different e values
rewards = np.zeros((num_steps,))
rewards_2 = np.zeros((num_steps,))
rewards_3 = np.zeros((num_steps,))

# epsilon = .1
for i in range (num_experiments):
    epsilon = .1
    q_table = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    q_table_freq = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    experiment_rewards = []
    for j in range(1, num_steps + 1):
        action = get_action(epsilon, q_table)
        reward = bandit(action)
        experiment_rewards.append(reward)
        #Updating the frequency of the chosen action
        q_table_freq[action] = q_table_freq[action] + 1;
        #Updating the value of the chosen action
        q_table[action] = q_table[action] + ((1/q_table_freq[action]) * (reward - q_table[action]))
    rewards += experiment_rewards

reward_average = rewards / np.float(num_experiments)
plt.plot(reward_average, "k.")

# epsilon = .03
for i in range (num_experiments):
    epsilon = .03
    q_table = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    q_table_freq = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    experiment_rewards = []
    for j in range(1, num_steps + 1):
        action = get_action(epsilon, q_table)
        reward = bandit(action)
        experiment_rewards.append(reward)
        #Updating the frequency of the chosen action
        q_table_freq[action] = q_table_freq[action] + 1;
        #Updating the value of the chosen action
        q_table[action] = q_table[action] + ((1/q_table_freq[action]) * (reward - q_table[action]))
    rewards_2 += experiment_rewards

reward_average = rewards_2 / np.float(num_experiments)
plt.plot(reward_average, "r.")

# epsilon = 0 or otherwise known as greedy
for i in range (num_experiments):
    epsilon = 0
    q_table = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    q_table_freq = dict.fromkeys([1,2,3,4,5,6,7,8,9,10], 0)
    experiment_rewards = []
    for j in range(1, num_steps + 1):
        action = get_action(epsilon, q_table)
        reward = bandit(action)
        experiment_rewards.append(reward)
        #Updating the frequency of the chosen action
        q_table_freq[action] = q_table_freq[action] + 1;
        #Updating the value of the chosen action
        q_table[action] = q_table[action] + ((1/q_table_freq[action]) * (reward - q_table[action]))
    rewards_3 += experiment_rewards

reward_average = rewards_3 / np.float(num_experiments)
plt.plot(reward_average, "g.")
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.grid()
plt.xlim([1, num_steps])
plt.show()
