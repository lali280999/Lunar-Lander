# plot data from mean_total_rewards_new7.npy. Each data point is a mean of previous 10 episodes.
# list of total rewards is a list of lists. Each lists contains the total rewards of the last 10 episodes.
# use list of total rewards to plot variance of total rewards of last 10 episodes for each element of mean_total_rewards.
import numpy as np
import matplotlib.pyplot as plt

mean_total_rewards = np.load("mean_total_rewards_new7.npy")
list_of_total_rewards = np.load("list_of_total_rewards_new7npy.npy")

print(mean_total_rewards)

plt.plot(mean_total_rewards)
plt.xlabel("Episodes")
plt.ylabel("Mean of total rewards of last 10 episodes")
plt.show()

plt.plot(list_of_total_rewards)
plt.xlabel("Episodes")
plt.ylabel("Total rewards of last 10 episodes")
plt.show()

