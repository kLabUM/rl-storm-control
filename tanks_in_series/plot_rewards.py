import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn

rewards = np.empty([0])
for i in os.listdir("training_rewards/"):
    if i.endswith(".npy"):
        rewards = np.append(rewards, np.load("training_rewards/"+i))

plt.plot(rewards)
plt.title("Training Rewards")
plt.xlabel("Episodes")
plt.show()
