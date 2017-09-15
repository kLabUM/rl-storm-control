import matplotlib.pyplot as plt
import numpy as np
import os

rewards_data = np.empty([0])
for i in os.listdir("./training_rewards"):
    if i.endswith(".npy"):
        i = "./training_rewards/" + i
        rewards_data = np.append(rewards_data, np.load(i))

plt.plot(rewards_data)
plt.tight_layout()
plt.show()
