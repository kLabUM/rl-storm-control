import numpy as np


def reward_funcion(outflow, depth, flood):
    # Flow part of the reward
    outflow = [1 if i < 0.1 else -1 for i in outflow]
    # Weighting the flow values
    weights = [1, 1, 1, 1, 1]
    # Rewards
    flow_reward = (np.dot(outflow, np.transpose(weights)))
    # Depth rewards
    depth = [-0.5*i if i <= 2.0 else -i**2 + 3 for i in depth]
    weights = [1, 1, 1, 1, 1]
    depth_reward = np.dot(depth, np.transpose(weights))
    # flooding reward
    flood = [-1 if i > 0.0 else 0.0 for i in flood]
    weights = [1, 1, 1, 1, 1]
    flood_reward = np.dot(flood, np.transpose(weights))
    # Sum the total reward
    total_reward = flow_reward + depth_reward + flood_reward
    return total_reward
