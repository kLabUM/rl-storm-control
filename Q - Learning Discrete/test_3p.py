import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '/Users/abhiram/Desktop/adaptive-systems/Test Cases Algorithms')
sys.path.insert(0, '/Users/abhiram/Desktop/adaptive-systems/Reward_Functions')
from pond_single_test import pond

# ------------ Discrete Height ----------------

def discrete_height(class_pond, number_of_bins=10):
    """Discrete Heights for the pond"""
    discrete_height_index = class_pond.max_height/float(number_of_bins)
    if class_pond.height <= class_pond.max_height:
        height = class_pond.height/discrete_height_index
    else:
        height = number_of_bins + 1
    height = int(np.floor(height))
    return height

test_pond= pond(100.0, 2.0, 0.0)

for i in range(100):
    test_pond = pond(100.0,2.0,np.random.ran)

print (discrete_height(test_pond))
