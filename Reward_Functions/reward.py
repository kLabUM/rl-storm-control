import numpy as np
import scipy as spy


#---------- Reward Functions ----------#

# Reward Function - 1

# Discrete Heights - Control Objectives

# Height levels --> 0.8 to 0.9 and discharge limits 2

def reward(height_in_pond, valve_position):
    """Reward Function 0.8-0.9; Discharge 0-2"""
    area = 1
    c_discharge = 1
    discharge = np.sqrt(2 * 9.81 * height_in_pond) * valve_position * area * c_discharge
    if height_in_pond >= 0.8 and height_in_pond <= 0.9:
        if  discharge > 0 and discharge < 100:
            return 10.0
        else:
            return 0.0
    elif height_in_pond < 0.8 and discharge < 0.1:
        return 10.0
    else:
        return 0.0



def reward1(height_in_pond, valve_position):
    """Reward Functions for 0.8-0.9"""
    return 0
