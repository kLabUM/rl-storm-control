import numpy as np


#---------- Reward Functions ----------#

# Reward Function - 1
# Discrete Heights - Control Objectives
# Height levels --> 0.8 to 0.9 with no limit on outflow

def reward12(height_in_pond, valve_position):
    """Reward Function 0.8-0.9; Discharge 0-100"""
    area = 1
    c_discharge = 1
    discharge = np.sqrt(2 * 9.81 * height_in_pond) * valve_position * area * c_discharge
    if height_in_pond >= 1.48 and height_in_pond <= 1.52:
        if  discharge > 0 and discharge <= 4:
            return 10.0
        else:
            return 0.0
    elif height_in_pond < 1.48 and discharge < 0.1:
        return 10.0
    else:
        return 0.0

def reward1(height_in_pond, valve_position):
    """Reward Functions for 0.8-0.9"""
    if height_in_pond <= 0.8 and valve_position == 0:
        return 10.0
    elif height_in_pond > 0.8 and valve_position != 5:
        return 10.0
    else:
        return 0.

def reward2(height_in_pond, valve_position):
    """Reward Functions for 0.8-0.9"""
    if height_in_pond < 0.8 and valve_position == 0:
        return 10.0
    else:
        return 0.0

# CONTINUOUS FUNCTION

def reward3(height_in_pond):
    """Continuous Function"""
    return (-80.0/8.0)*height_in_pond + 10
