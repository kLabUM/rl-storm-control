# Discharge Control Tools

# Module for widely used functions

import numpy as np
import swmm

# Function: flow_control_outlet

#  Input  : Pond ID, Area of the orifice, Coefficient of discharge and Desired flow
#  Output : % of gate opening and the desired flow is achieved
#  Action : Computes % of gate opening based on pressure head

# TODO Add an indicator if the flow value is not achieved --> need it ?


def flow_control_outlet(tag,area_orifice,cd,target_flow):
    g = 9.81                                    # Acceleration due to gravity
    h = float(swmm.get(tag,swmm.DEPTH,swmm.SI)) # Pressure Head

    if h > 0:
        percent_opening = target_flow/(cd * np.sqrt(2*g*h) * area_orifice)
    else:
        percent_opening = 0

    if percent_opening > 1:                     # Limit the flow to achievable value
        percent_opening = 1

    return percent_opening

# Function: Q_discharge

#  Input  : Pond
#  Output : Qout for the pond
#  Action : Set Q out based on the volume to be discharged

def Q_discharge(pond):
    v_d = pond.volume_discharge
    h = pond.depth_current
    q_lim = pond.Qlim
    if v_d > 0:
        qout = np.minimum(q_lim,np.sqrt(2*9.81*h))
        return qout
    else:
        return 0

# Unit test of Discharge Function
if __name__== '__main__':
    from physical_storm import pond
    a=pond(1,10,0.12,2,12,0,2)
    print a.Qlim
    q_rate = Q_discharge(a)
    print q_rate







