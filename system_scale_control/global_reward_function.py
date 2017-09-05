import numpy as np
import matplotlib.pyplot as plt

def reward_fun(outflow, overflow):
    if outflow < 0.1:
        r = 1.0
    else:
        r = -20.0*outflow + 3.0

    if overflow > 0.0:
        r += - 2.0

    return r
