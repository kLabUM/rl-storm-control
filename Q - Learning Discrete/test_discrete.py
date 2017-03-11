import numpy as np
import scipy as spy
import sys
sys.path.insert(0,'/Users/abhiram/Desktop/adaptive-systems/Test Cases Algorithms')
from pond_single_test import pond

def testfun(pond_class, Q_matrix):
    """Discrete Ponds"""
    height_discrete_value = Q_matrix.shape
    index_discrete = pond_class.max_height/height_discrete_value[0]
    height = pond_class.height/index_discrete
    height = int(np.floor(height))
    return height

test_pond= pond(100.0, 2.0, 1.59)

q=np.zeros(shape=(10,100))

print (testfun(test_pond, q))
